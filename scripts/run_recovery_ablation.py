#!/usr/bin/env python
r"""POST-HOC EXPLORATORY ABLATION: does a short global recovery phase move the joint gain?

NOT CONFIRMATORY. Writes to its own tree and cannot alter F-37, F-41, the confirmatory manifest or
any frozen decision. Validation split only.

THE QUESTION
------------
F-38 found the local objective improving where the global one does not. This gives both arms the
same short end-to-end recovery phase -- frozen pruning mask, live W4 fake quantisation, causal-LM
loss -- and reports the joint gain before and after.

WHY THIS IS A SEPARATE SCRIPT AND NOT A SWEEP ARM
-------------------------------------------------
The arm pipeline is prepare -> apply -> recover -> convert, and `convert` packs weights into
`PackedLinear`, which holds integer codes and cannot be trained. Recovery has to happen between
`apply` and `convert`, so the stages are driven by hand here. `LayerwiseArm.recover` stays the
deliberate no-op it is -- its docstring already reserved this hook for "the optional short
fine-tune ablation", and leaving it inert is what keeps every existing config unchanged.

FAIRNESS IS ENFORCED, NOT ASSUMED
---------------------------------
Recovery batches are materialised ONCE per replicate, before the arm loop, and both arms consume
the same list in the same order. Budgets are compared with `assert_budgets_match`. Either check
fails the run rather than producing a confounded number.

    python scripts/run_recovery_ablation.py --config configs/experiments/recovery_ablation_160m_w4.yaml --smoke
    python scripts/run_recovery_ablation.py --config configs/experiments/recovery_ablation_160m_w4.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "src") not in sys.path:  # pragma: no cover - script bootstrap
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from scale_aware_compression.config import ConfigError, load_config  # noqa: E402
from scale_aware_compression.logging_utils import configure_logging, get_logger  # noqa: E402

LOGGER = get_logger(__name__)

ARMS = ("sequential", "joint")
"""Only these two. The ablation compares the frozen baseline against joint, nothing else."""


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--replicates",
        type=int,
        default=None,
        help="Override the replicate count. Used by --smoke.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "One replicate, a handful of recovery steps, and a small evaluation window. Proves the "
            "path end to end and times it, without spending the real budget"
        ),
    )
    parser.add_argument("--log-level", default=None)
    return parser


def _retention(dense_ppl: float, ppl: float) -> float:
    """Perplexity retention as a percentage, the study's primary quality metric."""
    return 100.0 * dense_ppl / ppl


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0915 - a linear experiment script
    """Run the ablation.

    Returns:
        0 on success, 2 on a configuration error.
    """
    arguments = build_parser().parse_args(argv)
    try:
        config = load_config(arguments.config)
    except ConfigError as error:
        configure_logging(arguments.log_level or "INFO")
        LOGGER.error("Invalid configuration: %s", error)
        return 2
    configure_logging(arguments.log_level or config.runtime.log_level)

    if not config.compression.recovery.end_to_end:
        LOGGER.error(
            "compression.recovery.end_to_end is false, so this config would run no recovery. "
            "Refusing rather than silently producing a before/after pair that is identical."
        )
        return 2
    if config.data.eval_split != "validation":
        LOGGER.error(
            "This ablation is post-hoc and exploratory; it must evaluate on validation, not %r.",
            config.data.eval_split,
        )
        return 2

    import torch

    from scale_aware_compression.compression import COMPRESSOR_REGISTRY
    from scale_aware_compression.constants import CompressionMethod
    from scale_aware_compression.data.calibration import load_calibration_set
    from scale_aware_compression.data.loaders import build_evaluation_dataloader
    from scale_aware_compression.evaluation.perplexity import compute_perplexity
    from scale_aware_compression.models.loader import load_model_and_tokenizer
    from scale_aware_compression.training.end_to_end import (
        assert_budgets_match,
        bake_recovery_modules,
        install_recovery_modules,
        mask_sparsity,
        run_end_to_end_recovery,
    )
    from scale_aware_compression.training.recovery import recovery_step_budget

    recovery = config.compression.recovery
    replicates = arguments.replicates or config.sweep.replicates
    if arguments.smoke:
        replicates = 1
        recovery.max_steps = 4
        config.evaluation.max_samples = 16
        LOGGER.warning(
            "SMOKE: 1 replicate, %d recovery steps, %d evaluation sequences. Numbers from this "
            "mode are meaningless and are written with smoke=true in the record.",
            recovery.max_steps,
            config.evaluation.max_samples,
        )

    output_dir = Path(config.runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    LOGGER.info("Recovery device: %s", device)

    records: list[dict[str, Any]] = []
    for replicate in range(replicates):
        LOGGER.info("=" * 78)
        LOGGER.info("REPLICATE %d of %d", replicate, replicates)
        LOGGER.info("=" * 78)

        # Recovery batches ONCE per replicate, before the arm loop. Both arms consume this same
        # list in this same order: that is the fairness requirement, and materialising it here is
        # what makes it structurally true rather than asserted.
        # The replicate selects the calibration draw, exactly as the sweep does.
        config.data.calibration_replicate = replicate
        loaded = load_model_and_tokenizer(config.model)
        tokenizer = loaded.tokenizer

        calibration = load_calibration_set(config.data, tokenizer)
        calibration_fingerprint = calibration.summary.token_fingerprint

        # RECOVERY DATA IS DISJOINT FROM CALIBRATION, deliberately.
        #
        # The arms are FITTED on the calibration sequences, so recovering on those same tokens
        # would partly be re-fitting on seen data: the absolute improvements would be optimistic
        # and the comparison would measure memorisation as much as recovery. It would still be
        # fair -- both arms see identical data -- but it would answer a weaker question.
        #
        # Drawn from the same train split, excluding every calibration index, with a fixed
        # generator so the two arms and any re-run get byte-identical batches in identical order.
        used = {int(index) for index in calibration.indices}
        available = [index for index in range(len(calibration.dataset)) if index not in used]
        generator = torch.Generator().manual_seed(recovery.seed + replicate)
        order = torch.randperm(len(available), generator=generator).tolist()
        needed = budget_sequences = (
            recovery.max_steps * recovery.batch_size * recovery.gradient_accumulation_steps
        )
        chosen = [available[order[i % len(order)]] for i in range(needed)]
        recovery_batches = []
        micro = recovery.batch_size
        for start in range(0, len(chosen), micro):
            block = chosen[start : start + micro]
            if len(block) < micro:
                break
            recovery_batches.append(
                torch.stack([calibration.dataset[index]["input_ids"] for index in block])
            )
        LOGGER.info(
            "Recovery slice: %d sequence(s) from train, DISJOINT from the %d calibration "
            "sequences (overlap 0 by construction)",
            len(chosen),
            len(used),
        )
        del budget_sequences
        LOGGER.info(
            "Recovery data: %d batch(es) of shape %s, fingerprint %s -- shared by both arms",
            len(recovery_batches),
            tuple(recovery_batches[0].shape),
            calibration_fingerprint,
        )

        evaluation_loader, evaluation_summary = build_evaluation_dataloader(
            config.data, tokenizer, max_samples=config.evaluation.max_samples
        )
        dataset_fingerprint = evaluation_summary.fingerprint

        # Dense reference for retention, once per replicate on the same window.
        dense_model = loaded.model
        dense_model.to("cpu").eval()
        dense = compute_perplexity(
            dense_model,
            evaluation_loader,
            config.evaluation,
            dataset_fingerprint=dataset_fingerprint,
        )
        del dense_model, loaded
        LOGGER.info("Dense reference perplexity: %.4f", dense.perplexity)

        budgets = {}
        for arm_name in ARMS:
            started = time.perf_counter()
            method = CompressionMethod(arm_name)
            arm_class = COMPRESSOR_REGISTRY[method]
            # The arm refuses a config whose method is a different arm, because the budget is
            # derived from the config and a mismatched pair would compress to the wrong target.
            # Set it per arm; every other field -- sparsity, bits, solver, local steps -- is shared,
            # which is exactly what §3.11 requires the two arms to hold identical.
            config.compression.method = method
            arm = arm_class(config)
            arm.set_calibration(list(recovery_batches), fingerprint=calibration_fingerprint)

            arm_loaded = load_model_and_tokenizer(config.model)
            model = arm_loaded.model
            model = arm.prepare(model)
            model = arm.apply(model)

            masks = dict(arm.report.masks_by_module)
            if not masks:
                LOGGER.error("No masks retained; is compression.recovery.end_to_end set?")
                return 2

            model.to("cpu").eval()
            before = compute_perplexity(
                model, evaluation_loader, config.evaluation, dataset_fingerprint=dataset_fingerprint
            )

            installed = install_recovery_modules(
                model,
                masks,
                bits=config.compression.quantisation.bits,
                granularity=config.compression.quantisation.granularity,
                group_size=config.compression.quantisation.group_size,
            )
            sparsity_before = mask_sparsity(installed)
            budget = recovery_step_budget(
                recovery,
                steps_per_epoch=max(1, len(recovery_batches)),
                sequence_length=recovery.sequence_length or config.data.sequence_length,
            )
            budgets[arm_name] = budget

            outcome = run_end_to_end_recovery(
                model,
                recovery_batches,
                installed,
                config=config,
                budget=budget,
                device=device,
            )
            bake_recovery_modules(model, installed)

            model.to("cpu").eval()
            after = compute_perplexity(
                model, evaluation_loader, config.evaluation, dataset_fingerprint=dataset_fingerprint
            )

            statistics = arm.report
            record = {
                "experiment_id": f"{config.experiment.id}__{arm_name}_rep{replicate}",
                "kind": "recovery-ablation",
                "confirmatory": False,
                "smoke": bool(arguments.smoke),
                "tags": list(config.experiment.tags),
                "model_name": config.model.name,
                "model_revision": config.model.revision,
                "arm": arm_name,
                "budget_label": config.compression.budget_label,
                "calibration_replicate": replicate,
                "calibration_fingerprint": calibration_fingerprint,
                "eval_split": config.data.eval_split,
                "dataset_fingerprint": dataset_fingerprint,
                "target_sparsity": config.compression.effective_sparsity,
                "quantisation_bits": config.compression.quantisation.bits,
                "quantisation_granularity": config.compression.quantisation.granularity.value,
                "dense_perplexity": dense.perplexity,
                "before": {
                    "perplexity": before.perplexity,
                    "total_nll": before.total_nll,
                    "retention": _retention(dense.perplexity, before.perplexity),
                    "mask_sparsity": sparsity_before,
                },
                "after": {
                    "perplexity": after.perplexity,
                    "total_nll": after.total_nll,
                    "retention": _retention(dense.perplexity, after.perplexity),
                    "mask_sparsity": outcome.mask_sparsity_after,
                },
                "recovery": outcome.to_dict(),
                "layerwise": {
                    "num_layers": len(statistics.layers),
                    "total_local_steps": statistics.total_local_steps,
                    "mask_sparsity": (
                        sum(layer.mask_sparsity for layer in statistics.layers)
                        / max(1, len(statistics.layers))
                    ),
                },
                "duration_seconds": time.perf_counter() - started,
            }
            record["improvement_pp"] = record["after"]["retention"] - record["before"]["retention"]
            records.append(record)

            path = output_dir / f"{record['experiment_id']}.json"
            path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
            LOGGER.info(
                "%s rep%d: retention %.4f -> %.4f (%+.4f pp) in %.1f s",
                arm_name,
                replicate,
                record["before"]["retention"],
                record["after"]["retention"],
                record["improvement_pp"],
                record["duration_seconds"],
            )
            del model, arm_loaded

        # Fairness gate, after both arms have run and before any gain is reported.
        assert_budgets_match(budgets["sequential"], budgets["joint"])

        pair = [r for r in records if r["calibration_replicate"] == replicate]
        by_arm = {r["arm"]: r for r in pair}
        if len(by_arm) == len(ARMS):
            before_gain = (
                by_arm["joint"]["before"]["retention"] - by_arm["sequential"]["before"]["retention"]
            )
            after_gain = (
                by_arm["joint"]["after"]["retention"] - by_arm["sequential"]["after"]["retention"]
            )
            LOGGER.info(
                "REPLICATE %d joint gain: %+.4f pp before recovery -> %+.4f pp after (%+.4f)",
                replicate,
                before_gain,
                after_gain,
                after_gain - before_gain,
            )

    summary_path = output_dir / f"{config.experiment.id}__summary.json"
    summary_path.write_text(
        json.dumps({"records": records, "arms": list(ARMS)}, indent=2, default=str),
        encoding="utf-8",
    )
    LOGGER.info("Wrote %d record(s) and %s", len(records), summary_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
