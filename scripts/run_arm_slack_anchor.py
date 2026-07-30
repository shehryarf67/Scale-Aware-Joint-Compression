"""Measure whether solver slack differs between the arms (validity_threats.md).

[F-20](../docs/findings_log.md) found the reconstruction sweep captures only 0.6409 of the achievable
objective gain. The joint-versus-sequential difference this study measures has been about 1 pp of
retention. So the question that decides whether that number means anything:

    Is the sweep's slack the same under both arms' masks?

There is no reason to assume so. The arms produce different masks -- that is the mechanism under study
-- different masks give different ``H[S,S]`` conditioning, and conditioning is what determines how much
a one-pass error-compensated sweep recovers.

The sharp test is not the efficiency gap but a **ranking** check:

* our sweep says one mask gives a lower objective;
* the exact optimum says which mask *actually* does.

Where those disagree, that row's contribution to the measured joint gain has the wrong sign relative to
the masks' true quality. ``attributable_joint_benefit`` aggregates it: 1.0 means the sweep's joint
advantage is entirely real, 0.0 means it is entirely solver artefact.

The two masks follow the arms as specified. Sequential P->Q scores saliency on the **dense** weights,
because at that point no quantiser exists. Joint scores on the **fake-quantised** weights, per
decision **D3** -- ``S_ij = |Q_b(W_ij)| * ||X_j||_2`` -- so the mask is chosen against the grid it will
live on. Reconstruction itself runs pruning-only for both, which isolates the mask's effect on solver
efficiency from quantisation.

    python scripts/run_arm_slack_anchor.py --config configs/experiments/screening.yaml --bits 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "src") not in sys.path:  # pragma: no cover - script bootstrap
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from scale_aware_compression.logging_utils import configure_logging, get_logger  # noqa: E402

LOGGER = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path, help="Experiment config to match.")
    parser.add_argument("--sparsity", type=float, default=0.3, help="Target sparsity.")
    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        help="Bit width the joint mask is scored under. Defaults to 4, where F-05 found the mask "
        "mechanism is live; at W8 it is nearly inert and the two masks barely differ.",
    )
    parser.add_argument("--modules", type=int, default=12, help="Modules to sample.")
    parser.add_argument("--rows", type=int, default=8, help="Rows to solve exactly per module.")
    parser.add_argument("--output", type=Path, default=None, help="Where to write the JSON report.")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and exit.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the arm-slack measurement.

    Returns:
        0 on success, 2 on a configuration error. There is deliberately no failure exit code: this
        anchor measures the size of a confound rather than asserting a threshold.
    """
    arguments = build_parser().parse_args(argv)

    from scale_aware_compression.config import ConfigError, load_config

    try:
        config = load_config(arguments.config)
    except ConfigError as error:
        print(f"configuration error: {error}", file=sys.stderr)
        return 2

    configure_logging(config.runtime.log_level)
    print("Arm-dependent solver slack")
    print(f"  model    : {config.model.name} @ {config.model.revision}")
    print(f"  sparsity : {arguments.sparsity}   joint mask scored at W{arguments.bits}")
    print(f"  sampling : {arguments.modules} modules x {arguments.rows} rows")
    if arguments.dry_run:
        print("\n--dry-run: nothing loaded.")
        return 0

    import torch

    from scale_aware_compression.anchors import ArmRowComparison, ArmSlackReport, compare_row
    from scale_aware_compression.compression.activations import ActivationStatistics
    from scale_aware_compression.compression.masks import build_mask_from_scores
    from scale_aware_compression.compression.pruning import activation_weighted_saliency
    from scale_aware_compression.compression.quantisation import fake_quantise
    from scale_aware_compression.compression.reconstruct import sweep_reconstruct
    from scale_aware_compression.data.calibration import load_calibration_set
    from scale_aware_compression.models.adapters import select_compressible_modules
    from scale_aware_compression.models.loader import load_model_and_tokenizer

    sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))
    from run_reconstruction_anchor import _stratified_sample

    loaded = load_model_and_tokenizer(config.model)
    calibration = load_calibration_set(config.data, loaded.tokenizer)
    batches = list(calibration.loader)
    device = loaded.device.value

    all_names = list(
        select_compressible_modules(
            loaded.model,
            target_modules=config.compression.pruning.target_modules,
            exclude_patterns=config.compression.pruning.exclude_patterns,
        ).names
    )
    names = _stratified_sample(all_names, arguments.modules)
    print(f"\n  sampled : {len(names)} of {len(all_names)} modules")
    print(f"  calib   : {len(batches)} batches, {calibration.summary.token_fingerprint}")

    statistics: dict[str, ActivationStatistics] = {}
    handles = []

    def make_hook(name: str):
        def hook(_module, inputs):
            if inputs:
                statistics[name].update(inputs[0])

        return hook

    for name in names:
        module = loaded.model.get_submodule(name)
        statistics[name] = ActivationStatistics(module.in_features)
        handles.append(module.register_forward_pre_hook(make_hook(name)))
    try:
        loaded.model.eval()
        with torch.no_grad():
            for batch in batches:
                inputs = {
                    key: value.to(device)
                    for key, value in batch.items()
                    if key in {"input_ids", "attention_mask"} and hasattr(value, "to")
                }
                loaded.model(**inputs)
    finally:
        for handle in handles:
            handle.remove()

    report = ArmSlackReport(bits=int(arguments.bits), target_sparsity=float(arguments.sparsity))
    report.notes.append(
        "Sequential mask scored on dense weights; joint mask scored on fake-quantised weights per D3. "
        "Reconstruction is pruning-only for both, isolating the mask's effect on solver efficiency."
    )
    report.notes.append(
        "Efficiency is measured against each mask's OWN exact optimum, so it asks how well the solver "
        "does on that mask rather than which mask is better."
    )

    generator = torch.Generator().manual_seed(20260730)
    identical_masks = 0

    for name in names:
        module = loaded.model.get_submodule(name)
        weight = module.weight.detach().to("cpu", torch.float32)
        gram = statistics[name].gram().detach().to("cpu", torch.float32)
        norms = statistics[name].column_norms().detach().to("cpu")

        sequential_mask = build_mask_from_scores(
            activation_weighted_saliency(weight, norms), sparsity=float(arguments.sparsity)
        )
        quantised = fake_quantise(weight, bits=int(arguments.bits))
        joint_mask = build_mask_from_scores(
            activation_weighted_saliency(quantised, norms), sparsity=float(arguments.sparsity)
        )
        divergence = float((sequential_mask != joint_mask).float().mean())
        if divergence == 0.0:
            identical_masks += 1

        sequential_outcome = sweep_reconstruct(
            gram,
            weight,
            sequential_mask,
            damping=config.compression.reconstruction.damping,
            bits=None,
            block_size=config.compression.reconstruction.block_size,
            activation_order=config.compression.reconstruction.activation_order,
        )
        joint_outcome = sweep_reconstruct(
            gram,
            weight,
            joint_mask,
            damping=config.compression.reconstruction.damping,
            bits=None,
            block_size=config.compression.reconstruction.block_size,
            activation_order=config.compression.reconstruction.activation_order,
        )

        count = min(arguments.rows, weight.shape[0])
        rows = sorted(torch.randperm(weight.shape[0], generator=generator)[:count].tolist())
        comparisons = []
        for row in rows:
            comparisons.append(
                ArmRowComparison(
                    row=row,
                    sequential=compare_row(
                        row, gram, weight[row], sequential_outcome.weight[row], sequential_mask[row]
                    ),
                    joint=compare_row(
                        row, gram, weight[row], joint_outcome.weight[row], joint_mask[row]
                    ),
                )
            )
        report.module_rows[name] = comparisons
        gap = sum(c.efficiency_gap for c in comparisons) / len(comparisons)
        print(f"    {name}: mask divergence {divergence:.4%}, efficiency gap {gap:+.4f}")

    if identical_masks:
        report.notes.append(
            f"{identical_masks} of {len(names)} modules had IDENTICAL masks under both arms, so those "
            "contribute exactly zero to any measured difference. At W8 this is expected (F-05)."
        )

    print()
    for line in report.summary_lines():
        print(line)

    destination = arguments.output or (
        REPOSITORY_ROOT
        / "outputs"
        / "anchors"
        / f"arm_slack_{config.model.name}_w{arguments.bits}.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = report.to_dict()
    payload["model"] = config.model.name
    payload["revision"] = config.model.revision
    payload["calibration_fingerprint"] = calibration.summary.token_fingerprint
    payload["modules_with_identical_masks"] = identical_masks
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n  report written to {destination}")
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
