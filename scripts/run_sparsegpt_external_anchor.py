r"""Compare our pruning against the reference SparseGPT implementation (A1 §5.5b2).

The one check that speaks to **absolute quality** rather than internal consistency. [F-19](../docs/findings_log.md#f-19)
confirmed our mask matches an independent Wanda implementation exactly, and
[F-20](../docs/findings_log.md#f-20) confirmed our sweep never beats and never falls below the provable
optimum of its objective. Neither says whether ~57% retention is *competitive*. Only a real external
implementation does.

**What is matched, and what is deliberately not.**

Matched, so the comparison isolates the algorithm:

* the same model at the same pinned revision, loaded through our loader;
* the same calibration sequences, drawn once from our pipeline and handed to both;
* the same target sparsity and module coverage;
* the same evaluation, run by **our** evaluator on **our** window, on both compressed models.

Not matched, because it is the independent variable:

* the compression algorithm. Theirs is `SparseGPT.fasterprune`, used **unmodified** from
  `IST-DASLab/sparsegpt`. Only the model plumbing is ours, because their repo ships drivers for OPT,
  BLOOM and LLaMA but not GPT-NeoX.

**Why their `H` scaling does not matter.** They accumulate `H = (2/n) * sum(X X^T)` with a running
rescale; we accumulate raw `X^T X`. A positive constant factor does not move the argmin of a quadratic,
so the solutions are comparable even though the reported loss magnitudes are not.

**Reading the result.** Per A1 §5.5, the thresholds are *debugging alarms, not acceptance criteria* --
the algorithms are genuinely different and bit-for-bit agreement is not expected. A gap beyond roughly
3 retention points, or 5-10% relative perplexity, is the signal to investigate.

Requires the reference checkout:

    git clone --depth 1 https://github.com/IST-DASLab/sparsegpt.git

    python scripts/run_sparsegpt_external_anchor.py --config configs/experiments/screening.yaml \\
        --sparsegpt-path c:/Users/shehr/sajc_external/sparsegpt --sparsity 0.3
"""

from __future__ import annotations

import argparse
import contextlib
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
    parser.add_argument(
        "--sparsegpt-path",
        required=True,
        type=Path,
        help="Path to an IST-DASLab/sparsegpt checkout. Kept outside this repository.",
    )
    parser.add_argument("--sparsity", type=float, default=0.3, help="Target sparsity for both.")
    parser.add_argument("--output", type=Path, default=None, help="Where to write the JSON report.")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and exit.")
    return parser


def _load_reference(path: Path):
    """Import the reference ``SparseGPT`` class from an external checkout.

    Args:
        path: The checkout directory.

    Returns:
        The ``SparseGPT`` class, unmodified.

    Raises:
        SystemExit: If the checkout is missing or does not import.
    """
    resolved = path.resolve()
    if not (resolved / "sparsegpt.py").is_file():
        raise SystemExit(
            f"No sparsegpt.py in {resolved}. Clone it first:\n"
            "  git clone --depth 1 https://github.com/IST-DASLab/sparsegpt.git"
        )
    sys.path.insert(0, str(resolved))
    try:
        from sparsegpt import SparseGPT
    except Exception as error:  # noqa: BLE001 - report whatever their import raised
        raise SystemExit(
            f"Reference SparseGPT failed to import: {type(error).__name__}: {error}"
        ) from error
    return SparseGPT


def main(argv: list[str] | None = None) -> int:
    """Run the external comparison.

    Returns:
        0 on success, 1 when the gap exceeds the A1 alarm thresholds, 2 on a configuration error.
    """
    arguments = build_parser().parse_args(argv)

    from scale_aware_compression.config import ConfigError, load_config

    try:
        config = load_config(arguments.config)
    except ConfigError as error:
        print(f"configuration error: {error}", file=sys.stderr)
        return 2

    configure_logging(config.runtime.log_level)
    print("External SparseGPT comparison (A1 §5.5b2)")
    print(f"  model     : {config.model.name} @ {config.model.revision}")
    print(f"  sparsity  : {arguments.sparsity} (both arms)")
    print(f"  reference : {arguments.sparsegpt_path}")
    print("  matched   : model, revision, calibration, coverage, evaluation")
    print("  differs   : the compression algorithm only")
    if arguments.dry_run:
        print("\n--dry-run: nothing loaded.")
        return 0

    sparsegpt_class = _load_reference(arguments.sparsegpt_path)

    import torch

    from scale_aware_compression.compression.masks import build_mask_from_scores
    from scale_aware_compression.compression.pruning import activation_weighted_saliency
    from scale_aware_compression.compression.reconstruct import sweep_reconstruct
    from scale_aware_compression.data.calibration import load_calibration_set
    from scale_aware_compression.data.loaders import build_evaluation_dataloader
    from scale_aware_compression.evaluation.perplexity import compute_perplexity
    from scale_aware_compression.models.adapters import (
        get_decoder_blocks,
        get_linear_modules,
        select_compressible_modules,
    )
    from scale_aware_compression.models.loader import load_model_and_tokenizer

    sparsity = float(arguments.sparsity)
    results: dict[str, dict] = {}

    # --- dense reference -------------------------------------------------------------------------
    loaded = load_model_and_tokenizer(config.model)
    calibration = load_calibration_set(config.data, loaded.tokenizer)
    batches = list(calibration.loader)
    device = loaded.device.value
    print(f"\n  calibration: {len(batches)} batches, {calibration.summary.token_fingerprint}")

    # Built once and reused for every arm, so all three perplexities come from the identical token
    # stream in the identical order -- which is the whole point of a matched comparison.
    eval_loader, eval_summary = build_evaluation_dataloader(config.data, loaded.tokenizer)
    print(f"  evaluation : {len(eval_loader)} batches, {eval_summary.fingerprint}")

    def score(model) -> float:
        result = compute_perplexity(
            model,
            eval_loader,
            config.evaluation,
            dataset_fingerprint=eval_summary.fingerprint,
        )
        return float(result.perplexity)

    dense_ppl = score(loaded.model)
    results["dense"] = {"perplexity": dense_ppl}
    print(f"  dense perplexity: {dense_ppl:.4f}")

    def capture_gram_and_norms(model, names: list[str]):
        """Accumulate our Gram statistics for the named modules over the calibration set."""
        from scale_aware_compression.compression.activations import ActivationStatistics

        statistics = {
            name: ActivationStatistics(model.get_submodule(name).in_features) for name in names
        }
        handles = []

        def make_hook(name):
            def hook(_module, inputs):
                if inputs:
                    statistics[name].update(inputs[0])

            return hook

        for name in names:
            handles.append(model.get_submodule(name).register_forward_pre_hook(make_hook(name)))
        try:
            model.eval()
            with torch.no_grad():
                for batch in batches:
                    payload = {
                        key: value.to(device)
                        for key, value in batch.items()
                        if key in {"input_ids", "attention_mask"} and hasattr(value, "to")
                    }
                    model(**payload)
        finally:
            for handle in handles:
                handle.remove()
        return statistics

    # --- ours ------------------------------------------------------------------------------------
    # Reloaded so each arm starts from identical dense weights rather than inheriting the other's.
    ours_loaded = load_model_and_tokenizer(config.model)
    ours_names = list(
        select_compressible_modules(
            ours_loaded.model,
            target_modules=config.compression.pruning.target_modules,
            exclude_patterns=config.compression.pruning.exclude_patterns,
        ).names
    )
    print(f"\n  ours: compressing {len(ours_names)} modules")
    statistics = capture_gram_and_norms(ours_loaded.model, ours_names)
    with torch.no_grad():
        for name in ours_names:
            module = ours_loaded.model.get_submodule(name)
            weight = module.weight.detach().to("cpu", torch.float32)
            gram = statistics[name].gram().detach().to("cpu", torch.float32)
            norms = statistics[name].column_norms().detach().to("cpu")
            mask = build_mask_from_scores(
                activation_weighted_saliency(weight, norms), sparsity=sparsity
            )
            outcome = sweep_reconstruct(
                gram,
                weight,
                mask,
                damping=config.compression.reconstruction.damping,
                bits=None,
                block_size=config.compression.reconstruction.block_size,
                activation_order=config.compression.reconstruction.activation_order,
            )
            module.weight.data.copy_(
                outcome.weight.to(module.weight.dtype).to(module.weight.device)
            )

    ours_zeros = _zero_fraction(ours_loaded.model, ours_names)
    ours_ppl = score(ours_loaded.model)
    results["ours"] = {
        "perplexity": ours_ppl,
        "measured_sparsity": ours_zeros,
        "retention": dense_ppl / ours_ppl,
    }
    print(f"  ours perplexity: {ours_ppl:.4f} (sparsity {ours_zeros:.4f})")
    del ours_loaded, statistics

    # --- reference SparseGPT ---------------------------------------------------------------------
    reference_loaded = load_model_and_tokenizer(config.model)
    reference_names = list(
        select_compressible_modules(
            reference_loaded.model,
            target_modules=config.compression.pruning.target_modules,
            exclude_patterns=config.compression.pruning.exclude_patterns,
        ).names
    )
    print(f"\n  reference: compressing {len(reference_names)} modules with SparseGPT.fasterprune")
    # Identity-matched rather than name-matched: the same coverage rule picks the modules, and the
    # driver then recognises them inside each block by object identity. No name-prefix reconstruction,
    # which is where an adapter for a new architecture usually goes wrong.
    target_modules = get_linear_modules(reference_loaded.model, reference_names)
    _run_reference_sparsegpt(
        sparsegpt_class,
        reference_loaded.model,
        get_decoder_blocks(reference_loaded.model),
        batches,
        device=device,
        sparsity=sparsity,
        target_ids={id(module) for module in target_modules.values()},
    )

    reference_zeros = _zero_fraction(reference_loaded.model, reference_names)
    reference_ppl = score(reference_loaded.model)
    results["reference_sparsegpt"] = {
        "perplexity": reference_ppl,
        "measured_sparsity": reference_zeros,
        "retention": dense_ppl / reference_ppl,
    }
    print(f"  reference perplexity: {reference_ppl:.4f} (sparsity {reference_zeros:.4f})")

    # --- verdict ---------------------------------------------------------------------------------
    retention_gap_pp = 100.0 * (
        results["ours"]["retention"] - results["reference_sparsegpt"]["retention"]
    )
    relative_ppl_gap = (
        results["ours"]["perplexity"] - results["reference_sparsegpt"]["perplexity"]
    ) / results["reference_sparsegpt"]["perplexity"]
    within_alarms = abs(retention_gap_pp) <= 3.0 and abs(relative_ppl_gap) <= 0.10

    print(f"\nExternal comparison at {sparsity:.0%} sparsity")
    print(f"  dense                    : {dense_ppl:.4f}")
    print(
        f"  ours                     : {results['ours']['perplexity']:.4f}  "
        f"retention {results['ours']['retention']:.2%}"
    )
    print(
        f"  reference SparseGPT      : {results['reference_sparsegpt']['perplexity']:.4f}  "
        f"retention {results['reference_sparsegpt']['retention']:.2%}"
    )
    print(f"  retention gap            : {retention_gap_pp:+.2f} pp (alarm beyond +/-3.00)")
    print(f"  relative perplexity gap  : {relative_ppl_gap:+.2%} (alarm beyond +/-10%)")
    print(f"  VERDICT                  : {'WITHIN ALARMS' if within_alarms else 'INVESTIGATE'}")
    print("\n  Thresholds are debugging alarms, not acceptance criteria (A1 §5.5).")

    payload = {
        "anchor": "external_sparsegpt_comparison",
        "model": config.model.name,
        "revision": config.model.revision,
        "target_sparsity": sparsity,
        "calibration_fingerprint": calibration.summary.token_fingerprint,
        "results": results,
        "retention_gap_pp": retention_gap_pp,
        "relative_perplexity_gap": relative_ppl_gap,
        "within_alarms": within_alarms,
        "reference": "IST-DASLab/sparsegpt SparseGPT.fasterprune, unmodified",
    }
    destination = arguments.output or (
        REPOSITORY_ROOT / "outputs" / "anchors" / f"external_sparsegpt_{config.model.name}.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n  report written to {destination}")
    return 0 if within_alarms else 1


def _zero_fraction(model, names: list[str]) -> float:
    """Return the fraction of zero weights across the named modules."""
    zeros = 0
    total = 0
    for name in names:
        weight = model.get_submodule(name).weight
        zeros += int((weight == 0).sum())
        total += weight.numel()
    return zeros / total if total else 0.0


def _run_reference_sparsegpt(
    sparsegpt_class,
    model,
    blocks,
    batches,
    *,
    device: str,
    sparsity: float,
    target_ids: set[int],
) -> None:
    """Drive the reference ``SparseGPT`` block by block.

    Reproduces the structure of their ``opt_sequential`` -- capture the first block's inputs, then walk
    the blocks, pruning each and re-running it so the next block sees already-compressed activations.
    Only this plumbing is ours; ``fasterprune`` is theirs and untouched.

    Args:
        sparsegpt_class: Their class, unmodified.
        model: The model to compress in place.
        blocks: The decoder blocks, in depth order.
        batches: Calibration batches.
        device: Device to run on.
        sparsity: Target sparsity.
        target_ids: ``id()`` of every module our own coverage rule selected, so both arms compress
            exactly the same set without reconstructing dotted names per block.
    """
    import torch
    from torch import nn

    model.eval()

    # GPT-NeoX passes `layer_past` (a Cache) and `use_cache=True` into every block. This driver
    # *replays* each block over the captured inputs, and a live cache would accumulate across replays
    # -- growing the key/value length and silently changing the activations SparseGPT then fits to.
    # Their own opt_sequential disables caching for exactly this reason. Restored afterwards so the
    # subsequent perplexity evaluation is unaffected.
    previous_use_cache = getattr(model.config, "use_cache", None)
    model.config.use_cache = False

    captured: list[tuple] = []

    class Catcher(nn.Module):
        """Intercept the first block's inputs, then stop the forward pass."""

        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, hidden_states, **kwargs):
            captured.append((hidden_states.detach(), kwargs))
            raise _StopForwardError

    blocks[0] = Catcher(blocks[0])
    try:
        with torch.no_grad():
            for batch in batches:
                payload = {
                    key: value.to(device)
                    for key, value in batch.items()
                    if key in {"input_ids", "attention_mask"} and hasattr(value, "to")
                }
                with contextlib.suppress(_StopForwardError):
                    model(**payload)
    finally:
        blocks[0] = blocks[0].inner

    if not captured:
        raise SystemExit(
            "Captured no inputs to the first decoder block. The block signature or call convention "
            "has changed; the reference comparison would otherwise prune against nothing."
        )

    for index, block in enumerate(blocks):
        linears = {
            name: module
            for name, module in block.named_modules()
            if isinstance(module, nn.Linear) and id(module) in target_ids
        }
        if not linears:
            continue

        handlers = {name: sparsegpt_class(module) for name, module in linears.items()}
        handles = []

        def make_hook(name, sinks=handlers):
            # `sinks` is bound at definition time on purpose: `handlers` is rebound every block, and
            # a late-binding closure would attach this block's hook to a later block's accumulators.
            def hook(_module, inputs, output):
                sinks[name].add_batch(inputs[0].data, output.data)

            return hook

        for name, module in linears.items():
            handles.append(module.register_forward_hook(make_hook(name)))
        with torch.no_grad():
            for hidden_states, kwargs in captured:
                block(hidden_states, **kwargs)
        for handle in handles:
            handle.remove()

        for name in linears:
            handlers[name].fasterprune(sparsity, prunen=0, prunem=0, blocksize=128, percdamp=0.01)
            handlers[name].free()

        # Re-run the pruned block so the next one sees compressed activations, matching their driver
        # and matching our own dependency-group scheme.
        with torch.no_grad():
            for position, (hidden_states, kwargs) in enumerate(captured):
                advanced = block(hidden_states, **kwargs)
                if isinstance(advanced, tuple):
                    advanced = advanced[0]
                captured[position] = (advanced.detach(), kwargs)
        LOGGER.info("reference SparseGPT: block %d done (%d linears)", index, len(linears))

    if previous_use_cache is not None:
        model.config.use_cache = previous_use_cache


class _StopForwardError(Exception):
    """Raised by the catcher to abort the forward pass once inputs are captured."""


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
