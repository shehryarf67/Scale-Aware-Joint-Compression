"""Run the Wanda mask-agreement anchor (Amendment A1 §5.5a).

Checks our mask construction against an independent implementation of the same published criterion,
on identical activations from the dense model. Cheap: one forward pass over the calibration set, then
pure tensor comparisons. No compression, no reconstruction, no evaluation.

A1 §7 puts this **before** the screening re-run, so two hours of grid are not spent measuring a
pipeline that has never been checked against anything outside itself.

    python scripts/run_wanda_anchor.py --config configs/experiments/screening.yaml
    python scripts/run_wanda_anchor.py --config configs/experiments/screening.yaml --dry-run
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
    parser.add_argument(
        "--sparsity",
        type=float,
        default=None,
        help="Target sparsity to compare at. Defaults to the config's pruning target.",
    )
    parser.add_argument(
        "--max-modules",
        type=int,
        default=None,
        help="Compare only the first N modules. For a quick check; the full run compares all.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write the JSON report. Defaults to outputs/anchors/wanda_<model>.json.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the config and print the plan without loading a model.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the anchor.

    Args:
        argv: Command-line arguments, or ``None`` to read ``sys.argv``.

    Returns:
        0 when the anchor passes, 1 when it finds a divergence worth investigating, 2 on a
        configuration error.
    """
    arguments = build_parser().parse_args(argv)

    from scale_aware_compression.config import ConfigError, load_config

    try:
        config = load_config(arguments.config)
    except ConfigError as error:
        print(f"configuration error: {error}", file=sys.stderr)
        return 2

    configure_logging(config.runtime.log_level)
    sparsity = arguments.sparsity
    if sparsity is None:
        sparsity = config.compression.pruning.sparsity

    print("Wanda mask-agreement anchor (A1 §5.5a)")
    print(f"  config          : {arguments.config}")
    print(f"  model           : {config.model.name}")
    print(f"  revision        : {config.model.revision}")
    print(f"  sparsity        : {sparsity}")
    print(f"  calibration     : {config.data.calibration_samples} from {config.data.dataset}")
    print("  comparison group: per output row (matched to the shipped default)")
    if arguments.dry_run:
        print("\n--dry-run: nothing loaded, nothing compared.")
        return 0

    from scale_aware_compression.anchors import (
        WandaAnchorReport,
        compare_column_norms,
        compare_masks,
        independent_column_norms,
        independent_wanda_mask,
    )
    from scale_aware_compression.compression.activations import ActivationStatistics
    from scale_aware_compression.compression.masks import build_mask_from_scores
    from scale_aware_compression.compression.pruning import activation_weighted_saliency
    from scale_aware_compression.data.calibration import load_calibration_set
    from scale_aware_compression.models.adapters import select_compressible_modules
    from scale_aware_compression.models.loader import load_model_and_tokenizer

    loaded = load_model_and_tokenizer(config.model)
    calibration = load_calibration_set(config.data, loaded.tokenizer)
    batches = list(calibration.loader)
    print(f"\n  calibration batches : {len(batches)}")
    print(f"  token fingerprint   : {calibration.summary.token_fingerprint}")

    selection = select_compressible_modules(
        loaded.model,
        target_modules=config.compression.pruning.target_modules,
        exclude_patterns=config.compression.pruning.exclude_patterns,
    )
    names = list(selection.names)
    if arguments.max_modules is not None:
        names = names[: arguments.max_modules]
    print(f"  modules to compare  : {len(names)}")

    # The loader already resolved and applied the device; re-deriving it risks moving the model
    # somewhere its weights are not. GPU is fine here -- an anchor is a diagnostic, not a
    # measurement, so the CPU-only rule does not apply.
    device = loaded.device.value

    # The reference path: direct accumulation of column sums of squares, no Gram matrix.
    reference_norms = independent_column_norms(loaded.model, names, batches, device=device)

    # Our path: the streamed Gram accumulator the production pipeline uses, fed the same activations.
    import torch

    ours_statistics: dict[str, ActivationStatistics] = {}
    handles = []

    def make_hook(name: str):
        def hook(_module, inputs):
            if inputs:
                ours_statistics[name].update(inputs[0])

        return hook

    for name in names:
        module = loaded.model.get_submodule(name)
        ours_statistics[name] = ActivationStatistics(module.in_features)
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

    report = WandaAnchorReport(target_sparsity=float(sparsity))
    report.notes.append(
        "Both sides use dense-model activations, isolating the criterion from the "
        "compressed-prefix propagation scheme (which has its own tests)."
    )

    for name in names:
        module = loaded.model.get_submodule(name)
        weight = module.weight.detach().to("cpu", torch.float32)
        ours_norms = ours_statistics[name].column_norms().detach().to("cpu")
        their_norms = reference_norms[name].to("cpu")

        report.norms.append(compare_column_norms(name, ours_norms, their_norms))

        scores = activation_weighted_saliency(weight, ours_norms)
        our_mask = build_mask_from_scores(scores, sparsity=float(sparsity))
        their_mask = independent_wanda_mask(weight, their_norms, sparsity=float(sparsity))
        report.masks.append(compare_masks(name, our_mask, their_mask, scores))

    print()
    for line in report.summary_lines():
        print(line)

    destination = arguments.output
    if destination is None:
        destination = REPOSITORY_ROOT / "outputs" / "anchors" / f"wanda_{config.model.name}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = report.to_dict()
    payload["model"] = config.model.name
    payload["revision"] = config.model.revision
    payload["calibration_fingerprint"] = calibration.summary.token_fingerprint
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n  report written to {destination}")

    return 0 if report.passes else 1


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
