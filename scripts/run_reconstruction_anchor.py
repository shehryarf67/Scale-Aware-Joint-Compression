"""Check the reconstruction sweep against the exact optimum of its objective (A1 §5.5b).

Runs the real pipeline's sweep on real Pythia layers, then solves the same masked least-squares
problem exactly, per output row, in float64. Two invariants must hold on every row:

* the sweep may not score **below** the provable optimum -- impossible, so it proves a defect;
* the sweep may not end **worse than naive masking** -- the refinement loop keeps naive rounding as
  iterate zero and only accepts improvements, so a violation means that guard is broken.

Efficiency (share of the achievable gain captured) is reported but does not gate the verdict: a
one-pass sweep giving up some of the optimum is the documented trade for making wide layers tractable.

Rows are sampled rather than exhaustive because the exact solve costs ``|S|^3`` per row -- the very
cost the sweep exists to avoid. The objective is separable across output rows, so each sampled row is a
complete test of that row rather than an approximation of the layer.

    python scripts/run_reconstruction_anchor.py --config configs/experiments/screening.yaml
    python scripts/run_reconstruction_anchor.py --config configs/experiments/screening.yaml --dry-run
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
        "--sparsity", type=float, default=0.3, help="Target sparsity. Defaults to the frozen 0.3."
    )
    parser.add_argument(
        "--modules",
        type=int,
        default=6,
        help="How many modules to sample, spread through the depth.",
    )
    parser.add_argument(
        "--rows", type=int, default=8, help="Output rows to solve exactly per module."
    )
    parser.add_argument("--output", type=Path, default=None, help="Where to write the JSON report.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate the config and print the plan."
    )
    return parser


def _stratified_sample(names: list[str], budget: int) -> list[str]:
    """Sample modules covering every module *type*, spread through the depth.

    A plain stride is wrong here and quietly so: a GPT-NeoX block contributes four target modules in
    a fixed order, so striding by ``len(names) // 6`` on a 48-module model steps by exactly 8 and
    returns ``attention.query_key_value`` six times. The first run of this anchor did that and never
    touched an MLP projection -- which are the widest layers and the ones where a one-pass sweep has
    the most to compensate for.

    Args:
        names: All target module names, in depth order.
        budget: How many to return.

    Returns:
        Up to ``budget`` names, round-robin across module types and evenly spaced within each.
    """
    groups: dict[str, list[str]] = {}
    for name in names:
        # "gpt_neox.layers.7.mlp.dense_4h_to_h" -> "mlp.dense_4h_to_h"
        key = ".".join(name.split(".")[-2:])
        groups.setdefault(key, []).append(name)

    per_group = {key: [] for key in groups}
    for key, members in groups.items():
        wanted = max(1, budget // len(groups))
        step = max(1, len(members) // wanted)
        per_group[key] = members[::step][:wanted]

    sampled: list[str] = []
    ordered = list(groups)
    index = 0
    while len(sampled) < budget and any(per_group[key] for key in ordered):
        key = ordered[index % len(ordered)]
        if per_group[key]:
            sampled.append(per_group[key].pop(0))
        index += 1

    if len(sampled) < budget:
        # Say so rather than quietly returning fewer: a coverage shortfall that is not reported reads
        # as "we sampled what we asked for".
        LOGGER.warning(
            "asked for %d modules but stratification over %d type(s) yielded %d; "
            "raise --modules to a multiple of the type count for even coverage",
            budget,
            len(groups),
            len(sampled),
        )
    return sampled


def main(argv: list[str] | None = None) -> int:
    """Run the anchor.

    Args:
        argv: Command-line arguments, or ``None`` to read ``sys.argv``.

    Returns:
        0 when the anchor passes, 1 on a violated invariant, 2 on a configuration error.
    """
    arguments = build_parser().parse_args(argv)

    from scale_aware_compression.config import ConfigError, load_config

    try:
        config = load_config(arguments.config)
    except ConfigError as error:
        print(f"configuration error: {error}", file=sys.stderr)
        return 2

    configure_logging(config.runtime.log_level)
    print("Exact-optimum reconstruction anchor (A1 §5.5b)")
    print(f"  model      : {config.model.name} @ {config.model.revision}")
    print(f"  sparsity   : {arguments.sparsity}")
    print(f"  sampling   : {arguments.modules} modules x {arguments.rows} rows")
    print(f"  solver     : {config.compression.reconstruction.solver}")
    if arguments.dry_run:
        print("\n--dry-run: nothing loaded.")
        return 0

    import torch

    from scale_aware_compression.anchors import ExactReconstructionReport, compare_row
    from scale_aware_compression.compression.activations import ActivationStatistics
    from scale_aware_compression.compression.masks import build_mask_from_scores
    from scale_aware_compression.compression.pruning import activation_weighted_saliency
    from scale_aware_compression.compression.reconstruct import sweep_reconstruct
    from scale_aware_compression.data.calibration import load_calibration_set
    from scale_aware_compression.models.adapters import select_compressible_modules
    from scale_aware_compression.models.loader import load_model_and_tokenizer

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
    print(f"\n  sampled modules: {len(names)} of {len(all_names)}")
    print(f"  calibration    : {len(batches)} batches, {calibration.summary.token_fingerprint}")

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

    report = ExactReconstructionReport()
    report.notes.append(
        "Pruning-only (bits=None). The reference minimises the TRUE objective with no damping, so it "
        "is a genuine lower bound; our sweep regularises for conditioning and is expected to be "
        "close but never better."
    )
    report.notes.append(
        f"Rows sampled, not exhaustive: the exact solve costs |S|^3 per row. "
        f"{arguments.rows} rows per module, chosen by a fixed seed."
    )

    generator = torch.Generator().manual_seed(20260730)

    for name in names:
        module = loaded.model.get_submodule(name)
        weight = module.weight.detach().to("cpu", torch.float32)
        gram = statistics[name].gram().detach().to("cpu", torch.float32)
        norms = statistics[name].column_norms().detach().to("cpu")

        mask = build_mask_from_scores(
            activation_weighted_saliency(weight, norms), sparsity=float(arguments.sparsity)
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

        out_features = weight.shape[0]
        count = min(arguments.rows, out_features)
        rows = torch.randperm(out_features, generator=generator)[:count].tolist()

        comparisons = []
        for row in sorted(rows):
            comparisons.append(compare_row(row, gram, weight[row], outcome.weight[row], mask[row]))
        report.module_rows[name] = comparisons
        print(
            f"    {name}: mean efficiency "
            f"{sum(c.efficiency for c in comparisons) / len(comparisons):.4f}"
        )

    print()
    for line in report.summary_lines():
        print(line)

    destination = arguments.output or (
        REPOSITORY_ROOT / "outputs" / "anchors" / f"exact_reconstruction_{config.model.name}.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = report.to_dict()
    payload["model"] = config.model.name
    payload["revision"] = config.model.revision
    payload["target_sparsity"] = float(arguments.sparsity)
    payload["calibration_fingerprint"] = calibration.summary.token_fingerprint
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n  report written to {destination}")

    return 0 if report.passes else 1


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
