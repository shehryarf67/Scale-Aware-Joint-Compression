r"""Prove that per-block GPU offload changes no weight, and measure what it saves.

Offload holds **one decoder block** on the card at a time instead of the whole model. It is the
change that makes Pythia-1B runnable at all: with the model resident, 1B peaks at 6.31 GiB on a
6.00 GiB card and completes only by spilling into host memory, which made the widest-layer solve
33.9 s against 4.8 s standalone (findings_log.md F-29).

THE CLAIM UNDER TEST
--------------------
Residency is not arithmetic. Where a tensor lives should not reach the Gram, the mask, the scales
or the solve, so compressing with the model resident and compressing one block at a time should
produce **bit-identical weights**.

"Should" is doing real work in that sentence. Every retraction in this project was built on a
should. This script checks it.

WHAT IT COMPARES
----------------
Two runs of `compress_model_layerwise` over the same model, the same calibration draw and the same
batch order, differing only in `offload_blocks`:

* **resident**  -- the whole model on the card, the path every existing record was produced by;
* **offloaded** -- the model on the host, one block moved to the card and back at a time.

Every parameter is then compared with `torch.equal`, not `allclose`. A tolerance here would hide
exactly the class of difference worth finding: a near-tie in the saliency ranking flipping, which
changes a mask position rather than a low bit.

    python scripts/verify_block_offload.py --config configs/experiments/screening.yaml

For a model that does not fit resident -- which is the whole point -- the comparison is impossible
by construction, so measure the offloaded peak alone and check it against the ceiling. §5.2 wants
peak under 85% of the card, a 5.1 GiB ceiling on a 6.0 GiB one:

    python scripts/verify_block_offload.py --config configs/experiments/verify_offload_1b.yaml \
        --peak-only
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "src") not in sys.path:  # pragma: no cover - script bootstrap
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from scale_aware_compression.logging_utils import configure_logging, get_logger  # noqa: E402

LOGGER = get_logger(__name__)

GIB = 1024**3


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", required=True, type=Path, help="Experiment config to match.")
    parser.add_argument(
        "--arm",
        default="joint",
        help="Arm to run. Joint by default: it is the one with an outer loop, so it exercises the "
        "most of the driver per block (default: joint).",
    )
    parser.add_argument(
        "--peak-only",
        action="store_true",
        help="Run only the offloaded path and report its peak. For models that cannot be held "
        "resident, where the comparison is impossible rather than merely slow.",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=None,
        metavar="KEY=VALUE",
        help="Dotted config overrides, applied after includes.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate the config and stop.")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the comparison.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        0 when the weights match (or when only a peak was requested), 1 on any disagreement or
        when no CUDA device is available.
    """
    arguments = build_parser().parse_args(argv)
    configure_logging(arguments.log_level)

    import torch

    from scale_aware_compression.compression.arms import plan_from_config
    from scale_aware_compression.compression.layerwise import compress_model_layerwise
    from scale_aware_compression.config import load_config
    from scale_aware_compression.constants import Device
    from scale_aware_compression.data.calibration import load_calibration_set
    from scale_aware_compression.models.loader import load_model_and_tokenizer

    config = load_config(arguments.config, arguments.override)
    if arguments.dry_run:
        LOGGER.info("Config valid: %s, arm=%s", config.model.name, arguments.arm)
        return 0

    if not torch.cuda.is_available():
        LOGGER.error("No CUDA device. Offload has nothing to offload to, so there is nothing here.")
        return 1

    # Load on the host. The offloaded path needs it there anyway, and a caller that puts a large
    # model on the card first has paid the full-model transient this exists to avoid.
    config.model.device = Device.CPU
    loaded = load_model_and_tokenizer(config.model)
    calibration = load_calibration_set(config.data, loaded.tokenizer)
    # Same unwrapping the runner does: the loader yields dicts, the driver wants token-id tensors.
    batches = [
        batch["input_ids"] if isinstance(batch, dict) else batch[0] for batch in calibration.loader
    ]
    plan = plan_from_config(config)
    LOGGER.info(
        "%s: %d calibration batch(es), fingerprint %s",
        config.model.name,
        len(batches),
        calibration.summary.token_fingerprint,
    )

    def run(model, *, offload: bool) -> tuple[float, float]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        compress_model_layerwise(
            model,
            batches,
            plan,
            arm=arguments.arm,
            calibration_fingerprint=calibration.summary.token_fingerprint,
            device="cuda",
            offload_blocks=offload,
        )
        # Both, because they answer different questions and mixing them up would misreport the
        # headroom. `allocated` is what the tensors need; `reserved` is what the caching allocator
        # holds on the device, which is what `nvidia-smi` sees and what a second process would
        # collide with. F-29's 6.31 GiB against a 6.00 GiB card was a device-level figure, so
        # `reserved` is the one to compare it against.
        return (
            torch.cuda.max_memory_allocated() / GIB,
            torch.cuda.max_memory_reserved() / GIB,
        )

    # Deep-copied before either run: compression is destructive, and comparing a model against
    # itself after two passes would compare the second pass to nothing.
    offloaded_model = copy.deepcopy(loaded.model)
    offloaded_peak, offloaded_reserved = run(offloaded_model, offload=True)
    LOGGER.info(
        "offloaded peak: %.2f GiB allocated, %.2f GiB reserved",
        offloaded_peak,
        offloaded_reserved,
    )

    if arguments.peak_only:
        print(f"\n  offloaded peak, allocated : {offloaded_peak:.2f} GiB")
        print(f"  offloaded peak, reserved  : {offloaded_reserved:.2f} GiB")
        print("  resident path             : not run (--peak-only)")
        return 0

    resident_model = copy.deepcopy(loaded.model).to("cuda")
    resident_peak, resident_reserved = run(resident_model, offload=False)
    LOGGER.info(
        "resident peak: %.2f GiB allocated, %.2f GiB reserved", resident_peak, resident_reserved
    )

    mismatched: list[str] = []
    worst = 0.0
    for name, parameter in offloaded_model.named_parameters():
        expected = resident_model.get_parameter(name).detach().cpu()
        actual = parameter.detach().cpu()
        if not torch.equal(actual, expected):
            mismatched.append(name)
            worst = max(worst, float((actual - expected).abs().max()))

    total = sum(1 for _ in offloaded_model.named_parameters())
    print(f"\n  parameters compared       : {total}")
    print(f"  parameters disagreeing    : {len(mismatched)}")
    print(f"  worst absolute difference : {worst:.3e}")
    print(f"  resident peak  (alloc/res): {resident_peak:.2f} / {resident_reserved:.2f} GiB")
    print(f"  offloaded peak (alloc/res): {offloaded_peak:.2f} / {offloaded_reserved:.2f} GiB")
    if resident_reserved > 0:
        reduction = 100 * (1 - offloaded_reserved / resident_reserved)
        print(f"  reduction, reserved       : {reduction:.1f}%")

    if mismatched:
        print("\n  VERDICT: NOT EQUIVALENT")
        for name in mismatched[:10]:
            print(f"    {name}")
        return 1
    print("\n  VERDICT: EQUIVALENT")
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
