r"""Prefill and decode latency, timed separately, at two prompt lengths (plan §4.7, gap A5).

WHY THE SPLIT EXISTS
--------------------
A single end-to-end generation latency blends two workloads that scale differently and that
compression affects differently:

* **prefill** -- one forward over the whole prompt. Compute-bound; cost grows with prompt length.
* **decode** -- one forward per generated token against a populated cache. Memory-bandwidth-bound;
  per-token cost is roughly flat in prompt length, and it dominates a long generation.

Weight-only compression helps decode most, because decode moves weights and little else. Reporting
one blended number averages that away.

WHAT THIS MEASURES, AND WHAT IT DELIBERATELY DOES NOT
-----------------------------------------------------
Only arms whose runtime representation is **FP32** are timed: the dense baseline and the
pruning-only arm. Per decision D1 a packed W4/W8 layer dequantises on every forward, so timing it
measures the unpacking kernel rather than the compression. Those arms are skipped and the absence
is recorded, because an absent field is a question and a wrong field is an answer.

That is also why RQ4's sparsity-versus-latency curve comes from the pruning-only arm: it is the one
arm that stays FP32, so its timings mean what they appear to mean.

MODEL-ORDER ROTATION
--------------------
Benchmarks drift with temperature. Whichever arm runs first on a cold machine is measured under
different conditions from whichever runs last, and with a fixed order that difference lands on the
same arm every time -- a bias, not noise. Each round rotates the arm order, so over a full cycle
every arm leads exactly once and the drift is spread across arms.

Arms are built fresh inside each round rather than held resident, so rotation is real rather than
cosmetic and so two full-size FP32 models never sit in RAM at once (7.5 GiB at Pythia-1B, on a
13.7 GiB machine).

CPU-ONLY, on the designated benchmark host. This is a deployment measurement (§4.6).

    python scripts/run_prefill_decode.py --config configs/experiments/prefill_decode.yaml
    python scripts/run_prefill_decode.py --config ... --models pythia-160m --rounds 2
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

ARMS = ("dense", "pruning")
"""The arms whose latency is meaningful. See the module docstring and decision D1."""


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", required=True, type=Path, help="Experiment config.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Restrict to these registry names (default: the config's sweep.models).",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Rotation rounds. Defaults to the number of arms, so every arm leads exactly once.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/metrics/prefill_decode.json"),
        help="Where to write the aggregated record.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the plan.")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser


def _plan(models: list[str], rounds: int, prompt_lengths: tuple[int, ...]) -> dict:
    """Describe what would be measured, without measuring it."""
    return {
        "models": models,
        "arms": list(ARMS),
        "prompt_lengths": list(prompt_lengths),
        "phases": ["prefill", "decode"],
        "rotation_rounds": rounds,
        "measurements": len(models) * len(ARMS) * len(prompt_lengths) * 2 * rounds,
    }


def main(argv: list[str] | None = None) -> int:
    """Measure prefill and decode latency for every eligible arm.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        0 on success, 1 if no measurement could be taken.
    """
    arguments = build_parser().parse_args(argv)
    configure_logging(arguments.log_level)

    from scale_aware_compression.benchmarking.cpu import CpuBenchmarkRunner
    from scale_aware_compression.benchmarking.phases import (
        PROMPT_LENGTHS,
        build_decode_callable,
        build_prefill_callable,
        rotate,
    )
    from scale_aware_compression.compression.arms import plan_from_config
    from scale_aware_compression.compression.layerwise import compress_model_layerwise
    from scale_aware_compression.config import ExperimentConfig, deep_merge, load_config
    from scale_aware_compression.constants import Device
    from scale_aware_compression.data.calibration import load_calibration_set
    from scale_aware_compression.experiments.scale_sweep import _revision_for
    from scale_aware_compression.hardware import (
        cuda_available,
        get_hardware_info,
        get_software_versions,
        host_key,
    )
    from scale_aware_compression.models.loader import load_model_and_tokenizer
    from scale_aware_compression.models.registry import get_model_spec

    config = load_config(arguments.config)
    models = arguments.models or list(config.sweep.models)
    rounds = arguments.rounds or len(ARMS)

    if arguments.dry_run:
        LOGGER.info("Plan: %s", json.dumps(_plan(models, rounds, PROMPT_LENGTHS), indent=2))
        return 0

    # The CPU rule, enforced here as well as in the runner: this script exists to produce numbers
    # for a deployment table, and a GPU timing has no place in one.
    if config.benchmark.device is not Device.CPU:
        LOGGER.error(
            "benchmark.device=%s; deployment measurements are CPU-only.", config.benchmark.device
        )
        return 1

    samples: dict[tuple[str, str, int, str], list[float]] = {}

    for model_name in models:
        for round_index in range(rounds):
            for arm in rotate(list(ARMS), round_index):
                LOGGER.info("round %d: %s / %s", round_index, model_name, arm)

                # Re-resolve the pinned revision for THIS model rather than inheriting the base
                # config's. §2.7 requires every checkpoint pinned to a SHA, and a loop over models
                # that inherits one revision either fails to load or silently loads the wrong
                # weights if the SHA exists in both repos (B-13). Reusing the sweep's resolver
                # rather than re-implementing it, so the two cannot drift.
                spec = get_model_spec(model_name)
                document = deep_merge(
                    config.to_dict(),
                    {"model": {"name": spec.short_name, "size_label": spec.size_label}},
                )
                document["model"].pop("hf_id", None)
                document["model"]["revision"] = _revision_for(model_name, document)
                document["model"]["device"] = Device.CPU.value
                cell_config = ExperimentConfig.from_mapping(document)

                loaded = load_model_and_tokenizer(cell_config.model)
                model = loaded.model

                if arm == "pruning":
                    # Rebuilt every round rather than held resident: two full-size FP32 models is
                    # 7.5 GiB at 1B, and rebuilding is what makes the rotation real.
                    calibration = load_calibration_set(config.data, loaded.tokenizer)
                    batches = [
                        batch["input_ids"] if isinstance(batch, dict) else batch[0]
                        for batch in calibration.loader
                    ]
                    # Compression on the GPU, measurement on the CPU. §4.6 restricts the
                    # *measurement*, not how the artefact was produced, and the phase callables move
                    # the model back to CPU before timing anything. Without this, building the
                    # pruned 1B model would run the Gram accumulation and every solve on the host --
                    # hours of work the card does in minutes, for a mask that latency does not even
                    # depend on. Offload keeps 1B inside the 6 GiB card (F-31).
                    use_gpu = cuda_available()
                    compress_model_layerwise(
                        model,
                        batches,
                        plan_from_config(config),
                        arm="pruning",
                        calibration_fingerprint=calibration.summary.token_fingerprint,
                        device=Device.CUDA.value if use_gpu else None,
                        offload_blocks=use_gpu,
                    )

                for prompt_length in PROMPT_LENGTHS:
                    builders = {
                        "prefill": build_prefill_callable,
                        "decode": build_decode_callable,
                    }
                    for phase, builder in builders.items():
                        callable_ = builder(
                            model,
                            loaded.tokenizer,
                            batch_size=config.benchmark.batch_size,
                            prompt_length=prompt_length,
                        )
                        runner = CpuBenchmarkRunner(config.benchmark)
                        result = runner.run(callable_, label=f"{model_name}/{arm}/{phase}")
                        key = (model_name, arm, prompt_length, phase)
                        samples.setdefault(key, []).extend(
                            ms / 1000.0 for ms in result.per_run_latencies_ms
                        )
                        LOGGER.info(
                            "  %s prompt=%d  median %.2f ms  IQR %.2f ms",
                            phase,
                            prompt_length,
                            result.latency.median_ms,
                            result.latency.iqr_ms,
                        )

                del model, loaded

    if not samples:
        LOGGER.error("No measurements were taken.")
        return 1

    from scale_aware_compression.benchmarking.latency import summarise_latencies

    rows = []
    for (model_name, arm, prompt_length, phase), seconds in sorted(samples.items()):
        statistics = summarise_latencies(seconds)
        rows.append(
            {
                "model": model_name,
                "arm": arm,
                "prompt_length": prompt_length,
                "phase": phase,
                "rotation_rounds": rounds,
                **statistics.to_dict(),
            }
        )

    payload = {
        "schema": "prefill_decode/1",
        "plan": _plan(models, rounds, PROMPT_LENGTHS),
        "host": host_key(),
        "hardware": get_hardware_info(),
        "software": get_software_versions(),
        "benchmark": {
            "num_threads": config.benchmark.num_threads,
            "batch_size": config.benchmark.batch_size,
            "warmup_runs": config.benchmark.warmup_runs,
            "measured_runs": config.benchmark.measured_runs,
        },
        "excluded_arms": {
            "quantisation, sequential, joint": (
                "packed W4/W8 layers dequantise on every forward, so a timing would measure the "
                "unpacking kernel rather than the compression (decision D1)"
            )
        },
        "rows": rows,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    LOGGER.info("Wrote %d row(s) to %s", len(rows), arguments.output)

    print(f"\n  {'model':<14}{'arm':<10}{'prompt':>7}{'phase':>9}{'median ms':>12}{'IQR ms':>10}")
    for row in rows:
        print(
            f"  {row['model']:<14}{row['arm']:<10}{row['prompt_length']:>7}{row['phase']:>9}"
            f"{row['latency_median_ms']:>12.2f}{row['latency_iqr_ms']:>10.2f}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
