#!/usr/bin/env python
r"""Run the CPU deployment benchmark for one model or checkpoint.

Measures median and p95 latency, throughput, and peak process memory under the protocol in
docs/benchmarking_protocol.md: pinned thread count, fixed batch size and sequence length, warm-up
iterations, then repeated timed runs.

**CPU only.** The configuration loader rejects any other device: a latency measured on GPU is not a
deployment number for this study.

Before running, close everything else. A 30-run measurement taken while a compile or a browser is
running produces a plausible number that is not comparable with a clean one.

Examples:
    python scripts/run_cpu_benchmark.py --config configs/experiments/pilot.yaml --dry-run
    python scripts/run_cpu_benchmark.py --config configs/experiments/pilot.yaml
    python scripts/run_cpu_benchmark.py --config configs/experiments/pilot.yaml --threads 1
    python scripts/run_cpu_benchmark.py --config configs/experiments/pilot.yaml \\
        --checkpoint outputs/pilot/checkpoint
"""

from __future__ import annotations

import argparse

from scale_aware_compression.config import ConfigError, load_config
from scale_aware_compression.logging_utils import configure_logging, get_logger, log_key_values

LOGGER = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to a YAML experiment configuration")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config value, e.g. --override benchmark.measured_runs=50. Repeatable.",
    )
    parser.add_argument(
        "--checkpoint",
        help="Benchmark a saved checkpoint directory instead of the configured base model",
    )
    parser.add_argument(
        "--threads",
        type=int,
        help=(
            "Override benchmark.num_threads. Report each thread count as its own series; never "
            "average across them."
        ),
    )
    parser.add_argument(
        "--warmup-runs", type=int, help="Override benchmark.warmup_runs (must be >= 0)"
    )
    parser.add_argument(
        "--measured-runs", type=int, help="Override benchmark.measured_runs (must be >= 2)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved benchmark protocol without loading a model or timing anything",
    )
    parser.add_argument("--log-level", default=None, help="Override runtime.log_level")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CPU benchmark.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        0 on success, 2 on a configuration error, 3 when the pipeline is not implemented yet.
    """
    arguments = build_parser().parse_args(argv)
    try:
        config = load_config(arguments.config, arguments.override)
    except ConfigError as error:
        configure_logging(arguments.log_level or "INFO")
        LOGGER.error("Invalid configuration: %s", error)
        return 2

    benchmark = config.benchmark
    if arguments.threads is not None:
        benchmark.num_threads = arguments.threads
    if arguments.warmup_runs is not None:
        benchmark.warmup_runs = arguments.warmup_runs
    if arguments.measured_runs is not None:
        benchmark.measured_runs = arguments.measured_runs

    configure_logging(arguments.log_level or config.runtime.log_level)
    log_key_values(
        LOGGER,
        "CPU benchmark protocol",
        {
            "model": config.model.name,
            "checkpoint": arguments.checkpoint or "(base model from config)",
            "device": benchmark.device.value,
            "num_threads": benchmark.num_threads,
            "batch_size": benchmark.batch_size,
            "sequence_length": benchmark.sequence_length,
            "generated_tokens": benchmark.generated_tokens,
            "warmup_runs": benchmark.warmup_runs,
            "measured_runs": benchmark.measured_runs,
            "workload": "generate" if benchmark.generated_tokens else "forward",
        },
    )

    if benchmark.warmup_runs < 5:
        LOGGER.warning(
            "warmup_runs=%d is below the protocol's 5. The first calls to a quantised CPU kernel "
            "are several times slower than the steady state and will skew the result.",
            benchmark.warmup_runs,
        )
    if benchmark.measured_runs < 30:
        LOGGER.warning(
            "measured_runs=%d is below the protocol's 30, so the p95 rests on few samples.",
            benchmark.measured_runs,
        )

    if arguments.dry_run:
        from scale_aware_compression.hardware import get_hardware_info

        log_key_values(LOGGER, "This machine", get_hardware_info())
        LOGGER.info("Dry run: no model was loaded and nothing was timed")
        return 0

    from scale_aware_compression.benchmarking.cpu import benchmark_model
    from scale_aware_compression.models.loader import load_model_and_tokenizer

    try:
        loaded = load_model_and_tokenizer(config.model)
        result = benchmark_model(
            loaded.model,
            loaded.tokenizer,
            benchmark,
            label=f"{config.model.name}/{config.compression.method.value}",
        )
    except NotImplementedError as error:
        LOGGER.error("Not implemented yet: %s", error)
        return 3

    LOGGER.info("%s", result.summary_line())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
