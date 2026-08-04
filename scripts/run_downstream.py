r"""Downstream task accuracy across the arms (plan §4.3, gap A4).

Perplexity says a compressed model still predicts text. It does not say the model is still *useful*,
and §4.3 requires three multiple-choice tasks alongside it: HellaSwag, PIQA and ARC-Easy, scored by a
pinned lm-evaluation-harness with the **task versions recorded** (§4.8).

DEVICE, AND WHY IT IS NOT CPU
-----------------------------
Multiple-choice scoring is one forward per *candidate continuation*, not per sequence: ~53,000
forwards and ~8M tokens across the three tasks, roughly 32x a perplexity evaluation. That is ~15-20 h
on GPU against ~150 h on CPU across the sweep, so CPU is not feasible.

Running on GPU is legitimate and it is declared rather than assumed. §4.6 restricts **deployment**
measurements -- latency, throughput, memory, checkpoint size -- to CPU, because those are properties
of the machine. A multiple-choice accuracy is a quality metric: it is a property of the weights and
the data, device-invariant far below the ~1 pp differences being reported. The device is written into
every record either way, so the choice is auditable.

The alternative was `--limit` subsampling, which was rejected: it weakens comparability with
published numbers, and comparability is the whole reason the harness was chosen over a
reimplementation.

WHAT TO READ CAREFULLY
----------------------
* **There is a floor.** 25% on HellaSwag and ARC-Easy, 50% on PIQA. A model at chance has *stopped
  doing the task*, not done it worse. Both the per-task chance level and an at-chance flag are in
  every record, and the run logs a warning.
* **The scoring path is partly uncompressed.** §2.6 excludes embeddings and the head, so the logits
  come from an FP32 layer at every budget.
* **Report against dense.** An absolute accuracy without its dense reference does not say whether
  compression cost anything, which is why dense runs first here exactly as it does for perplexity.

    python scripts/run_downstream.py --config configs/experiments/downstream.yaml
    python scripts/run_downstream.py --config ... --models pythia-160m --limit 200
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

ARMS = ("dense", "sequential", "joint")
"""Dense first: every compressed arm's retention is a ratio against it.

Pruning-only and quantisation-only are omitted -- §4.3 asks whether the *compared* arms stay usable,
and each extra arm is another ~1.9 h per scale.
"""


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", required=True, type=Path, help="Experiment config.")
    parser.add_argument("--models", nargs="+", default=None, help="Restrict to these models.")
    parser.add_argument("--arms", nargs="+", default=None, help="Restrict to these arms.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Samples per task. Overrides the config. A subsampled score is not comparable with a "
        "published number and is recorded as such.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/metrics/downstream.json"),
        help="Where to write the aggregated record.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the plan.")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Score every arm on the downstream tasks.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        0 on success, 1 if nothing could be scored.
    """
    arguments = build_parser().parse_args(argv)
    configure_logging(arguments.log_level)

    from scale_aware_compression.compression.arms import plan_from_config
    from scale_aware_compression.compression.layerwise import compress_model_layerwise
    from scale_aware_compression.config import ExperimentConfig, deep_merge, load_config
    from scale_aware_compression.constants import CompressionMethod, Device
    from scale_aware_compression.data.calibration import load_calibration_set
    from scale_aware_compression.evaluation.downstream import (
        DOWNSTREAM_TASKS,
        accuracy_retention,
        evaluate_downstream,
    )
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
    arms = tuple(arguments.arms) if arguments.arms else ARMS
    tasks = tuple(config.evaluation.downstream_tasks or DOWNSTREAM_TASKS)
    limit = arguments.limit if arguments.limit is not None else config.evaluation.downstream_limit
    device = config.evaluation.effective_downstream_device.value

    if arguments.dry_run:
        LOGGER.info(
            "Plan: %s",
            json.dumps(
                {
                    "models": models,
                    "arms": list(arms),
                    "tasks": list(tasks),
                    "device": device,
                    "limit": limit,
                    "evaluations": len(models) * len(arms),
                },
                indent=2,
            ),
        )
        return 0

    if device != Device.CPU.value and not cuda_available():
        LOGGER.error("evaluation.downstream_device=%s but no CUDA device is available.", device)
        return 1

    rows: list[dict] = []
    dense_by_model: dict[str, dict[str, float]] = {}

    for model_name in models:
        # Dense first, so each compressed arm has a retention reference by the time it runs.
        for arm in sorted(arms, key=lambda a: (a != "dense", a)):
            LOGGER.info("=== %s / %s ===", model_name, arm)

            spec = get_model_spec(model_name)
            document = deep_merge(
                config.to_dict(),
                {"model": {"name": spec.short_name, "size_label": spec.size_label}},
            )
            document["model"].pop("hf_id", None)
            document["model"]["revision"] = _revision_for(model_name, document)
            document["model"]["device"] = Device.CPU.value
            if arm != "dense":
                document["compression"]["method"] = (
                    CompressionMethod.JOINT.value
                    if arm == "joint"
                    else CompressionMethod.SEQUENTIAL.value
                )
            cell_config = ExperimentConfig.from_mapping(document)

            loaded = load_model_and_tokenizer(cell_config.model)
            model = loaded.model

            if arm != "dense":
                calibration = load_calibration_set(cell_config.data, loaded.tokenizer)
                batches = [
                    batch["input_ids"] if isinstance(batch, dict) else batch[0]
                    for batch in calibration.loader
                ]
                use_gpu = cuda_available()
                compress_model_layerwise(
                    model,
                    batches,
                    plan_from_config(cell_config),
                    arm=arm,
                    calibration_fingerprint=calibration.summary.token_fingerprint,
                    device=Device.CUDA.value if use_gpu else None,
                    offload_blocks=use_gpu,
                )

            report = evaluate_downstream(
                model,
                loaded.tokenizer,
                tasks=tasks,
                device=device,
                batch_size=config.evaluation.downstream_batch_size,
                limit=limit,
            )

            if arm == "dense":
                dense_by_model[model_name] = {r.task: r.accuracy for r in report.tasks}

            reference = dense_by_model.get(model_name, {})
            for result in report.tasks:
                payload = result.to_dict()
                dense_accuracy = reference.get(result.task)
                # Retention only where a dense reference for THIS model exists. Normalising against
                # another model's dense score would be meaningless, and a missing field is a
                # question while a wrong one is an answer.
                payload["dense_accuracy"] = dense_accuracy
                payload["accuracy_retention"] = (
                    accuracy_retention(result.accuracy, dense_accuracy) if dense_accuracy else None
                )
                payload.update(
                    {
                        "model": model_name,
                        "arm": arm,
                        "harness_version": report.harness_version,
                        "device": report.device,
                        "limit": report.limit,
                    }
                )
                rows.append(payload)

            del model, loaded

    if not rows:
        LOGGER.error("Nothing was scored.")
        return 1

    payload = {
        "schema": "downstream/1",
        "host": host_key(),
        "hardware": get_hardware_info(),
        "software": get_software_versions(),
        "device_rationale": (
            "Downstream accuracy is a quality metric, not a deployment measurement: it is a "
            "property of the weights and the data, device-invariant far below the ~1 pp "
            "differences reported. §4.6 binds latency, throughput, memory and checkpoint size to "
            "CPU; those remain CPU-only. CPU here would be ~150 h against ~15-20 h on GPU."
        ),
        "omitted_arms": {
            "pruning, quantisation": (
                "§4.3 asks whether the compared arms stay usable; each extra arm adds ~1.9 h per "
                "scale"
            )
        },
        "rows": rows,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    LOGGER.info("Wrote %d row(s) to %s", len(rows), arguments.output)

    print(f"\n  {'model':<13}{'arm':<12}{'task':<11}{'acc':>8}{'chance':>8}{'ret.':>8}  version")
    for row in rows:
        retention = f"{row['accuracy_retention']:.3f}" if row["accuracy_retention"] else "—"
        flag = "" if row["above_chance"] else "  <-- AT CHANCE"
        print(
            f"  {row['model']:<13}{row['arm']:<12}{row['task']:<11}{row['accuracy']:>8.4f}"
            f"{row['chance_level']:>8.2f}{retention:>8}  {row['task_version']}{flag}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
