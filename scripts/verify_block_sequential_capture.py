r"""Prove that block-sequential capture produces the same Gram as full-model capture.

Run BEFORE refactoring `compress_model_layerwise`. The refactor is worth doing (see below) but it
touches the most safety-critical code in the project -- the code three anchors currently vouch for --
so the equivalence it depends on is established first, on a model where the right answer is known.

WHY THE REFACTOR IS WANTED
--------------------------
`_compress_group` captures activations by running the **entire model** forward, once per dependency
group:

    with torch.inference_mode():
        for batch in batches:
            model(batch.to(next(model.parameters()).device))

Two consequences, and one fix addresses both.

* **Wasted work.** Blocks *after* the current one cannot influence its inputs, so running them is pure
  waste; and blocks *before* it are re-run from the embedding for every group instead of having their
  output cached. That is O(blocks x groups) full-model forwards where O(blocks) single-block forwards
  suffice -- roughly 16x more forward work at Pythia-1B.
* **Memory.** The whole model must be resident on the capture device. At 1B that is 3.77 GiB of FP32
  weights, and the widest layer's inverse-Cholesky needs ~2.5 GiB of temporaries on top. Measured peak
  was 6.31 GiB on a 6.00 GiB card: it only completed by spilling into host memory, which made the
  widest-layer solve 33.9 s against 4.8 s standalone.

The standard alternative -- what SparseGPT and Wanda both do, and what this repository's own external
anchor driver already does in `run_sparsegpt_external_anchor.py` -- caches the hidden states entering
the current block and runs **only that block**, advancing the cache as it goes. One block resident
instead of the whole model.

THE CLAIM UNDER TEST
--------------------
Blocks after block *k* cannot affect block *k*'s inputs, so capturing by running the full model and
capturing by running only blocks 0..k should give **the same activations**, hence the same Gram, hence
the same mask, scales and reconstruction.

"Should" is doing real work in that sentence. This script checks it.

WHAT IT COMPARES
----------------
For every targeted module, the Gram matrix and column norms captured two ways:

* **reference** -- the current path: full-model forward, hooks on the target modules;
* **candidate** -- block-sequential: capture block 0's inputs once, then run one block at a time over
  the cached hidden states, advancing the cache after each block.

Both paths use the identical model, calibration draw and batch order, so any disagreement is the
capture strategy and nothing else.

    python scripts/verify_block_sequential_capture.py --config configs/experiments/screening.yaml
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "src") not in sys.path:  # pragma: no cover - script bootstrap
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from scale_aware_compression.logging_utils import configure_logging, get_logger  # noqa: E402

LOGGER = get_logger(__name__)


class _StopForwardError(Exception):
    """Raised to abort a forward pass once block 0's inputs have been captured."""


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--batches",
        type=int,
        default=8,
        help="Calibration batches to use. Fewer is fine: this tests equivalence, not quality.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _make_hook(statistics, name):
    """Return a forward pre-hook that accumulates into ``statistics[name]``."""

    def hook(_module, inputs):
        if inputs:
            statistics[name].update(inputs[0])

    return hook


def main(argv: list[str] | None = None) -> int:
    """Compare the two capture strategies.

    Returns:
        0 when every module's Gram agrees, 1 on any disagreement, 2 on a config error.
    """
    arguments = build_parser().parse_args(argv)

    from scale_aware_compression.config import ConfigError, load_config

    try:
        config = load_config(arguments.config)
    except ConfigError as error:
        print(f"configuration error: {error}", file=sys.stderr)
        return 2

    configure_logging(config.runtime.log_level)
    print("Block-sequential capture equivalence check")
    print(f"  model   : {config.model.name} @ {config.model.revision}")
    print(f"  batches : {arguments.batches}")
    if arguments.dry_run:
        print("\n--dry-run: nothing loaded.")
        return 0

    import torch
    from torch import nn

    from scale_aware_compression.compression.activations import ActivationStatistics
    from scale_aware_compression.data.calibration import load_calibration_set
    from scale_aware_compression.models.adapters import (
        get_decoder_blocks,
        get_linear_modules,
        select_compressible_modules,
    )
    from scale_aware_compression.models.loader import load_model_and_tokenizer

    loaded = load_model_and_tokenizer(config.model)
    model = loaded.model
    model.eval()
    device = loaded.device.value

    calibration = load_calibration_set(config.data, loaded.tokenizer)
    batches = [
        batch["input_ids"] if isinstance(batch, dict) else batch[0] for batch in calibration.loader
    ][: arguments.batches]
    print(f"  calib   : {len(batches)} batches, {calibration.summary.token_fingerprint}")

    names = list(select_compressible_modules(model).names)
    modules = get_linear_modules(model, names)
    blocks = get_decoder_blocks(model)
    print(f"  modules : {len(names)} across {len(blocks)} blocks\n")

    # `use_cache` must be off for both paths. A live Cache accumulates across the repeated block
    # forwards the candidate path performs, which would change the activations it captures. Learned
    # from B-28, where exactly this bit the external SparseGPT driver.
    previous_use_cache = getattr(model.config, "use_cache", None)
    model.config.use_cache = False

    def fresh_statistics():
        return {
            name: ActivationStatistics(
                modules[name].in_features, dtype=torch.float32, device=device
            )
            for name in names
        }

    # ---------------------------------------------------------------- reference: full-model forward
    reference = fresh_statistics()
    handles = [
        modules[name].register_forward_pre_hook(_make_hook(reference, name)) for name in names
    ]
    try:
        with torch.inference_mode():
            for batch in batches:
                model(batch.to(device))
    finally:
        for handle in handles:
            handle.remove()
    print("reference capture done (full-model forward, all modules at once)")

    # ------------------------------------------------------- candidate: block-sequential with cache
    candidate = fresh_statistics()
    captured: list[tuple] = []

    def catcher(_module, args, kwargs):
        hidden = args[0] if args else kwargs.get("hidden_states")
        if hidden is None:
            raise _StopForwardError
        forwarded = {key: value for key, value in kwargs.items() if key != "hidden_states"}
        captured.append((hidden.detach().clone(), forwarded))
        raise _StopForwardError

    handle = blocks[0].register_forward_pre_hook(catcher, with_kwargs=True)
    try:
        with torch.inference_mode():
            for batch in batches:
                with contextlib.suppress(_StopForwardError):
                    model(batch.to(device))
    finally:
        handle.remove()

    if len(captured) != len(batches):
        print(f"FAIL: captured {len(captured)} block-0 inputs for {len(batches)} batches")
        return 1
    print(f"candidate: cached block-0 inputs for {len(captured)} batches")

    # Fully-qualified name per block, so local module names can be mapped back to the keys the
    # statistics are held under. Matching on identity avoids guessing at name prefixes.
    identity_to_name = {id(modules[name]): name for name in names}

    for block in blocks:
        in_block = {
            module: identity_to_name[id(module)]
            for module in block.modules()
            if isinstance(module, nn.Linear) and id(module) in identity_to_name
        }
        block_handles = [
            module.register_forward_pre_hook(_make_hook(candidate, name))
            for module, name in in_block.items()
        ]
        try:
            with torch.inference_mode():
                for hidden, kwargs in captured:
                    block(hidden, **kwargs)
        finally:
            for handle in block_handles:
                handle.remove()

        # Advance the cache so the next block sees this block's outputs, exactly as the full-model
        # forward would have produced them.
        with torch.inference_mode():
            for position, (hidden, kwargs) in enumerate(captured):
                advanced = block(hidden, **kwargs)
                if isinstance(advanced, tuple):
                    advanced = advanced[0]
                captured[position] = (advanced.detach(), kwargs)

    print("candidate capture done (one block at a time, cache advanced)\n")

    if previous_use_cache is not None:
        model.config.use_cache = previous_use_cache

    # ------------------------------------------------------------------------------------- compare
    worst_gram = 0.0
    worst_norm = 0.0
    worst_name = ""
    failures = 0
    for name in names:
        left = reference[name].gram().to(torch.float64)
        right = candidate[name].gram().to(torch.float64)
        scale = left.abs().max().clamp_min(1e-30)
        gram_error = float(((left - right).abs() / scale).max())

        left_norms = reference[name].column_norms().to(torch.float64)
        right_norms = candidate[name].column_norms().to(torch.float64)
        norm_scale = left_norms.abs().max().clamp_min(1e-30)
        norm_error = float(((left_norms - right_norms).abs() / norm_scale).max())

        if gram_error > worst_gram:
            worst_gram, worst_name = gram_error, name
        worst_norm = max(worst_norm, norm_error)
        if gram_error > 1e-5:
            failures += 1
            if failures <= 5:
                print(f"  DISAGREES {name}: relative Gram error {gram_error:.3e}")

    print(f"modules compared            : {len(names)}")
    print(f"worst relative Gram error   : {worst_gram:.3e}  ({worst_name})")
    print(f"worst relative norm error   : {worst_norm:.3e}")
    print(f"modules disagreeing > 1e-5  : {failures}")
    verdict = failures == 0
    print(f"\nVERDICT: {'EQUIVALENT' if verdict else 'NOT EQUIVALENT'}")
    if verdict:
        print("  Block-sequential capture reproduces full-model capture. The refactor is safe to")
        print("  build, and a full-cell reproduction of F-23 remains the next gate.")
    else:
        print(
            "  Do NOT refactor on this evidence. Something about the block-sequential path changes"
        )
        print("  the activations -- most likely the replay kwargs or a cache that is still live.")
    return 0 if verdict else 1


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
