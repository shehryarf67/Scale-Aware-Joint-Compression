"""Load a compressed checkpoint back from disk, independently of the process that wrote it.

§4.8 requires verifying that "the compressed checkpoint reloads independently from disk" and that the
bit width and sparsity survive. Saving the state dict is not sufficient to do that. A packed model
replaces some ``nn.Linear`` modules with :class:`PackedLinear`, and a freshly constructed model from
the same config has plain ``nn.Linear`` everywhere -- so it has no idea which modules to instantiate
as packed, and loading the state dict into it fails on unexpected keys.

The missing piece is a **manifest**: a small JSON file recorded alongside the weights saying which
modules were packed and under what scheme. With it, reloading is mechanical -- build the base model,
swap the named modules for empty packed ones, load the state dict, verify.

Why this matters beyond tidiness: the quality figure in the paper should come from the artefact a
deployment would actually load, not from the in-memory object that happened to exist at the end of a
compression run. Those are the same model only if this path works.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from scale_aware_compression.compression.packed import packed_linear_class
from scale_aware_compression.constants import QuantisationGranularity
from scale_aware_compression.logging_utils import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torch import nn

LOGGER = get_logger(__name__)

MANIFEST_NAME = "compression_manifest.json"
"""Filename the manifest is written to, inside the checkpoint directory."""

MANIFEST_VERSION = "1"
"""Format version. Bump when a field changes meaning, so an old manifest fails loudly."""


class ReloadError(RuntimeError):
    """Raised when a compressed checkpoint cannot be reloaded or fails verification."""


def write_manifest(
    destination: Path,
    *,
    module_names: list[str],
    bits: int,
    granularity: QuantisationGranularity,
    group_size: int,
    shapes: dict[str, tuple[int, int]],
    target_sparsity: float,
    method: str,
) -> Path:
    """Record what a reader needs in order to rebuild this artefact.

    Args:
        destination: The checkpoint directory.
        module_names: Modules that were replaced with packed equivalents.
        bits: Bit width of the stored codes.
        granularity: Scope each scale covers.
        group_size: Elements per quantisation group.
        shapes: ``(out_features, in_features)`` per module, so empty modules can be built before the
            state dict arrives.
        target_sparsity: What the run asked for, so the reload can check what it got.
        method: The arm that produced the artefact, for provenance.

    Returns:
        The manifest path.
    """
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / MANIFEST_NAME
    payload = {
        "manifest_version": MANIFEST_VERSION,
        "method": method,
        "bits": bits,
        "granularity": granularity.value,
        "group_size": group_size,
        "target_sparsity": target_sparsity,
        "packed_modules": {name: list(shapes[name]) for name in module_names},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    LOGGER.info("Wrote compression manifest for %d module(s) to %s", len(module_names), path)
    return path


def read_manifest(checkpoint: Path) -> dict[str, Any]:
    """Read and validate a compression manifest.

    Args:
        checkpoint: The checkpoint directory.

    Returns:
        The manifest payload.

    Raises:
        ReloadError: If the manifest is missing, unreadable, or a version this code cannot read.
    """
    path = Path(checkpoint) / MANIFEST_NAME
    if not path.is_file():
        raise ReloadError(
            f"No {MANIFEST_NAME} in {checkpoint}. Without it the packed modules cannot be "
            "identified, and a base model built from the config has plain nn.Linear everywhere. "
            "The artefact is not independently loadable."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReloadError(f"{path} is unreadable: {error}") from error

    version = payload.get("manifest_version")
    if version != MANIFEST_VERSION:
        raise ReloadError(
            f"{path} is manifest version {version!r}, and this code reads {MANIFEST_VERSION!r}. "
            "Refusing to guess at a format whose fields may have changed meaning."
        )
    return payload


def load_packed_model(checkpoint: Path, base_model: nn.Module) -> nn.Module:
    """Rebuild a compressed model from disk and verify it against its manifest.

    Args:
        checkpoint: Directory containing the weights and the manifest.
        base_model: A freshly constructed dense model of the same architecture. Modified in place.

    Returns:
        The model with packed modules restored and the state dict loaded.

    Raises:
        ReloadError: If the manifest is unusable, a named module is missing, the state dict does not
            fit, or the reloaded artefact fails its bit-width and sparsity checks.
    """
    import torch
    from safetensors.torch import load_file

    checkpoint = Path(checkpoint)
    manifest = read_manifest(checkpoint)
    granularity = QuantisationGranularity(manifest["granularity"])
    bits = int(manifest["bits"])
    group_size = int(manifest["group_size"])
    packed_class = packed_linear_class()

    # Swap in empty packed modules first. Their buffer shapes are wrong at this point and that is
    # fine: PackedLinear resizes them on load, because the code-buffer length depends on the bit
    # width being loaded.
    for name, shape in manifest["packed_modules"].items():
        out_features, in_features = int(shape[0]), int(shape[1])
        parent_name, _, attribute = name.rpartition(".")
        try:
            parent = base_model.get_submodule(parent_name) if parent_name else base_model
        except AttributeError as error:
            raise ReloadError(f"{name!r} does not resolve on the base model") from error
        existing = getattr(parent, attribute, None)
        if existing is None:
            raise ReloadError(f"{name!r} is in the manifest but absent from the base model")

        setattr(
            parent,
            attribute,
            packed_class(
                in_features,
                out_features,
                bits=bits,
                granularity=granularity,
                group_size=group_size,
                packed=torch.zeros(1, dtype=torch.uint8),
                scales=torch.ones(out_features, 1),
                bias=torch.zeros(out_features)
                if getattr(existing, "bias", None) is not None
                else None,
            ),
        )

    state = _load_state_dict(checkpoint, load_file)
    missing, unexpected = base_model.load_state_dict(state, strict=False)
    # `strict=False` then inspect, rather than strict=True: a Transformers checkpoint legitimately
    # omits tied or buffer entries, so a blanket failure would be noise. What must not happen is a
    # *packed* buffer going missing, because that silently leaves a zeroed layer behind.
    packed_missing = [
        key for key in missing if key.split(".")[-1] in {"packed", "scales", "scheme"}
    ]
    if packed_missing:
        raise ReloadError(
            f"packed buffers missing from the checkpoint: {packed_missing[:6]}. The reloaded model "
            "would contain zeroed layers that still run and still score."
        )
    if unexpected:
        LOGGER.debug("Ignoring %d unexpected key(s) on reload", len(unexpected))

    _verify_reloaded(base_model, manifest, bits=bits)
    return base_model


def _load_state_dict(checkpoint: Path, load_file: Any) -> dict[str, Any]:
    """Read whichever weight format the checkpoint was written in."""
    import torch

    safetensors = sorted(checkpoint.glob("*.safetensors"))
    if safetensors:
        merged: dict[str, Any] = {}
        for shard in safetensors:
            merged.update(load_file(str(shard)))
        return merged

    for candidate in ("pytorch_model.bin", "state_dict.pt"):
        path = checkpoint / candidate
        if path.is_file():
            return torch.load(path, map_location="cpu", weights_only=False)

    raise ReloadError(f"No weight file found in {checkpoint}")


def _verify_reloaded(model: nn.Module, manifest: dict[str, Any], *, bits: int) -> None:
    """Check the reloaded artefact still has the precision and sparsity it claims.

    §4.8 asks for exactly this: confirm the bit width is real rather than silently dequantised, and
    that reported sparsity survives serialisation. Both are checked on the modules as reloaded, not
    on anything carried over in memory.

    Args:
        model: The reloaded model.
        manifest: Its manifest.
        bits: Expected bit width.

    Raises:
        ReloadError: If a packed module lost its precision or its sparsity.
    """
    packed_class = packed_linear_class()
    target = float(manifest.get("target_sparsity") or 0.0)

    for name in manifest["packed_modules"]:
        module = model.get_submodule(name)
        if not isinstance(module, packed_class):
            raise ReloadError(f"{name!r} did not come back as a packed module")
        if module.bits != bits:
            raise ReloadError(f"{name!r} reloaded at {module.bits} bits, expected {bits}")

        weight = module.dequantise()
        if target > 0.0:
            zeros = float((weight == 0).float().mean())
            # Quantisation rounds some survivors to zero, so the numeric zero fraction is
            # legitimately *higher* than the mask budget. Only a shortfall is suspicious.
            #
            # But the budget is not exactly attainable, and that was B-46. The default comparison
            # group is the output row, and `build_mask` prunes `round(in_features * sparsity)` per
            # row -- an integer count. So the realised fraction is quantised to multiples of
            # 1/in_features and lands up to 0.5/in_features either side of the target. A 768-wide
            # module at 30% prunes round(230.4) = 230, realising 0.299479: short of target by
            # 5.2e-04, which the old 1e-6 tolerance read as corruption. Every sequential and joint
            # cell of the confirmatory run failed here, on masks that were exactly right.
            #
            # The allowance is derived from the row width rather than fixed, so it stays tight: at
            # 768 it is 1.3e-03, still three orders of magnitude below any real serialisation loss
            # (a dropped mask leaves the fraction near zero, not one row-step short).
            in_features = int(manifest["packed_modules"][name][1])
            allowance = 1.0 / in_features if in_features else 1e-6
            if zeros < target - allowance:
                raise ReloadError(
                    f"{name!r} reloaded with {zeros:.4f} zeros against a {target:.4f} target "
                    f"(allowance {allowance:.2e} for per-row integer rounding at "
                    f"in_features={in_features}): sparsity did not survive serialisation"
                )
