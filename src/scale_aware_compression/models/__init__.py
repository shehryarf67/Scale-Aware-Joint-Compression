"""Model registry, safe loading, and per-architecture adapters.

Only the registry is imported eagerly: it is pure data and stays importable without torch.
The loader and adapters are exposed lazily so ``import scale_aware_compression.models`` does
not pull in transformers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from scale_aware_compression.models.registry import (
    MODEL_REGISTRY,
    ModelSpec,
    UnknownModelError,
    get_model_spec,
    list_models,
    normalise_model_name,
    registry_table,
    resolve_model_id,
    scale_sweep_models,
    validation_models,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from scale_aware_compression.models.adapters import ArchitectureAdapter, get_adapter
    from scale_aware_compression.models.loader import (
        LoadedModel,
        ModelLoadError,
        load_model,
        load_model_and_tokenizer,
        load_tokenizer,
    )

_LAZY_EXPORTS: dict[str, str] = {
    "LoadedModel": "scale_aware_compression.models.loader",
    "ModelLoadError": "scale_aware_compression.models.loader",
    "load_model": "scale_aware_compression.models.loader",
    "load_model_and_tokenizer": "scale_aware_compression.models.loader",
    "load_tokenizer": "scale_aware_compression.models.loader",
    "ArchitectureAdapter": "scale_aware_compression.models.adapters",
    "get_adapter": "scale_aware_compression.models.adapters",
}

__all__ = [
    "MODEL_REGISTRY",
    "ArchitectureAdapter",
    "LoadedModel",
    "ModelLoadError",
    "ModelSpec",
    "UnknownModelError",
    "get_adapter",
    "get_model_spec",
    "list_models",
    "load_model",
    "load_model_and_tokenizer",
    "load_tokenizer",
    "normalise_model_name",
    "registry_table",
    "resolve_model_id",
    "scale_sweep_models",
    "validation_models",
]


def __getattr__(name: str) -> Any:
    """Import loader/adapter symbols on first access."""
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(module_path), name)
    globals()[name] = value
    return value
