"""Scale-aware joint compression: does model scale change joint-vs-sequential outcomes?

Research question:
    How does model scale influence the effectiveness of joint versus sequential pruning and
    quantisation in decoder-only language models?

Importing this package is cheap and has no side effects: no torch import, no model
download, no logging configuration, no filesystem writes. Heavier submodules are exposed
lazily through :func:`__getattr__`, so ``import scale_aware_compression`` stays fast even
though ``scale_aware_compression.compression`` pulls in torch when it is finally used.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from scale_aware_compression.constants import (
    JOINT_STAGES,
    SEQUENTIAL_STAGES,
    CompressionMethod,
    CompressionStage,
    Device,
    DType,
)

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from scale_aware_compression.config import ExperimentConfig, load_config
    from scale_aware_compression.logging_utils import configure_logging, get_logger
    from scale_aware_compression.models.registry import MODEL_REGISTRY, resolve_model_id
    from scale_aware_compression.seed import set_global_seed

__version__ = "0.1.0"

RESEARCH_QUESTION = (
    "How does model scale influence the effectiveness of joint versus sequential pruning "
    "and quantisation in decoder-only language models?"
)

_LAZY_EXPORTS: dict[str, str] = {
    "ExperimentConfig": "scale_aware_compression.config",
    "load_config": "scale_aware_compression.config",
    "configure_logging": "scale_aware_compression.logging_utils",
    "get_logger": "scale_aware_compression.logging_utils",
    "set_global_seed": "scale_aware_compression.seed",
    "MODEL_REGISTRY": "scale_aware_compression.models.registry",
    "resolve_model_id": "scale_aware_compression.models.registry",
}

__all__ = [
    "JOINT_STAGES",
    "MODEL_REGISTRY",
    "RESEARCH_QUESTION",
    "SEQUENTIAL_STAGES",
    "CompressionMethod",
    "CompressionStage",
    "DType",
    "Device",
    "ExperimentConfig",
    "__version__",
    "configure_logging",
    "get_logger",
    "load_config",
    "resolve_model_id",
    "set_global_seed",
]


def __getattr__(name: str) -> Any:
    """Import a lazily exported symbol on first access.

    Args:
        name: Attribute name requested on the package.

    Returns:
        The resolved object.

    Raises:
        AttributeError: If ``name`` is not a known export.
    """
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(module_path), name)
    globals()[name] = value  # cache so subsequent lookups skip __getattr__
    return value


def __dir__() -> list[str]:
    """Return the public attribute names, including lazy exports."""
    return sorted(__all__)
