"""Registry mapping short model names to Hugging Face identifiers.

Short names are what appear in configs, filenames, plot legends, and run records. The Hub
identifier appears in exactly one place — this module — so a checkpoint can be repointed
without touching configs or results tooling.

Nothing here touches the network. Looking a model up is a dictionary access.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final

from scale_aware_compression.logging_utils import get_logger

LOGGER = get_logger(__name__)


class UnknownModelError(KeyError):
    """Raised when a short model name is not in the registry."""


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Static facts about one checkpoint in the study."""

    short_name: str
    """Registry key used in configs and filenames."""
    hf_id: str
    """Hugging Face Hub repository identifier."""
    size_label: str
    """Scale label for plot axes and tables, e.g. ``410M``."""
    parameter_count: int
    """Total parameters as published, used to order the sweep and label the x axis. The
    value actually measured at load time is what gets written to a run record."""
    family: str
    """Model family. Only models sharing a family belong in the same scale sweep."""
    architecture: str
    """Transformers architecture class name, all decoder-only here."""
    role: str
    """``scale_sweep``, ``scale_sweep_optional``, or ``external_validation``."""
    notes: str = ""
    aliases: tuple[str, ...] = field(default_factory=tuple)
    """Alternative spellings accepted by :func:`get_model_spec`, e.g. filename-safe forms."""

    @property
    def is_optional(self) -> bool:
        """Whether this model is only run when hardware permits."""
        return self.role == "scale_sweep_optional"

    @property
    def in_scale_sweep(self) -> bool:
        """Whether this model is part of the controlled Pythia scale sweep."""
        return self.role in {"scale_sweep", "scale_sweep_optional"}


PYTHIA_FAMILY: Final[str] = "pythia"
QWEN_FAMILY: Final[str] = "qwen2.5"

_REGISTRY: Final[dict[str, ModelSpec]] = {
    "pythia-160m": ModelSpec(
        short_name="pythia-160m",
        hf_id="EleutherAI/pythia-160m",
        size_label="160M",
        parameter_count=162_322_944,
        family=PYTHIA_FAMILY,
        architecture="GPTNeoXForCausalLM",
        role="scale_sweep",
        notes="Smallest point in the sweep; fast enough for pilot runs and CI-scale smoke tests.",
        aliases=("pythia_160m", "160m"),
    ),
    "pythia-410m": ModelSpec(
        short_name="pythia-410m",
        hf_id="EleutherAI/pythia-410m",
        size_label="410M",
        parameter_count=405_334_016,
        family=PYTHIA_FAMILY,
        architecture="GPTNeoXForCausalLM",
        role="scale_sweep",
        notes="Middle point in the sweep.",
        aliases=("pythia_410m", "410m"),
    ),
    "pythia-1b": ModelSpec(
        short_name="pythia-1b",
        hf_id="EleutherAI/pythia-1b",
        size_label="1B",
        parameter_count=1_011_781_632,
        family=PYTHIA_FAMILY,
        architecture="GPTNeoXForCausalLM",
        role="scale_sweep",
        notes="Third point in the sweep; the smallest size where a scale trend can be fitted.",
        aliases=("pythia_1b", "1b"),
    ),
    "pythia-1.4b": ModelSpec(
        short_name="pythia-1.4b",
        hf_id="EleutherAI/pythia-1.4b",
        size_label="1.4B",
        parameter_count=1_414_647_808,
        family=PYTHIA_FAMILY,
        architecture="GPTNeoXForCausalLM",
        role="scale_sweep_optional",
        notes="Fourth sweep point, run only if joint training fits in available memory.",
        aliases=("pythia_1_4b", "pythia-1_4b", "pythia_1.4b", "1.4b"),
    ),
    "qwen2.5-0.5b": ModelSpec(
        short_name="qwen2.5-0.5b",
        hf_id="Qwen/Qwen2.5-0.5B",
        size_label="0.5B",
        parameter_count=494_032_768,
        family=QWEN_FAMILY,
        architecture="Qwen2ForCausalLM",
        role="external_validation",
        notes=(
            "Different family, tokeniser, and training recipe. Used only to test whether a "
            "trend found within Pythia transfers; never averaged with Pythia results."
        ),
        aliases=("qwen2_5_0_5b", "qwen2.5_0.5b", "qwen-0.5b", "qwen2.5-0.5B"),
    ),
}

MODEL_REGISTRY: Final[MappingProxyType[str, ModelSpec]] = MappingProxyType(_REGISTRY)
"""Read-only view of the registry, keyed by canonical short name."""

_ALIASES: Final[dict[str, str]] = {
    alias: spec.short_name for spec in _REGISTRY.values() for alias in spec.aliases
}


def normalise_model_name(name: str) -> str:
    """Canonicalise a user-supplied model name.

    Trims whitespace, lowercases, and resolves registered aliases, so
    ``"Pythia_1_4B"`` and ``"pythia-1.4b"`` reach the same entry.

    Args:
        name: Name as written in a config, filename, or command-line flag.

    Returns:
        The canonical short name, which may or may not exist in the registry.

    Raises:
        UnknownModelError: If ``name`` is empty or not a string.
    """
    if not isinstance(name, str) or not name.strip():
        raise UnknownModelError("Model name must be a non-empty string")
    candidate = name.strip().lower()
    if candidate in _REGISTRY:
        return candidate
    return _ALIASES.get(candidate, candidate)


def get_model_spec(name: str) -> ModelSpec:
    """Look up the full specification for a short model name.

    Args:
        name: Short name or registered alias, case-insensitive.

    Returns:
        The matching :class:`ModelSpec`.

    Raises:
        UnknownModelError: If the name is not registered. The message lists the valid
            names, since a typo here is the most likely cause.
    """
    canonical = normalise_model_name(name)
    spec = _REGISTRY.get(canonical)
    if spec is None:
        raise UnknownModelError(
            f"Unknown model {name!r} (normalised to {canonical!r}). "
            f"Registered models: {', '.join(list_models())}"
        )
    return spec


def resolve_model_id(name: str) -> str:
    """Return the Hugging Face identifier for a short model name.

    Args:
        name: Short name or registered alias.

    Returns:
        The Hub repository identifier, e.g. ``EleutherAI/pythia-410m``.

    Raises:
        UnknownModelError: If the name is not registered.
    """
    return get_model_spec(name).hf_id


def list_models(*, family: str | None = None, role: str | None = None) -> list[str]:
    """List canonical short names, ordered by parameter count.

    Args:
        family: Restrict to one family, e.g. ``"pythia"``.
        role: Restrict to one role, e.g. ``"scale_sweep"``.

    Returns:
        Short names ascending by published parameter count, which is the order plots and
        tables use for the scale axis.
    """
    selected = [
        spec
        for spec in _REGISTRY.values()
        if (family is None or spec.family == family) and (role is None or spec.role == role)
    ]
    return [spec.short_name for spec in sorted(selected, key=lambda spec: spec.parameter_count)]


def scale_sweep_models(*, include_optional: bool = False) -> list[str]:
    """Return the Pythia sweep in ascending size order.

    Args:
        include_optional: Include ``pythia-1.4b``, which is only run if hardware permits.

    Returns:
        Short names for the scale sweep.
    """
    roles = {"scale_sweep", "scale_sweep_optional"} if include_optional else {"scale_sweep"}
    specs = [
        spec for spec in _REGISTRY.values() if spec.family == PYTHIA_FAMILY and spec.role in roles
    ]
    return [spec.short_name for spec in sorted(specs, key=lambda spec: spec.parameter_count)]


def validation_models() -> list[str]:
    """Return the models reserved for external validation."""
    return list_models(role="external_validation")


def registry_table() -> list[dict[str, object]]:
    """Return the registry as row dictionaries, for logging or documentation tables.

    Returns:
        One row per model, ordered by parameter count.
    """
    return [
        {
            "short_name": spec.short_name,
            "hf_id": spec.hf_id,
            "size_label": spec.size_label,
            "parameter_count": spec.parameter_count,
            "family": spec.family,
            "role": spec.role,
        }
        for spec in sorted(_REGISTRY.values(), key=lambda spec: spec.parameter_count)
    ]
