"""Architecture adapters: locate the parts of a decoder-only model that get compressed.

Pythia is ``GPTNeoXForCausalLM`` and Qwen2.5 is ``Qwen2ForCausalLM``. They name their
blocks and projections differently, so every compressor would otherwise hard-code both
naming schemes. This module is the single place that knows the difference.

Status: placeholder. The module-name conventions below are recorded so the eventual
implementation has a specification to work from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from scale_aware_compression.logging_utils import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch
    from torch import nn

LOGGER = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ArchitectureAdapter:
    """Naming conventions for one decoder-only architecture.

    Attributes:
        architecture: Transformers class name this adapter applies to.
        block_path: Dotted attribute path from the model to the list of decoder blocks.
        attention_projections: Suffixes of the attention weight matrices.
        mlp_projections: Suffixes of the feed-forward weight matrices.
        embedding_names: Suffixes of embedding and output-head modules, which are excluded
            from compression by default.
        tied_embeddings: Whether the output head shares weights with the input embedding.
    """

    architecture: str
    block_path: str
    attention_projections: tuple[str, ...]
    mlp_projections: tuple[str, ...]
    embedding_names: tuple[str, ...]
    tied_embeddings: bool = False
    notes: str = ""

    @property
    def compressible_suffixes(self) -> tuple[str, ...]:
        """All linear-projection suffixes eligible for pruning and quantisation."""
        return self.attention_projections + self.mlp_projections


GPT_NEOX_ADAPTER = ArchitectureAdapter(
    architecture="GPTNeoXForCausalLM",
    block_path="gpt_neox.layers",
    # GPT-NeoX fuses q, k, and v into one matrix, which matters for structured pruning:
    # a head-level pattern has to be applied consistently across all three slices.
    attention_projections=("attention.query_key_value", "attention.dense"),
    mlp_projections=("mlp.dense_h_to_4h", "mlp.dense_4h_to_h"),
    embedding_names=("gpt_neox.embed_in", "embed_out"),
    tied_embeddings=False,
    notes="Pythia suite. Fused QKV projection; untied input/output embeddings.",
)

QWEN2_ADAPTER = ArchitectureAdapter(
    architecture="Qwen2ForCausalLM",
    block_path="model.layers",
    attention_projections=(
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
    ),
    mlp_projections=("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"),
    embedding_names=("model.embed_tokens", "lm_head"),
    tied_embeddings=True,
    notes=(
        "Qwen2.5-0.5B. Separate q/k/v with grouped-query attention, gated MLP, and tied "
        "embeddings: pruning the head would also prune the input embedding."
    ),
)

ADAPTERS: dict[str, ArchitectureAdapter] = {
    GPT_NEOX_ADAPTER.architecture: GPT_NEOX_ADAPTER,
    QWEN2_ADAPTER.architecture: QWEN2_ADAPTER,
}


class UnsupportedArchitectureError(NotImplementedError):
    """Raised when no adapter is registered for a model's architecture."""


class AdapterResolutionError(RuntimeError):
    """Raised when an adapter's naming assumptions no longer match the installed model class."""


class EmptySelectionError(RuntimeError):
    """Raised when module selection matches nothing.

    Its own type because it is the most dangerous silent failure in the pipeline: an arm that
    compresses zero modules returns a model identical to the dense one, which then scores perfect
    quality retention at the requested sparsity. That reads as an excellent result.
    """


@dataclass(slots=True)
class CompressibleModules:
    """The modules a compressor is allowed to touch, resolved for one model instance."""

    names: list[str] = field(default_factory=list)
    total_parameters: int = 0
    excluded_names: list[str] = field(default_factory=list)
    excluded_parameters: int = 0

    @property
    def count(self) -> int:
        """Number of modules selected for compression."""
        return len(self.names)


def get_adapter(model_or_architecture: Any) -> ArchitectureAdapter:
    """Return the adapter for a model instance or an architecture name.

    Args:
        model_or_architecture: A loaded model, or the Transformers class name as a string.

    Returns:
        The matching :class:`ArchitectureAdapter`.

    Raises:
        UnsupportedArchitectureError: If the architecture has no registered adapter.
    """
    if isinstance(model_or_architecture, str):
        architecture = model_or_architecture
    else:
        architecture = type(model_or_architecture).__name__

    adapter = ADAPTERS.get(architecture)
    if adapter is None:
        raise UnsupportedArchitectureError(
            f"No architecture adapter for {architecture!r}. Registered: "
            f"{sorted(ADAPTERS)}. Add one in models/adapters.py before compressing this "
            "model family."
        )
    return adapter


def get_decoder_blocks(model: nn.Module) -> list[nn.Module]:
    """Return the model's decoder blocks in depth order.

    Depth order is not cosmetic. The layerwise method propagates activations through the
    already-compressed prefix, so compressing block 5 before block 3 would capture activations
    from a model state that never occurs at inference.

    Args:
        model: A loaded decoder-only model.

    Returns:
        The decoder blocks, shallowest first.

    Raises:
        UnsupportedArchitectureError: If no adapter is registered for the architecture.
        AdapterResolutionError: If the adapter's ``block_path`` does not resolve, which means the
            adapter is stale relative to the installed transformers version.
    """
    adapter = get_adapter(model)
    current: Any = model
    for attribute in adapter.block_path.split("."):
        current = getattr(current, attribute, None)
        if current is None:
            raise AdapterResolutionError(
                f"block_path {adapter.block_path!r} does not resolve on "
                f"{type(model).__name__}: stopped at {attribute!r}. The adapter in "
                "models/adapters.py is stale relative to the installed transformers version."
            )
    try:
        blocks = list(current)
    except TypeError as error:
        raise AdapterResolutionError(
            f"block_path {adapter.block_path!r} resolved to {type(current).__name__}, "
            "which is not iterable"
        ) from error
    if not blocks:
        raise AdapterResolutionError(f"block_path {adapter.block_path!r} resolved to zero blocks")
    return blocks


def select_compressible_modules(
    model: nn.Module,
    *,
    target_modules: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
) -> CompressibleModules:
    """Choose which submodules pruning and quantisation may modify.

    The default policy excludes embeddings and the output head. Both dominate the parameter
    count at small scale and shrink in relative terms as models grow, so including them
    would confound the scale trend this study is trying to measure.

    Args:
        model: A loaded decoder-only model.
        target_modules: Module class names to include, default ``["Linear"]``.
        exclude_patterns: Substrings that disqualify a module by name.

    Returns:
        The resolved selection, with parameter totals for reporting.

    Raises:
        UnsupportedArchitectureError: If no adapter is registered for the architecture.
        EmptySelectionError: If nothing was selected.
    """
    adapter = get_adapter(model)
    wanted_classes = tuple(target_modules or ["Linear"])
    excluded_substrings = tuple(
        exclude_patterns
        if exclude_patterns is not None
        else ["embed", "embed_out", "lm_head", "wte", "wpe"]
    )
    block_prefix = f"{adapter.block_path}."
    suffixes = adapter.compressible_suffixes

    selection = CompressibleModules()
    for name, module in model.named_modules():
        if type(module).__name__ not in wanted_classes:
            continue
        weight = getattr(module, "weight", None)
        if weight is None:
            continue

        # Two independent gates, deliberately. The adapter gate is the specification -- decoder
        # block linears only (§3.10). The substring gate is a safety net that catches anything the
        # adapter's suffix list would let through if a model family renamed a projection.
        inside_a_block = name.startswith(block_prefix) and name.endswith(suffixes)
        excluded_by_name = any(pattern in name for pattern in excluded_substrings)

        if inside_a_block and not excluded_by_name:
            selection.names.append(name)
            selection.total_parameters += weight.numel()
        else:
            selection.excluded_names.append(name)
            selection.excluded_parameters += weight.numel()

    if not selection.names:
        raise EmptySelectionError(
            f"module selection matched nothing on {type(model).__name__}. Looked for classes "
            f"{list(wanted_classes)} under {adapter.block_path!r} ending in {list(suffixes)}. "
            f"Skipped {len(selection.excluded_names)} candidate(s). A compression arm that "
            "touches zero modules would return the dense model and score perfect retention."
        )

    LOGGER.debug(
        "Selected %d module(s) totalling %d parameters; excluded %d module(s) totalling %d",
        selection.count,
        selection.total_parameters,
        len(selection.excluded_names),
        selection.excluded_parameters,
    )
    return selection


def count_targeted_parameters(
    model: nn.Module,
    *,
    target_modules: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
) -> int:
    """Count the parameters compression actually touches.

    This is the study's **scale x-axis** (§2.6, gap A6), not total model size. Pythia's names
    reflect total parameters, but the embedding share falls sharply with scale -- embeddings are a
    large fraction of 160M and a small one of 1.4B. Plotting joint gain against total parameters
    would therefore confound a scale trend with a trend in how much of each model was compressed.

    Args:
        model: A loaded decoder-only model.
        target_modules: Module class names to include.
        exclude_patterns: Substrings that disqualify a module by name.

    Returns:
        Summed weight elements across the selected modules.
    """
    return select_compressible_modules(
        model, target_modules=target_modules, exclude_patterns=exclude_patterns
    ).total_parameters


def get_weight_tensors(model: nn.Module, module_names: list[str]) -> dict[str, torch.Tensor]:
    """Return the weight tensor of each named module.

    Args:
        model: A loaded model.
        module_names: Names as returned by :func:`select_compressible_modules`.

    Returns:
        Mapping from module name to its ``weight`` tensor (not a copy), in the order requested.

    Raises:
        AdapterResolutionError: If a name does not resolve to a submodule.
    """
    tensors: dict[str, torch.Tensor] = {}
    for name in module_names:
        try:
            module = model.get_submodule(name)
        except AttributeError as error:
            raise AdapterResolutionError(
                f"{name!r} does not resolve on {type(model).__name__}; the module list is stale"
            ) from error
        weight = getattr(module, "weight", None)
        if weight is None:
            LOGGER.warning("Skipping %s: no weight attribute", name)
            continue
        tensors[name] = weight
    return tensors


def get_linear_modules(model: nn.Module, module_names: list[str]) -> dict[str, nn.Module]:
    """Return the named submodules themselves, not just their weights.

    The layerwise driver needs the modules in order to install activation hooks on them, and needs
    to write reconstructed weights back in place.

    Args:
        model: A loaded model.
        module_names: Names as returned by :func:`select_compressible_modules`.

    Returns:
        Mapping from module name to the module, in the order requested.

    Raises:
        AdapterResolutionError: If a name does not resolve to a submodule.
    """
    modules: dict[str, nn.Module] = {}
    for name in module_names:
        try:
            modules[name] = model.get_submodule(name)
        except AttributeError as error:
            raise AdapterResolutionError(
                f"{name!r} does not resolve on {type(model).__name__}; the module list is stale"
            ) from error
    return modules


def describe_architecture(model: nn.Module) -> dict[str, Any]:
    """Summarise a model's shape for the run record.

    Args:
        model: A loaded model.

    Returns:
        Mapping with the architecture name, adapter notes, and the config fields that
        matter for interpreting compression results (depth, width, head count). Fields absent on a
        given family are reported as ``None`` rather than omitted, so a record's shape does not
        depend on which model produced it.
    """
    adapter: ArchitectureAdapter | None
    try:
        adapter = get_adapter(model)
    except UnsupportedArchitectureError:
        adapter = None

    config = getattr(model, "config", None)
    fields = (
        "num_hidden_layers",
        "hidden_size",
        "num_attention_heads",
        "num_key_value_heads",
        "intermediate_size",
        "vocab_size",
        "tie_word_embeddings",
        "max_position_embeddings",
    )
    description: dict[str, Any] = {
        "architecture": type(model).__name__,
        "adapter_notes": adapter.notes if adapter else None,
        "tied_embeddings": adapter.tied_embeddings if adapter else None,
    }
    for field_name in fields:
        description[field_name] = getattr(config, field_name, None) if config else None
    return description
