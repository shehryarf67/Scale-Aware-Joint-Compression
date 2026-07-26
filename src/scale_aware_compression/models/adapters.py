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

    Layer-wise pruning criteria and per-layer sparsity reporting both need this ordering.

    Args:
        model: A loaded decoder-only model.

    Returns:
        The decoder blocks, shallowest first.

    Raises:
        NotImplementedError: Always, in the current scaffold.
    """
    # TODO(adapters): resolve `adapter.block_path` with functools.reduce(getattr, ...) and
    # return list(module). Raise a clear error if the path does not resolve, since that
    # means the adapter is stale relative to the installed transformers version.
    raise NotImplementedError(
        "get_decoder_blocks is not implemented yet; see the TODO in models/adapters.py"
    )


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
        NotImplementedError: Always, in the current scaffold.
    """
    # TODO(adapters): walk model.named_modules(), keep modules whose class name is in
    # target_modules and whose name contains no exclude pattern, and sum
    # module.weight.numel() into total_parameters. Log the selection at DEBUG: an
    # accidentally empty selection is the most likely silent failure in the pipeline.
    raise NotImplementedError(
        "select_compressible_modules is not implemented yet; see the TODO in models/adapters.py"
    )


def get_weight_tensors(model: nn.Module, module_names: list[str]) -> dict[str, torch.Tensor]:
    """Return the weight tensor of each named module.

    Args:
        model: A loaded model.
        module_names: Names as returned by :func:`select_compressible_modules`.

    Returns:
        Mapping from module name to its ``weight`` tensor (not a copy).

    Raises:
        NotImplementedError: Always, in the current scaffold.
    """
    # TODO(adapters): resolve each name via model.get_submodule(name) and return
    # module.weight. Skip and warn for modules without a `weight` attribute.
    raise NotImplementedError(
        "get_weight_tensors is not implemented yet; see the TODO in models/adapters.py"
    )


def describe_architecture(model: nn.Module) -> dict[str, Any]:
    """Summarise a model's shape for the run record.

    Args:
        model: A loaded model.

    Returns:
        Mapping with the architecture name, adapter notes, and the config fields that
        matter for interpreting compression results (depth, width, head count).

    Raises:
        NotImplementedError: Always, in the current scaffold.
    """
    # TODO(adapters): read num_hidden_layers, hidden_size, num_attention_heads,
    # num_key_value_heads, intermediate_size, vocab_size, and tie_word_embeddings from
    # model.config, tolerating absent fields across families.
    raise NotImplementedError(
        "describe_architecture is not implemented yet; see the TODO in models/adapters.py"
    )
