"""Safe loading of tokenisers and decoder-only causal language models.

Design constraints:

* Importing this module never imports ``transformers`` and never touches the network.
  Both happen only inside :func:`load_tokenizer` / :func:`load_model`.
* The model name is validated against the registry before anything is downloaded, so a
  typo fails in milliseconds rather than after a partial download.
* ``trust_remote_code`` defaults to ``False``. All models in this study are standard
  architectures, so executing repository code is never required and is opt-in.
* Errors are wrapped in :class:`ModelLoadError` with the cause attached, so callers see one
  exception type with an actionable message.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from scale_aware_compression.config import ModelConfig
from scale_aware_compression.constants import Device, DType
from scale_aware_compression.hardware import resolve_device
from scale_aware_compression.logging_utils import get_logger
from scale_aware_compression.models.registry import ModelSpec, get_model_spec

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps torch out of import time
    import torch
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

LOGGER = get_logger(__name__)


class ModelLoadError(RuntimeError):
    """Raised when a tokeniser or model cannot be loaded."""


@dataclass(slots=True)
class LoadedModel:
    """A loaded model with the metadata needed to describe it in a run record."""

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    spec: ModelSpec
    device: Device
    dtype: DType
    parameter_count: int
    revision: str | None = None

    def describe(self) -> dict[str, Any]:
        """Return a serialisable summary for the run record."""
        return {
            "model_name": self.spec.short_name,
            "hf_id": self.spec.hf_id,
            "size_label": self.spec.size_label,
            "revision": self.revision,
            "device": self.device.value,
            "dtype": self.dtype.value,
            "parameter_count": self.parameter_count,
            "architecture": self.spec.architecture,
        }


def _require_transformers() -> Any:
    """Import ``transformers`` on demand, with an actionable error if it is missing."""
    try:
        import transformers
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ModelLoadError(
            "transformers is required to load models. Install the project dependencies "
            "with `pip install -e .` or `pip install -r requirements.txt`."
        ) from error
    return transformers


def _torch_dtype(dtype: DType) -> torch.dtype:
    """Map the config dtype enum onto a torch dtype."""
    try:
        import torch
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ModelLoadError(
            "PyTorch is required to load models. Install the wheel for your platform first, "
            "e.g. `pip install torch --index-url https://download.pytorch.org/whl/cpu`."
        ) from error
    mapping = {
        DType.FLOAT32: torch.float32,
        DType.FLOAT16: torch.float16,
        DType.BFLOAT16: torch.bfloat16,
    }
    return mapping[dtype]


def load_tokenizer(config: ModelConfig) -> PreTrainedTokenizerBase:
    """Load the tokeniser for a registered model.

    A padding token is added when the checkpoint has none, which is the case for both
    Pythia and Qwen base models; batched evaluation needs one.

    Args:
        config: Model section of an experiment config.

    Returns:
        The tokeniser.

    Raises:
        ModelLoadError: If transformers is missing or the tokeniser cannot be fetched.
        UnknownModelError: If ``config.name`` is not registered.
    """
    spec = get_model_spec(config.name)
    hf_id = config.hf_id or spec.hf_id
    transformers = _require_transformers()
    LOGGER.info("Loading tokenizer %s (revision=%s)", hf_id, config.revision or "default")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            hf_id,
            revision=config.revision,
            trust_remote_code=config.trust_remote_code,
            local_files_only=config.local_files_only,
        )
    except Exception as error:
        raise ModelLoadError(
            f"Could not load tokenizer for {config.name!r} ({hf_id}): {error}"
        ) from error

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        LOGGER.debug("Tokenizer had no pad token; using eos_token as pad_token")
    return tokenizer


def load_model(config: ModelConfig) -> tuple[PreTrainedModel, Device, DType]:
    """Load a decoder-only causal language model.

    Args:
        config: Model section of an experiment config.

    Returns:
        A tuple of the model, the resolved device, and the dtype it was loaded in.

    Raises:
        ModelLoadError: If transformers/torch are missing, or loading fails.
        HardwareError: If ``device: cuda`` is requested without a CUDA device.
        UnknownModelError: If ``config.name`` is not registered.
    """
    spec = get_model_spec(config.name)
    hf_id = config.hf_id or spec.hf_id
    device = resolve_device(config.device)
    dtype = config.dtype
    transformers = _require_transformers()
    torch_dtype = _torch_dtype(dtype)

    if device is Device.CPU and dtype is not DType.FLOAT32:
        LOGGER.warning(
            "dtype=%s on CPU: reduced-precision CPU kernels are uneven, and the dense "
            "baseline for this study is defined as FP32.",
            dtype.value,
        )

    keyword_arguments: dict[str, Any] = {
        "revision": config.revision,
        "torch_dtype": torch_dtype,
        "trust_remote_code": config.trust_remote_code,
        "local_files_only": config.local_files_only,
    }
    if config.attn_implementation:
        keyword_arguments["attn_implementation"] = config.attn_implementation

    LOGGER.info(
        "Loading model %s (revision=%s, dtype=%s, device=%s)",
        hf_id,
        config.revision or "default",
        dtype.value,
        device.value,
    )
    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(hf_id, **keyword_arguments)
    except Exception as error:
        raise ModelLoadError(f"Could not load model {config.name!r} ({hf_id}): {error}") from error

    try:
        model.to(device.value)
    except Exception as error:
        raise ModelLoadError(
            f"Could not move {config.name!r} to {device.value}: {error}"
        ) from error

    if config.eval_mode:
        model.eval()
        LOGGER.debug("Model set to eval mode")

    return model, device, dtype


def load_model_and_tokenizer(config: ModelConfig) -> LoadedModel:
    """Load a model and its tokeniser together and count the parameters.

    Args:
        config: Model section of an experiment config.

    Returns:
        A :class:`LoadedModel` bundling both objects with their metadata.

    Raises:
        ModelLoadError: If either artefact cannot be loaded.
    """
    from scale_aware_compression.metrics.compression import count_parameters

    spec = get_model_spec(config.name)
    tokenizer = load_tokenizer(config)
    model, device, dtype = load_model(config)
    parameter_count = count_parameters(model)

    if (
        spec.parameter_count
        and abs(parameter_count - spec.parameter_count) > spec.parameter_count // 100
    ):
        LOGGER.warning(
            "%s reports %d parameters but the registry expects %d; check the revision pin.",
            spec.short_name,
            parameter_count,
            spec.parameter_count,
        )

    LOGGER.info(
        "Loaded %s: %.1fM parameters on %s",
        spec.short_name,
        parameter_count / 1e6,
        device.value,
    )
    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        spec=spec,
        device=device,
        dtype=dtype,
        parameter_count=parameter_count,
        revision=config.revision,
    )


def prefetch(config: ModelConfig) -> ModelSpec:
    """Download a checkpoint into the local cache without instantiating it.

    Used by ``scripts/download_models.py`` so that a sweep does not stall on downloads
    part-way through.

    Args:
        config: Model section of an experiment config.

    Returns:
        The registry spec for the fetched model.

    Raises:
        ModelLoadError: If the snapshot cannot be downloaded.
    """
    spec = get_model_spec(config.name)
    hf_id = config.hf_id or spec.hf_id
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:  # pragma: no cover - shipped with transformers
        raise ModelLoadError(
            "huggingface_hub is required to prefetch checkpoints; it is installed with "
            "transformers."
        ) from error

    LOGGER.info("Prefetching %s into the local Hugging Face cache", hf_id)
    try:
        snapshot_download(repo_id=hf_id, revision=config.revision)
    except Exception as error:
        raise ModelLoadError(f"Could not prefetch {hf_id}: {error}") from error
    return spec
