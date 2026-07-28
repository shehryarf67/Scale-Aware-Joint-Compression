"""Shared fixtures.

Everything here is offline and cheap. No test downloads a model, imports torch, trains, or runs a
real benchmark; the fakes below stand in for tensors and modules so the metric and benchmark
functions can be tested against known values.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml

from scale_aware_compression.constants import PROJECT_ROOT


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def project_root() -> Path:
    """Repository root."""
    return PROJECT_ROOT


# ---------------------------------------------------------------------------
# Import side-effect checking
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def imported_after():
    """Return :func:`_imported_after`, for tests that assert on import side effects."""
    return _imported_after


@pytest.fixture(scope="session")
def environment_after_import():
    """Return :func:`_environment_after_import`."""
    return _environment_after_import


def _imported_after(module: str, candidates: Sequence[str]) -> list[str]:
    """Import ``module`` in a fresh interpreter and report which candidates it pulled in.

    A subprocess rather than ``sys.modules.pop()``: torch registers C++ operators at import, and
    re-importing it in a live interpreter raises ``Only a single TORCH_LIBRARY can be used``.
    A clean process is both safe and a more honest test of what a cold import actually costs.

    Args:
        module: Dotted module path to import.
        candidates: Module names to look for afterwards.

    Returns:
        The subset of ``candidates`` present in ``sys.modules`` after the import.
    """
    program = textwrap.dedent(f"""
        import json
        import sys

        import {module}  # noqa: F401

        print(json.dumps([name for name in {list(candidates)!r} if name in sys.modules]))
    """)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    completed = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        env=environment,
        timeout=120,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"Importing {module} in a fresh interpreter failed:\n{completed.stderr}"
        )
    return list(json.loads(completed.stdout.strip().splitlines()[-1]))


def _environment_after_import(module: str, variables: Sequence[str]) -> dict[str, str | None]:
    """Import ``module`` in a fresh interpreter and report the named environment variables.

    Args:
        module: Dotted module path to import.
        variables: Environment variable names to read afterwards.

    Returns:
        Mapping from variable name to its value, or ``None`` when unset.
    """
    program = textwrap.dedent(f"""
        import json
        import os

        import {module}  # noqa: F401

        print(json.dumps({{name: os.environ.get(name) for name in {list(variables)!r}}}))
    """)
    environment = {key: value for key, value in os.environ.items() if key not in set(variables)}
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    completed = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        env=environment,
        timeout=120,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"Importing {module} in a fresh interpreter failed:\n{completed.stderr}"
        )
    return dict(json.loads(completed.stdout.strip().splitlines()[-1]))


@pytest.fixture(scope="session")
def configs_dir(project_root: Path) -> Path:
    """The shipped ``configs/`` directory."""
    return project_root / "configs"


# ---------------------------------------------------------------------------
# Torch stand-ins
# ---------------------------------------------------------------------------
class FakeTensor:
    """Minimal stand-in for a torch tensor.

    Implements only ``numel()`` and ``count_nonzero()``, which is exactly the surface the metric
    functions use. ``count_nonzero`` is real torch API, so the production code path is the one
    being exercised rather than a test-only branch.
    """

    def __init__(self, values: list[float], *, requires_grad: bool = True) -> None:
        self.values = list(values)
        self.requires_grad = requires_grad

    def numel(self) -> int:
        return len(self.values)

    def count_nonzero(self) -> int:
        return sum(1 for value in self.values if value != 0)

    def tolist(self) -> list[float]:
        return list(self.values)


class FakeModule:
    """Minimal stand-in for ``torch.nn.Module``: just ``parameters()``."""

    def __init__(self, tensors: list[FakeTensor]) -> None:
        self._tensors = tensors

    def parameters(self) -> Iterator[FakeTensor]:
        yield from self._tensors


@pytest.fixture
def dense_module() -> FakeModule:
    """A module with 10 parameters and no zeros."""
    return FakeModule([FakeTensor([1.0] * 6), FakeTensor([2.0] * 4)])


@pytest.fixture
def half_sparse_module() -> FakeModule:
    """A module with 10 parameters, 5 of them zero."""
    return FakeModule(
        [FakeTensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0]), FakeTensor([0.0, 2.0, 0.0, 2.0])]
    )


@pytest.fixture
def frozen_module() -> FakeModule:
    """A module with 4 trainable and 6 frozen parameters."""
    return FakeModule(
        [FakeTensor([1.0] * 4, requires_grad=True), FakeTensor([1.0] * 6, requires_grad=False)]
    )


# ---------------------------------------------------------------------------
# Configuration documents
# ---------------------------------------------------------------------------
@pytest.fixture
def minimal_config_document() -> dict[str, Any]:
    """A small but complete configuration document."""
    return {
        "experiment": {"id": "unit-test", "name": "Unit test"},
        "runtime": {"seed": 42, "output_dir": "outputs", "log_level": "WARNING"},
        "model": {"name": "pythia-160m", "size_label": "160M", "device": "cpu", "dtype": "float32"},
        "data": {
            "dataset": "Salesforce/wikitext",
            "subset": "wikitext-2-raw-v1",
            "sequence_length": 128,
            "batch_size": 2,
            "calibration_samples": 8,
        },
        "compression": {
            "method": "sequential",
            "budget_label": "moderate",
            "pruning": {"enabled": True, "sparsity": 0.5, "schedule": "cubic"},
            "quantisation": {"enabled": True, "bits": 8},
            "recovery": {"enabled": True, "max_steps": 10},
        },
        "evaluation": {"device": "cpu", "batch_size": 1, "max_samples": 8},
        "benchmark": {
            "device": "cpu",
            "num_threads": 1,
            "warmup_runs": 1,
            "measured_runs": 2,
            "batch_size": 1,
            "sequence_length": 16,
        },
    }


@pytest.fixture
def write_yaml(tmp_path: Path):
    """Return a helper that writes a mapping to a YAML file under ``tmp_path``."""

    def _write(name: str, document: dict[str, Any]) -> Path:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        return path

    return _write


@pytest.fixture
def config_file(write_yaml, minimal_config_document: dict[str, Any]) -> Path:
    """A written YAML file holding the minimal configuration."""
    return write_yaml("config.yaml", minimal_config_document)


# ---------------------------------------------------------------------------
# Offline stand-ins for a tokeniser and a model
#
# Everything below stays offline. The "model" is a randomly initialised two-layer GPT-NeoX built
# from a config object -- same architecture class as Pythia, a few thousand parameters instead of
# 160 million, and no download. That is what lets the whole pipeline be exercised in CI.
# ---------------------------------------------------------------------------
class FakeTokenizer:
    """Byte-level tokeniser with the slice of the Transformers API this project uses.

    Deterministic and dependency-free, so data-pipeline tests need neither a download nor the
    `tokenizers` library.
    """

    name_or_path = "fake-byte-tokenizer"
    vocab_size = 259
    eos_token = "</s>"
    eos_token_id = 256
    pad_token = "<pad>"
    pad_token_id = 257
    unk_token_id = 258
    model_max_length = 1_000_000

    def __call__(
        self,
        text: str | list[str],
        add_special_tokens: bool = False,
        return_tensors: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        texts = [text] if isinstance(text, str) else list(text)
        encoded = [list(item.encode("utf-8")) for item in texts]
        if add_special_tokens:
            encoded = [[*ids, self.eos_token_id] for ids in encoded]

        if return_tensors == "pt":
            import torch

            tensor = torch.tensor(encoded, dtype=torch.long)
            return {"input_ids": tensor, "attention_mask": torch.ones_like(tensor)}
        if isinstance(text, str):
            return {"input_ids": encoded[0], "attention_mask": [1] * len(encoded[0])}
        return {"input_ids": encoded, "attention_mask": [[1] * len(ids) for ids in encoded]}

    def decode(self, token_ids: Any, skip_special_tokens: bool = False) -> str:
        ids = [int(value) for value in token_ids]
        if skip_special_tokens:
            ids = [value for value in ids if value < 256]
        return bytes(value for value in ids if value < 256).decode("utf-8", errors="replace")


@pytest.fixture
def fake_tokenizer() -> FakeTokenizer:
    """A deterministic offline tokeniser."""
    return FakeTokenizer()


@pytest.fixture
def sample_corpus() -> list[str]:
    """A small corpus with blank entries, as WikiText itself has."""
    return [
        "The quick brown fox jumps over the lazy dog. " * 4,
        "",
        "   ",
        "Compression removes weights and reduces precision. " * 4,
        "Perplexity measures how surprised a model is by text. " * 4,
    ]


@pytest.fixture(scope="session")
def tiny_causal_lm() -> Any:
    """A randomly initialised two-layer GPT-NeoX model, built offline from a config.

    Same architecture class as the Pythia suite, so adapters and compression code meet the real
    module names, but small enough to run a full pipeline in under a second.
    """
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    config = transformers.GPTNeoXConfig(
        vocab_size=259,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=128,
        rotary_pct=0.25,
        use_cache=True,
    )
    torch.manual_seed(0)
    model = transformers.GPTNeoXForCausalLM(config)
    model.eval()
    return model


@pytest.fixture
def token_block_dataset() -> Any:
    """A small :class:`TokenBlockDataset` of deterministic token ids."""
    from scale_aware_compression.data.loaders import TokenBlockDataset

    return TokenBlockDataset(
        [[(index * 7 + step) % 259 for step in range(16)] for index in range(8)]
    )


# ---------------------------------------------------------------------------
# Latency samples
# ---------------------------------------------------------------------------
@pytest.fixture
def latency_samples_seconds() -> list[float]:
    """Ten known latencies in seconds, 10 ms to 100 ms in 10 ms steps.

    Chosen so every statistic is hand-checkable: mean 55 ms, median 55 ms, p95 95.5 ms with linear
    interpolation, sample std 30.2765 ms.
    """
    return [round(0.01 * index, 10) for index in range(1, 11)]
