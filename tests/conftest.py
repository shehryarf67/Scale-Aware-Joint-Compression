"""Shared fixtures.

Everything here is offline and cheap. No test downloads a model, imports torch, trains, or runs a
real benchmark; the fakes below stand in for tensors and modules so the metric and benchmark
functions can be tested against known values.
"""

from __future__ import annotations

from collections.abc import Iterator
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
            "dataset": "wikitext",
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
# Latency samples
# ---------------------------------------------------------------------------
@pytest.fixture
def latency_samples_seconds() -> list[float]:
    """Ten known latencies in seconds, 10 ms to 100 ms in 10 ms steps.

    Chosen so every statistic is hand-checkable: mean 55 ms, median 55 ms, p95 95.5 ms with linear
    interpolation, sample std 30.2765 ms.
    """
    return [round(0.01 * index, 10) for index in range(1, 11)]
