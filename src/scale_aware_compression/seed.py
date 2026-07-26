"""Deterministic seeding across Python, NumPy, and PyTorch.

Every entry point seeds once, at the top of ``main()``, and records the returned mapping in
the run record so a run can be reproduced. Optional dependencies are imported lazily and
their absence is reported rather than raised, so seeding works in a torch-free environment
such as the test suite.
"""

from __future__ import annotations

import os
import random
from typing import Any

from scale_aware_compression.logging_utils import get_logger

LOGGER = get_logger(__name__)

CUBLAS_WORKSPACE_ENV_VAR = "CUBLAS_WORKSPACE_CONFIG"
PYTHONHASHSEED_ENV_VAR = "PYTHONHASHSEED"


def set_global_seed(seed: int, *, deterministic: bool = True) -> dict[str, Any]:
    """Seed every random source this project uses.

    Args:
        seed: Non-negative seed applied to ``random``, NumPy, and PyTorch (CPU and CUDA).
        deterministic: Also request deterministic algorithms from cuDNN and PyTorch. This
            can slow training down; it is on by default because reproducibility matters
            more here than throughput.

    Returns:
        A mapping describing what was actually seeded, e.g.
        ``{"seed": 1234, "python": True, "numpy": True, "torch": False, ...}``. Suitable
        for embedding in a run record.

    Raises:
        ValueError: If ``seed`` is negative.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    record: dict[str, Any] = {
        "seed": seed,
        "deterministic": deterministic,
        "python": True,
        "numpy": False,
        "torch": False,
        "torch_cuda": False,
    }

    # PYTHONHASHSEED only affects interpreters started after it is set; recorded so a
    # reproduction attempt knows to export it.
    os.environ.setdefault(PYTHONHASHSEED_ENV_VAR, str(seed))
    random.seed(seed)

    try:
        import numpy as np
    except ImportError:
        LOGGER.debug("NumPy not installed; skipping NumPy seeding")
    else:
        np.random.seed(seed)
        record["numpy"] = True

    try:
        import torch
    except ImportError:
        LOGGER.debug("PyTorch not installed; skipping torch seeding")
        return record

    torch.manual_seed(seed)
    record["torch"] = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        record["torch_cuda"] = True

    if deterministic:
        # Required for deterministic CUDA matmuls; harmless on CPU-only machines.
        os.environ.setdefault(CUBLAS_WORKSPACE_ENV_VAR, ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except (
            AttributeError,
            RuntimeError,
        ) as error:  # pragma: no cover - torch version dependent
            LOGGER.warning("Could not enable deterministic algorithms: %s", error)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    LOGGER.debug("Seeded: %s", record)
    return record


def seed_worker(worker_id: int) -> None:
    """Seed a DataLoader worker deterministically from the parent process seed.

    Pass as ``DataLoader(worker_init_fn=seed_worker)``.

    Args:
        worker_id: Index supplied by PyTorch. Unused directly; the per-worker seed comes
            from ``torch.initial_seed()``, which already differs per worker.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover - only reached without torch
        LOGGER.debug("PyTorch not installed; cannot seed worker %d", worker_id)
        return

    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - only reached without numpy
        return
    np.random.seed(worker_seed)
