"""Hardware and software introspection, plus device resolution.

Every run record embeds the output of :func:`get_hardware_info` and
:func:`get_software_versions`. Without those, CPU latency numbers from different machines
are indistinguishable in the results table, and the CPU-only comparison protocol cannot be
audited after the fact.

All third-party imports are lazy and failures degrade to ``None`` rather than raising, so
this module works in a minimal environment.
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import sys
from typing import Any

from scale_aware_compression.constants import Device
from scale_aware_compression.logging_utils import get_logger

LOGGER = get_logger(__name__)

_VERSION_PACKAGES = (
    "torch",
    "transformers",
    "datasets",
    "accelerate",
    "evaluate",
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
    "safetensors",
)

_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


class HardwareError(RuntimeError):
    """Raised when a requested device is unavailable."""


def cuda_available() -> bool:
    """Report whether a usable CUDA device is present.

    Returns:
        ``True`` only if torch is installed and reports at least one CUDA device.
    """
    try:
        import torch
    except ImportError:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception as error:  # pragma: no cover - driver-level failures
        LOGGER.warning("torch.cuda.is_available() raised: %s", error)
        return False


def resolve_device(requested: Device | str) -> Device:
    """Turn a requested device into a concrete one.

    Args:
        requested: ``auto``, ``cpu``, or ``cuda``.

    Returns:
        :attr:`Device.CPU` or :attr:`Device.CUDA`. ``auto`` prefers CUDA when available.

    Raises:
        HardwareError: If CUDA is requested explicitly but is not available.
    """
    device = Device(requested) if not isinstance(requested, Device) else requested
    if device is Device.CUDA:
        if not cuda_available():
            raise HardwareError(
                "device='cuda' was requested but no CUDA device is available. Install a CUDA "
                "build of torch, or set device to 'cpu' / 'auto'."
            )
        return Device.CUDA
    if device is Device.AUTO:
        resolved = Device.CUDA if cuda_available() else Device.CPU
        LOGGER.debug("device='auto' resolved to %s", resolved.value)
        return resolved
    return Device.CPU


def set_cpu_threads(num_threads: int, interop_threads: int | None = None) -> dict[str, Any]:
    """Pin the PyTorch CPU thread count for reproducible latency measurement.

    Thread-count environment variables are also set, because BLAS libraries read them at
    first use and would otherwise ignore the torch setting.

    Args:
        num_threads: Intra-op thread count. Must be >= 1.
        interop_threads: Optional inter-op thread count. PyTorch only allows this to be set
            once per process and before any parallel work; a failure is logged, not raised.

    Returns:
        A mapping of what was requested and what torch reports afterwards.

    Raises:
        ValueError: If ``num_threads`` is less than 1.
    """
    if num_threads < 1:
        raise ValueError(f"num_threads must be >= 1, got {num_threads}")

    for variable in _THREAD_ENV_VARS:
        os.environ[variable] = str(num_threads)

    report: dict[str, Any] = {
        "requested_num_threads": num_threads,
        "requested_interop_threads": interop_threads,
        "torch_num_threads": None,
        "torch_num_interop_threads": None,
    }

    try:
        import torch
    except ImportError:
        LOGGER.warning("PyTorch not installed; only thread environment variables were set")
        return report

    torch.set_num_threads(num_threads)
    if interop_threads is not None:
        try:
            torch.set_num_interop_threads(interop_threads)
        except RuntimeError as error:
            LOGGER.warning(
                "Could not set inter-op threads to %d (must be set before any parallel work): %s",
                interop_threads,
                error,
            )
    report["torch_num_threads"] = torch.get_num_threads()
    report["torch_num_interop_threads"] = torch.get_num_interop_threads()
    LOGGER.debug("CPU threads pinned: %s", report)
    return report


def get_hardware_info() -> dict[str, Any]:
    """Collect the machine description embedded in every run record.

    Returns:
        A mapping with platform, CPU, memory, thread, and accelerator fields. Values that
        cannot be determined are ``None``.
    """
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_model": platform.processor() or platform.machine() or None,
        "cpu_count_logical": os.cpu_count(),
        "cpu_count_physical": None,
        "cpu_max_frequency_mhz": None,
        "total_memory_gb": None,
        "available_memory_gb": None,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "torch_num_threads": None,
        "torch_num_interop_threads": None,
        "thread_env": {name: os.environ.get(name) for name in _THREAD_ENV_VARS},
        "cuda_available": False,
        "cuda_device_count": 0,
        "cuda_device_names": [],
    }

    try:
        import psutil
    except ImportError:
        LOGGER.debug("psutil not installed; memory and physical core count unavailable")
    else:
        info["cpu_count_physical"] = psutil.cpu_count(logical=False)
        memory = psutil.virtual_memory()
        info["total_memory_gb"] = round(memory.total / 1024**3, 3)
        info["available_memory_gb"] = round(memory.available / 1024**3, 3)
        frequency = psutil.cpu_freq()
        if frequency is not None:
            info["cpu_max_frequency_mhz"] = frequency.max or None

    try:
        import torch
    except ImportError:
        LOGGER.debug("PyTorch not installed; thread and CUDA details unavailable")
        return info

    info["torch_num_threads"] = torch.get_num_threads()
    info["torch_num_interop_threads"] = torch.get_num_interop_threads()
    if cuda_available():
        info["cuda_available"] = True
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_device_names"] = [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ]
    return info


def get_software_versions() -> dict[str, str | None]:
    """Resolve installed versions of the packages that can change results.

    Returns:
        A mapping from distribution name to version string, with ``None`` for packages that
        are not installed.
    """
    versions: dict[str, str | None] = {
        "python": sys.version.split()[0],
        "scale_aware_compression": None,
    }
    try:
        versions["scale_aware_compression"] = importlib.metadata.version("scale-aware-compression")
    except importlib.metadata.PackageNotFoundError:
        # Running from a source checkout without an install; fall back to the package
        # constant so the field is never silently empty.
        from scale_aware_compression import __version__

        versions["scale_aware_compression"] = f"{__version__}+source"

    for package in _VERSION_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def describe_environment() -> dict[str, Any]:
    """Return hardware and software information as a single nested mapping.

    Returns:
        ``{"hardware": ..., "software": ...}``, the shape stored in a run record.
    """
    return {"hardware": get_hardware_info(), "software": get_software_versions()}
