"""Exception types for the data subpackage.

These live in their own module because :mod:`loaders`, :mod:`preprocessing`, and
:mod:`calibration` all raise them and all import from one another; a shared leaf module keeps
that from becoming a cycle.
"""

from __future__ import annotations


class DataError(RuntimeError):
    """Raised when a dataset cannot be loaded or is unusable as configured."""


class CalibrationError(DataError):
    """Raised when a calibration set cannot be built as configured.

    A subclass of :class:`DataError`, so a caller that only wants to know "the data step failed"
    can catch the parent.
    """
