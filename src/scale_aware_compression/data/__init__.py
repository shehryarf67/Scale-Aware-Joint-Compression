"""Corpus loading, tokenisation, and calibration sampling.

No dataset is downloaded at import time; every loader is an explicit call.
"""

from __future__ import annotations

from scale_aware_compression.data.calibration import (
    CalibrationError,
    CalibrationSummary,
    cache_calibration_set,
    load_calibration_set,
    select_calibration_indices,
)
from scale_aware_compression.data.loaders import (
    DataError,
    DatasetSummary,
    build_dataloader,
    build_evaluation_dataloader,
    build_language_modelling_dataset,
    load_raw_dataset,
)
from scale_aware_compression.data.preprocessing import (
    chunk_sequence,
    fingerprint_token_ids,
    prepare_dataset,
    tokenise_corpus,
)

__all__ = [
    "CalibrationError",
    "CalibrationSummary",
    "DataError",
    "DatasetSummary",
    "build_dataloader",
    "build_evaluation_dataloader",
    "build_language_modelling_dataset",
    "cache_calibration_set",
    "chunk_sequence",
    "fingerprint_token_ids",
    "load_calibration_set",
    "load_raw_dataset",
    "prepare_dataset",
    "select_calibration_indices",
    "tokenise_corpus",
]
