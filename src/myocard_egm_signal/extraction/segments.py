"""Segment dataclasses produced by the extractors.

Two parallel types — :class:`HealthySegment` and :class:`NoiseSegment`
— with identical fields but distinct names so consumer code's type
signatures communicate which kind of segment they accept.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HealthySegment:
    """One high-voltage bipolar segment selected for the healthy class.

    ``signal`` carries the calibrated, band-pass-filtered samples
    (length ``end_sample - start_sample``).
    """

    record_name: str
    patient: str
    channel: str
    start_sample: int
    end_sample: int
    fs: float
    peak_to_peak_mv: float
    signal: np.ndarray


@dataclass(frozen=True)
class NoiseSegment:
    """One quiet bipolar window selected as background noise.

    ``signal`` carries the band-pass-filtered samples (calibrated if
    the caller pre-calibrated the record; raw otherwise).
    """

    record_name: str
    patient: str
    channel: str
    start_sample: int
    end_sample: int
    fs: float
    peak_to_peak_mv: float
    signal: np.ndarray
