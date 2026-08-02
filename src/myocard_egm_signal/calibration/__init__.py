"""Per-record amplitude calibration.

A :class:`CalibrationStrategy` computes a single scalar per record
such that ``calibrated_signal = scalar * raw_signal`` is in
physiological units. The shipped :class:`RWaveAnchoring` strategy
uses the median surface-ECG QRS peak-to-peak amplitude as the
in-band reference.

Future strategies (percentile-based, fixed-gain, manual) drop into
their own sibling modules here and are re-exported below.

Extending
---------
Write a class that satisfies the :class:`CalibrationStrategy`
Protocol::

    class PercentileBasedCalibration:
        name = "percentile_based"
        def __init__(self, ...): ...
        def compute(self, record: Record) -> Calibration:
            ...
"""

from __future__ import annotations

from ._helpers import compute_calibration
from .base import Calibration, CalibrationStrategy
from .qrs_estimation import DEFAULT_PREFERRED_LEADS, estimate_qrs_peak_to_peak
from .r_wave_anchoring import RWaveAnchoring

__all__ = [
    "DEFAULT_PREFERRED_LEADS",
    "Calibration",
    "CalibrationStrategy",
    "RWaveAnchoring",
    "compute_calibration",
    "estimate_qrs_peak_to_peak",
]
