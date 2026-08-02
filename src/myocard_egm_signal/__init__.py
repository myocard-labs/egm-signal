"""myocard_egm_signal — generic signal-processing primitives for intracardiac EGM data.

Pure numpy + scipy. No internal dependencies on other myocard-labs
repos. Consumed by the producer pipelines (iafdb-pipeline,
synthetic-egm-pipeline) and the feature library (egm-features).

Public surface (re-exported for ergonomic import):

- :class:`Record` — Protocol the extractors consume.
- :func:`bandpass` — zero-phase Butterworth band-pass.
- :func:`sliding_window_peak_to_peak` — generic sliding p-p.
- :class:`ThresholdStrategy` + :class:`AbsoluteThreshold` +
  :class:`PercentileThreshold` + :class:`NoThreshold` — keep-above
  selection.
- :class:`NoiseSegmentStrategy` + :class:`AbsoluteQuietThreshold` +
  :class:`PercentileQuietThreshold` — keep-below selection.
- :class:`Calibration` + :class:`CalibrationStrategy` +
  :class:`RWaveAnchoring` + :func:`compute_calibration` +
  :func:`estimate_qrs_peak_to_peak` — signal-amplitude calibration
  (R-wave anchoring against IAFDB recordings).
- :class:`HealthySegment` + :class:`NoiseSegment` +
  :func:`extract_healthy_segments` + :func:`extract_noise_segments` +
  :data:`DEFAULT_BIPOLAR_BAND_HZ` — record-Protocol-driven extractors.
- :func:`fit_temperature` + :func:`apply_temperature` +
  :data:`DEFAULT_TEMPERATURE_BOUNDS` — probability calibration
  (temperature scaling) for ML model outputs.

Subpackage layout (light grouping per project/architecture.md):
``filters/``, ``calibration/`` (signal-amplitude), ``thresholds/``,
``extraction/``, and ``model/`` (ML pre/post-processing math) are
folders that group multiple files or are likely to grow; ``records``
and ``windowing`` stay flat as single-module domains.
"""

from __future__ import annotations

from importlib import metadata

from .calibration import (
    DEFAULT_PREFERRED_LEADS,
    Calibration,
    CalibrationStrategy,
    RWaveAnchoring,
    compute_calibration,
    estimate_qrs_peak_to_peak,
)
from .extraction import (
    DEFAULT_BIPOLAR_BAND_HZ,
    HealthySegment,
    NoiseSegment,
    extract_healthy_segments,
    extract_noise_segments,
)
from .filters import bandpass, lowpass
from .model import (
    DEFAULT_TEMPERATURE_BOUNDS,
    apply_temperature,
    fit_temperature,
)
from .records import Record
from .thresholds import (
    AbsoluteQuietThreshold,
    AbsoluteThreshold,
    NoiseSegmentStrategy,
    NoThreshold,
    PercentileQuietThreshold,
    PercentileThreshold,
    ThresholdStrategy,
)
from .windowing import sliding_window_peak_to_peak

try:
    __version__ = metadata.version("myocard-egm-signal")
except metadata.PackageNotFoundError:  # pragma: no cover — editable install without metadata
    __version__ = "0.0.0+unknown"


__all__ = [
    "DEFAULT_BIPOLAR_BAND_HZ",
    "DEFAULT_PREFERRED_LEADS",
    "DEFAULT_TEMPERATURE_BOUNDS",
    "AbsoluteQuietThreshold",
    "AbsoluteThreshold",
    "Calibration",
    "CalibrationStrategy",
    "HealthySegment",
    "NoThreshold",
    "NoiseSegment",
    "NoiseSegmentStrategy",
    "PercentileQuietThreshold",
    "PercentileThreshold",
    "RWaveAnchoring",
    "Record",
    "ThresholdStrategy",
    "__version__",
    "apply_temperature",
    "bandpass",
    "compute_calibration",
    "estimate_qrs_peak_to_peak",
    "extract_healthy_segments",
    "extract_noise_segments",
    "fit_temperature",
    "lowpass",
    "sliding_window_peak_to_peak",
]
