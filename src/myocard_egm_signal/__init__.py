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
from .exceptions import (
    ConstantSignalError,
    DegenerateSignalError,
    EmptySignalError,
)
from .extraction import (
    DEFAULT_BIPOLAR_BAND_HZ,
    DEFAULT_BOTTERON_BAND_HZ,
    DEFAULT_BOTTERON_LOWPASS_HZ,
    MIN_WINDOW_LENGTH_SAMPLES,
    ActivationCandidate,
    ActivationComplex,
    ActivationPositionGenerator,
    ActivationWindower,
    AnchoredWindow,
    BotteronEnvelope,
    CandidateSelector,
    DetectionPreprocessor,
    GreedyHeightSuppressor,
    HealthySegment,
    LocalMaximaSelector,
    MultiActivationWindower,
    NoiseSegment,
    RectifiedDerivative,
    RefractorySuppressor,
    SingleActivationWindower,
    TeagerKaiser,
    TwoStageRefiner,
    UniformPositionGenerator,
    WindowSet,
    detect_activation,
    detect_activation_train,
    extract_healthy_segments,
    extract_noise_segments,
    measure_complex,
    measure_complexes,
    window_train,
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
    MedianMadThreshold,
    NoiseSegmentStrategy,
    NoThreshold,
    PeakFractionThreshold,
    PercentileQuietThreshold,
    PercentileSignalThreshold,
    PercentileThreshold,
    PositionAwareSignalThreshold,
    SignalThreshold,
    SignalThresholdLike,
    ThresholdStrategy,
    median_absolute_deviation,
)
from .windowing import sliding_window_peak_to_peak

try:
    __version__ = metadata.version("myocard-egm-signal")
except metadata.PackageNotFoundError:  # pragma: no cover — editable install without metadata
    __version__ = "0.0.0+unknown"


__all__ = [
    "DEFAULT_BIPOLAR_BAND_HZ",
    "DEFAULT_BOTTERON_BAND_HZ",
    "DEFAULT_BOTTERON_LOWPASS_HZ",
    "DEFAULT_PREFERRED_LEADS",
    "DEFAULT_TEMPERATURE_BOUNDS",
    "MIN_WINDOW_LENGTH_SAMPLES",
    "AbsoluteQuietThreshold",
    "AbsoluteThreshold",
    "ActivationCandidate",
    "ActivationComplex",
    "ActivationPositionGenerator",
    "ActivationWindower",
    "AnchoredWindow",
    "BotteronEnvelope",
    "Calibration",
    "CalibrationStrategy",
    "CandidateSelector",
    "ConstantSignalError",
    "DegenerateSignalError",
    "DetectionPreprocessor",
    "EmptySignalError",
    "GreedyHeightSuppressor",
    "HealthySegment",
    "LocalMaximaSelector",
    "MedianMadThreshold",
    "MultiActivationWindower",
    "NoThreshold",
    "NoiseSegment",
    "NoiseSegmentStrategy",
    "PeakFractionThreshold",
    "PercentileQuietThreshold",
    "PercentileSignalThreshold",
    "PercentileThreshold",
    "PositionAwareSignalThreshold",
    "RWaveAnchoring",
    "Record",
    "RectifiedDerivative",
    "RefractorySuppressor",
    "SignalThreshold",
    "SignalThresholdLike",
    "SingleActivationWindower",
    "TeagerKaiser",
    "ThresholdStrategy",
    "TwoStageRefiner",
    "UniformPositionGenerator",
    "WindowSet",
    "__version__",
    "apply_temperature",
    "bandpass",
    "compute_calibration",
    "detect_activation",
    "detect_activation_train",
    "estimate_qrs_peak_to_peak",
    "extract_healthy_segments",
    "extract_noise_segments",
    "fit_temperature",
    "lowpass",
    "measure_complex",
    "measure_complexes",
    "median_absolute_deviation",
    "sliding_window_peak_to_peak",
    "window_train",
]
