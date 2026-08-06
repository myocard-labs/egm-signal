"""Record-Protocol-driven segment extractors.

Public surface:

- :class:`HealthySegment`, :class:`NoiseSegment` — segment dataclasses.
- :func:`extract_healthy_segments`, :func:`extract_noise_segments` —
  the extractors themselves.
- :data:`DEFAULT_BIPOLAR_BAND_HZ` — clinical 30-300 Hz default band.

The :mod:`.activation_based` subpackage cuts a record **relative to the
activations it contains** rather than at a fixed stride; its surface is
re-exported here and from the package root.
"""

from __future__ import annotations

from .activation_based import (
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
    LocalMaximaSelector,
    MultiActivationWindower,
    RectifiedDerivative,
    RefractorySuppressor,
    SingleActivationWindower,
    TeagerKaiser,
    TwoStageRefiner,
    UniformPositionGenerator,
    WindowSet,
    detect_activation,
    detect_activation_train,
    measure_complex,
    measure_complexes,
    window_train,
)
from .extractors import (
    DEFAULT_BIPOLAR_BAND_HZ,
    extract_healthy_segments,
    extract_noise_segments,
)
from .segments import HealthySegment, NoiseSegment

__all__ = [
    "DEFAULT_BIPOLAR_BAND_HZ",
    "DEFAULT_BOTTERON_BAND_HZ",
    "DEFAULT_BOTTERON_LOWPASS_HZ",
    "MIN_WINDOW_LENGTH_SAMPLES",
    "ActivationCandidate",
    "ActivationComplex",
    "ActivationPositionGenerator",
    "ActivationWindower",
    "AnchoredWindow",
    "BotteronEnvelope",
    "CandidateSelector",
    "DetectionPreprocessor",
    "GreedyHeightSuppressor",
    "HealthySegment",
    "LocalMaximaSelector",
    "MultiActivationWindower",
    "NoiseSegment",
    "RectifiedDerivative",
    "RefractorySuppressor",
    "SingleActivationWindower",
    "TeagerKaiser",
    "TwoStageRefiner",
    "UniformPositionGenerator",
    "WindowSet",
    "detect_activation",
    "detect_activation_train",
    "extract_healthy_segments",
    "extract_noise_segments",
    "measure_complex",
    "measure_complexes",
    "window_train",
]
