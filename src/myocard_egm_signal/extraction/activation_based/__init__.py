"""Activation-based extraction: detect activations, then window on them.

Where the sliding-window extractors in :mod:`..extractors` cut a record
at a fixed stride, this subpackage cuts it **relative to the activations
it contains** — detect where activation happens, then place a
fixed-length window at a chosen position relative to that anchor.

Vocabulary (see :mod:`.base`): detection is a chain of three stages, and
only the last one detects.

1. **Detection preprocessing** — :class:`DetectionPreprocessor` and its
   three implementations transform a channel into a curve ``g`` that
   emphasises activations.
2. **Detection thresholding** — the decision rule over ``g``
   (``thresholds.detection``).
3. **Candidate selection** — the local maxima of ``g`` at or above the
   threshold (:class:`CandidateSelector`).
4. **Refractory suppression** — which candidates are *distinct*
   activations (:class:`RefractorySuppressor`).
5. **The detection function** — the whole chain
   (:func:`detect_activation_train`, and :func:`detect_activation` for
   the single-activation synthetic case), which is what actually
   answers "where are the activations?"

Public surface, built up across SIG1; more lands in later steps.
"""

from __future__ import annotations

from .base import DetectionPreprocessor
from .candidates import ActivationCandidate, CandidateSelector, LocalMaximaSelector
from .detection import TwoStageRefiner, detect_activation, detect_activation_train
from .preprocessors import (
    DEFAULT_BOTTERON_BAND_HZ,
    DEFAULT_BOTTERON_LOWPASS_HZ,
    BotteronEnvelope,
    RectifiedDerivative,
    TeagerKaiser,
)
from .suppression import GreedyHeightSuppressor, RefractorySuppressor

__all__ = [
    "DEFAULT_BOTTERON_BAND_HZ",
    "DEFAULT_BOTTERON_LOWPASS_HZ",
    "ActivationCandidate",
    "BotteronEnvelope",
    "CandidateSelector",
    "DetectionPreprocessor",
    "GreedyHeightSuppressor",
    "LocalMaximaSelector",
    "RectifiedDerivative",
    "RefractorySuppressor",
    "TeagerKaiser",
    "TwoStageRefiner",
    "detect_activation",
    "detect_activation_train",
]
