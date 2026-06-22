"""Pre/post-processing math for ML models.

Sits alongside the signal-processing subpackages (``filters``,
``thresholds``, ``calibration`` for **signal-amplitude** calibration,
``extraction``) but is conceptually separate: these functions operate
on classifier *outputs* (logits, probabilities) rather than raw EGM
signals.

Today: temperature scaling for probability calibration. Future
additions (Platt scaling, isotonic regression, multi-class softmax
calibration, alternative reliability-aware fits) land here as
sibling modules.

Naming note: the sibling :mod:`myocard_egm_signal.calibration`
subpackage handles signal-amplitude calibration (R-wave anchoring +
QRS estimation against IAFDB recordings). Probability calibration is
a different concept that shares the English word; we keep them in
separate subpackages so the namespace stays unambiguous.
"""

from __future__ import annotations

from .temperature_scaling import (
    DEFAULT_TEMPERATURE_BOUNDS,
    apply_temperature,
    fit_temperature,
)

__all__ = [
    "DEFAULT_TEMPERATURE_BOUNDS",
    "apply_temperature",
    "fit_temperature",
]
