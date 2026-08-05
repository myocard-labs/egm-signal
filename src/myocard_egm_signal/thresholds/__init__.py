"""Threshold strategies for segment selection and activation detection.

Four families, each with its own interface in :mod:`.base`:

- :mod:`.healthy` — keep-above over pooled peak-to-peak amplitudes.
  ``ThresholdStrategy``, ``AbsoluteThreshold``, ``PercentileThreshold``,
  ``NoThreshold``.
- :mod:`.noise` — keep-below over the same. ``NoiseSegmentStrategy``,
  ``AbsoluteQuietThreshold``, ``PercentileQuietThreshold``.
- :mod:`.detection` — keep-above over one 1-D signal, read from the
  **whole** array. ``SignalThreshold``, ``MedianMadThreshold``,
  ``PercentileSignalThreshold``.
- :mod:`.detection` also — keep-above over one 1-D signal **at a given
  sample**. ``PositionAwareSignalThreshold``, ``PeakFractionThreshold``.

The first two consume a pooled amplitude distribution across a record's
channels; the last two consume one channel of signal. Different arrays,
different families — and what separates the last two is how much context
the rule needs, not what the caller does with the answer.
"""

from __future__ import annotations

from .base import (
    NoiseSegmentStrategy,
    PositionAwareSignalThreshold,
    SignalThreshold,
    SignalThresholdLike,
    ThresholdStrategy,
)
from .detection import (
    MedianMadThreshold,
    PeakFractionThreshold,
    PercentileSignalThreshold,
    median_absolute_deviation,
)
from .healthy import AbsoluteThreshold, NoThreshold, PercentileThreshold
from .noise import AbsoluteQuietThreshold, PercentileQuietThreshold

__all__ = [
    "AbsoluteQuietThreshold",
    "AbsoluteThreshold",
    "MedianMadThreshold",
    "NoThreshold",
    "NoiseSegmentStrategy",
    "PeakFractionThreshold",
    "PercentileQuietThreshold",
    "PercentileSignalThreshold",
    "PercentileThreshold",
    "PositionAwareSignalThreshold",
    "SignalThreshold",
    "SignalThresholdLike",
    "ThresholdStrategy",
    "median_absolute_deviation",
]
