"""Threshold strategies for segment selection and activation detection.

Three families, each with its own interface in :mod:`.base`:

- :mod:`.healthy` — keep-above over pooled peak-to-peak amplitudes.
  ``ThresholdStrategy``, ``AbsoluteThreshold``, ``PercentileThreshold``,
  ``NoThreshold``.
- :mod:`.noise` — keep-below over the same. ``NoiseSegmentStrategy``,
  ``AbsoluteQuietThreshold``, ``PercentileQuietThreshold``.
- :mod:`.detection` — keep-above over a *detection curve* from a
  detection preprocessor. ``DetectionThreshold``,
  ``MedianMadThreshold``, ``PercentileDetectionThreshold``.

The first two consume a pooled amplitude distribution across a record's
channels; the third consumes one channel's detection curve. Different
arrays, different families.
"""

from __future__ import annotations

from .base import DetectionThreshold, NoiseSegmentStrategy, ThresholdStrategy
from .detection import (
    MedianMadThreshold,
    PercentileDetectionThreshold,
    median_absolute_deviation,
)
from .healthy import AbsoluteThreshold, NoThreshold, PercentileThreshold
from .noise import AbsoluteQuietThreshold, PercentileQuietThreshold

__all__ = [
    "AbsoluteQuietThreshold",
    "AbsoluteThreshold",
    "DetectionThreshold",
    "MedianMadThreshold",
    "NoThreshold",
    "NoiseSegmentStrategy",
    "PercentileDetectionThreshold",
    "PercentileQuietThreshold",
    "PercentileThreshold",
    "ThresholdStrategy",
    "median_absolute_deviation",
]
