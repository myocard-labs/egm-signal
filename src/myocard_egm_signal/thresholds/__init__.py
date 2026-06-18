"""Threshold strategies for segment selection.

Two parallel hierarchies share the same Protocol shape but
distinguish the comparison direction:

- :mod:`.healthy` (re-exported below): keep-above strategies.
  ``ThresholdStrategy``, ``AbsoluteThreshold``,
  ``PercentileThreshold``, ``NoThreshold``.
- :mod:`.noise` (re-exported below): keep-below strategies.
  ``NoiseSegmentStrategy``, ``AbsoluteQuietThreshold``,
  ``PercentileQuietThreshold``.

The Protocols both live in :mod:`.base`.
"""

from __future__ import annotations

from .base import NoiseSegmentStrategy, ThresholdStrategy
from .healthy import AbsoluteThreshold, NoThreshold, PercentileThreshold
from .noise import AbsoluteQuietThreshold, PercentileQuietThreshold

__all__ = [
    "AbsoluteQuietThreshold",
    "AbsoluteThreshold",
    "NoThreshold",
    "NoiseSegmentStrategy",
    "PercentileQuietThreshold",
    "PercentileThreshold",
    "ThresholdStrategy",
]
