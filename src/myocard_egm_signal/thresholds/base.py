"""Threshold-strategy Protocols.

Two Protocols share an identical shape but signal direction:

- :class:`ThresholdStrategy` — consumed by the keep-above
  (healthy-class) extractor.
- :class:`NoiseSegmentStrategy` — consumed by the keep-below (noise)
  extractor.

Parallel hierarchies make the comparison direction obvious at the call
site — a strategy implementing one Protocol is not interchangeable
with the other even though they have the same method signature.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ThresholdStrategy(Protocol):
    """Plug-in interface for healthy-segment threshold computation.

    The extractor keeps windows whose peak-to-peak is **at or above**
    :meth:`compute_threshold`'s return value.
    """

    name: str

    def compute_threshold(self, pooled_peak_to_peaks: np.ndarray) -> float:
        """Return the effective threshold for this record."""
        ...


@runtime_checkable
class NoiseSegmentStrategy(Protocol):
    """Plug-in interface for noise-segment threshold computation.

    The extractor keeps windows whose peak-to-peak is **at or below**
    :meth:`compute_threshold`'s return value.
    """

    name: str

    def compute_threshold(self, pooled_peak_to_peaks: np.ndarray) -> float:
        """Return the upper-bound peak-to-peak for a window to count as noise."""
        ...
