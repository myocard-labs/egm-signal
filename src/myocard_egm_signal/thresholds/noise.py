"""Threshold strategies for noise (keep-below) selection.

A :class:`~.base.NoiseSegmentStrategy` is structurally identical to
:class:`~.base.ThresholdStrategy` (same protocol shape, same inputs,
same return type) but is consumed by the noise extractor, which keeps
windows with peak-to-peak *at or below* the returned threshold. Two
parallel hierarchies make the orchestrator's comparison direction
obvious at the call site.

Two concrete strategies ship:

- :class:`AbsoluteQuietThreshold` — fixed mV upper bound. Use with
  calibrated signals if you want a true mV cutoff. Sanders 2003's
  "electrically silent" tier (≤ 0.05 mV) is a defensible conservative
  starting point for atrial bipolar noise.
- :class:`PercentileQuietThreshold` — keep the bottom Nth percentile
  of the pooled distribution. Scale-invariant.

There is intentionally no ``NoQuietThreshold`` analog — "every window
is noise" is not a meaningful operation; the producer should write a
plain (non-noise) bank if it wants every window.
"""

from __future__ import annotations

import numpy as np


class AbsoluteQuietThreshold:
    """Fixed peak-to-peak upper bound (units must match the input).

    Use with calibrated signals if you want a true mV cutoff. Sanders
    2003's ≤ 0.05 mV "electrically silent" tier is a defensible
    conservative starting point for atrial bipolar EGM noise.
    """

    name: str = "absolute_quiet"

    def __init__(self, value: float) -> None:
        if value <= 0:
            raise ValueError("threshold value must be positive.")
        self.value = float(value)

    def compute_threshold(self, pooled_peak_to_peaks: np.ndarray) -> float:
        return self.value

    def __repr__(self) -> str:
        return f"AbsoluteQuietThreshold(value={self.value})"


class PercentileQuietThreshold:
    """Per-record percentile-based upper bound (no calibration needed).

    Keeps windows whose peak-to-peak is at or below the
    ``percentile``-th percentile of the pooled distribution across all
    bipolar channels in the record. E.g. ``PercentileQuietThreshold(20)``
    keeps the quietest 20% of windows in each record.

    Scale-invariant — the natural choice for IAFDB and any other
    uncalibrated source.
    """

    name: str = "percentile_quiet"

    def __init__(self, percentile: float) -> None:
        if not 0 < percentile < 100:
            raise ValueError("percentile must be in (0, 100).")
        self.percentile = float(percentile)

    def compute_threshold(self, pooled_peak_to_peaks: np.ndarray) -> float:
        if pooled_peak_to_peaks.size == 0:
            # No data -> accept nothing. Returning -inf means the noise-side
            # filter (pp <= threshold) rejects every window. (Don't return
            # +inf here: that would accept everything, masking an empty-input
            # bug as a full bank.)
            return float("-inf")
        return float(np.percentile(pooled_peak_to_peaks, self.percentile))

    def __repr__(self) -> str:
        return f"PercentileQuietThreshold(percentile={self.percentile})"
