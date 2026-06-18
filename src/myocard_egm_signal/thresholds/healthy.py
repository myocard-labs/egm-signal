"""Threshold strategies for healthy-class (keep-above) selection.

A :class:`~.base.ThresholdStrategy` consumes the pooled distribution
of peak-to-peak amplitudes across all bipolar channels in a record
and returns a scalar threshold. Segments whose peak-to-peak is *at or
above* the returned value are kept.

Three concrete strategies ship:

- :class:`AbsoluteThreshold` — fixed mV cap. Use with calibrated input.
- :class:`PercentileThreshold` — per-record percentile of the pooled
  p-p distribution. Scale-invariant.
- :class:`NoThreshold` — pass-through: every windowed segment is kept.

Extending
---------
Implement a new strategy by writing a class that satisfies the
:class:`~.base.ThresholdStrategy` Protocol::

    class RelativeVoltageIndex:
        name = "rvi"
        def __init__(self, ...): ...
        def compute_threshold(self, pooled_peak_to_peaks: np.ndarray) -> float:
            ...
"""

from __future__ import annotations

import numpy as np


class AbsoluteThreshold:
    """Fixed peak-to-peak amplitude threshold (units must match the input).

    Use with calibrated signals if you want a true mV threshold
    (e.g. ``AbsoluteThreshold(0.5)`` for the Sánchez sinus-rhythm
    convention or ``AbsoluteThreshold(0.2)`` for the Kosiuk AF-rhythm
    adjustment).
    """

    name: str = "absolute"

    def __init__(self, value: float) -> None:
        if value <= 0:
            raise ValueError("threshold value must be positive.")
        self.value = float(value)

    def compute_threshold(self, pooled_peak_to_peaks: np.ndarray) -> float:
        return self.value

    def __repr__(self) -> str:
        return f"AbsoluteThreshold(value={self.value})"


class PercentileThreshold:
    """Per-record percentile threshold (no calibration needed).

    Keeps windows whose peak-to-peak exceeds the ``percentile``-th
    percentile of the pooled distribution across all bipolar channels
    in the record. E.g. ``PercentileThreshold(70)`` keeps the top 30%.
    """

    name: str = "percentile"

    def __init__(self, percentile: float) -> None:
        if not 0 < percentile < 100:
            raise ValueError("percentile must be in (0, 100).")
        self.percentile = float(percentile)

    def compute_threshold(self, pooled_peak_to_peaks: np.ndarray) -> float:
        if pooled_peak_to_peaks.size == 0:
            return float("inf")
        return float(np.percentile(pooled_peak_to_peaks, self.percentile))

    def __repr__(self) -> str:
        return f"PercentileThreshold(percentile={self.percentile})"


class NoThreshold:
    """Pass-through strategy: every windowed segment is retained.

    Returns ``-inf`` from :meth:`compute_threshold`, which combined
    with the orchestrator's ``pp >= threshold`` filter retains every
    window regardless of peak-to-peak. The noise-side orchestrator
    uses ``pp <= threshold`` and would invert this behavior — never
    mix :class:`NoThreshold` into a noise extraction.
    """

    name: str = "none"

    def compute_threshold(self, pooled_peak_to_peaks: np.ndarray) -> float:
        return float("-inf")

    def __repr__(self) -> str:
        return "NoThreshold()"
