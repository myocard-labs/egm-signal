"""Record-Protocol-driven segment extractors.

Two parallel functions:

- :func:`extract_healthy_segments` keeps windows whose calibrated p-p
  is *at or above* the threshold (used by producers building healthy /
  pretraining / "high-voltage" banks).
- :func:`extract_noise_segments` keeps windows whose p-p is *at or
  below* the threshold (used by producers building quiet / noise
  banks).

Both consume any object satisfying :class:`~..records.Record`. Both
reuse :func:`~..filters.bandpass` and
:func:`~..windowing.sliding_window_peak_to_peak` so the math is one
implementation.

Channel selection
-----------------
``channels`` is required: the extractors do not assume a default set
because that's source-specific. The producer passes its own bipolar
channel list. Channels not present in the record are silently
skipped — different records have different bipolar pairs available.

Calibration
-----------
The healthy extractor accepts an optional :class:`Calibration`; when
provided, ``calibrated = scalar * raw`` is computed before the
band-pass. The noise extractor does NOT take a calibration: noise
extraction is typically used on raw signal (percentile strategies are
scale-invariant) and the producer can pre-calibrate the record itself
if it wants absolute-mV noise thresholds.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from ..calibration import Calibration
from ..filters import bandpass
from ..records import Record
from ..thresholds import NoiseSegmentStrategy, ThresholdStrategy
from ..windowing import sliding_window_peak_to_peak
from .segments import HealthySegment, NoiseSegment

DEFAULT_BIPOLAR_BAND_HZ: tuple[float, float] = (30.0, 300.0)
"""Default band-pass edges for bipolar EGM extraction.

30-300 Hz follows the clinical convention in Sánchez 2021 / Unger
2019 / Deno 2017. Override per call if your dataset uses a different
frontend.
"""


# ---------------------------------------------------------------------------
# Healthy (keep-above) extraction
# ---------------------------------------------------------------------------


def extract_healthy_segments(
    record: Record,
    threshold: ThresholdStrategy,
    *,
    channels: Iterable[str],
    calibration: Calibration | None = None,
    window_ms: float = 200.0,
    hop_ms: float = 100.0,
    band_hz: tuple[float, float] = DEFAULT_BIPOLAR_BAND_HZ,
) -> list[HealthySegment]:
    """Extract high-voltage bipolar segments from a record.

    Pipeline per channel: (1) calibrate (raw * scalar, if a
    :class:`Calibration` is provided), (2) band-pass with the clinical
    bipolar filter, (3) slide a window of ``window_ms`` with ``hop_ms``
    stride and compute peak-to-peak, then (4) keep windows whose
    peak-to-peak is **at or above** the strategy's threshold.

    Parameters
    ----------
    record
        Any object satisfying :class:`Record`.
    threshold
        A :class:`ThresholdStrategy` — see :mod:`..thresholds.healthy`.
    channels
        Required. Bipolar channel names to process. Channels not
        present in ``record.channel_names`` are silently skipped.
    calibration
        Optional. If provided, multiply each channel by
        ``calibration.scalar`` before filtering. Required if you want
        the threshold strategy interpreted in calibrated mV.
    window_ms, hop_ms
        Sliding-window length and stride in milliseconds.
    band_hz
        Butterworth band-pass edges (Hz). Defaults to the clinical
        bipolar EGM band (30-300 Hz).
    """
    if window_ms <= 0 or hop_ms <= 0:
        raise ValueError("window_ms and hop_ms must be positive.")

    scalar = calibration.scalar if calibration is not None else 1.0
    window_samples = round(window_ms * 1e-3 * record.fs)
    hop_samples = max(1, round(hop_ms * 1e-3 * record.fs))

    present = [c for c in channels if c in record.channel_names]

    # First pass: filter each channel and compute per-window peak-to-peak,
    # caching so we don't recompute when the strategy needs the pooled
    # distribution.
    per_chan: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    pooled_buckets: list[np.ndarray] = []
    for chan in present:
        col_idx = record.channel_index(chan)
        raw = record.signal[:, col_idx]
        calibrated = scalar * raw
        filtered = bandpass(calibrated, record.fs, band_hz[0], band_hz[1])
        starts, pps = sliding_window_peak_to_peak(filtered, window_samples, hop_samples)
        per_chan.append((chan, filtered, starts, pps))
        pooled_buckets.append(pps)

    pooled = np.concatenate(pooled_buckets) if pooled_buckets else np.empty(0, dtype=np.float64)
    effective_threshold = threshold.compute_threshold(pooled)

    segments: list[HealthySegment] = []
    for chan, filtered, starts, pps in per_chan:
        for s, pp in zip(starts, pps, strict=True):
            if pp >= effective_threshold:
                seg = filtered[s : s + window_samples].copy()
                segments.append(
                    HealthySegment(
                        record_name=record.name,
                        patient=record.patient,
                        channel=chan,
                        start_sample=int(s),
                        end_sample=int(s + window_samples),
                        fs=record.fs,
                        peak_to_peak_mv=float(pp),
                        signal=seg,
                    )
                )
    return segments


# ---------------------------------------------------------------------------
# Noise (keep-below) extraction
# ---------------------------------------------------------------------------


def extract_noise_segments(
    record: Record,
    strategy: NoiseSegmentStrategy,
    *,
    channels: Iterable[str],
    window_ms: float = 200.0,
    hop_ms: float = 100.0,
    band_hz: tuple[float, float] = DEFAULT_BIPOLAR_BAND_HZ,
) -> list[NoiseSegment]:
    """Extract low-amplitude (quiet) bipolar windows from a record.

    Mirror of :func:`extract_healthy_segments` with the comparison
    direction inverted: keeps windows whose peak-to-peak is **at or
    below** the strategy's threshold.

    Signals are NOT calibrated inside this function. If you want
    absolute-mV noise thresholds, calibrate the record before calling
    (e.g. by mutating ``record.signal`` in place to the calibrated
    values, or by passing a wrapper that exposes the calibrated
    signal). The percentile strategies are scale-invariant and work on
    raw signal directly.
    """
    if window_ms <= 0 or hop_ms <= 0:
        raise ValueError("window_ms and hop_ms must be positive.")

    window_samples = round(window_ms * 1e-3 * record.fs)
    hop_samples = max(1, round(hop_ms * 1e-3 * record.fs))

    present = [c for c in channels if c in record.channel_names]

    per_chan: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    pooled_buckets: list[np.ndarray] = []
    for chan in present:
        col_idx = record.channel_index(chan)
        raw = record.signal[:, col_idx]
        filtered = bandpass(raw, record.fs, band_hz[0], band_hz[1])
        starts, pps = sliding_window_peak_to_peak(filtered, window_samples, hop_samples)
        per_chan.append((chan, filtered, starts, pps))
        pooled_buckets.append(pps)

    pooled = np.concatenate(pooled_buckets) if pooled_buckets else np.empty(0, dtype=np.float64)
    upper_bound = strategy.compute_threshold(pooled)

    segments: list[NoiseSegment] = []
    for chan, filtered, starts, pps in per_chan:
        for s, pp in zip(starts, pps, strict=True):
            if pp <= upper_bound:
                seg = filtered[s : s + window_samples].copy()
                segments.append(
                    NoiseSegment(
                        record_name=record.name,
                        patient=record.patient,
                        channel=chan,
                        start_sample=int(s),
                        end_sample=int(s + window_samples),
                        fs=record.fs,
                        peak_to_peak_mv=float(pp),
                        signal=seg,
                    )
                )
    return segments
