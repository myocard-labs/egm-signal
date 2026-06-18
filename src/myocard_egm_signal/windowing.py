"""Sliding-window primitives shared by the segment extractors.

Today: :func:`sliding_window_peak_to_peak`. Future additions
(``sliding_window_rms``, ``sliding_window_zero_crossings``, ...) land
here so multiple extractors / feature pipelines share one
implementation.
"""

from __future__ import annotations

import numpy as np


def sliding_window_peak_to_peak(
    signal: np.ndarray,
    window_samples: int,
    hop_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sliding peak-to-peak on a 1-D signal along axis 0.

    For each non-overlapping (or hop-spaced) window of length
    ``window_samples``, returns the start index and the peak-to-peak
    amplitude (``max - min``, NaN-aware via ``nanmax``/``nanmin``).

    Parameters
    ----------
    signal
        1-D ``(n_samples,)`` array.
    window_samples
        Window length in samples. Must be positive and not exceed
        ``signal.shape[0]``; if it does, returns empty arrays.
    hop_samples
        Stride between adjacent window starts.

    Returns
    -------
    starts
        ``(n_windows,)`` int array of start indices.
    pps
        ``(n_windows,)`` float array of peak-to-peak amplitudes per window.

    Raises
    ------
    ValueError
        On non-positive ``hop_samples``.
    """
    n = signal.shape[0]
    if window_samples <= 0 or window_samples > n:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    if hop_samples <= 0:
        raise ValueError("hop_samples must be positive.")
    starts = np.arange(0, n - window_samples + 1, hop_samples, dtype=np.int64)
    if starts.size == 0:
        return starts, np.empty(0, dtype=np.float64)
    pps = np.empty(starts.size, dtype=np.float64)
    for i, s in enumerate(starts):
        window = signal[s : s + window_samples]
        pps[i] = float(np.nanmax(window) - np.nanmin(window))
    return starts, pps
