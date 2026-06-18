"""Signal-level filters.

Today: band-pass. Future additions (notch, smoothing, decimation) land
here so consumers keep a single import for filtering needs.

All filters operate along axis 0 (the sample axis). They accept either a
1-D ``(n_samples,)`` array or a 2-D ``(n_samples, n_channels)`` array
and return an output of the same shape.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt


def bandpass(
    signal: np.ndarray,
    fs: float,
    low_hz: float,
    high_hz: float,
    order: int = 2,
) -> np.ndarray:
    """Zero-phase Butterworth band-pass along axis 0.

    Used across producers for the clinical bipolar EGM band
    (~30-300 Hz per Sánchez 2021 / Unger 2019 / Deno 2017) but the
    function itself is band-agnostic — any (low_hz, high_hz) pair the
    Nyquist check accepts will work.

    The high edge is automatically capped to ``0.99 * fs/2`` so the
    filter is well-defined even when callers pass a high cutoff above
    Nyquist (a common copy-paste hazard at lower sample rates).

    Parameters
    ----------
    signal
        1-D ``(n_samples,)`` or 2-D ``(n_samples, n_channels)`` array.
    fs
        Sampling frequency in Hz.
    low_hz, high_hz
        Band edges. ``high_hz`` is auto-capped at ``0.99 * fs/2``.
    order
        Butterworth order. Default 2.

    Returns
    -------
    np.ndarray
        Same shape as ``signal``. Zero-phase via ``sosfiltfilt``, so the
        output is non-causal — no group-delay correction needed.

    Raises
    ------
    ValueError
        On invalid array shape, ``low_hz <= 0``, or capped
        ``high <= low_hz``.
    """
    if signal.ndim not in (1, 2):
        raise ValueError(f"signal must be 1-D or 2-D, got {signal.ndim}-D")

    nyq = 0.5 * fs
    high = min(high_hz, 0.99 * nyq)
    if low_hz <= 0 or high <= low_hz:
        raise ValueError(f"Invalid band: low={low_hz}, high={high_hz}, fs={fs}")

    sos = butter(order, [low_hz / nyq, high / nyq], btype="bandpass", output="sos")
    # scipy's sosfiltfilt returns Any; pin to the declared ndarray type.
    out: np.ndarray = sosfiltfilt(sos, signal, axis=0)
    return out
