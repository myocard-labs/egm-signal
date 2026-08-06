"""Zero-phase Butterworth low-pass.

Sibling of :func:`~.bandpass.bandpass`, sharing its conventions: axis-0
filtering, 1-D or 2-D input, an auto-capped cutoff, and ``sosfiltfilt``
so the output is zero-phase.

Why this exists
---------------
The Botteron activation envelope
(:mod:`~..extraction.activation_based.preprocessors`) is
``LP(|BP(x)|)`` — rectification followed by a low-pass. The low-pass is
the load-bearing step: it is what bridges the dips *between* the
deflections of a fractionated activation, so the smoothed envelope
crosses a detection threshold once on each side of the complex instead
of several times. Faking it with ``bandpass(x, fs, low_hz=eps, ...)``
would work numerically but misstates the intent and puts a spurious
high-pass corner near DC, right where the rectified envelope carries
most of its energy.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt


def lowpass(
    signal: np.ndarray,
    fs: float,
    cutoff_hz: float,
    order: int = 2,
) -> np.ndarray:
    """Zero-phase Butterworth low-pass along axis 0.

    The cutoff is automatically capped at ``0.99 * fs/2`` so the filter
    stays well-defined when a caller passes a cutoff above Nyquist —
    the same guard :func:`bandpass` applies to its high edge.

    Parameters
    ----------
    signal
        1-D ``(n_samples,)`` or 2-D ``(n_samples, n_channels)`` array.
    fs
        Sampling frequency in Hz.
    cutoff_hz
        Low-pass corner in Hz. Auto-capped at ``0.99 * fs/2``.
    order
        Butterworth order. Default 2, matching :func:`bandpass`.

    Returns
    -------
    np.ndarray
        Same shape as ``signal``. Zero-phase via ``sosfiltfilt``, so the
        output is non-causal — no group-delay correction needed, and an
        envelope's onset/offset crossings are not shifted in time.

    Raises
    ------
    ValueError
        On invalid array shape, or non-positive ``cutoff_hz``.
    """
    if signal.ndim not in (1, 2):
        raise ValueError(f"signal must be 1-D or 2-D, got {signal.ndim}-D")

    nyq = 0.5 * fs
    cutoff = min(cutoff_hz, 0.99 * nyq)
    if cutoff_hz <= 0:
        raise ValueError(f"Invalid cutoff: cutoff_hz={cutoff_hz}, fs={fs}")

    sos = butter(order, cutoff / nyq, btype="lowpass", output="sos")
    # scipy's sosfiltfilt returns Any; pin to the declared ndarray type.
    out: np.ndarray = sosfiltfilt(sos, signal, axis=0)
    return out
