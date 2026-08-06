"""Decimation — anti-alias filter, then keep every ``factor``-th sample.

Downsampling without filtering first is not a shortcut, it is a bug:
every component above the *new* Nyquist folds back into the retained
band and lands on top of real signal, indistinguishable from it
afterwards. At ``fs = 1000`` decimated by 4, a 200 Hz component
reappears at ``|200 - 250| = 50`` Hz — squarely inside the atrial band
this library exists to look at. So the low-pass is not optional
pre-processing; it is half of the operation.

Why this matters here specifically
----------------------------------
The anti-alias filter is applied **zero-phase**, which is not the usual
choice for a resampler. It is the right one for this library because the
quantity everything downstream depends on is *when* an activation
happens: a causal filter would delay the whole trace, shifting every
detected activation time by its group delay, and that shift would flow
straight into the stored activation position. Zero-phase costs nothing
here — this is offline batch processing, not a real-time pipeline — and
it keeps the timing exact.

Math + sources: [`docs/theory.md`](../../../docs/theory.md) §1.4.
"""

from __future__ import annotations

import numpy as np

from .lowpass import lowpass

DEFAULT_ANTIALIAS_ORDER = 4
"""Butterworth order for the anti-alias low-pass.

Higher than the order-2 default the other filters carry, and measured
rather than assumed. Decimation needs a steeper roll-off than general
filtering: at ``fs = 1000``, factor 4, cutoff ``0.8 * new_nyquist``, an
out-of-band 200 Hz tone survives at

===========  ==========================
order        amplitude in the result
===========  ==========================
2            0.0385  (3.9% — audible junk)
**4**        **0.0017**  (0.17%)
8            0.0002
===========  ==========================

while the passband stays flat (a 50 Hz tone retains 0.997 at order 4).
Order 8 buys another factor of ten in the stopband but collapses the
transition band — a 110 Hz component, still legitimately in range, drops
from 0.31 to 0.16 — so order 4 is the balance point.

Note that :func:`lowpass` is zero-phase, so the achieved roll-off is
that of a one-pass filter of twice this order.
"""

DEFAULT_ANTIALIAS_CUTOFF_FRACTION = 0.8
"""Anti-alias cutoff, as a fraction of the **new** Nyquist frequency.

Filtering exactly at the new Nyquist would leave the whole transition
band to fold back, since a Butterworth is only 3 dB down at its own
cutoff. Backing off to 0.8 puts the transition inside the discarded
band. The same fraction ``scipy.signal.decimate`` uses for its IIR path.

These two constants are *filter-design* values, not policy: they
describe how faithfully this function resamples, not a choice about the
data. That is why they carry defaults where the threshold multipliers
and position ranges elsewhere in this library deliberately do not.
"""


def minimum_length_samples(order: int = DEFAULT_ANTIALIAS_ORDER) -> int:
    """Shortest signal :func:`decimate` can anti-alias filter at ``order``.

    The zero-phase filter pads each end before filtering, so a signal
    shorter than the padding has nothing to pad from. Derived from
    scipy's default ``padlen`` for ``sosfiltfilt``, which is
    ``3 * (2 * n_sections + 1)`` — a Butterworth of order ``N`` has
    ``ceil(N / 2)`` second-order sections. Verified against the measured
    limits: 10 samples at order 2, 16 at order 4, 28 at order 8.
    """
    n_sections = (order + 1) // 2
    return 3 * (2 * n_sections + 1) + 1


def decimate(
    signal: np.ndarray,
    factor: int,
    order: int = DEFAULT_ANTIALIAS_ORDER,
    cutoff_fraction: float = DEFAULT_ANTIALIAS_CUTOFF_FRACTION,
) -> np.ndarray:
    """Low-pass then downsample by an integer ``factor``, along axis 0.

    .. important::
       **The sampling rate of the result is** ``fs / factor``. Nothing in
       the returned array records that, so every rate-dependent value
       downstream — filter cutoffs in Hz, refractory intervals in
       samples, window lengths in samples — has to be recomputed for the
       new rate. A stale ``fs`` after decimation fails silently and
       plausibly, which is the worst way for it to fail.

    Parameters
    ----------
    signal
        1-D ``(n_samples,)`` or 2-D ``(n_samples, n_channels)``. A 2-D
        input is filtered and downsampled per channel along axis 0.
    factor
        Integer decimation factor, at least 1. ``factor = 1`` returns the
        signal unchanged and **does not filter** — there is no aliasing
        to prevent, so filtering would only distort it.
    order, cutoff_fraction
        Anti-alias filter design; see
        :data:`DEFAULT_ANTIALIAS_ORDER` and
        :data:`DEFAULT_ANTIALIAS_CUTOFF_FRACTION`.

    Returns
    -------
    np.ndarray
        Length ``ceil(n_samples / factor)`` along axis 0.

    Raises
    ------
    ValueError
        On a non-integer or sub-1 ``factor``, a ``cutoff_fraction``
        outside ``(0, 1]``, or a signal that is neither 1-D nor 2-D.
    """
    if signal.ndim not in (1, 2):
        raise ValueError(f"signal must be 1-D or 2-D, got {signal.ndim}-D")
    if isinstance(factor, bool) or not isinstance(factor, (int, np.integer)):
        raise ValueError(f"factor must be an integer, got {type(factor).__name__}")
    if factor < 1:
        raise ValueError(f"factor must be at least 1, got {factor}")
    if not 0.0 < cutoff_fraction <= 1.0:
        raise ValueError(f"cutoff_fraction must be in (0, 1], got {cutoff_fraction}")

    if factor == 1:
        return signal

    minimum = minimum_length_samples(order)
    if signal.shape[0] < minimum:
        raise ValueError(
            f"a {signal.shape[0]}-sample signal is too short to anti-alias filter at "
            f"order {order}, which needs at least {minimum}. The zero-phase filter pads "
            f"each end before filtering, and there is not enough signal to pad from. "
            f"Use a lower order, or do not decimate this trace."
        )

    # The anti-alias cutoff is expressed against the *new* Nyquist, so it
    # is a pure function of the factor: with fs normalised to 1, the new
    # Nyquist sits at 1 / (2 * factor). Passing fs = 1.0 keeps this
    # function rate-agnostic — the caller never has to tell us the rate,
    # and cannot tell us a wrong one.
    cutoff = cutoff_fraction / (2.0 * factor)
    filtered = lowpass(signal, fs=1.0, cutoff_hz=cutoff, order=order)
    decimated: np.ndarray = filtered[::factor]
    return decimated
