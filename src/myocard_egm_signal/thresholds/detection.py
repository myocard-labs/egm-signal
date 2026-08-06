"""Threshold rules over a 1-D signal.

Three strategies, all adaptive — they read their level off the signal
they are given, so the same configuration works across records,
channels and patients whose amplitudes differ by an order of magnitude:

- :class:`MedianMadThreshold` — ``c * median(g) + lam * MAD(g)``. The
  default, and the one the method spec specifies.
- :class:`PercentileSignalThreshold` — a high percentile of ``g``.
  Simpler, and useful when you want to fix the *rate* of candidates
  rather than their prominence.
- :class:`PeakFractionThreshold` — ``fraction * g[position]``, the only
  one of the three that needs to know *where* it is being asked.

The first two are :class:`~.base.SignalThreshold`; the third is a
:class:`~.base.PositionAwareSignalThreshold`. Nothing here is
detection-specific despite the module name — the same median/MAD rule
serves as the detection threshold ``tau`` and as the complex-boundary
level ``theta`` (§4). What distinguishes the classes is how much
context each needs, not what a caller does with the answer.

Only :class:`PercentileSignalThreshold` carries a qualifier in its name.
Everything in this package is re-exported flat from the package root and
``healthy.py`` already owns ``PercentileThreshold``, so that one needs
disambiguating and the others do not. Qualifying a name with no clash
would be noise.

None of them ship default constants. The multipliers decide how
aggressive detection is, which is a policy question belonging to the
pipeline that knows its data — not to a library (the same rule that
removed the calibration target's default). Study §8.1 sets them for
this project.

Math + primary sources: [`docs/theory.md`](../../../docs/theory.md) §3.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from .base import PositionAwareSignalThreshold, SignalThreshold


def median_absolute_deviation(values: np.ndarray) -> float:
    """Median absolute deviation: ``median(|x - median(x)|)``.

    A robust scale estimate. Returned in **raw MAD units**, not scaled
    to be a standard-deviation estimate — multiply by ``1.4826`` if you
    want the Gaussian-consistent version.
    """
    med = float(np.median(values))
    return float(np.median(np.abs(values - med)))


class MedianMadThreshold(SignalThreshold):
    """``tau = c * median(g) + lam * MAD(g)``.

    A robust analogue of "mean plus k standard deviations": locate the
    curve's baseline with the median, measure its spread with the median
    absolute deviation, and sit a configurable distance above it.

    Serves both roles this package needs a global level for. As a
    detection threshold it is ``tau``. As a complex-boundary level
    (§4.2) it is ``theta`` with ``c = 1``, which the method spec
    describes as "a small multiple of the baseline-noise MAD" — the
    median term is what makes that phrase a *level* rather than a bare
    spread, since a MAD multiple alone would sit below the noise floor
    on any curve whose baseline is above zero, and the outward walk
    would never terminate.

    Why MAD rather than standard deviation
    --------------------------------------
    The curve contains the very activations being detected, and they are
    large outliers in it. Standard deviation has a breakdown point of
    zero — a single large activation inflates it — so a sigma-based
    threshold **rises with the activations it is supposed to find**. A
    channel with more or larger activations would get a proportionally
    higher bar and detect a smaller fraction of them, which is precisely
    backwards. MAD has a breakdown point of 50%: up to half the samples
    can be arbitrary without moving it. Since activations occupy a few
    percent of a record's samples, MAD measures the *baseline* — which
    is what a detection threshold should be referenced to.

    Behavior when MAD is zero
    -------------------------
    Worth understanding, because it is not an error case. ``MAD == 0``
    means over half the samples share one value — true of a **clean,
    sparse** detection curve (a synthetic trace that is exactly flat
    between activations) just as much as of a dead channel. The two are
    indistinguishable from the curve alone.

    So this is deliberately *not* special-cased. The formula degrades on
    its own to ``tau = c * median(g)``, which for a sparse curve with a
    zero baseline is ``tau = 0``. That is harmless, because the
    threshold gates **local maxima**, not samples: a flat baseline
    contains no strict local maximum, so a ``tau`` of 0 admits the
    deflections and nothing else. Measured on an 8-activation clean
    curve: 8 candidates. The genuinely empty case (a *constant* curve)
    is caught in the base class and returns ``+inf``.

    On a clean Gabor-pulse trace through :class:`RectifiedDerivative`,
    where 59% of the curve is exactly zero and ``tau`` is 0, local-maxima
    extraction yields **4** candidates rather than the 206 samples that
    merely sit above the threshold — and the refractory step then reduces
    those to one activation. This is why candidate extraction, not the
    threshold, is what makes detection selective on clean signal.

    Parameters
    ----------
    c
        Multiplier on the baseline median. Required.
    lam
        Multiplier on the MAD. Required. Larger is more conservative
        (fewer, more prominent candidates). Multiply by ``1.4826`` to
        read it as a multiple of a Gaussian standard deviation.
    """

    name: ClassVar[str] = "median_mad"

    def __init__(self, c: float, lam: float) -> None:
        if c < 0:
            raise ValueError(f"c must be non-negative, got {c}.")
        if lam < 0:
            raise ValueError(f"lam must be non-negative, got {lam}.")
        self.c = float(c)
        self.lam = float(lam)

    def _compute_threshold(self, detection_curve: np.ndarray) -> float:
        med = float(np.median(detection_curve))
        mad = median_absolute_deviation(detection_curve)
        return self.c * med + self.lam * mad

    def __repr__(self) -> str:
        return f"MedianMadThreshold(c={self.c}, lam={self.lam})"


class PercentileSignalThreshold(SignalThreshold):
    """``tau = percentile(g, q)`` — fix the candidate *rate*.

    Where :class:`MedianMadThreshold` asks "how prominent must a sample
    be?", this asks "what fraction of samples may be candidates?" — at
    ``q = 99`` at most 1% of samples clear the bar, whatever the curve
    looks like.

    That makes it predictable and shape-independent, which is useful for
    a first pass on an unfamiliar record. The cost is that it always
    returns *something*: on a channel with no activations at all it
    still promotes its top 1% of samples, where the median/MAD rule
    would put the bar above everything present. Prefer it for
    exploration, not for a production splitter.

    Parameters
    ----------
    q
        Percentile in ``(0, 100)``. Required.
    """

    name: ClassVar[str] = "percentile_signal"

    def __init__(self, q: float) -> None:
        if not 0 < q < 100:
            raise ValueError(f"q must be in (0, 100), got {q}.")
        self.q = float(q)

    def _compute_threshold(self, detection_curve: np.ndarray) -> float:
        return float(np.percentile(detection_curve, self.q))

    def __repr__(self) -> str:
        return f"PercentileSignalThreshold(q={self.q})"


class PeakFractionThreshold(PositionAwareSignalThreshold):
    """``theta = fraction * g[position]`` — a fraction of *this* peak.

    The motivating case for the position-aware family: the level scales
    with each complex individually, so a low-voltage fibrotic activation
    is measured against its own amplitude rather than against the
    record's largest. That is usually what you want when comparing
    complex *shapes* across a corpus whose amplitudes vary — and it is
    why the index cannot be optional here.

    The trade-off is that it says nothing about the noise floor. On a
    weak activation barely above baseline, a small fraction of its peak
    can fall *into* the noise, and an outward walk then runs until it
    happens to dip — potentially into a neighbouring activation.
    Measured on a four-activation train, ``fraction = 0.02`` produced a
    597-sample "complex" that swallowed both neighbours. Bound the
    search (see :func:`~..extraction.activation_based.measure_complex`)
    or use :class:`MedianMadThreshold`, which is defined relative to the
    floor and so cannot fall beneath it.

    Parameters
    ----------
    fraction
        In ``(0, 1)``. Required.
    """

    name: ClassVar[str] = "peak_fraction"

    def __init__(self, fraction: float) -> None:
        if not 0.0 < fraction < 1.0:
            raise ValueError(f"fraction must be in (0, 1), got {fraction}.")
        self.fraction = float(fraction)

    def _compute_threshold(self, signal: np.ndarray, position: int) -> float:
        return self.fraction * float(signal[position])

    def __repr__(self) -> str:
        return f"PeakFractionThreshold(fraction={self.fraction})"
