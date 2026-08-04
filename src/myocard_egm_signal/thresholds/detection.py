"""Threshold rules over a detection curve.

Two strategies, both adaptive — they read their level off the curve
they are given, so the same configuration works across records,
channels and patients whose amplitudes differ by an order of magnitude:

- :class:`MedianMadThreshold` — ``c * median(g) + lam * MAD(g)``. The
  default, and the one the method spec specifies.
- :class:`PercentileDetectionThreshold` — a high percentile of ``g``.
  Simpler, and useful when you want to fix the *rate* of candidates
  rather than their prominence.

Only the second name is qualified with "Detection". Everything in this
package is re-exported flat from the package root, and ``healthy.py``
already owns ``PercentileThreshold`` — so that one needs
disambiguating and ``MedianMadThreshold`` does not. Qualifying a name
that has no clash would be noise.

Neither ships default constants. The multipliers decide how aggressive
detection is, which is a policy question belonging to the pipeline that
knows its data — not to a library (the same rule that removed the
calibration target's default). Study §8.1 sets them for this project.

Math + primary sources: [`docs/theory.md`](../../../docs/theory.md) §3.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from .base import DetectionThreshold


def median_absolute_deviation(values: np.ndarray) -> float:
    """Median absolute deviation: ``median(|x - median(x)|)``.

    A robust scale estimate. Returned in **raw MAD units**, not scaled
    to be a standard-deviation estimate — multiply by ``1.4826`` if you
    want the Gaussian-consistent version.
    """
    med = float(np.median(values))
    return float(np.median(np.abs(values - med)))


class MedianMadThreshold(DetectionThreshold):
    """``tau = c * median(g) + lam * MAD(g)``.

    A robust analogue of "mean plus k standard deviations": locate the
    curve's baseline with the median, measure its spread with the median
    absolute deviation, and sit a configurable distance above it.

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


class PercentileDetectionThreshold(DetectionThreshold):
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

    name: ClassVar[str] = "percentile_detection"

    def __init__(self, q: float) -> None:
        if not 0 < q < 100:
            raise ValueError(f"q must be in (0, 100), got {q}.")
        self.q = float(q)

    def _compute_threshold(self, detection_curve: np.ndarray) -> float:
        return float(np.percentile(detection_curve, self.q))

    def __repr__(self) -> str:
        return f"PercentileDetectionThreshold(q={self.q})"
