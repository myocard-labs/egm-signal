"""Candidate selection — turning a detection curve into candidate peaks.

Stage three of the detection chain, between thresholding and refractory
suppression. Given a detection curve ``g`` and a level ``tau``, return
the points that *might* be activations, leaving the question of which
are **distinct** activations to the refractory step.

Like the preprocessors, thresholds and suppressors, this is a strategy
family behind an interface (:class:`CandidateSelector`), so an
alternative selection rule can be tried without touching working code.
:class:`LocalMaximaSelector` is what the method spec specifies.

Why not one candidate per above-``tau`` segment
-----------------------------------------------
The tempting simplification is to collapse each contiguous above-``tau``
run to its single ``argmax``. The method spec rules this out
explicitly, and measurement shows why: it lets the *threshold* do the
merging. Two genuine activations riding **one** above-``tau`` run —
common in fast AF, where the envelope never fully returns to baseline
between beats — would collapse into a single peak, a silent miss that
no refractory tuning can recover. Measured with two activations 80 ms
apart under a 10 Hz envelope: one above-``tau`` run 168 samples wide,
spanning nearly three refractory intervals, which segment-``argmax``
reduces to one activation and local maxima keep as two.

So the threshold gates *prominence* (amplitude) and the refractory step
decides *distinctness* (time). Keeping those jobs separate is what stops
the threshold from silently deleting activations — and raising ``tau``
to suppress fractionation would also remove genuine **low-voltage**
activations, which are exactly the fibrotic regions of interest.

The above-``tau`` **segment** is still computed and returned: it is the
activation-complex extent that onset/offset measurement works within.
It is simply not the merge rule.

Reference: the local-peaks -> threshold -> refractory chain is the
standard event-detection pattern (Pan & Tompkins 1985). Math + sources:
[`docs/theory.md`](../../../../docs/theory.md) §3.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from scipy.signal import find_peaks


@dataclass(frozen=True)
class ActivationCandidate:
    """One candidate activation, with the complex it sits in.

    ``segment_start`` / ``segment_end`` bound the contiguous
    above-``tau`` run containing this peak, half-open as usual. Several
    candidates can share one segment — that is the fractionated case.
    The segment is the raw activation-complex extent that onset/offset
    measurement refines.
    """

    peak_sample: int
    segment_start: int
    segment_end: int
    height: float
    prominence: float

    @property
    def segment_width(self) -> int:
        """Width of the enclosing above-``tau`` run, in samples."""
        return self.segment_end - self.segment_start


def _above_threshold_segments(curve: np.ndarray, tau: float) -> list[tuple[int, int]]:
    """Find the contiguous stretches where the curve is at or above ``tau``.

    Each stretch is one *activation complex* as the threshold sees it:
    the curve rises above ``tau``, stays there for a while, and drops
    back below. Returned as half-open ``(start, end)`` sample bounds, so
    ``curve[start:end]`` is the stretch and ``end - start`` its width.

    Worked example, ``tau = 1``::

        curve   = [0, 5, 6, 0, 0, 9, 8, 0]
        above   = [F, T, T, F, F, T, T, F]     step 1
        idx     = [   1, 2,       5, 6   ]     step 2 -> [1, 2, 5, 6]
        diff    = [    1,    3,    1     ]     step 3 -> a gap of 3 at position 2
        runs    = [[1, 2], [5, 6]]             step 4
        result  = [(1, 3), (5, 7)]             step 5

    Returns an empty list when nothing reaches ``tau``.
    """
    # Step 1 — a boolean mask: which samples are at or above the level.
    above = curve >= tau
    if not above.any():
        return []

    # Step 2 — the *indices* of those samples, in increasing order.
    # Samples belonging to one stretch are consecutive integers here.
    idx = np.flatnonzero(above)

    # Step 3 — consecutive indices differ by exactly 1, so any larger
    # difference marks a gap, i.e. the boundary between two stretches.
    # `+ 1` converts "position of the last index before the gap" into
    # "position where the next stretch begins", which is what np.split
    # wants.
    breaks = np.flatnonzero(np.diff(idx) != 1) + 1

    # Step 4 — cut the index list at those boundaries: one sub-array of
    # consecutive indices per stretch.
    runs = np.split(idx, breaks)

    # Step 5 — each run's first and last index become half-open bounds.
    # `+ 1` on the last index makes `end` exclusive, matching Python
    # slicing so callers can write curve[start:end] directly.
    return [(int(run[0]), int(run[-1]) + 1) for run in runs]


def _enclosing_segment(segments: list[tuple[int, int]], peak: int) -> tuple[int, int]:
    """Return the above-``tau`` stretch containing ``peak``.

    Every peak is inside exactly one stretch, by construction: a peak
    only exists because ``curve[peak] >= tau``, and the stretches are
    precisely the runs where ``curve >= tau``. The two are derived from
    the *same* comparison against the *same* ``tau``.

    So failing to find one is not a data condition to absorb — it means
    that invariant has been broken, most plausibly by the peak finder
    and :func:`_above_threshold_segments` drifting apart on the
    comparison (one using ``>``, the other ``>=``). Hence
    :exc:`RuntimeError` rather than a fabricated single-sample segment:
    a made-up extent would flow into the activation-complex width that
    onset/offset measurement works within, quietly corrupting a
    downstream measurement instead of failing where the fault is.

    Verified unreachable over 3000 randomised curve/threshold
    combinations, plateau peaks, and NaN-containing curves.
    """
    for start, end in segments:
        if start <= peak < end:
            return start, end
    raise RuntimeError(
        f"internal inconsistency: peak at sample {peak} lies outside every above-threshold "
        f"segment {segments}. Peak selection and _above_threshold_segments must apply the "
        "same comparison to the same tau."
    )


class CandidateSelector(ABC):
    """Turn a detection curve and a threshold into candidate activations.

    Subclasses implement :meth:`_select` and set :attr:`name`; they do
    not override :meth:`select`, which is the template method holding
    the shared contract.

    Contract guaranteed to every caller
    -----------------------------------
    - Input is one channel's curve, 1-D.
    - Output is **ordered by sample index**.
    - A non-finite ``tau`` — the threshold's fail-closed ``+inf``
      sentinel — yields no candidates rather than an error, so the
      sentinel propagates through the chain as "detect nothing".
    - An empty curve yields no candidates.
    """

    name: ClassVar[str]

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "name", None):
            raise TypeError(
                f"{cls.__name__} must define a class-level `name`; it is recorded "
                "in provenance and identifies the selection rule in run records."
            )

    def select(self, detection_curve: np.ndarray, tau: float) -> list[ActivationCandidate]:
        """Return the candidate activations, ordered by sample index."""
        if detection_curve.ndim != 1:
            raise ValueError(
                f"a detection curve is one channel (1-D), got {detection_curve.ndim}-D."
            )
        if detection_curve.size == 0 or not np.isfinite(tau):
            return []
        return self._select(detection_curve.astype(np.float64, copy=False), tau)

    @abstractmethod
    def _select(self, detection_curve: np.ndarray, tau: float) -> list[ActivationCandidate]:
        """The selection rule itself.

        Called only with a non-empty 1-D float64 curve and a finite
        ``tau``, so implementations carry no defensive checks.
        """


class LocalMaximaSelector(CandidateSelector):
    """Local maxima at or above ``tau`` — the method spec's rule.

    A candidate is a point where ``g[i-1] < g[i] > g[i+1]`` and
    ``g[i] >= tau``, filtered by a minimum prominence. A fractionated
    complex may contribute several candidates, and that is intended.

    Parameters
    ----------
    min_prominence
        Minimum peak prominence, in the curve's own units — how far a
        peak rises above the higher of the two saddles that bound it.
        This is the noise-blip and far-field filter.

        **Required, but accepts ``None``.** There is no default, because
        omitting the filter is not a neutral choice: an adaptive
        threshold sitting a few MAD above a quiet baseline is low in
        absolute terms, so filter ringing around a strong activation and
        ordinary noise bumps both clear it — and being spaced further
        apart than the refractory interval, suppression *keeps* them.
        Measured on one fractionated complex with realistic noise:
        **3** activations under the Botteron envelope and **9** under
        the rectified derivative with ``None``; **1** and **1** with a
        floor at 20% of the curve maximum. The failure is silent,
        because spurious detections look like activations. Passing
        ``None`` is therefore a decision the caller should make on
        purpose, which is why it cannot be reached by omission.

        The right value is data-dependent, so the library does not
        choose one — the same rule that removed the calibration
        target's default.

    Notes
    -----
    Uses ``scipy.signal.find_peaks``, whose ``height`` comparison is
    inclusive — matching the ``g[i] >= tau`` the method spec specifies —
    and which resolves a **flat-topped** peak to its midpoint. A naive
    ``g[i-1] < g[i] > g[i+1]`` test drops plateaus entirely, which on
    quantized or synthetic data silently loses activations.
    """

    name: ClassVar[str] = "local_maxima"

    def __init__(self, min_prominence: float | None) -> None:
        if min_prominence is not None and min_prominence < 0:
            raise ValueError(f"min_prominence must be non-negative, got {min_prominence}.")
        self.min_prominence = None if min_prominence is None else float(min_prominence)

    def _select(self, detection_curve: np.ndarray, tau: float) -> list[ActivationCandidate]:
        # `prominence=0.0` rather than `None` when the caller wants no
        # filtering, and the reason is reporting, not filtering:
        # find_peaks only populates `props["prominences"]` when it has
        # been asked to filter on prominence. We record the prominence
        # on every candidate regardless, so we always need that key.
        #
        # 0.0 is a genuine no-op filter, not an approximation of one:
        # find_peaks' prominence comparison is **inclusive**, so a peak
        # of prominence exactly 0.0 still passes. Verified over 4000
        # curves (including quantized and flat-heavy ones, where
        # plateaus are likely) that `None` and `0.0` select identical
        # peaks, and pinned by
        # `test_no_prominence_floor_and_a_zero_floor_agree` — so if a
        # future scipy made that comparison strict, the suite says so
        # rather than silently dropping peaks.
        peaks, props = find_peaks(
            detection_curve,
            height=tau,
            prominence=self.min_prominence if self.min_prominence is not None else 0.0,
        )
        if peaks.size == 0:
            return []

        segments = _above_threshold_segments(detection_curve, tau)
        candidates: list[ActivationCandidate] = []
        for peak, height, prominence in zip(
            peaks, props["peak_heights"], props["prominences"], strict=True
        ):
            start, end = _enclosing_segment(segments, int(peak))
            candidates.append(
                ActivationCandidate(
                    peak_sample=int(peak),
                    segment_start=start,
                    segment_end=end,
                    height=float(height),
                    prominence=float(prominence),
                )
            )
        return candidates

    def __repr__(self) -> str:
        return f"LocalMaximaSelector(min_prominence={self.min_prominence})"
