"""Refractory suppression — deciding which candidates are distinct activations.

Stage four of the detection chain. Candidate extraction may return
several peaks per activation (a fractionated complex under a sharp
preprocessor, or residual bumps under a smoothed one); this collapses
each cluster to one activation by enforcing a minimum spacing
``Delta_refr``.

This is the *time*-axis decision, and it is deliberately the only one.
The threshold decides *amplitude* — how prominent a peak must be — and
the two must not be traded against each other. Raising the threshold to
suppress fractionation peaks would also remove genuine **low-voltage**
activations, which are exactly the fibrotic regions this project exists
to find.

Selectable at run time
----------------------
The method spec makes the suppressor a **strategy chosen at run time**,
like the preprocessors, so alternatives drop in without touching the
pipeline. Phase 1.5 ships the greedy-by-height default; the rest of the
menu is recorded here and in ``roadmap.md``:

===================== ============================================ ==================
algorithm             how it works                                 guarantee
===================== ============================================ ==================
greedy-by-height *    sort tall->short, accept if none kept within tallest in its
                      ``Delta_refr``                               neighbourhood wins
causal blanking       time-ordered scan, blank after each accept   order-dependent
sliding-window max    keep peaks maximal within +/-``Delta_refr``  needs a second pass
segment-merge         merge runs whose sub-threshold gap is small  merges by return
                                                                   to baseline
DP-optimal            max total height with all gaps >= Delta_refr global optimum
===================== ============================================ ==================

``*`` = the Phase-1.5 default.

**Complexity is not the deciding factor.** There are tens to hundreds
of candidates per channel and this is offline batch work, so every
option above runs in microseconds. Choose on correctness and
simplicity. (Speed could matter in a real-time mapping application —
noted, not a Phase-1.5 concern.)

Math + sources: [`docs/theory.md`](../../../../docs/theory.md) §3.5;
greedy/blanking lineage: Pan & Tompkins 1985.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from .candidates import ActivationCandidate


class RefractorySuppressor(ABC):
    """Collapse candidate peaks into distinct activations.

    Subclasses implement :meth:`_suppress` and set :attr:`name`. They do
    not override :meth:`suppress`, which is the template method holding
    the shared contract.

    Contract guaranteed to every caller
    -----------------------------------
    - Input candidates may arrive in any order; output is **ordered by
      sample index**, because the result is an activation *train* and
      every consumer indexes neighbours by position.
    - The output is a subset of the input — suppression never invents,
      moves or merges peaks into new positions. Repositioning is the
      optional refinement step's job, not this one's.
    - Empty input yields empty output.

    The refractory interval is **configuration**, set on the instance
    rather than passed per call — the same shape as
    :class:`~.preprocessors.BotteronEnvelope`'s band edges or
    :class:`~...thresholds.detection.MedianMadThreshold`'s multipliers.
    A suppressor is a configured rule, and one instance means one
    interval.
    """

    name: ClassVar[str]

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "name", None):
            raise TypeError(
                f"{cls.__name__} must define a class-level `name`; it is recorded "
                "in provenance and identifies the suppressor in run records."
            )

    def suppress(self, candidates: list[ActivationCandidate]) -> list[ActivationCandidate]:
        """Return the accepted activations, ordered by sample index."""
        if not candidates:
            return []
        return sorted(self._suppress(candidates), key=lambda c: c.peak_sample)

    @abstractmethod
    def _suppress(self, candidates: list[ActivationCandidate]) -> list[ActivationCandidate]:
        """The suppression rule itself.

        Called only with a non-empty candidate list, and need not sort
        its output.
        """


class GreedyHeightSuppressor(RefractorySuppressor):
    """Greedy by height — the Phase-1.5 default.

    Sort candidates tallest to shortest and accept each only if no
    already-accepted peak lies within ``Delta_refr``. Every survivor is
    therefore the tallest peak in its own refractory neighbourhood.

    Why height-ordered rather than time-ordered: a time-ordered scan
    (causal blanking) accepts whichever peak of a cluster comes *first*
    and blanks the rest, so a small precursor deflection can mask the
    genuine activation behind it. Ordering by height makes the outcome
    independent of which end of the record you start from, which matters
    for an offline splitter where there is no causality requirement to
    respect.

    It is greedy, not optimal: it maximises height locally, so a
    pathological arrangement can yield a lower total height than the
    DP-optimal alternative. On activation trains that case is not
    realistic, and the spec keeps DP-optimal on the menu should it ever
    become one.

    Parameters
    ----------
    refractory_interval_samples
        The minimum spacing between two accepted activations, as a
        **number of samples**. This layer is sample-domain throughout
        and holds no sampling rate, so convert at the call site with
        ``round(refractory_ms * 1e-3 * fs)``.

        Starts at a physiological floor — the shortest plausible atrial
        cycle — and is refined from the low tail of the measured
        inter-activation-interval distribution. Must be positive: a
        non-positive interval would make the step a silent no-op while
        looking like it ran.
    """

    name: ClassVar[str] = "greedy_height"

    def __init__(self, refractory_interval_samples: int) -> None:
        if refractory_interval_samples <= 0:
            raise ValueError(
                f"refractory_interval_samples must be positive, got "
                f"{refractory_interval_samples}. A non-positive interval suppresses nothing."
            )
        self.refractory_interval_samples = int(refractory_interval_samples)

    def _suppress(self, candidates: list[ActivationCandidate]) -> list[ActivationCandidate]:
        accepted: list[ActivationCandidate] = []
        # Tallest first; ties broken by position so the result is
        # deterministic rather than dependent on input order.
        for cand in sorted(candidates, key=lambda c: (-c.height, c.peak_sample)):
            if all(
                abs(cand.peak_sample - kept.peak_sample) >= self.refractory_interval_samples
                for kept in accepted
            ):
                accepted.append(cand)
        return accepted

    def __repr__(self) -> str:
        return f"GreedyHeightSuppressor(refractory_interval_samples={self.refractory_interval_samples})"
