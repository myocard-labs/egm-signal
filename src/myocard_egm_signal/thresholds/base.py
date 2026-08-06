"""Threshold-strategy interfaces.

Four families, distinguished by *which array they consume*, *which
direction they keep*, and *how much context they need*:

- :class:`ThresholdStrategy` — pooled peak-to-peak amplitudes; the
  keep-above (healthy-class) extractor.
- :class:`NoiseSegmentStrategy` — pooled peak-to-peak amplitudes; the
  keep-below (noise) extractor.
- :class:`SignalThreshold` — one 1-D signal; a level read from the
  **whole** array.
- :class:`PositionAwareSignalThreshold` — one 1-D signal **and a sample
  index**; a level that varies with where in the signal you ask.

Keeping them apart is the point. The first two have identical method
signatures and differ only in direction, so separate names are what stop
a keep-below strategy being passed where keep-above was meant. The last
two consume a completely different array — one channel of signal, not a
pooled amplitude distribution — so confusing them with either of the
others is a category error the type checker can catch.

Global versus position-aware is the real divide
-----------------------------------------------
It is tempting to name these two after what they are *used for* —
"detection threshold", "boundary level" — and the earlier revision did
exactly that. That was the wrong axis. A median/MAD rule is the same
computation whether it is deciding "is this an activation?" or "where
does this complex end?", while a peak-relative rule genuinely cannot be
evaluated without knowing which peak. What varies is **how much context
the rule needs**, so that is what the types encode. A caller then cannot
forget to supply the index, and a global rule is not made to accept one
it will ignore.

The two are **siblings, not parent and child.**
:class:`PositionAwareSignalThreshold` looks like a specialization, but
inheriting would let it be passed where the base signature is expected
and then fail — Liskov substitution does not hold when a subclass
*requires* an argument the base does not have. Shared behavior lives in
free functions instead. A future ``RangeAwareSignalThreshold`` (a level
read over a local window, the standard answer to a drifting noise floor
on long records) would be a third sibling, not a subclass of either.

Keep-above by contract
----------------------
Both signal families are **keep-above**: the returned value is a floor a
sample must reach. This matters because it is not implied by the generic
name — :class:`NoiseSegmentStrategy` is keep-below over its own array,
and a sentinel or comparison that fails closed for one direction fails
*open* for the other.

Interface style differs deliberately:

- **Protocol** where *someone else's* type must satisfy the interface
  structurally without inheriting from us. The two amplitude families
  are documented as user-extensible in ``docs/usage.md`` ("define your
  own threshold strategy"), so a caller's class qualifies by shape alone.
- **ABC** where the family is one we ship and extend in-repo *and* has
  shared behavior worth enforcing rather than restating. The two signal
  families are that case: their degenerate-input rules are easy to get
  subtly wrong and must not be re-derived per strategy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Protocol, TypeAlias, runtime_checkable

import numpy as np

from ..exceptions import ConstantSignalError, EmptySignalError


@runtime_checkable
class ThresholdStrategy(Protocol):
    """Plug-in interface for healthy-segment threshold computation.

    The extractor keeps windows whose peak-to-peak is **at or above**
    :meth:`compute_threshold`'s return value.
    """

    name: str

    def compute_threshold(self, pooled_peak_to_peaks: np.ndarray) -> float:
        """Return the effective threshold for this record."""
        ...


@runtime_checkable
class NoiseSegmentStrategy(Protocol):
    """Plug-in interface for noise-segment threshold computation.

    The extractor keeps windows whose peak-to-peak is **at or below**
    :meth:`compute_threshold`'s return value.
    """

    name: str

    def compute_threshold(self, pooled_peak_to_peaks: np.ndarray) -> float:
        """Return the upper-bound peak-to-peak for a window to count as noise."""
        ...


# ---------------------------------------------------------------------------
# Shared behavior for the two signal families
#
# Free functions rather than a common base class, so that neither family
# inherits from the other (see the module docstring on Liskov).
# ---------------------------------------------------------------------------


def _require_name(cls: type) -> None:
    """Enforce the ``name`` ClassVar at subclass-definition time."""
    if not getattr(cls, "name", None):
        raise TypeError(
            f"{cls.__name__} must define a class-level `name`; it is recorded "
            "in provenance and identifies the threshold rule in run records."
        )


def _validated_signal(signal: np.ndarray) -> np.ndarray:
    """Return ``signal`` as 1-D float64, or raise.

    The degenerate cases are handled here, once, because getting them
    wrong is silent and costly. Neither returns a sentinel: a sentinel
    threshold propagates into the next stage and re-emerges as a
    plausible-looking result. See :mod:`..exceptions` for why the two
    conditions are separate exception types.
    """
    if signal.ndim != 1:
        raise ValueError(f"a signal is one channel (1-D), got {signal.ndim}-D.")
    if signal.size == 0:
        raise EmptySignalError(
            "cannot compute a threshold from a zero-length signal; "
            "this usually means an empty slice or a bad index range upstream."
        )
    curve = signal.astype(np.float64, copy=False)
    if float(curve.max()) == float(curve.min()):
        raise ConstantSignalError(
            f"every sample is {float(curve.min())}, so there are no peaks for a "
            "threshold to separate; the usual causes are a dead or disconnected "
            "electrode and a channel saturated against a rail."
        )
    return curve


class SignalThreshold(ABC):
    """A level read from a whole 1-D signal.

    Given one channel — in this package usually a **detection curve**
    ``g`` from a detection preprocessor — return the level a sample must
    reach to qualify. Nothing about the rule is detection-specific: the
    same median/MAD computation serves activation detection (where the
    level is ``tau``, and a local maximum qualifies when ``g[i] >= tau``)
    and complex-boundary measurement (where it is ``theta``, and the
    outward walk stops when the curve drops below it).

    What a threshold gets applied to is the caller's business
    ---------------------------------------------------------
    In the detection chain the answer is **local maxima, not every
    sample**: candidate selection takes points where
    ``g[i-1] < g[i] > g[i+1]`` *and* ``g[i] >= tau``, then filters by
    prominence. The threshold gates *how prominent*; it deliberately
    does **not** decide which candidates are distinct activations — that
    is the refractory interval's job, on the time axis. Keeping those
    separate is what stops the threshold silently merging two genuine
    activations riding a single above-``tau`` run, which happens in fast
    AF when the envelope never fully returns to baseline between beats.

    *(Earlier revisions specified a strictly-greater comparison. That
    mattered when the threshold was imagined as applied sample-wise: on
    a clean sparse curve ``tau`` is 0.0 and ``>=`` would admit all 1000
    samples where ``>`` admitted 8. Under local-maxima extraction the
    two are identical — a flat baseline contains no strict local maximum
    — so ``>=`` is used, matching the method spec and
    ``scipy.signal.find_peaks(height=...)``. Verified: 8 vs 8.)*

    Levels are computed **from the signal itself** rather than supplied
    as absolute numbers, because the three preprocessors produce
    differently-scaled curves — a derivative is spiky, an envelope broad
    — and because amplitude varies across records, channels and
    patients. A fixed number would need retuning for every combination.
    """

    name: ClassVar[str]

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        _require_name(cls)

    def compute_threshold(self, signal: np.ndarray) -> float:
        """Return the level a sample must reach.

        Template method: validates, then delegates to
        :meth:`_compute_threshold`. Subclasses override that, not this.

        Raises
        ------
        EmptySignalError
            The signal has zero length.
        ConstantSignalError
            Every sample holds the same value.
        """
        return self._compute_threshold(_validated_signal(signal))

    @abstractmethod
    def _compute_threshold(self, signal: np.ndarray) -> float:
        """The rule itself.

        Called only with a non-empty, non-constant 1-D float64 array, so
        implementations carry no defensive checks.
        """


class PositionAwareSignalThreshold(ABC):
    """A level that depends on *where* in the signal you ask.

    Same contract as :class:`SignalThreshold` plus a sample index, for
    rules that cannot be evaluated without one — a fraction of the local
    peak's height being the motivating case, since it scales with each
    activation individually rather than with the record.

    Deliberately **not** a subclass of :class:`SignalThreshold`; see the
    module docstring. A caller that accepts either kind distinguishes
    them with a single ``isinstance`` check.
    """

    name: ClassVar[str]

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        _require_name(cls)

    def compute_threshold(self, signal: np.ndarray, position: int) -> float:
        """Return the level at ``position``.

        Raises
        ------
        EmptySignalError
            The signal has zero length.
        ConstantSignalError
            Every sample holds the same value.
        ValueError
            ``position`` is outside the signal.
        """
        curve = _validated_signal(signal)
        if not 0 <= position < curve.size:
            raise ValueError(f"position {position} is outside the signal (0..{curve.size - 1}).")
        return self._compute_threshold(curve, position)

    @abstractmethod
    def _compute_threshold(self, signal: np.ndarray, position: int) -> float:
        """The rule itself, given a validated signal and in-range position."""


SignalThresholdLike: TypeAlias = SignalThreshold | PositionAwareSignalThreshold
"""Either signal-threshold family.

For callers that accept both and branch on ``isinstance``. Spelled as an
alias so the two names do not have to be written out in every signature.
"""
