"""Anchor windowing — cutting a signal into activation-anchored windows.

Where the sliding-window extractors cut a record at a fixed stride, this
cuts it **relative to detected activations**: given an activation at
index ``t_a``, a fractional position ``p``, and a window length ``T``,
the window starts at::

    s = round(t_a - p * (T - 1))

so ``p = 0.0`` puts the activation on the window's first sample,
``p = 1.0`` on its last, and ``p = 0.5`` centres it.

:func:`window_train` is the windowing primitive — window a *whole
activation train*. Both corpora go through it: the real side passes a
detected train, the synthetic side its single activation as a **train of
one**. That is the point rather than a convenience, because one code
path means the window geometry cannot drift between the corpora, which
is what makes their activation-position distributions comparable at all.
For the detect-and-window flow in one call, see :mod:`.windowers`.

Naming
------
Counts and durations carry a ``_samples`` suffix; positions in an array
carry ``_index``. ``window_length_samples`` is a length, ``start_index``
is an index. (Modules written before this convention —
``complex_bounds``, ``detection`` — still say ``activation_sample``.)

Note that *sample* means a **signal sample** everywhere in this library,
never a draw from a distribution. Hence
:meth:`ActivationPositionGenerator.generate` rather than any spelling
involving "sample", which would read as though it returned signal.

``realized_position`` is a fleet contract
-----------------------------------------
Because ``s`` is an integer, the position a window achieves is generally
not the one requested. Both are recorded, and the **realized** one is
what gets stored, under the name ``activation_position``. Its definition
is shared fleet-wide via ``common.schema.json#/$defs/ActivationPosition``
in egm-contracts, ``$ref``'d by both ``iafdb_bank`` and
``synthetic_bank`` so the corpora cannot drift. Three points this module
is responsible for honouring:

- ``[0, 1]`` where **0.0 is the first sample and 1.0 the last**. Convert
  at point of use with ``idx = round(frac * (T - 1))``.
- It is what the crop produced — **never a value measured back off the
  waveform**. A consumer wanting the measured dV/dt-max position uses
  egm-features. Storing a measurement here would silently turn a
  comparison of *what we asked for* into one of *what a detector found*.
- ``0.0`` is a legitimate value. Nothing downstream may read absence as
  zero, which would fabricate a spike at the low edge of the
  distribution.

Classify, don't drop
--------------------
:func:`window_train` labels every window and discards none. Which to
keep is the **producer's** decision, because the two corpora have
deliberately different policies and because the drop rate is itself a
measured quantity — see :class:`WindowSet` and ``docs/theory.md`` §5.6.

Math + the position-dependent drop:
[`docs/theory.md`](../../../../docs/theory.md) §5.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

MIN_WINDOW_LENGTH_SAMPLES = 2
"""A window shorter than this has no meaningful activation position.

``realized_position`` divides by ``T - 1``, and in a one-sample window
the activation is simultaneously at the first and the last sample.
Rejected rather than special-cased: any convention chosen for the
degenerate case would flow straight into the stored distribution.
"""


# ---------------------------------------------------------------------------
# Where the activation goes
# ---------------------------------------------------------------------------


class ActivationPositionGenerator(ABC):
    """Produces the fractional positions windows are anchored at.

    Varying the position per window is what **blocks the positional
    shortcut**: if every window placed the activation identically, a
    classifier could key on position instead of morphology, which is the
    hypothesis this whole apparatus exists to test.

    A strategy family rather than a fixed rule. Uniform-over-an-interval
    is the Phase-1.5 policy, but it is a policy — a shaped distribution
    is a plausible later need — so the interface is the family and
    :class:`UniformPositionGenerator` is one member.

    **Stateful, unlike the rest of this library.** The generator owns its
    random stream, seeded at construction, so a caller configures it once
    instead of threading a generator through every call. Two consequences
    worth knowing: repeated calls continue the stream rather than
    repeating it (which is what you want per trace across a corpus), and
    two generators built with the same arguments are independent, so
    reproducibility comes from the seed, not from equality.
    """

    name: ClassVar[str]

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "name", None):
            raise TypeError(
                f"{cls.__name__} must define a class-level `name`; it is recorded "
                "in provenance and identifies the position policy in run records."
            )

    def __init__(self, seed: int | np.random.Generator | None = None) -> None:
        self._rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)

    def generate(self, count: int) -> np.ndarray:
        """Return ``count`` activation positions in ``[0, 1]``.

        Template method: validates, then delegates to :meth:`_generate`.

        Deliberately not called ``sample`` — in this library *sample*
        means a signal sample everywhere else, so a method of that name
        returning position fractions would read as though it returned
        signal.
        """
        if count < 0:
            raise ValueError(f"count must be non-negative, got {count}.")
        positions = self._generate(count)
        if positions.shape != (count,):
            raise ValueError(
                f"{type(self).__name__} produced shape {positions.shape}, expected ({count},)."
            )
        if count and (positions.min() < 0.0 or positions.max() > 1.0):
            raise ValueError(
                f"{type(self).__name__} produced positions outside [0, 1]: "
                f"min {positions.min()}, max {positions.max()}."
            )
        return positions

    @abstractmethod
    def _generate(self, count: int) -> np.ndarray:
        """The rule itself, given a validated non-negative ``count``."""


class UniformPositionGenerator(ActivationPositionGenerator):
    """Uniform over ``[low, high]`` — the Phase-1.5 position policy.

    **Point-collapsible.** ``UniformPositionGenerator(0.5, 0.5)`` is a
    fixed position, which is the baseline arm of the anchored-versus-
    varied A/B. One class covers both arms, so switching between them is
    a value change rather than a code path — the same idiom the noise
    mixer uses for ``snr_db_range=(X, X)``. A collapsed generator does
    not consume its random stream, so the fixed arm is reproducible
    however the seed is chosen.

    **A range is shared as a mechanism, not as a value.** The two corpora
    deliberately use different ranges: the synthetic one covers the real
    one and may be wider, but is *back-bounded*, because a single-beat
    synthetic trace has no signal before the upstroke and a far-back
    position would fill the window with flat, unrealistic
    pre-activation. So a range is **not required to be symmetric about
    0.5** and this class does not impose that — a central band is a
    choice a producer makes, not a property of the type.

    No default range is provided. The values are policy and belong to
    the pipeline that knows its data.

    Parameters
    ----------
    low, high
        Fractions in ``[0, 1]`` with ``low <= high``.
    seed
        Seed or generator for the random stream.
    """

    name: ClassVar[str] = "uniform_position"

    def __init__(
        self, low: float, high: float, seed: int | np.random.Generator | None = None
    ) -> None:
        super().__init__(seed)
        for label, value in (("low", low), ("high", high)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{label} must be in [0, 1], got {value}.")
        if low > high:
            raise ValueError(f"low must not exceed high, got low={low}, high={high}.")
        self.low = float(low)
        self.high = float(high)

    @property
    def is_fixed(self) -> bool:
        """True when the range has collapsed to a point."""
        return self.low == self.high

    @property
    def width(self) -> float:
        return self.high - self.low

    def _generate(self, count: int) -> np.ndarray:
        if self.is_fixed:
            return np.full(count, self.low, dtype=np.float64)
        drawn: np.ndarray = self._rng.uniform(self.low, self.high, size=count)
        return drawn

    def __repr__(self) -> str:
        return f"UniformPositionGenerator(low={self.low}, high={self.high})"


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnchoredWindow:
    """One window placed on one activation, with its classification.

    Carries both positions because they differ whenever ``p * (T - 1)``
    is not an integer, and only :attr:`realized_position` is meaningful
    downstream — it is the value stored as ``activation_position``.

    The classification fields are **reports, not filters**. A window that
    is out of bounds or holds a neighbour is still returned; see
    :class:`WindowSet`.
    """

    activation_index: int
    """The anchor, as an index into the **original** signal."""

    start_index: int
    end_index: int
    """Exclusive: the window is ``signal[start_index:end_index]``."""

    requested_position: float
    realized_position: float
    """``(t_a - s) / (T - 1)`` — the position this crop actually produced."""

    in_bounds: bool
    """Whether ``[start_index, end_index)`` lies inside the signal."""

    single_activation: bool
    """Whether exactly one member of the **detected train** is inside.

    Named for what is counted. It is *not* a claim that the window holds
    one beat: a fractionated or low-voltage complex that the detector
    splits registers as several activations, so a genuinely single-beat
    window can land here as ``False``. Whether detections correspond to
    beats is the detector's tuning problem, not this flag's.
    """

    iai_prev_samples: int | None
    iai_next_samples: int | None
    """Intervals to the neighbouring activations, or ``None`` for none.

    ``None`` at the ends of a train, and on both sides of a train of one.
    Deliberately not a large sentinel: "there is no neighbour" is a
    different statement from "the neighbour is very far away", and only
    the first is true here.

    Reported because the per-anchor probability that a window survives
    the single-activation test reads directly off these two numbers,
    which is what lets a study measure the position-dependent drop
    without fitting an interval distribution (theory §5.7).
    """

    signal: np.ndarray | None = field(compare=False, repr=False, default=None)
    """The cropped window, or ``None`` when out of bounds.

    There is no valid length-``T`` array to return in that case, while
    every other field remains defined — they are pure arithmetic.

    Excluded from equality and repr: an ndarray in a generated ``__eq__``
    raises on the ambiguous truth value, and it would make the frozen
    dataclass unhashable. The index fields identify a window on their
    own.
    """

    @property
    def window_length_samples(self) -> int:
        return self.end_index - self.start_index

    @property
    def position_error_samples(self) -> float:
        """How far the realized position sits from the requested one.

        In samples, so it is comparable across window lengths. Bounded by
        half a sample — integer rounding of ``s`` is its only source.
        """
        return abs(self.realized_position - self.requested_position) * (
            self.window_length_samples - 1
        )


@dataclass(frozen=True)
class WindowSet:
    """Every window cut from one signal, each labelled, none discarded.

    **Keep/drop is the caller's.** This reports ``in_bounds`` and
    ``single_activation``; it does not act on them. Two reasons the seam
    sits here: the corpora have deliberately different policies (the real
    side drops boundary and multi-activation windows, the synthetic side
    sizes its simulation so neither can occur), and the drop rate is a
    quantity a study needs to *measure* — drops that happened invisibly
    inside a library could not be counted.

    Accordingly this exposes masks and a :meth:`select`, never a
    ``keep()``: the mechanism to filter, with the policy left outside.

    Carries no configuration. The window length and the position of each
    window are already on the records, so repeating them here would be a
    second copy that could disagree with the first.
    """

    windows: tuple[AnchoredWindow, ...]

    def __len__(self) -> int:
        return len(self.windows)

    def __iter__(self):  # type: ignore[no-untyped-def]  # Iterator[AnchoredWindow]
        return iter(self.windows)

    def __getitem__(self, item: int) -> AnchoredWindow:
        return self.windows[item]

    @property
    def realized_positions(self) -> np.ndarray:
        """``(n,)`` float array — the stored ``activation_position`` values."""
        return np.array([w.realized_position for w in self.windows], dtype=np.float64)

    @property
    def requested_positions(self) -> np.ndarray:
        return np.array([w.requested_position for w in self.windows], dtype=np.float64)

    @property
    def in_bounds_mask(self) -> np.ndarray:
        return np.array([w.in_bounds for w in self.windows], dtype=bool)

    @property
    def single_activation_mask(self) -> np.ndarray:
        return np.array([w.single_activation for w in self.windows], dtype=bool)

    @classmethod
    def concat(cls, window_sets: Iterable[WindowSet]) -> WindowSet:
        """Pool several sets into one, preserving order.

        The corpus-building step: one set comes out of each channel of
        each record, and the bank is built from all of them together.

        An alternate constructor rather than a free function, so a
        subclass gets its own type back. Empty sets contribute nothing,
        and concatenating nothing yields an empty set.

        **Every window must share one length.** ``T`` is a single value
        coupled across the simulator, the real-data splitter and the
        classifier's input, so a pooled set of mixed lengths is not a
        choice anyone makes deliberately — it is a wiring mistake. Caught
        here because the alternative is discovering it much later, when
        the windows fail to stack into an array, with nothing left to say
        which source disagreed.

        .. warning::
           **The index fields stop being meaningful across sources.**
           ``activation_index``, ``start_index`` and ``end_index`` are
           positions in *their own* source signal, so once windows from
           different channels or records sit in one set, two of them can
           carry identical indices while referring to entirely different
           places. The positions, flags, intervals and crops all survive
           pooling; the indices only identify a window within the set it
           came from.

           A producer that needs to know where each window came from —
           for a patient-aware split, say — should keep its own
           ``(source, WindowSet)`` pairs and pool at the very end, or not
           pool at all. This class deliberately carries no source
           identity: inventing one here would be guessing at a
           provenance scheme that belongs to the producer.
        """
        sets = list(window_sets)
        lengths = {w.window_length_samples for s in sets for w in s}
        if len(lengths) > 1:
            raise ValueError(
                f"cannot pool windows of differing lengths: found {sorted(lengths)}. "
                "T is one value shared across the corpora and the classifier input."
            )
        return cls(windows=tuple(w for s in sets for w in s))

    def select(self, mask: np.ndarray) -> WindowSet:
        """Return the subset where ``mask`` is true, as a new set.

        The caller supplies the mask, so the policy stays outside:
        ``ws.select(ws.in_bounds_mask & ws.single_activation_mask)`` is
        the usual real-data filter, spelled at the call site rather than
        assumed here.
        """
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (len(self.windows),):
            raise ValueError(f"mask must have shape ({len(self.windows)},), got {mask.shape}.")
        return WindowSet(
            windows=tuple(w for w, keep in zip(self.windows, mask, strict=True) if keep)
        )


# ---------------------------------------------------------------------------
# Internals
#
# These were public before the train primitive existed. They are the
# arithmetic `window_train` is built from, and everything they answered
# for a caller is now reported on the records instead.
# ---------------------------------------------------------------------------


def _validate_window_length(window_length_samples: int) -> None:
    if window_length_samples < MIN_WINDOW_LENGTH_SAMPLES:
        raise ValueError(
            f"window_length_samples must be at least {MIN_WINDOW_LENGTH_SAMPLES}, got "
            f"{window_length_samples}; a one-sample window has no meaningful "
            "activation position."
        )


def _start_index(activation_index: int, activation_position: float, length: int) -> int:
    """``s = round(t_a - p(T-1))``.

    Ties round to even (Python's ``round``). Which way a tie breaks is
    arbitrary — it moves the window one sample and the realized position
    half a sample, inside the rounding error already accepted — but it is
    fixed, so the same request always yields the same crop.
    """
    return round(activation_index - float(activation_position) * (length - 1))


def _fits(start_index: int, length: int, n_samples: int) -> bool:
    """Whether ``[s, s+T)`` lies inside ``[0, n_samples)``.

    A predicate, never a clamp. Sliding a window to fit would change the
    realized position without saying so — the activation would no longer
    sit where it was asked to, and the stored value would record the
    slide as though it were intended.
    """
    return start_index >= 0 and start_index + length <= n_samples


def _count_inside(train: np.ndarray, start_index: int, length: int) -> int:
    """How many train members fall in the half-open ``[s, s+T)``.

    Half-open matches the crop: a member at ``s`` is inside, one at
    ``s + T`` is not.
    """
    inside = (train >= start_index) & (train < start_index + length)
    return int(np.count_nonzero(inside))


def _build_window(
    *,
    signal: np.ndarray,
    activation_index: int,
    requested_position: float,
    start_index: int,
    window_length_samples: int,
    single_activation: bool,
    iai_prev_samples: int | None,
    iai_next_samples: int | None,
) -> AnchoredWindow:
    """Assemble one record. Assumes its inputs are already validated."""
    end_index = start_index + window_length_samples
    in_bounds = _fits(start_index, window_length_samples, signal.size)
    return AnchoredWindow(
        activation_index=activation_index,
        start_index=start_index,
        end_index=end_index,
        requested_position=requested_position,
        realized_position=(activation_index - start_index) / (window_length_samples - 1),
        in_bounds=in_bounds,
        single_activation=single_activation,
        iai_prev_samples=iai_prev_samples,
        iai_next_samples=iai_next_samples,
        signal=signal[start_index:end_index] if in_bounds else None,
    )


# ---------------------------------------------------------------------------
# The primitive
# ---------------------------------------------------------------------------


def window_train(
    signal: np.ndarray,
    activation_train: np.ndarray,
    *,
    position_generator: ActivationPositionGenerator,
    window_length_samples: int,
) -> WindowSet:
    """Cut one window per activation, classifying each.

    Both corpora go through this: the real side passes a detected train,
    the synthetic side its single activation as a **train of one**.
    Sharing the path is what stops the window geometry drifting between
    them. For detection and windowing in one call see
    :class:`~.windowers.ActivationWindower`; use this directly when the
    train is already in hand — a probe sweeping crop offsets over one
    simulation, for instance, which must detect *once* on the source
    rather than re-detect per crop.

    Nothing is dropped. Each window is labelled ``in_bounds`` and
    ``single_activation`` and carries its neighbouring intervals; the
    caller decides what to keep. See :class:`WindowSet`.

    Parameters
    ----------
    signal
        1-D ``(n_samples,)`` array.
    activation_train
        Strictly increasing activation indices, as produced by
        :func:`~.detection.detect_activation_train`.
    position_generator
        Supplies one position per activation, and owns its own random
        stream — see :class:`ActivationPositionGenerator`.
    window_length_samples
        ``T``, at least :data:`MIN_WINDOW_LENGTH_SAMPLES`.

    Raises
    ------
    ValueError
        On a non-1-D signal, a ``T`` below the minimum, or a train that
        is not strictly increasing or strays outside the signal. An
        **empty train is not an error** — a channel with no detected
        activations is an ordinary outcome, and yields an empty set.
    """
    if signal.ndim != 1:
        raise ValueError(f"a signal is one channel (1-D), got {signal.ndim}-D.")
    _validate_window_length(window_length_samples)

    train = np.asarray(activation_train, dtype=np.int64)
    if train.ndim != 1:
        raise ValueError(f"activation_train must be 1-D, got {train.ndim}-D.")
    if train.size and (train.min() < 0 or train.max() >= signal.size):
        raise ValueError(
            f"activation_train has indices outside the signal (0..{signal.size - 1}): "
            f"min {int(train.min())}, max {int(train.max())}."
        )
    if train.size > 1 and not np.all(np.diff(train) > 0):
        raise ValueError(
            "activation_train must be strictly increasing; the intervals reported per "
            "window are meaningless otherwise."
        )

    positions = position_generator.generate(len(train))
    gaps = np.diff(train) if train.size > 1 else np.empty(0, dtype=np.int64)

    windows = []
    for k, (activation_index, requested_position) in enumerate(zip(train, positions, strict=True)):
        start = _start_index(
            int(activation_index), float(requested_position), window_length_samples
        )
        windows.append(
            _build_window(
                signal=signal,
                activation_index=int(activation_index),
                requested_position=float(requested_position),
                start_index=start,
                window_length_samples=window_length_samples,
                single_activation=_count_inside(train, start, window_length_samples) == 1,
                iai_prev_samples=int(gaps[k - 1]) if k > 0 else None,
                iai_next_samples=int(gaps[k]) if k < len(train) - 1 else None,
            )
        )

    return WindowSet(windows=tuple(windows))
