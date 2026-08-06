"""Tests for anchor windowing.

What leaves this module — ``realized_position`` — is a stored fleet
contract compared across two corpora, so these tests are less about "does
it crop" and more about the properties that make the stored distributions
comparable: the fraction must mean the same thing at every window length,
the round trip through the contract's own conversion must be exact, a
window must never quietly slide to fit, and the selection applied on top
must not silently reshape the distribution without saying so.
"""

from __future__ import annotations

import numpy as np
import pytest

from myocard_egm_signal.extraction.activation_based import (
    ActivationPositionGenerator,
    AnchoredWindow,
    UniformPositionGenerator,
    WindowSet,
    window_train,
)

T = 100  # T-1 is odd, so p=0.5 lands on a rounding tie
FULL = UniformPositionGenerator(0.0, 1.0, seed=0)


def _ramp(n: int) -> np.ndarray:
    """A signal whose value equals its index, so a crop is self-locating."""
    return np.arange(n, dtype=np.float64)


def _one(signal: np.ndarray, activation_index: int, position: float, length: int) -> AnchoredWindow:
    """Window a single anchor at a fixed position — a train of one."""
    return window_train(
        signal,
        np.array([activation_index]),
        position_generator=UniformPositionGenerator(position, position),
        window_length_samples=length,
    )[0]


def _train_from_gaps(gaps: np.ndarray, lead_in: int = 600) -> np.ndarray:
    return np.cumsum(np.r_[lead_in, gaps]).astype(np.int64)


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------


def test_the_three_named_positions() -> None:
    """p=0 first sample, p=1 last sample, p=0.5 centred."""
    x = _ramp(1000)
    first, last, mid = (_one(x, 500, p, T) for p in (0.0, 1.0, 0.5))

    assert first.start_index == 500
    assert first.signal is not None and first.signal[0] == 500.0
    assert last.end_index == 501
    assert last.signal is not None and last.signal[-1] == 500.0
    assert abs(mid.activation_index - mid.start_index - (T - 1) / 2) <= 0.5


def test_the_window_is_the_crop_it_claims_to_be() -> None:
    x = _ramp(1000)
    w = _one(x, 400, 0.3, T)
    assert w.window_length_samples == T
    assert w.signal is not None
    np.testing.assert_array_equal(w.signal, x[w.start_index : w.end_index])


def test_the_activation_lands_where_the_realized_position_says_it_does() -> None:
    """The contract's own conversion, applied to our output, must recover
    the anchor. ``idx = round(frac * (T - 1))`` is how every consumer
    turns the stored fraction back into a sample; a lossy round trip
    would mean the stored value did not mean what the schema says."""
    x = _ramp(2000)
    for p in (0.0, 0.1, 0.25, 1 / 3, 0.5, 0.75, 0.9, 1.0):
        for length in (2, 3, 64, 99, 100, 501):
            w = _one(x, 1000, p, length)
            idx = round(w.realized_position * (length - 1))
            assert w.start_index + idx == w.activation_index
            assert w.signal is not None and w.signal[idx] == float(w.activation_index)


def test_realized_position_is_within_half_a_sample_of_requested() -> None:
    """Integer rounding of the start index is the only difference between
    the two, so the gap cannot exceed half a sample at any length."""
    x = _ramp(4000)
    rng = np.random.default_rng(0)
    for _ in range(300):
        w = _one(x, 2000, float(rng.uniform(0, 1)), int(rng.integers(2, 800)))
        assert w.position_error_samples <= 0.5 + 1e-9
        assert 0.0 <= w.realized_position <= 1.0


def test_the_fraction_means_the_same_thing_at_any_window_length() -> None:
    """Why the contract stores a fraction rather than a sample offset: the
    same request must place the activation at the same *relative* spot, or
    the corpora's position distributions would not be comparable."""
    x = _ramp(4000)
    for length in (50, 200, 999):
        w = _one(x, 2000, 0.25, length)
        assert w.realized_position == pytest.approx(0.25, abs=0.5 / (length - 1))


# ---------------------------------------------------------------------------
# ActivationPositionGenerator
# ---------------------------------------------------------------------------


def test_a_collapsed_range_is_the_fixed_position_arm() -> None:
    """The A/B baseline is a value, not a code path."""
    fixed = UniformPositionGenerator(0.35, 0.35, seed=0)
    assert fixed.is_fixed and fixed.width == 0.0
    assert np.all(fixed.generate(100) == 0.35)


def test_a_collapsed_range_does_not_consume_its_stream() -> None:
    """So the fixed arm is reproducible however the seed is chosen —
    nothing about it depends on how many positions were produced."""
    gen = UniformPositionGenerator(0.5, 0.5, seed=0)
    gen.generate(1000)
    reference = UniformPositionGenerator(0.2, 0.8, seed=0).generate(5)
    after = UniformPositionGenerator(0.2, 0.8, seed=0)
    np.testing.assert_array_equal(after.generate(5), reference)


def test_a_range_generates_uniformly_inside_its_bounds() -> None:
    drawn = UniformPositionGenerator(0.2, 0.8, seed=0).generate(20000)
    assert drawn.min() >= 0.2
    assert drawn.max() <= 0.8
    assert drawn.mean() == pytest.approx(0.5, abs=0.01)


def test_the_generator_owns_its_stream_and_advances_it() -> None:
    """Config once, then call per trace. Repeated calls continue the
    stream rather than repeating it, which is what a corpus build wants;
    reproducibility comes from the seed, not from object equality."""
    gen = UniformPositionGenerator(0.0, 1.0, seed=7)
    first, second = gen.generate(50), gen.generate(50)
    assert not np.array_equal(first, second)
    fresh = UniformPositionGenerator(0.0, 1.0, seed=7)
    np.testing.assert_array_equal(fresh.generate(50), first)


def test_a_generator_accepts_a_shared_stream() -> None:
    shared = np.random.default_rng(3)
    a = UniformPositionGenerator(0.0, 1.0, seed=shared).generate(10)
    b = UniformPositionGenerator(0.0, 1.0, seed=np.random.default_rng(3)).generate(10)
    np.testing.assert_array_equal(a, b)


def test_an_asymmetric_range_is_allowed() -> None:
    """The synthetic range is *back-bounded* — it may extend further one
    way than the other — so symmetry about 0.5 must not be enforced."""
    back_bounded = UniformPositionGenerator(0.1, 0.6, seed=0)
    assert back_bounded.width == pytest.approx(0.5)
    assert back_bounded.generate(500).max() <= 0.6


def test_generators_reject_bad_bounds_and_counts() -> None:
    for lo, hi in ((-0.1, 0.5), (0.5, 1.1)):
        with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
            UniformPositionGenerator(lo, hi)
    with pytest.raises(ValueError, match="low must not exceed high"):
        UniformPositionGenerator(0.8, 0.2)
    with pytest.raises(ValueError, match="count must be non-negative"):
        UniformPositionGenerator(0.0, 1.0).generate(-1)


def test_a_generator_must_declare_a_name() -> None:
    with pytest.raises(TypeError, match="must define a class-level `name`"):

        class Unnamed(ActivationPositionGenerator):
            def _generate(self, count: int) -> np.ndarray:
                return np.zeros(count)


def test_the_base_class_catches_an_out_of_range_generator() -> None:
    """A new policy cannot quietly emit positions outside [0, 1]; the
    schema constrains the stored value, so this must fail here rather
    than at the bank writer."""

    class Rogue(ActivationPositionGenerator):
        name = "rogue"

        def _generate(self, count: int) -> np.ndarray:
            return np.full(count, 1.5)

    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        Rogue().generate(3)


# ---------------------------------------------------------------------------
# window_train
# ---------------------------------------------------------------------------


def test_one_window_per_anchor_in_train_order() -> None:
    train = np.array([300, 600, 900, 1200])
    ws = window_train(
        _ramp(2000),
        train,
        position_generator=UniformPositionGenerator(0.5, 0.5),
        window_length_samples=T,
    )
    assert len(ws) == len(train)
    assert [w.activation_index for w in ws] == list(train)


def test_a_train_of_one_is_the_synthetic_path() -> None:
    """Both corpora share this function, so the geometry cannot drift."""
    ws = window_train(
        _ramp(1000),
        np.array([500]),
        position_generator=UniformPositionGenerator(0.4, 0.4),
        window_length_samples=T,
    )
    assert len(ws) == 1
    assert ws[0].single_activation is True
    assert ws[0].iai_prev_samples is None and ws[0].iai_next_samples is None


def test_intervals_are_reported_and_absent_only_at_the_ends() -> None:
    """These are what let a study compute the per-anchor keep probability
    without fitting an interval distribution."""
    ws = window_train(
        _ramp(2000),
        np.array([300, 600, 1000, 1500]),
        position_generator=UniformPositionGenerator(0.5, 0.5),
        window_length_samples=T,
    )
    assert [w.iai_prev_samples for w in ws] == [None, 300, 400, 500]
    assert [w.iai_next_samples for w in ws] == [300, 400, 500, None]


def test_out_of_bounds_windows_are_reported_not_dropped_or_raised() -> None:
    """The classify-not-drop seam. The record still carries its start and
    realized position — both pure arithmetic — and only the crop is
    absent, because there is no valid length-T array to hand back."""
    ws = window_train(
        _ramp(2000),
        np.array([10, 1000, 1990]),
        position_generator=UniformPositionGenerator(0.5, 0.5),
        window_length_samples=200,
    )
    assert len(ws) == 3, "nothing is dropped"
    assert list(ws.in_bounds_mask) == [False, True, False]
    for w in ws:
        assert (w.signal is None) is (not w.in_bounds)
        assert 0.0 <= w.realized_position <= 1.0


def test_single_activation_is_exact_at_the_half_open_edges() -> None:
    x = _ramp(3000)
    # Two activations exactly T apart: at p=0 the second sits at s+T, outside.
    ws = window_train(
        x,
        np.array([1000, 1000 + T]),
        position_generator=UniformPositionGenerator(0.0, 0.0),
        window_length_samples=T,
    )
    assert ws[0].single_activation is True
    ws = window_train(
        x,
        np.array([1000, 1000 + T - 1]),
        position_generator=UniformPositionGenerator(0.0, 0.0),
        window_length_samples=T,
    )
    assert ws[0].single_activation is False


def test_the_flag_counts_detections_not_beats() -> None:
    """Why it is named ``single_activation``. Two entries a few samples
    apart — what a detector does to one fractionated complex — read as
    multi-activation, though physiologically that is one beat. The flag
    reports the train it was given; whether the train matches beats is
    the detector's problem."""
    ws = window_train(
        _ramp(2000),
        np.array([1000, 1004]),  # one complex, split by an over-eager detector
        position_generator=UniformPositionGenerator(0.5, 0.5),
        window_length_samples=T,
    )
    assert ws[0].single_activation is False


def test_the_set_filters_only_when_asked() -> None:
    ws = window_train(
        _ramp(2000),
        np.array([10, 1000, 1990]),
        position_generator=UniformPositionGenerator(0.5, 0.5),
        window_length_samples=200,
    )
    kept = ws.select(ws.in_bounds_mask & ws.single_activation_mask)
    assert len(ws) == 3 and len(kept) == 1
    with pytest.raises(ValueError, match="mask must have shape"):
        ws.select(np.array([True, False]))


def test_windowing_is_reproducible_from_the_seed() -> None:
    x, train = _ramp(4000), np.arange(300, 3700, 150)
    runs = [
        window_train(
            x,
            train,
            position_generator=UniformPositionGenerator(0.0, 1.0, seed=7),
            window_length_samples=T,
        ).realized_positions
        for _ in range(2)
    ]
    np.testing.assert_array_equal(runs[0], runs[1])


def test_an_empty_train_is_an_empty_set_not_an_error() -> None:
    """A channel with no detected activations is an ordinary outcome."""
    ws = window_train(
        _ramp(1000),
        np.array([], dtype=np.int64),
        position_generator=FULL,
        window_length_samples=T,
    )
    assert len(ws) == 0
    assert ws.realized_positions.size == 0


def test_window_train_rejects_bad_input() -> None:
    x = _ramp(1000)
    for train, match in (
        (np.array([500, 400]), "strictly increasing"),
        (np.array([500, 5000]), "outside the signal"),
    ):
        with pytest.raises(ValueError, match=match):
            window_train(x, train, position_generator=FULL, window_length_samples=T)
    with pytest.raises(ValueError, match="1-D"):
        window_train(
            np.zeros((10, 3)), np.array([5]), position_generator=FULL, window_length_samples=4
        )
    with pytest.raises(ValueError, match="at least 2"):
        window_train(x, np.array([50]), position_generator=FULL, window_length_samples=1)


# ---------------------------------------------------------------------------
# The position-dependent drop (CL-126 / CL-127)
# ---------------------------------------------------------------------------


def _surviving_position_shape(gaps: np.ndarray, window_length_samples: int) -> np.ndarray:
    """Histogram of the positions that survive the single-activation
    filter, normalised to its own peak so only the *shape* is asserted."""
    train = _train_from_gaps(gaps)
    ws = window_train(
        np.zeros(int(train[-1]) + 600),
        train,
        position_generator=UniformPositionGenerator(0.0, 1.0, seed=0),
        window_length_samples=window_length_samples,
    )
    kept = ws.select(ws.in_bounds_mask & ws.single_activation_mask)
    counts, _ = np.histogram(kept.realized_positions, bins=5, range=(0.0, 1.0))
    shape: np.ndarray = counts / counts.max()
    return shape


def _edge_over_centre(shape: np.ndarray) -> float:
    return float((shape[0] + shape[-1]) / 2 / shape[2])


def test_memoryless_intervals_leave_the_position_distribution_alone() -> None:
    """The analytic result, measured. For an exponential interval
    distribution ``S(pT)S((1-p)T) = e^(-T/mu)`` — independent of p,
    because the window always demands a total of T of clear space and p
    only splits that budget between the two sides. So windows are lost
    without the survivors' positions being reshaped."""
    rng = np.random.default_rng(1)
    gaps = np.maximum(rng.exponential(200, 20000), 1).astype(int)
    assert _edge_over_centre(_surviving_position_shape(gaps, 192)) > 0.85


def test_a_regular_rhythm_near_its_cycle_length_biases_toward_the_centre() -> None:
    """The CL-126 finding: with T near the cycle length the surviving
    positions skew central, so the stored distribution is not the one
    that was requested. Sidestepped by choosing a central band (CL-127)
    rather than corrected after the fact."""
    rng = np.random.default_rng(1)
    gaps = np.clip(rng.normal(180, 15, 20000), 40, None).astype(int)
    assert _edge_over_centre(_surviving_position_shape(gaps, 192)) < 0.80


def test_the_criterion_is_whether_half_the_window_fits_under_the_interval_floor() -> None:
    """Sharper than 'regular versus memoryless'. What decides the bias is
    whether ``T/2`` sits below the shortest intervals: a central anchor
    needs about T/2 of margin on *each* side, an edge anchor nearly all
    of T on *one*. So a floor above T/2 protects the centre only.

    Two exponential trains differing *only* in their floor, at the same
    T, land on opposite sides — which isolates the mechanism as the floor
    relative to T/2, not the regularity of the rhythm."""
    rng = np.random.default_rng(1)
    draws = rng.exponential(200, 20000)
    below = _edge_over_centre(_surviving_position_shape(np.clip(draws, 40, None).astype(int), 192))
    above = _edge_over_centre(_surviving_position_shape(np.clip(draws, 100, None).astype(int), 192))
    assert below > 0.85, "a floor under T/2 protects nothing, so no reshaping"
    assert above < 0.80, "a floor over T/2 protects the centre only"


# ---------------------------------------------------------------------------
# The records themselves
# ---------------------------------------------------------------------------


def test_windows_compare_and_hash_without_touching_the_array() -> None:
    """A generated ``__eq__`` over an ndarray raises on the ambiguous
    truth value, and the array would make the frozen dataclass
    unhashable. The index fields identify a window on their own."""
    x = _ramp(1000)
    a, b = _one(x, 500, 0.5, T), _one(x, 500, 0.5, T)
    assert a == b
    assert len({a, b}) == 1
    assert isinstance(a, AnchoredWindow)
    assert "array" not in repr(a)


def test_zero_is_a_real_position_not_a_missing_one() -> None:
    """The contract is explicit that consumers must not read absence as
    zero, because 0.0 legitimately means 'on the first sample'."""
    w = _one(_ramp(1000), 500, 0.0, T)
    assert w.realized_position == 0.0
    assert w.signal is not None and w.signal[0] == 500.0


def test_the_set_carries_no_configuration_copy() -> None:
    """Window length and position are already on every record, so
    repeating them on the set would be a second copy free to disagree."""
    ws = window_train(
        _ramp(2000),
        np.array([500, 1000]),
        position_generator=UniformPositionGenerator(0.3, 0.7, seed=0),
        window_length_samples=T,
    )
    assert isinstance(ws, WindowSet)
    assert not hasattr(ws, "position_range")
    assert not hasattr(ws, "window_length_samples")
    assert all(w.window_length_samples == T for w in ws)
    assert np.all(ws.requested_positions >= 0.3) and np.all(ws.requested_positions <= 0.7)


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------


def _set_of(indices: list[int], length: int = T, n: int = 4000) -> WindowSet:
    return window_train(
        _ramp(n),
        np.array(indices),
        position_generator=UniformPositionGenerator(0.5, 0.5),
        window_length_samples=length,
    )


def test_concat_pools_in_order() -> None:
    """The corpus-building step: one set per channel, all pooled."""
    a, b = _set_of([300, 600]), _set_of([1000, 1400, 1800])
    pooled = WindowSet.concat([a, b])
    assert len(pooled) == 5
    assert [w.activation_index for w in pooled] == [300, 600, 1000, 1400, 1800]


def test_concat_handles_empty_sets_and_no_sets() -> None:
    empty = _set_of([])
    assert len(WindowSet.concat([])) == 0
    assert len(WindowSet.concat([empty, empty])) == 0
    assert len(WindowSet.concat([empty, _set_of([500]), empty])) == 1


def test_concat_preserves_the_flags_and_positions() -> None:
    """Pooling must not touch what it pools — the distributions are the
    whole reason the pooled set exists."""
    a, b = _set_of([300, 600]), _set_of([10, 1000])  # 10 cannot fit a window
    pooled = WindowSet.concat([a, b])
    np.testing.assert_array_equal(
        pooled.realized_positions, np.r_[a.realized_positions, b.realized_positions]
    )
    np.testing.assert_array_equal(pooled.in_bounds_mask, np.r_[a.in_bounds_mask, b.in_bounds_mask])
    assert list(pooled.in_bounds_mask) == [True, True, False, True]


def test_concat_rejects_mixed_window_lengths() -> None:
    """T is one value coupled across the corpora and the classifier's
    input, so a mixed-length pool is a wiring mistake, not a choice. It
    is caught here because the alternative is a much later failure with
    nothing left to say which source disagreed."""
    with pytest.raises(ValueError, match="differing lengths"):
        WindowSet.concat([_set_of([500], length=T), _set_of([500], length=T + 8)])


def test_concat_composes_with_select() -> None:
    """The real-side idiom: filter per channel, then pool the survivors."""
    sets = [_set_of([10, 1000, 2000]), _set_of([20, 1500])]
    pooled = WindowSet.concat(s.select(s.in_bounds_mask & s.single_activation_mask) for s in sets)
    assert len(pooled) == 3
    assert all(w.in_bounds for w in pooled)
