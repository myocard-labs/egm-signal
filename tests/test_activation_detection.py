"""Tests for the detection function — candidates, suppression, the chain.

The properties that matter downstream: an activation train must have the
right *count* (neither over- nor under-counting), the right *times*, and
the threshold must never be the thing that merges two activations —
that is the refractory interval's job alone.
"""

from __future__ import annotations

import numpy as np
import pytest

from myocard_egm_signal.extraction.activation_based import (
    ActivationCandidate,
    BotteronEnvelope,
    CandidateSelector,
    GreedyHeightSuppressor,
    LocalMaximaSelector,
    RectifiedDerivative,
    RefractorySuppressor,
    TwoStageRefiner,
    detect_activation,
    detect_activation_train,
)
from myocard_egm_signal.extraction.activation_based.candidates import _enclosing_segment
from myocard_egm_signal.thresholds import MedianMadThreshold

FS = 1000.0
REFRACTORY = 60  # samples; a 60 ms physiological floor at 1 kHz


def _prominence(curve: np.ndarray, fraction: float = 0.2) -> float:
    """A prominence floor scaled to the curve, as a caller would set it.

    Measured across every fixture here, 20% of the curve maximum gives
    the correct activation count for both preprocessor families; with no
    prominence filter both over-detect (see
    ``test_without_prominence_the_chain_over_detects``)."""
    return fraction * float(curve.max())


def _activation(n: int, centre: int, width_ms: float = 6.0, amp: float = 1.0) -> np.ndarray:
    """A Gabor pulse — a sine cycle under a Gaussian envelope.

    Tapered rather than truncated, so its steepest slope is at ``centre``
    by construction and no edge discontinuity competes with it.
    """
    u = np.arange(n, dtype=np.float64) - centre
    period = width_ms * 1e-3 * FS
    sigma = period / 3.0
    pulse: np.ndarray = amp * -np.sin(2 * np.pi * u / period) * np.exp(-0.5 * (u / sigma) ** 2)
    return pulse


def _train(n: int, centres: list[int], noise: float = 0.02, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = sum(_activation(n, c) for c in centres)
    return np.asarray(x + noise * rng.normal(size=n))


def _cand(peak: int, height: float, start: int = 0, end: int = 1) -> ActivationCandidate:
    return ActivationCandidate(
        peak_sample=peak, segment_start=start, segment_end=end, height=height, prominence=height
    )


def _unfiltered() -> LocalMaximaSelector:
    return LocalMaximaSelector(min_prominence=None)


# ---------------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------------


def test_candidates_are_local_maxima_at_or_above_tau() -> None:
    """The rule, on a hand-checkable curve."""
    g = np.array([0.0, 1.0, 5.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0])
    assert [c.peak_sample for c in _unfiltered().select(g, 3.0)] == [2, 6]
    # Inclusive: a peak exactly at tau is a candidate.
    assert [c.peak_sample for c in _unfiltered().select(g, 5.0)] == [2]
    assert _unfiltered().select(g, 5.1) == []


def test_candidates_resolve_a_flat_topped_peak() -> None:
    """A plateau must not be dropped.

    The naive `g[i-1] < g[i] > g[i+1]` test finds nothing here, which on
    quantized or synthetic data silently loses activations."""
    g = np.array([0.0, 1.0, 5.0, 5.0, 1.0, 0.0])
    naive = np.flatnonzero((g[1:-1] > g[:-2]) & (g[1:-1] > g[2:])) + 1
    assert naive.size == 0
    assert [c.peak_sample for c in _unfiltered().select(g, 1.0)] == [2]


def test_candidates_carry_the_enclosing_segment() -> None:
    """The above-tau run is the activation-complex extent that S5's
    onset/offset measurement refines. Several candidates can share one
    segment — that is the fractionated case."""
    g = np.array([0.0, 2.0, 5.0, 3.0, 4.0, 2.0, 0.0])
    cands = _unfiltered().select(g, 1.0)
    assert [c.peak_sample for c in cands] == [2, 4]
    assert all(c.segment_start == 1 and c.segment_end == 6 for c in cands)
    assert all(c.segment_width == 5 for c in cands)


def test_segments_are_split_at_every_gap() -> None:
    """Two separated complexes must not be merged into one extent."""
    g = np.array([0.0, 5.0, 6.0, 0.0, 0.0, 9.0, 8.0, 0.0])
    cands = _unfiltered().select(g, 1.0)
    assert [(c.segment_start, c.segment_end) for c in cands] == [(1, 3), (5, 7)]


def test_every_peak_lies_inside_a_segment() -> None:
    """The invariant `_enclosing_segment` relies on.

    A peak exists only because `curve[peak] >= tau`, and the segments are
    the runs where `curve >= tau` — same comparison, same tau — so every
    peak must fall inside exactly one. Swept over randomised curves,
    thresholds, plateaus and a NaN-containing curve."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        size = int(rng.integers(3, 200))
        curve = rng.normal(size=size) * float(rng.choice([1.0, 100.0]))
        tau = float(rng.choice([-5.0, 0.0, float(np.median(curve))]))
        for cand in LocalMaximaSelector(min_prominence=None).select(curve, tau):
            assert cand.segment_start <= cand.peak_sample < cand.segment_end

    for curve in (
        np.array([0.0, 1.0, 5.0, 5.0, 5.0, 1.0, 0.0]),  # plateau
        np.array([0.0, 1.0, 5.0, np.nan, 5.0, 1.0, 0.0]),  # NaN
    ):
        for cand in LocalMaximaSelector(min_prominence=None).select(curve, 1.0):
            assert cand.segment_start <= cand.peak_sample < cand.segment_end


def test_a_peak_outside_every_segment_is_an_error_not_a_guess() -> None:
    """If the invariant above ever breaks, fail loudly.

    Fabricating a single-sample segment would flow a made-up complex
    extent into onset/offset measurement, corrupting a downstream number
    rather than failing where the fault is."""
    with pytest.raises(RuntimeError, match="internal inconsistency"):
        _enclosing_segment([(10, 20), (30, 40)], peak=25)


def test_prominence_filter_drops_noise_blips() -> None:
    """A small bump riding on the shoulder of a real activation has low
    prominence even though its absolute height clears tau."""
    g = np.array([0.0, 1.0, 10.0, 8.0, 8.5, 8.0, 1.0, 0.0])
    assert len(_unfiltered().select(g, 1.0)) == 2
    assert [c.peak_sample for c in LocalMaximaSelector(min_prominence=5.0).select(g, 1.0)] == [2]


def test_prominence_is_reported_even_when_not_filtered_on() -> None:
    """Every candidate carries its prominence whether or not a floor was
    applied.

    This is what the `prominence=0.0` substitution in the selector buys:
    `find_peaks` only populates its prominences when asked to filter on
    them, and we want the value recorded regardless."""
    g = np.array([0.0, 1.0, 5.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0])
    cands = _unfiltered().select(g, 1.0)
    assert [c.prominence for c in cands] == [5.0, 3.0]


def test_no_prominence_floor_and_a_zero_floor_agree() -> None:
    """Guards the assumption the selector *does* rely on.

    With `min_prominence=None` the selector passes `0.0` to `find_peaks`
    so that prominences are still reported. That is only sound because
    the prominence comparison is inclusive, making `0.0` a true no-op
    filter. If a future scipy made it strict, a zero-prominence peak
    would start being dropped — silently, since the two configurations
    are meant to be interchangeable. This test is what makes that noisy
    instead. Quantized curves are used because they make plateaus, the
    likeliest source of a low-prominence peak."""
    rng = np.random.default_rng(0)
    for _ in range(100):
        size = int(rng.integers(3, 200))
        curve = np.round(rng.normal(size=size))  # quantized: plateaus likely
        tau = float(np.median(curve))
        assert [
            c.peak_sample for c in LocalMaximaSelector(min_prominence=None).select(curve, tau)
        ] == [c.peak_sample for c in LocalMaximaSelector(min_prominence=0.0).select(curve, tau)]


def test_infinite_tau_yields_nothing() -> None:
    """The threshold's fail-closed sentinel must propagate, not explode."""
    assert _unfiltered().select(np.array([0.0, 5.0, 0.0]), float("inf")) == []


def test_selector_rejects_multichannel_input() -> None:
    with pytest.raises(ValueError, match="1-D"):
        _unfiltered().select(np.zeros((10, 3)), 0.5)


def test_selector_requires_an_explicit_prominence_choice() -> None:
    """No default: omitting the filter changes the answer (it
    over-detects), so it must be a decision, not an omission."""
    with pytest.raises(TypeError):
        LocalMaximaSelector()  # type: ignore[call-arg]


def test_selector_subclass_must_declare_a_name() -> None:
    with pytest.raises(TypeError, match="must define a class-level `name`"):

        class Unnamed(CandidateSelector):
            def _select(self, detection_curve: np.ndarray, tau: float) -> list[ActivationCandidate]:
                return []


def test_selector_contract_applies_to_any_subclass() -> None:
    """A new rule inherits the degenerate-input handling for free."""

    class Careless(CandidateSelector):
        name = "careless"

        def _select(self, detection_curve: np.ndarray, tau: float) -> list[ActivationCandidate]:
            raise AssertionError("should not be reached for degenerate input")

    assert Careless().select(np.empty(0), 1.0) == []
    assert Careless().select(np.array([1.0, 2.0]), float("inf")) == []
    with pytest.raises(ValueError, match="1-D"):
        Careless().select(np.zeros((4, 2)), 1.0)


# ---------------------------------------------------------------------------
# Suppression
# ---------------------------------------------------------------------------


def test_greedy_keeps_the_tallest_in_each_neighbourhood() -> None:
    cands = [_cand(100, 1.0), _cand(120, 5.0), _cand(135, 2.0), _cand(300, 3.0)]
    kept = GreedyHeightSuppressor(refractory_interval_samples=50).suppress(cands)
    assert [c.peak_sample for c in kept] == [120, 300]


def test_greedy_is_independent_of_input_order() -> None:
    """Height-ordered, not time-ordered: the outcome must not depend on
    which end of the record you started from."""
    cands = [_cand(100, 1.0), _cand(120, 5.0), _cand(135, 2.0)]
    suppressor = GreedyHeightSuppressor(refractory_interval_samples=50)
    forward = suppressor.suppress(cands)
    reverse = suppressor.suppress(list(reversed(cands)))
    assert [c.peak_sample for c in forward] == [c.peak_sample for c in reverse]


def test_greedy_does_not_merge_beyond_the_refractory_interval() -> None:
    """Two peaks further apart than Delta_refr are distinct activations."""
    cands = [_cand(100, 5.0), _cand(161, 4.0)]
    assert len(GreedyHeightSuppressor(refractory_interval_samples=60).suppress(cands)) == 2
    assert len(GreedyHeightSuppressor(refractory_interval_samples=62).suppress(cands)) == 1


def test_suppression_output_is_ordered_by_time() -> None:
    """The result is a train; consumers index neighbours by position."""
    cands = [_cand(500, 9.0), _cand(100, 5.0), _cand(900, 7.0)]
    kept = GreedyHeightSuppressor(refractory_interval_samples=50).suppress(cands)
    assert [c.peak_sample for c in kept] == [100, 500, 900]


def test_suppressor_rejects_a_non_positive_interval_at_construction() -> None:
    """Caught when the rule is configured, not when it silently no-ops."""
    for bad in (0, -60):
        with pytest.raises(ValueError, match="refractory_interval_samples must be positive"):
            GreedyHeightSuppressor(refractory_interval_samples=bad)


def test_suppression_of_nothing_is_nothing() -> None:
    assert GreedyHeightSuppressor(refractory_interval_samples=50).suppress([]) == []


def test_suppressor_subclass_must_declare_a_name() -> None:
    with pytest.raises(TypeError, match="must define a class-level `name`"):

        class Unnamed(RefractorySuppressor):
            def _suppress(self, candidates: list[ActivationCandidate]) -> list[ActivationCandidate]:
                return candidates


def test_suppressor_contract_applies_to_any_subclass() -> None:
    """A new strategy inherits the output ordering for free."""

    class KeepEverything(RefractorySuppressor):
        name = "keep_everything"

        def _suppress(self, candidates: list[ActivationCandidate]) -> list[ActivationCandidate]:
            return list(reversed(candidates))  # deliberately unordered

    kept = KeepEverything().suppress([_cand(9, 1.0), _cand(3, 2.0)])
    assert [c.peak_sample for c in kept] == [3, 9]
    assert KeepEverything().suppress([]) == []


# ---------------------------------------------------------------------------
# The chain
# ---------------------------------------------------------------------------


def test_detect_activation_finds_the_single_synthetic_activation() -> None:
    x = _activation(500, 250)
    assert detect_activation(x, preprocessor=RectifiedDerivative()) == 250


def test_detect_activation_refuses_a_flat_trace() -> None:
    """Returning sample 0 would be a silent lie about an empty trace."""
    with pytest.raises(ValueError, match="no activation"):
        detect_activation(np.zeros(500), preprocessor=RectifiedDerivative())


def test_train_finds_every_activation_at_the_right_time() -> None:
    centres = [200, 500, 800, 1100]
    x = _train(1400, centres)
    preproc = BotteronEnvelope(fs=FS)
    times = detect_activation_train(
        x,
        preprocessor=preproc,
        threshold=MedianMadThreshold(c=1.0, lam=4.0),
        selector=LocalMaximaSelector(min_prominence=_prominence(preproc.compute(x))),
        suppressor=GreedyHeightSuppressor(refractory_interval_samples=REFRACTORY),
    )
    assert len(times) == len(centres)
    assert np.all(np.abs(times - np.array(centres)) <= 5)


def test_train_is_empty_on_a_flat_channel() -> None:
    times = detect_activation_train(
        np.zeros(1000),
        preprocessor=BotteronEnvelope(fs=FS),
        threshold=MedianMadThreshold(c=1.0, lam=4.0),
        selector=_unfiltered(),
        suppressor=GreedyHeightSuppressor(refractory_interval_samples=REFRACTORY),
    )
    assert times.size == 0


def test_no_suppressor_means_no_suppression() -> None:
    """`suppressor=None` skips the step rather than quietly substituting
    a default. On a fractionated complex that over-counts, which is the
    honest consequence of asking for no suppression."""
    n, centre = 1000, 500
    x = _train(n, [], noise=0.02) + sum(
        _activation(n, centre + k * 8, width_ms=4.0) for k in (-1, 0, 1)
    )
    x = np.asarray(x)
    preproc = BotteronEnvelope(fs=FS)
    threshold = MedianMadThreshold(c=1.0, lam=4.0)
    selector = LocalMaximaSelector(min_prominence=_prominence(preproc.compute(x)))

    with_suppression = detect_activation_train(
        x,
        preprocessor=preproc,
        threshold=threshold,
        selector=selector,
        suppressor=GreedyHeightSuppressor(refractory_interval_samples=REFRACTORY),
    )
    without = detect_activation_train(
        x,
        preprocessor=preproc,
        threshold=threshold,
        selector=selector,
        suppressor=None,
    )

    assert len(with_suppression) == 1
    assert len(without) >= len(with_suppression)
    # And the output is still an ordered train.
    assert list(without) == sorted(without)


def test_train_requires_an_explicit_suppressor_choice() -> None:
    """Omitting it must not silently pick one — the step changes the
    answer, so the caller says `None` on purpose or passes a rule."""
    with pytest.raises(TypeError):
        detect_activation_train(  # type: ignore[call-arg]
            _train(500, [250]),
            preprocessor=RectifiedDerivative(),
            threshold=MedianMadThreshold(c=1.0, lam=4.0),
            selector=_unfiltered(),
        )


def test_fractionation_yields_one_activation_under_both_families() -> None:
    """The two routes the method spec distinguishes.

    A fractionated complex is ONE activation made of several
    deflections. The smoothed envelope merges them at the `g` level; the
    sharp derivative does not, and relies on refractory suppression. The
    chain must land on one activation either way."""
    n, centre = 1000, 500
    x = _train(n, [], noise=0.02) + sum(
        _activation(n, centre + k * 8, width_ms=4.0) for k in (-1, 0, 1)
    )
    x = np.asarray(x)

    for preproc in (BotteronEnvelope(fs=FS), RectifiedDerivative()):
        times = detect_activation_train(
            x,
            preprocessor=preproc,
            threshold=MedianMadThreshold(c=1.0, lam=4.0),
            selector=LocalMaximaSelector(min_prominence=_prominence(preproc.compute(x))),
            suppressor=GreedyHeightSuppressor(refractory_interval_samples=REFRACTORY),
        )
        assert len(times) == 1, f"{preproc.name} saw {len(times)} activations, expected 1"
        assert abs(int(times[0]) - centre) <= 15


def test_two_genuine_activations_survive_a_merged_above_tau_run() -> None:
    """The case that decided candidate selection.

    Under heavy smoothing the envelope never returns below tau between
    two genuine activations, so they share ONE above-tau run wider than
    the refractory interval. Local-maxima selection keeps both;
    collapsing the run to its single argmax would lose one, and no
    refractory tuning could recover it."""
    n, first, gap = 1200, 400, 90
    x = _train(n, [first, first + gap])
    envelope = BotteronEnvelope(fs=FS, lowpass_hz=6.0)
    g = envelope.compute(x)
    tau = MedianMadThreshold(c=1.0, lam=4.0).compute_threshold(g)

    # Precondition: the two activations really do share one wide run.
    above = np.flatnonzero(g >= tau)
    runs = np.split(above, np.flatnonzero(np.diff(above) != 1) + 1)
    assert len(runs) == 1, "fixture no longer exercises the merged-run case"
    assert len(runs[0]) > REFRACTORY, "the run must span more than one refractory interval"

    times = detect_activation_train(
        x,
        preprocessor=envelope,
        threshold=MedianMadThreshold(c=1.0, lam=4.0),
        selector=_unfiltered(),
        suppressor=GreedyHeightSuppressor(refractory_interval_samples=REFRACTORY),
    )
    assert len(times) == 2, "both genuine activations must survive a merged above-tau run"


def test_without_prominence_the_chain_over_detects() -> None:
    """The prominence filter is effectively required, not optional.

    An adaptive threshold sitting a few MAD above a quiet baseline is
    low in absolute terms, so filter ringing around a strong activation
    and ordinary noise bumps both clear it — and being spaced further
    apart than the refractory interval, suppression keeps them. Measured
    on a single fractionated complex: 3 activations under the envelope
    and 9 under the derivative with no prominence floor, 1 and 1 once a
    floor of 20% of the curve maximum is applied.

    Pinned because it is the difference between a working splitter and a
    plausible-looking one, and because the failure is silent: the extra
    detections look like activations."""
    n, centre = 1000, 500
    x = _train(n, [], noise=0.02) + sum(
        _activation(n, centre + k * 8, width_ms=4.0) for k in (-1, 0, 1)
    )
    x = np.asarray(x)
    threshold = MedianMadThreshold(c=1.0, lam=4.0)

    for preproc in (BotteronEnvelope(fs=FS), RectifiedDerivative()):
        unfiltered = detect_activation_train(
            x,
            preprocessor=preproc,
            threshold=threshold,
            selector=_unfiltered(),
            suppressor=GreedyHeightSuppressor(refractory_interval_samples=REFRACTORY),
        )
        filtered = detect_activation_train(
            x,
            preprocessor=preproc,
            threshold=threshold,
            selector=LocalMaximaSelector(min_prominence=_prominence(preproc.compute(x))),
            suppressor=GreedyHeightSuppressor(refractory_interval_samples=REFRACTORY),
        )
        assert len(unfiltered) > 1, f"{preproc.name}: fixture no longer over-detects"
        assert len(filtered) == 1, f"{preproc.name}: prominence floor should leave one"


# ---------------------------------------------------------------------------
# Two-stage refinement
# ---------------------------------------------------------------------------


def test_refinement_moves_the_time_toward_the_sharp_maximum() -> None:
    """Refinement undoes the envelope's late bias on an asymmetric
    complex."""
    n, centre = 1000, 500
    # A strong activation with a long, weaker tail — the fibrotic shape
    # that drags a smoothed peak late.
    x = _activation(n, centre) + 0.45 * _activation(n, centre + 25, width_ms=10.0)
    x = np.asarray(x) + 0.02 * np.random.default_rng(11).normal(size=n)
    envelope = BotteronEnvelope(fs=FS)
    selector = LocalMaximaSelector(min_prominence=_prominence(envelope.compute(x)))

    blurred = detect_activation_train(
        x,
        preprocessor=envelope,
        threshold=MedianMadThreshold(c=1.0, lam=4.0),
        selector=selector,
        suppressor=GreedyHeightSuppressor(refractory_interval_samples=REFRACTORY),
    )
    refined = detect_activation_train(
        x,
        preprocessor=envelope,
        threshold=MedianMadThreshold(c=1.0, lam=4.0),
        selector=selector,
        suppressor=GreedyHeightSuppressor(refractory_interval_samples=REFRACTORY),
        refiner=TwoStageRefiner(RectifiedDerivative(), radius_samples=20),
    )
    assert len(blurred) == len(refined) == 1
    assert abs(int(refined[0]) - centre) <= abs(int(blurred[0]) - centre)


def test_refinement_is_inert_when_absent() -> None:
    x = _train(1000, [300, 700])
    envelope = BotteronEnvelope(fs=FS)
    selector = LocalMaximaSelector(min_prominence=_prominence(envelope.compute(x)))
    threshold = MedianMadThreshold(c=1.0, lam=4.0)
    suppressor = GreedyHeightSuppressor(refractory_interval_samples=REFRACTORY)

    implicit = detect_activation_train(
        x,
        preprocessor=envelope,
        threshold=threshold,
        selector=selector,
        suppressor=suppressor,
    )
    explicit_none = detect_activation_train(
        x,
        preprocessor=envelope,
        threshold=threshold,
        selector=selector,
        suppressor=suppressor,
        refiner=None,
    )
    assert np.array_equal(implicit, explicit_none)


def test_refiner_requires_a_positive_radius() -> None:
    with pytest.raises(ValueError, match="radius_samples must be positive"):
        TwoStageRefiner(RectifiedDerivative(), radius_samples=0)


def test_refining_nothing_is_nothing() -> None:
    out = TwoStageRefiner(RectifiedDerivative(), radius_samples=5).refine(
        np.zeros(100), np.array([], dtype=np.int64)
    )
    assert out.size == 0
