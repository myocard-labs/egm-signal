"""Tests for the detection-threshold rules.

These decide *how prominent is prominent enough* on a detection curve.
The two properties that matter downstream: the level must adapt to the
curve's own scale (so one configuration works across records whose
amplitudes differ), and degenerate curves must fail closed rather than
promoting noise to activations.
"""

from __future__ import annotations

import numpy as np
import pytest

from myocard_egm_signal.extraction.activation_based import RectifiedDerivative
from myocard_egm_signal.thresholds import (
    DetectionThreshold,
    MedianMadThreshold,
    PercentileDetectionThreshold,
    median_absolute_deviation,
)

ALL_RULES: list[DetectionThreshold] = [
    MedianMadThreshold(c=1.0, lam=5.0),
    PercentileDetectionThreshold(q=99.0),
]


def _sparse_curve(n: int = 1000, n_spikes: int = 8, baseline: float = 0.0) -> np.ndarray:
    """A detection curve shaped like a real one: mostly baseline, a few
    tall narrow spikes where the activations are."""
    g = np.full(n, baseline, dtype=np.float64)
    g[np.linspace(100, n - 100, n_spikes).astype(int)] = 10.0
    return g


# ---------------------------------------------------------------------------
# median_absolute_deviation
# ---------------------------------------------------------------------------


def test_mad_matches_its_definition() -> None:
    """median(|x - median(x)|), pinned on a hand-computed case."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    # median = 3; deviations = [2,1,0,1,97]; median of those = 1
    assert median_absolute_deviation(x) == pytest.approx(1.0)


def test_mad_ignores_outliers_where_std_does_not() -> None:
    """The whole reason MAD is used: one huge outlier moves the standard
    deviation enormously and MAD not at all. On a detection curve the
    'outliers' are the activations being detected, so a sigma-based
    threshold would rise with the very signal it is looking for."""
    base = np.random.default_rng(0).normal(size=999)
    with_activation = np.append(base, 1000.0)
    assert np.std(with_activation) > 10 * np.std(base)
    assert median_absolute_deviation(with_activation) == pytest.approx(
        median_absolute_deviation(base), rel=0.01
    )


# ---------------------------------------------------------------------------
# Shared contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rule", ALL_RULES, ids=lambda r: r.name)
def test_empty_curve_fails_closed(rule: DetectionThreshold) -> None:
    """Nothing to detect, so admit nothing."""
    assert rule.compute_threshold(np.empty(0)) == float("inf")


@pytest.mark.parametrize("rule", ALL_RULES, ids=lambda r: r.name)
@pytest.mark.parametrize("value", [0.0, 5.0, -3.0])
def test_constant_curve_fails_closed(rule: DetectionThreshold, value: float) -> None:
    """A constant curve has no peaks. Any finite threshold would admit
    either everything or nothing, and admitting everything is the
    dangerous direction — a dead channel would yield an activation at
    every sample."""
    assert rule.compute_threshold(np.full(200, value)) == float("inf")


@pytest.mark.parametrize("rule", ALL_RULES, ids=lambda r: r.name)
def test_rejects_multichannel_input(rule: DetectionThreshold) -> None:
    """A detection curve is one channel."""
    with pytest.raises(ValueError, match="1-D"):
        rule.compute_threshold(np.zeros((100, 4)))


@pytest.mark.parametrize("rule", ALL_RULES, ids=lambda r: r.name)
def test_adapts_to_curve_scale(rule: DetectionThreshold) -> None:
    """Scaling the curve scales the threshold with it — that is what
    makes one configuration usable across records and patients whose
    amplitudes differ by an order of magnitude."""
    g = _sparse_curve()
    rng = np.random.default_rng(1)
    g = g + rng.normal(scale=0.1, size=g.size)
    tau_1 = rule.compute_threshold(g)
    tau_10 = rule.compute_threshold(10.0 * g)
    assert tau_10 == pytest.approx(10.0 * tau_1, rel=1e-9)


@pytest.mark.parametrize("rule", ALL_RULES, ids=lambda r: r.name)
def test_separates_spikes_from_baseline(rule: DetectionThreshold) -> None:
    """The point of the exercise: every activation must clear the level
    and almost nothing else should.

    Asserted as the property rather than a numeric window — the two
    rules legitimately land at different levels (measured: 0.55 and
    0.61 on this curve), and pinning a range would test the curve's
    arithmetic rather than the rule's job."""
    rng = np.random.default_rng(2)
    spikes = np.linspace(100, 900, 8).astype(int)
    g = _sparse_curve() + np.abs(rng.normal(scale=0.2, size=1000))
    tau = rule.compute_threshold(g)

    assert np.all(g[spikes] > tau), "every activation must clear the threshold"
    assert int(np.sum(g > tau)) < 0.05 * g.size, "and the baseline mostly must not"


def test_subclass_must_declare_a_name() -> None:
    """Same structural enforcement as the preprocessors."""
    with pytest.raises(TypeError, match="must define a class-level `name`"):

        class Unnamed(DetectionThreshold):
            def _compute_threshold(self, detection_curve: np.ndarray) -> float:
                return 0.0


def test_degenerate_rules_are_applied_before_the_subclass() -> None:
    """A new rule inherits the fail-closed handling; its own code is
    never even reached for empty or constant input."""

    class Careless(DetectionThreshold):
        name = "careless"

        def _compute_threshold(self, detection_curve: np.ndarray) -> float:
            raise AssertionError("should not be reached for degenerate input")

    assert Careless().compute_threshold(np.empty(0)) == float("inf")
    assert Careless().compute_threshold(np.zeros(50)) == float("inf")


# ---------------------------------------------------------------------------
# MedianMadThreshold
# ---------------------------------------------------------------------------


def test_median_mad_matches_its_formula() -> None:
    """tau = c*median(g) + lam*MAD(g), pinned by hand."""
    g = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    tau = MedianMadThreshold(c=2.0, lam=3.0).compute_threshold(g)
    assert tau == pytest.approx(2.0 * 3.0 + 3.0 * 1.0)


def test_median_mad_threshold_does_not_chase_the_activations() -> None:
    """Adding more activations must not raise the bar.

    A standard-deviation rule would: each new spike inflates sigma, so
    the threshold climbs and a busy channel detects a smaller fraction
    of its activations than a quiet one. MAD's 50% breakdown point is
    what prevents it."""
    rng = np.random.default_rng(3)
    noise = np.abs(rng.normal(scale=0.2, size=2000))
    rule = MedianMadThreshold(c=1.0, lam=5.0)

    few = noise.copy()
    few[np.linspace(100, 1900, 5).astype(int)] = 50.0
    many = noise.copy()
    many[np.linspace(100, 1900, 200).astype(int)] = 50.0

    def sigma_rule(g: np.ndarray) -> float:
        return float(np.mean(g) + 5.0 * np.std(g))

    mad_drift = rule.compute_threshold(many) / rule.compute_threshold(few)
    sigma_drift = sigma_rule(many) / sigma_rule(few)

    # Measured: MAD 0.53 -> 0.62 (x1.16); sigma 12.7 -> 79.9 (x6.3).
    assert mad_drift < 1.5, f"MAD threshold drifted x{mad_drift:.2f} with activation count"
    assert sigma_drift > 3.0, f"sigma rule only drifted x{sigma_drift:.2f} - premise broken"


def test_median_mad_on_a_clean_sparse_curve_still_detects() -> None:
    """The MAD == 0 case, which is NOT an error.

    A clean synthetic trace is exactly flat between activations, so over
    half its detection-curve samples are identical and MAD is 0 — the
    same signature as a dead channel. Rather than fail closed (which
    would make the synthetic pipeline silently detect nothing), the
    formula degrades to c*median(g): with a zero baseline that means
    'anything above baseline is a candidate', which is correct when
    there is no noise to sit above."""
    g = _sparse_curve(baseline=0.0)
    assert median_absolute_deviation(g) == 0.0  # the degenerate signature
    tau = MedianMadThreshold(c=1.0, lam=5.0).compute_threshold(g)
    assert np.isfinite(tau)
    assert tau == 0.0
    # The threshold gates LOCAL MAXIMA, not samples (see DetectionThreshold).
    # A flat baseline has no strict local maximum, so `>=` and `>` agree —
    # both admit exactly the 8 deflections. Applied sample-wise they would
    # not: `>=` would take all 1000. That difference is why extraction, not
    # the comparison operator, is what makes detection selective here.
    local_max = np.flatnonzero((g[1:-1] > g[:-2]) & (g[1:-1] > g[2:])) + 1
    assert int(np.sum(g[local_max] >= tau)) == 8
    assert int(np.sum(g[local_max] > tau)) == 8
    assert int(np.sum(g >= tau)) == g.size  # the sample-wise reading, for contrast


def test_median_mad_on_a_clean_synthetic_curve_marks_a_region_not_a_peak() -> None:
    """A documented characteristic, pinned so it cannot regress silently.

    On a *clean* synthetic trace the detection curve is exactly 0.0 over
    most of its length (the pulse's Gaussian tail underflows), so
    median and MAD are both 0 and the rule degrades to "any non-zero
    sample". Measured here: 59% of samples are exactly zero, tau = 0,
    and 206 of 500 samples become candidates.

    That is not a failure, but it is not discrimination either — the
    threshold marks one contiguous *region* around the activation and
    all the real work of picking a time falls to the refractory
    suppression downstream. Worth knowing for the synthetic pipeline,
    where the method spec anyway treats detection as a cross-check
    (the generator knows where it put the activation) rather than a
    measurement. On real signal, which has a noise floor everywhere,
    MAD is non-zero and the rule discriminates normally."""
    fs = 1000.0
    n, centre = 500, 250
    u = np.arange(n, dtype=np.float64) - centre
    period, sigma = 8e-3 * fs, (8e-3 * fs) / 3.0
    x = -np.sin(2 * np.pi * u / period) * np.exp(-0.5 * (u / sigma) ** 2)

    g = RectifiedDerivative().compute(x)
    tau = MedianMadThreshold(c=1.0, lam=5.0).compute_threshold(g)
    assert np.isfinite(tau)

    idx = np.flatnonzero(g >= tau)
    runs = np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1)
    assert len(runs) == 1, "the above-threshold samples form one region"
    assert idx.min() < centre < idx.max(), "and that region contains the activation"
    assert int(np.argmax(g)) == centre, "with the peak itself exactly on it"

    # Local-maxima extraction is what makes this selective: 500 samples ->
    # a handful of candidates, which the refractory step then reduces to one.
    local_max = np.flatnonzero((g[1:-1] > g[:-2]) & (g[1:-1] > g[2:])) + 1
    candidates = local_max[g[local_max] >= tau]
    assert 0 < len(candidates) <= 8, f"expected a handful of candidates, got {len(candidates)}"


def test_median_mad_rejects_negative_constants() -> None:
    """Negative multipliers would invert the rule's meaning."""
    with pytest.raises(ValueError, match="c must be non-negative"):
        MedianMadThreshold(c=-1.0, lam=5.0)
    with pytest.raises(ValueError, match="lam must be non-negative"):
        MedianMadThreshold(c=1.0, lam=-5.0)


def test_median_mad_requires_both_constants() -> None:
    """No defaults: how aggressive detection should be is a policy the
    calling pipeline owns, not one this library picks."""
    with pytest.raises(TypeError):
        MedianMadThreshold()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        MedianMadThreshold(c=1.0)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# PercentileDetectionThreshold
# ---------------------------------------------------------------------------


def test_percentile_fixes_the_candidate_rate() -> None:
    """q = 99 admits at most 1% of samples, whatever the curve's shape."""
    rng = np.random.default_rng(4)
    g = np.abs(rng.normal(size=10_000))
    tau = PercentileDetectionThreshold(q=99.0).compute_threshold(g)
    assert int(np.sum(g >= tau)) == pytest.approx(100, rel=0.2)


def test_percentile_always_promotes_something() -> None:
    """The documented weakness: on a channel with no activations it
    still returns its top q% — which is why it is an exploration tool,
    not a production splitter."""
    rng = np.random.default_rng(5)
    pure_noise = np.abs(rng.normal(scale=0.01, size=5000))
    tau = PercentileDetectionThreshold(q=99.0).compute_threshold(pure_noise)
    assert np.isfinite(tau)
    assert int(np.sum(pure_noise >= tau)) > 0


def test_percentile_rejects_out_of_range_q() -> None:
    for bad in (0.0, 100.0, -5.0, 150.0):
        with pytest.raises(ValueError, match=r"q must be in \(0, 100\)"):
            PercentileDetectionThreshold(q=bad)
