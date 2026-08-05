"""Tests for activation-complex bounds.

This is a measurement instrument, so what matters is that the numbers
mean what they claim: a fibrotic-shaped complex must measure asymmetric,
a boundary that ran out of room must be distinguishable from one that
genuinely crossed the level, and the walk must not wander into a
neighbouring activation.
"""

from __future__ import annotations

import numpy as np
import pytest

from myocard_egm_signal.exceptions import ConstantSignalError, EmptySignalError
from myocard_egm_signal.extraction.activation_based import (
    ActivationComplex,
    BotteronEnvelope,
    RectifiedDerivative,
    measure_complex,
    measure_complexes,
)
from myocard_egm_signal.thresholds import (
    MedianMadThreshold,
    PeakFractionThreshold,
    PositionAwareSignalThreshold,
    median_absolute_deviation,
)

FS = 1000.0


def _activation(n: int, centre: int, width_ms: float = 6.0, amp: float = 1.0) -> np.ndarray:
    u = np.arange(n, dtype=np.float64) - centre
    period = width_ms * 1e-3 * FS
    sigma = period / 3.0
    pulse: np.ndarray = amp * -np.sin(2 * np.pi * u / period) * np.exp(-0.5 * (u / sigma) ** 2)
    return pulse


def _triangle(n: int, peak: int, rise: int, fall: int) -> np.ndarray:
    """A curve whose rise and fall lengths are known exactly."""
    g = np.zeros(n, dtype=np.float64)
    g[peak - rise : peak + 1] = np.linspace(0.0, 1.0, rise + 1)
    g[peak : peak + fall + 1] = np.linspace(1.0, 0.0, fall + 1)
    return g


# ---------------------------------------------------------------------------
# Boundary levels
# ---------------------------------------------------------------------------


def test_peak_fraction_level_scales_with_the_local_peak() -> None:
    """Each complex is measured against its own amplitude, so a weak
    activation is not judged by the record's largest."""
    g = np.array([0.0, 2.0, 10.0, 2.0, 0.0])
    assert PeakFractionThreshold(fraction=0.5).compute_threshold(g, 2) == pytest.approx(5.0)
    assert PeakFractionThreshold(fraction=0.5).compute_threshold(g, 1) == pytest.approx(1.0)


def test_baseline_mad_level_is_a_level_above_the_floor() -> None:
    """median + k*MAD, not k*MAD alone. A bare MAD multiple is a spread,
    not a level: on a curve with a raised baseline it would sit below the
    noise floor and the outward walk would never terminate."""
    rng = np.random.default_rng(0)
    g = 5.0 + rng.normal(scale=0.1, size=1000)  # baseline well above zero
    theta = MedianMadThreshold(c=1.0, lam=3.0).compute_threshold(g)
    expected = float(np.median(g)) + 3.0 * median_absolute_deviation(g)
    assert theta == pytest.approx(expected)
    assert theta > float(np.median(g)), "must sit above the baseline, not below it"


def test_baseline_mad_level_ignores_the_activations() -> None:
    """Robust for the same reason the detection threshold is: activations
    are large outliers in the curve and must not raise the level."""
    rng = np.random.default_rng(1)
    quiet = np.abs(rng.normal(scale=0.1, size=2000))
    busy = quiet.copy()
    busy[np.linspace(100, 1900, 40).astype(int)] = 50.0
    level = MedianMadThreshold(c=1.0, lam=3.0)
    assert level.compute_threshold(busy) == pytest.approx(level.compute_threshold(quiet), rel=0.1)


def test_levels_reject_bad_configuration() -> None:
    for bad in (0.0, 1.0, -0.5, 1.5):
        with pytest.raises(ValueError, match=r"fraction must be in \(0, 1\)"):
            PeakFractionThreshold(fraction=bad)
    with pytest.raises(ValueError, match="lam must be non-negative"):
        MedianMadThreshold(c=1.0, lam=-1.0)


def test_levels_reject_an_out_of_range_activation() -> None:
    # Deliberately not np.zeros: the signal is validated before the
    # position, so a flat curve would raise ConstantSignalError first and
    # this test would pass without exercising the bounds check at all.
    g = np.arange(10, dtype=np.float64)
    with pytest.raises(ValueError, match="outside the signal"):
        PeakFractionThreshold(fraction=0.5).compute_threshold(g, 10)


def test_a_flat_curve_is_rejected_before_the_position_is_looked_at() -> None:
    """Ordering matters. A dead channel is the more fundamental problem
    and should be the reported one even when the position is also bad."""
    with pytest.raises(ConstantSignalError):
        PeakFractionThreshold(fraction=0.5).compute_threshold(np.zeros(10), 999)


def test_measuring_a_dead_channel_raises_rather_than_reporting_zero_width() -> None:
    """The reason the sentinel was removed.

    With a ``+inf`` return the walk terminated immediately and produced
    ``width=0, is_complete=True, clamped=False`` — a fabricated complex
    indistinguishable from a real instantaneous one, which would then be
    pooled into the duration distribution the whole study rests on."""
    with pytest.raises(ConstantSignalError):
        measure_complex(np.zeros(500), 250, boundary_threshold=MedianMadThreshold(c=1.0, lam=3.0))
    with pytest.raises(EmptySignalError):
        measure_complexes(
            np.array([]), np.array([0]), boundary_threshold=MedianMadThreshold(c=1.0, lam=3.0)
        )


def test_level_subclass_must_declare_a_name() -> None:
    with pytest.raises(TypeError, match="must define a class-level `name`"):

        class Unnamed(PositionAwareSignalThreshold):
            def _compute_threshold(self, signal: np.ndarray, position: int) -> float:
                return 0.0


# ---------------------------------------------------------------------------
# Measuring a complex
# ---------------------------------------------------------------------------


def test_onset_and_offset_are_the_theta_crossings() -> None:
    """Hand-checkable: a triangle rising over 10 samples and falling over
    20, measured at half height, has its edges at the half-height points."""
    g = _triangle(200, peak=100, rise=10, fall=20)
    c = measure_complex(g, 100, boundary_threshold=PeakFractionThreshold(fraction=0.5))
    assert c.onset_sample == 94  # first sample below 0.5 walking back
    assert c.offset_sample == 111  # first sample below 0.5 walking forward
    assert c.is_complete


def test_a_fibrotic_shape_measures_asymmetric() -> None:
    """The property the study depends on: a long trailing tail must
    produce fall > rise. That asymmetry is what sets the *back* margin,
    which the method spec says needs to be the more generous one."""
    g = _triangle(400, peak=200, rise=10, fall=60)
    c = measure_complex(g, 200, boundary_threshold=PeakFractionThreshold(fraction=0.25))
    assert c.fall_samples > 3 * c.rise_samples
    assert c.width_samples == c.rise_samples + c.fall_samples


def test_a_clamped_boundary_is_reported_not_hidden() -> None:
    """A side that ran out of room is not a measurement. Pooling it into
    a duration distribution would bias the tail the study exists to
    characterise, so it must be filterable."""
    # A curve that never returns below theta on the right.
    g = np.concatenate([np.zeros(50), np.linspace(0.0, 1.0, 50), np.ones(100)])
    c = measure_complex(g, 120, boundary_threshold=PeakFractionThreshold(fraction=0.5))
    assert c.offset_clamped
    assert not c.is_complete


def test_the_search_radius_stops_the_walk_reaching_a_neighbour() -> None:
    """The runaway walk, and the condition that causes it.

    It happens exactly when ``theta`` falls **below the noise floor**:
    the curve never dips under it between beats, so the walk continues
    until it meets a neighbour. Measured on a four-activation train with
    realistic noise, at ``theta = 0.02 * peak`` (below the baseline
    median): a 597-sample 'complex' reaching back past the previous
    activation. Without noise the same fraction sits *above* the floor
    and behaves normally — which is why this fixture is the noisy one.

    A search radius bounds it, and flags the result clamped rather than
    passing the bound off as a measured edge."""
    n, centres = 1400, [200, 500, 800, 1100]
    rng = np.random.default_rng(0)
    x = np.asarray(sum(_activation(n, c) for c in centres)) + 0.02 * rng.normal(size=n)
    g = BotteronEnvelope(fs=FS).compute(x)
    level = PeakFractionThreshold(fraction=0.02)

    # Precondition: theta really is under the noise floor here.
    assert level.compute_threshold(g, 500) < float(np.median(g))

    unbounded = measure_complex(g, 500, boundary_threshold=level)
    bounded = measure_complex(g, 500, boundary_threshold=level, search_radius_samples=60)

    assert unbounded.width_samples > 300, "fixture no longer exercises the runaway walk"
    assert unbounded.onset_sample < 200, "the unbounded walk reaches the previous activation"
    assert bounded.width_samples <= 120
    assert bounded.onset_clamped and bounded.offset_clamped


def test_a_level_above_the_noise_floor_does_not_run_away() -> None:
    """The counterpart: on the same noisy train, a level chosen above the
    floor terminates on genuine crossings without needing a radius.

    This is the argument for a median/MAD level when complexes are weak —
    it is defined *relative to the floor*, so it cannot fall beneath it
    the way a small fraction of a small peak can."""
    n, centres = 1400, [200, 500, 800, 1100]
    rng = np.random.default_rng(0)
    x = np.asarray(sum(_activation(n, c) for c in centres)) + 0.02 * rng.normal(size=n)
    g = BotteronEnvelope(fs=FS).compute(x)

    c = measure_complex(g, 500, boundary_threshold=MedianMadThreshold(c=1.0, lam=3.0))
    assert c.is_complete
    assert c.width_samples < 200


def test_measurement_rejects_bad_input() -> None:
    g = np.zeros(100)
    with pytest.raises(ValueError, match="1-D"):
        measure_complex(
            np.zeros((10, 3)), 5, boundary_threshold=PeakFractionThreshold(fraction=0.5)
        )
    with pytest.raises(ValueError, match="outside the curve"):
        measure_complex(g, 500, boundary_threshold=PeakFractionThreshold(fraction=0.5))
    with pytest.raises(ValueError, match="search_radius_samples must be positive"):
        measure_complex(
            g, 50, boundary_threshold=PeakFractionThreshold(fraction=0.5), search_radius_samples=0
        )


# ---------------------------------------------------------------------------
# Asymmetric search radius
# ---------------------------------------------------------------------------


def test_an_asymmetric_radius_bounds_each_side_independently() -> None:
    """A tuple is ``(before, after)``, and the two really are separate.

    The fixture has a long trailing tail, which is the shape the
    asymmetric form exists for: a tight cap behind the peak and a
    generous one in front of it."""
    g = _triangle(600, peak=300, rise=10, fall=200)
    c = measure_complex(
        g,
        300,
        boundary_threshold=PeakFractionThreshold(fraction=0.1),
        search_radius_samples=(5, 150),
    )
    assert c.rise_samples == 5 and c.onset_clamped, "backward side capped at 5"
    assert c.fall_samples == 150 and c.offset_clamped, "forward side capped at 150"


def test_an_asymmetric_radius_can_clamp_one_side_only() -> None:
    """The point of splitting the cap: the generous side still terminates
    on a genuine crossing while the tight side is the one that runs out."""
    g = _triangle(600, peak=300, rise=100, fall=20)
    c = measure_complex(
        g,
        300,
        boundary_threshold=PeakFractionThreshold(fraction=0.5),
        search_radius_samples=(10, 200),
    )
    assert c.onset_clamped, "10 samples cannot reach the half-height point 50 back"
    assert not c.offset_clamped, "200 is ample for a 20-sample fall"
    assert not c.is_complete


def test_a_scalar_radius_equals_the_symmetric_tuple() -> None:
    g = _triangle(600, peak=300, rise=80, fall=80)
    level = PeakFractionThreshold(fraction=0.25)
    scalar = measure_complex(g, 300, boundary_threshold=level, search_radius_samples=30)
    tup = measure_complex(g, 300, boundary_threshold=level, search_radius_samples=(30, 30))
    assert scalar == tup


def test_asymmetric_radius_rejects_bad_input() -> None:
    g = _triangle(600, peak=300, rise=10, fall=10)
    level = PeakFractionThreshold(fraction=0.5)
    with pytest.raises(ValueError, match=r"must be positive.*before"):
        measure_complex(g, 300, boundary_threshold=level, search_radius_samples=(0, 10))
    with pytest.raises(ValueError, match=r"must be positive.*after"):
        measure_complex(g, 300, boundary_threshold=level, search_radius_samples=(10, -1))
    with pytest.raises(ValueError, match=r"must be \(before, after\)"):
        measure_complex(
            g,
            300,
            boundary_threshold=level,
            search_radius_samples=(1, 2, 3),  # type: ignore[arg-type]  # deliberately wrong arity
        )


def test_measuring_a_whole_train() -> None:
    n, centres = 1400, [200, 500, 800, 1100]
    x = np.asarray(sum(_activation(n, c) for c in centres))
    g = BotteronEnvelope(fs=FS).compute(x)
    complexes = measure_complexes(
        g,
        np.array(centres),
        boundary_threshold=PeakFractionThreshold(fraction=0.25),
        search_radius_samples=60,
    )
    assert len(complexes) == len(centres)
    assert [c.activation_sample for c in complexes] == centres
    assert all(c.is_complete for c in complexes)


def test_measuring_an_empty_train_is_empty() -> None:
    assert (
        measure_complexes(
            np.zeros(100),
            np.array([], dtype=np.int64),
            boundary_threshold=MedianMadThreshold(c=1.0, lam=3.0),
        )
        == []
    )


# ---------------------------------------------------------------------------
# Why the curve must be smoothed
# ---------------------------------------------------------------------------


def test_a_sharp_curve_measures_a_fragment_not_the_complex() -> None:
    """Why the docstring insists on a smoothed curve.

    A fractionated complex dips below theta *between* its deflections on
    a sharp curve, so the walk stops at the first dip and measures one
    deflection. The envelope bridges those dips and measures the whole
    complex — which is the number the study wants."""
    n, centre = 1000, 500
    x = np.asarray(sum(_activation(n, centre + k * 8, width_ms=4.0) for k in (-1, 0, 1)))

    sharp = measure_complex(
        RectifiedDerivative().compute(x),
        centre,
        boundary_threshold=PeakFractionThreshold(fraction=0.25),
    )
    smoothed = measure_complex(
        BotteronEnvelope(fs=FS).compute(x),
        centre,
        boundary_threshold=PeakFractionThreshold(fraction=0.25),
    )
    assert smoothed.width_samples > sharp.width_samples, (
        f"envelope should span the whole complex (got {smoothed.width_samples}) "
        f"where the sharp curve sees a fragment (got {sharp.width_samples})"
    )


def test_complex_arithmetic_is_self_consistent() -> None:
    """rise + fall == width, on arbitrary shapes."""
    rng = np.random.default_rng(2)
    for _ in range(50):
        rise, fall = int(rng.integers(2, 40)), int(rng.integers(2, 40))
        g = _triangle(300, peak=150, rise=rise, fall=fall)
        c: ActivationComplex = measure_complex(
            g, 150, boundary_threshold=PeakFractionThreshold(fraction=0.5)
        )
        assert c.rise_samples + c.fall_samples == c.width_samples
        assert c.onset_sample <= c.activation_sample <= c.offset_sample
