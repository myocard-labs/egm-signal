"""Tests for the threshold-strategy Protocols and concrete classes."""

from __future__ import annotations

import numpy as np
import pytest

from myocard_egm_signal.thresholds import (
    AbsoluteQuietThreshold,
    AbsoluteThreshold,
    NoiseSegmentStrategy,
    NoThreshold,
    PercentileQuietThreshold,
    PercentileThreshold,
    ThresholdStrategy,
)

# ---------------------------------------------------------------------------
# Healthy (keep-above) strategies
# ---------------------------------------------------------------------------


def test_absolute_threshold_returns_constant() -> None:
    """AbsoluteThreshold ignores the pooled distribution and returns
    the constructor value. The distribution argument is consumed for
    Protocol compatibility but not used."""
    s = AbsoluteThreshold(0.5)
    assert s.compute_threshold(np.array([])) == 0.5
    assert s.compute_threshold(np.array([0.01, 0.02, 100.0])) == 0.5


def test_absolute_threshold_rejects_non_positive() -> None:
    """A 0 or negative threshold has no clinical meaning and would
    accept every window with non-negative p-p; raise instead."""
    with pytest.raises(ValueError, match="must be positive"):
        AbsoluteThreshold(0.0)
    with pytest.raises(ValueError, match="must be positive"):
        AbsoluteThreshold(-1.0)


def test_percentile_threshold_returns_numpy_percentile() -> None:
    """PercentileThreshold returns the requested percentile of the
    pooled p-p distribution. 50th-percentile of [1..9] should be 5."""
    s = PercentileThreshold(50.0)
    pooled = np.arange(1, 10, dtype=float)
    assert s.compute_threshold(pooled) == 5.0


def test_percentile_threshold_rejects_out_of_range() -> None:
    """Percentiles outside (0, 100) are not interpretable; raise."""
    for bad in (0.0, 100.0, -10.0, 150.0):
        with pytest.raises(ValueError, match="must be in"):
            PercentileThreshold(bad)


def test_percentile_threshold_empty_distribution_returns_inf() -> None:
    """An empty pool means no usable threshold; return +inf so the
    healthy-side filter (pp >= threshold) rejects everything. Better
    to return an empty bank than silently accept all."""
    s = PercentileThreshold(50.0)
    assert s.compute_threshold(np.array([])) == float("inf")


def test_no_threshold_returns_neg_inf() -> None:
    """NoThreshold returns -inf so the healthy-side filter
    (pp >= threshold) keeps every window. Pre-training use case."""
    assert NoThreshold().compute_threshold(np.array([0.5, 1.0])) == float("-inf")


def test_threshold_strategy_protocol_satisfied() -> None:
    """All three concrete strategies should structurally satisfy
    ThresholdStrategy. Catches a regression where someone renames
    `name` or `compute_threshold`."""
    assert isinstance(AbsoluteThreshold(0.5), ThresholdStrategy)
    assert isinstance(PercentileThreshold(70.0), ThresholdStrategy)
    assert isinstance(NoThreshold(), ThresholdStrategy)


# ---------------------------------------------------------------------------
# Noise (keep-below) strategies
# ---------------------------------------------------------------------------


def test_absolute_quiet_threshold_returns_constant() -> None:
    """Same shape as AbsoluteThreshold; the distinct name is the
    direction hint (the caller compares with pp <= threshold)."""
    s = AbsoluteQuietThreshold(0.05)
    assert s.compute_threshold(np.array([])) == 0.05
    assert s.compute_threshold(np.array([100.0])) == 0.05


def test_absolute_quiet_threshold_rejects_non_positive() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        AbsoluteQuietThreshold(0.0)


def test_percentile_quiet_threshold_returns_numpy_percentile() -> None:
    """20th-percentile of [1..10] should be 2.8 (numpy linear default)."""
    s = PercentileQuietThreshold(20.0)
    pooled = np.arange(1, 11, dtype=float)
    assert s.compute_threshold(pooled) == pytest.approx(2.8)


def test_percentile_quiet_threshold_empty_distribution_returns_neg_inf() -> None:
    """Empty pool -> -inf so the noise-side filter (pp <= threshold)
    rejects everything. Inverse of the healthy-side convention; both
    are 'reject everything when there's no data' but the empty-pool
    sentinel is opposite per side."""
    s = PercentileQuietThreshold(20.0)
    assert s.compute_threshold(np.array([])) == float("-inf")


def test_noise_segment_strategy_protocol_satisfied() -> None:
    """Both noise strategies satisfy the NoiseSegmentStrategy Protocol."""
    assert isinstance(AbsoluteQuietThreshold(0.05), NoiseSegmentStrategy)
    assert isinstance(PercentileQuietThreshold(20.0), NoiseSegmentStrategy)
