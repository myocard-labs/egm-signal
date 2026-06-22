"""Tests for the temperature-scaling probability calibration module.

Three behaviors pinned:

1. **fit_temperature converges to the right region** for synthetic
   logit distributions with known miscalibration. We don't assert
   point-equality (the search is bounded scalar minimization, not
   closed-form), but we do assert "T > 1 for overconfident,
   T < 1 for underconfident, T ~ 1 for already-calibrated".
2. **apply_temperature preserves rank** (the property that makes
   calibration safe for AUROC). For any T > 0 and any logit pair,
   the relative ordering of sigmoid outputs is unchanged.
3. **The fit + apply pipeline reduces BCE loss** compared to the
   uncalibrated logits on the same data — a sanity check that the
   optimizer is doing what it advertises.

We also cover the input-validation edge cases (empty arrays, shape
mismatch, invalid bounds, non-positive temperature).
"""

from __future__ import annotations

import numpy as np
import pytest

from myocard_egm_signal.model import (
    DEFAULT_TEMPERATURE_BOUNDS,
    apply_temperature,
    fit_temperature,
)


def _make_overconfident_logits(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Generate logits that are 2x too sharp for their true class probabilities.

    Construct ground-truth probabilities, draw labels from them, then
    scale the corresponding "true logits" by 2.0 so the model is
    artificially overconfident. T = 2.0 should restore calibration.
    """
    rng = np.random.default_rng(seed)
    true_logits = rng.normal(loc=0.0, scale=1.5, size=n)
    true_probs = 1.0 / (1.0 + np.exp(-true_logits))
    labels = (rng.random(n) < true_probs).astype(np.int64)
    overconfident_logits = (2.0 * true_logits).astype(np.float64)
    return overconfident_logits, labels


def _bce(logits: np.ndarray, labels: np.ndarray) -> float:
    """Reference binary cross-entropy used to verify calibration improves it."""
    z = logits.astype(np.float64)
    y = labels.astype(np.float64)
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


def test_fit_temperature_softens_overconfident_logits() -> None:
    """Overconfident logits (true scaled x2) -> fitted T > 1, near 2.0."""
    logits, labels = _make_overconfident_logits(n=2000, seed=0)
    t = fit_temperature(logits, labels)
    # Tolerance is loose — finite-sample noise + bounded optimization
    # don't recover T == 2.0 exactly. The point is "T is well above 1
    # and in the right ballpark".
    assert t > 1.3
    assert t < 3.0


def test_fit_temperature_sharpens_underconfident_logits() -> None:
    """Underconfident logits (true scaled x0.5) -> fitted T < 1."""
    rng = np.random.default_rng(1)
    true_logits = rng.normal(loc=0.0, scale=2.0, size=2000)
    true_probs = 1.0 / (1.0 + np.exp(-true_logits))
    labels = (rng.random(2000) < true_probs).astype(np.int64)
    underconfident_logits = 0.5 * true_logits
    t = fit_temperature(underconfident_logits, labels)
    assert t < 0.8


def test_fit_temperature_near_one_for_calibrated_logits() -> None:
    """If logits are already calibrated, T should land near 1.0."""
    rng = np.random.default_rng(2)
    true_logits = rng.normal(loc=0.0, scale=2.0, size=4000)
    true_probs = 1.0 / (1.0 + np.exp(-true_logits))
    labels = (rng.random(4000) < true_probs).astype(np.int64)
    t = fit_temperature(true_logits, labels)
    # Sampling noise means T isn't exactly 1.0; broad tolerance.
    assert 0.8 < t < 1.25


def test_calibration_reduces_bce_loss() -> None:
    """Applying the fitted T should produce lower BCE than the raw logits."""
    logits, labels = _make_overconfident_logits(n=2000, seed=3)
    t = fit_temperature(logits, labels)
    calibrated = apply_temperature(logits, t)
    assert _bce(calibrated, labels) < _bce(logits, labels)


def test_apply_temperature_preserves_rank() -> None:
    """Monotonic transform: argsort of sigmoid(logits/T) == argsort of logits."""
    rng = np.random.default_rng(4)
    logits = rng.normal(size=500)
    for t in (0.2, 0.5, 1.0, 2.0, 5.0):
        scaled = apply_temperature(logits, t)
        assert np.array_equal(np.argsort(logits), np.argsort(scaled)), f"rank changed at T={t}"


def test_apply_temperature_identity_at_one() -> None:
    """T = 1.0 is the identity operation."""
    rng = np.random.default_rng(5)
    logits = rng.normal(size=100).astype(np.float32)
    out = apply_temperature(logits, 1.0)
    np.testing.assert_array_equal(out, logits)


def test_apply_temperature_preserves_dtype_for_float32() -> None:
    """float32 logits stay float32 through apply_temperature (numpy NEP 50).

    Avoids silent float64 promotion that would otherwise inflate
    downstream memory on large prediction banks.
    """
    rng = np.random.default_rng(6)
    logits = rng.normal(size=10).astype(np.float32)
    out = apply_temperature(logits, 2.5)
    assert out.dtype == np.float32


def test_fit_temperature_accepts_2d_logits() -> None:
    """[N, 1] logits get flattened to [N] internally; same answer as [N] input."""
    logits_1d, labels = _make_overconfident_logits(n=1000, seed=7)
    logits_2d = logits_1d.reshape(-1, 1)
    t_1d = fit_temperature(logits_1d, labels)
    t_2d = fit_temperature(logits_2d, labels)
    assert t_1d == pytest.approx(t_2d, abs=1e-6)


def test_fit_temperature_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match=r"empty"):
        fit_temperature(np.array([]), np.array([]))


def test_fit_temperature_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match=r"shape mismatch"):
        fit_temperature(np.zeros(10), np.zeros(8))


def test_fit_temperature_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError, match=r"bounds"):
        fit_temperature(np.zeros(4), np.zeros(4), bounds=(0.0, 1.0))
    with pytest.raises(ValueError, match=r"bounds"):
        fit_temperature(np.zeros(4), np.zeros(4), bounds=(2.0, 1.0))


def test_apply_temperature_rejects_non_positive_T() -> None:
    logits = np.zeros(4)
    with pytest.raises(ValueError, match=r"> 0"):
        apply_temperature(logits, 0.0)
    with pytest.raises(ValueError, match=r"> 0"):
        apply_temperature(logits, -1.0)


def test_default_bounds_constant_shape() -> None:
    """Sanity-check the default bounds are a (low, high) tuple with low < high."""
    low, high = DEFAULT_TEMPERATURE_BOUNDS
    assert 0 < low < high
