"""Tests for myocard_egm_signal.calibration."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from myocard_egm_signal.calibration import (
    Calibration,
    CalibrationStrategy,
    RWaveAnchoring,
    compute_calibration,
    estimate_qrs_peak_to_peak,
)

# SyntheticRecord type-hint only; the conftest factory builds them.
from tests.conftest import SyntheticRecord


def test_estimate_qrs_peak_to_peak_returns_lead_pp_count(
    synthetic_record: SyntheticRecord,
) -> None:
    """The default preferred-lead order finds II in the synthetic
    record (it's the first preferred lead AND it's present); the
    measured peak-to-peak should be ~ 1.0 mV (the synthetic
    scaling)."""
    lead, pp, n = estimate_qrs_peak_to_peak(synthetic_record)
    assert lead == "II"
    assert n > 0
    assert 0.85 < pp < 1.15


def test_estimate_qrs_falls_back_to_alternative_lead(
    synthetic_record_factory: Callable[..., SyntheticRecord],
) -> None:
    """When II is absent, the preferred-lead search should fall through
    to the next available lead. V1 in the synthetic builder is scaled
    to 0.6 mV peak-to-peak."""
    rec = synthetic_record_factory(surface_leads=("V1", "aVF"))
    lead, pp, _ = estimate_qrs_peak_to_peak(rec)
    assert lead == "V1"
    assert 0.5 < pp < 0.75


def test_estimate_qrs_raises_without_annotations(
    synthetic_record_no_qrs: SyntheticRecord,
) -> None:
    """No QRS annotations means there are no beats to measure; raise
    rather than return 0 or guess from the signal."""
    with pytest.raises(ValueError, match="QRS annotations"):
        estimate_qrs_peak_to_peak(synthetic_record_no_qrs)


def test_estimate_qrs_raises_when_no_preferred_lead_present(
    synthetic_record_factory: Callable[..., SyntheticRecord],
) -> None:
    """If none of the preferred leads are in the record's
    channel_names, raise — silently picking an arbitrary lead would
    lead to a wrong calibration scalar."""
    rec = synthetic_record_factory(surface_leads=("foo", "bar"))
    with pytest.raises(ValueError, match="preferred leads"):
        estimate_qrs_peak_to_peak(rec)


def test_r_wave_anchoring_scalar_correctness(synthetic_record: SyntheticRecord) -> None:
    """Calibration scalar should map measured pp to target pp:
    scalar = target / measured."""
    target = 1.5
    cal = RWaveAnchoring(target_qrs_pp_mv=target).compute(synthetic_record)
    measured = cal.metadata["measured_qrs_pp"]
    assert cal.scalar == pytest.approx(target / measured, rel=1e-9)


def test_r_wave_anchoring_metadata_fields(synthetic_record: SyntheticRecord) -> None:
    """Diagnostic metadata: lead used, target, beat count, measured pp.
    Catches a regression where the strategy stops recording one of
    these audit fields."""
    cal = RWaveAnchoring(target_qrs_pp_mv=2.0).compute(synthetic_record)
    assert cal.method == "r_wave_anchoring"
    assert cal.metadata["lead"] == "II"
    assert cal.metadata["target_qrs_pp_mv"] == 2.0
    assert cal.metadata["n_beats"] > 0
    assert cal.metadata["measured_qrs_pp"] > 0


def test_r_wave_anchoring_preferred_leads_kwarg(
    synthetic_record_factory: Callable[..., SyntheticRecord],
) -> None:
    """Custom preferred_leads should override the default order.
    Building a record where only V1 is available and constructing
    a strategy with preferred_leads=('V1',) should pick V1, not II."""
    rec = synthetic_record_factory(surface_leads=("V1",))
    cal = RWaveAnchoring(preferred_leads=("V1",)).compute(rec)
    assert cal.metadata["lead"] == "V1"


def test_compute_calibration_default_uses_r_wave_anchoring(
    synthetic_record: SyntheticRecord,
) -> None:
    """The convenience function defaults to R-wave anchoring with the
    supplied target."""
    cal = compute_calibration(synthetic_record, target_qrs_pp_mv=1.5)
    assert cal.method == "r_wave_anchoring"
    assert cal.metadata["target_qrs_pp_mv"] == 1.5


def test_compute_calibration_accepts_strategy_object(
    synthetic_record: SyntheticRecord,
) -> None:
    """When given an explicit strategy, the function uses it directly."""
    strat = RWaveAnchoring(target_qrs_pp_mv=2.0)
    cal = compute_calibration(synthetic_record, strategy=strat)
    assert cal.metadata["target_qrs_pp_mv"] == 2.0


def test_compute_calibration_rejects_strategy_plus_kwargs(
    synthetic_record: SyntheticRecord,
) -> None:
    """Passing both is ambiguous — raise rather than pick one and
    silently ignore the other."""
    strat = RWaveAnchoring()
    with pytest.raises(TypeError, match="strategy OR keyword"):
        compute_calibration(synthetic_record, strategy=strat, target_qrs_pp_mv=2.0)


def test_r_wave_anchoring_rejects_bad_config() -> None:
    """Defensive constructor checks: target / window can't be
    non-positive; preferred_leads can't be empty."""
    for target, window in [(-1.0, 100.0), (0.0, 100.0), (1.5, 0.0), (1.5, -10.0)]:
        with pytest.raises(ValueError):
            RWaveAnchoring(target_qrs_pp_mv=target, window_ms=window)
    with pytest.raises(ValueError, match="preferred_leads"):
        RWaveAnchoring(preferred_leads=())


def test_calibration_strategy_protocol_satisfied() -> None:
    """RWaveAnchoring is a structural subtype of CalibrationStrategy."""
    assert isinstance(RWaveAnchoring(), CalibrationStrategy)


def test_apply_calibration_scales_signal(synthetic_record: SyntheticRecord) -> None:
    """End-to-end: multiplying the raw signal by the calibration
    scalar should bring the recomputed QRS pp on lead II to ~ the
    target. Confirms the scalar is mathematically correct."""
    cal = RWaveAnchoring(target_qrs_pp_mv=1.5).compute(synthetic_record)
    scaled = synthetic_record.signal * cal.scalar
    half = int(0.05 * synthetic_record.fs)
    pps = []
    for s in synthetic_record.qrs_samples:
        lo = max(0, int(s) - half)
        hi = min(scaled.shape[0], int(s) + half + 1)
        if hi - lo < 3:
            continue
        col = scaled[lo:hi, 0]  # lead II is col 0
        pps.append(float(np.max(col) - np.min(col)))
    assert np.median(pps) == pytest.approx(1.5, rel=0.05)


def test_calibration_metadata_dict_is_separate_per_instance() -> None:
    """Frozen dataclass with mutable dict default — verify no shared
    state across instances."""
    a = Calibration(scalar=1.0, method="a")
    b = Calibration(scalar=2.0, method="b")
    a.metadata["x"] = 1
    assert "x" not in b.metadata
