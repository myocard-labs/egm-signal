"""Tests for myocard_egm_signal.extraction (both extractors)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from myocard_egm_signal.calibration import RWaveAnchoring
from myocard_egm_signal.extraction import (
    HealthySegment,
    NoiseSegment,
    extract_healthy_segments,
    extract_noise_segments,
)
from myocard_egm_signal.thresholds import (
    AbsoluteQuietThreshold,
    AbsoluteThreshold,
    NoThreshold,
    PercentileQuietThreshold,
    PercentileThreshold,
)
from tests.conftest import SyntheticRecord

# ---------------------------------------------------------------------------
# Healthy (keep-above) extraction
# ---------------------------------------------------------------------------


def test_extract_healthy_with_absolute_threshold(
    synthetic_record: SyntheticRecord,
) -> None:
    """Pick a low threshold so the synthetic record's bumps clear it;
    confirm we get segments back of the expected shape + provenance."""
    cal = RWaveAnchoring(target_qrs_pp_mv=1.5).compute(synthetic_record)
    segs = extract_healthy_segments(
        synthetic_record,
        threshold=AbsoluteThreshold(0.01),
        channels=("CS12", "CS34"),
        calibration=cal,
        window_ms=100.0,
        hop_ms=50.0,
    )
    assert len(segs) > 0
    for s in segs:
        assert isinstance(s, HealthySegment)
        assert s.record_name == "synthetic_001"
        assert s.patient == "syn1"
        assert s.channel in {"CS12", "CS34"}
        assert s.fs == 1000.0


def test_extract_healthy_no_threshold_keeps_everything(
    synthetic_record: SyntheticRecord,
) -> None:
    """NoThreshold returns -inf so every window is kept. Confirms
    the keep-above filter doesn't accidentally reject anything."""
    segs = extract_healthy_segments(
        synthetic_record,
        threshold=NoThreshold(),
        channels=("CS12", "CS34"),
        window_ms=200.0,
        hop_ms=200.0,
    )
    # 4 seconds * 1 kHz / 200 ms = 20 windows per channel, 2 channels = 40.
    assert len(segs) == 40


def test_extract_healthy_high_threshold_rejects_all(
    synthetic_record: SyntheticRecord,
) -> None:
    """An impossibly high threshold should return zero segments rather
    than crash."""
    segs = extract_healthy_segments(
        synthetic_record,
        threshold=AbsoluteThreshold(1e6),
        channels=("CS12", "CS34"),
    )
    assert segs == []


def test_extract_healthy_silently_skips_missing_channels(
    synthetic_record: SyntheticRecord,
) -> None:
    """Channels not in record.channel_names are skipped. Producers
    pass a fixed channel list (e.g. all 5 IAFDB CS pairs) and individual
    records may only have a subset; we shouldn't crash on those."""
    segs = extract_healthy_segments(
        synthetic_record,
        threshold=NoThreshold(),
        channels=("CS12", "CS_NOT_PRESENT", "CS34"),
        window_ms=200.0,
        hop_ms=200.0,
    )
    # Still 40 segments from the two channels actually present.
    assert len(segs) == 40


def test_extract_healthy_percentile_threshold(
    synthetic_record: SyntheticRecord,
) -> None:
    """50th-percentile selection keeps approximately the top half by
    p-p. Confirms the per-record pooled distribution path works."""
    segs = extract_healthy_segments(
        synthetic_record,
        threshold=PercentileThreshold(50.0),
        channels=("CS12", "CS34"),
        window_ms=200.0,
        hop_ms=200.0,
    )
    # 40 windows total; keeping >= median should give ~20-21 segments.
    # Allow a small margin for ties.
    assert 18 <= len(segs) <= 23


# ---------------------------------------------------------------------------
# Noise (keep-below) extraction
# ---------------------------------------------------------------------------


def test_extract_noise_with_absolute_threshold(
    synthetic_record: SyntheticRecord,
) -> None:
    """A high absolute cutoff should keep the quiet baseline windows;
    confirm segment shape + provenance fields."""
    segs = extract_noise_segments(
        synthetic_record,
        strategy=AbsoluteQuietThreshold(1.0),  # well above the bumps
        channels=("CS12", "CS34"),
        window_ms=200.0,
        hop_ms=200.0,
    )
    assert len(segs) > 0
    for s in segs:
        assert isinstance(s, NoiseSegment)
        assert s.record_name == "synthetic_001"
        assert s.patient == "syn1"


def test_extract_noise_percentile_keeps_bottom_fraction(
    synthetic_record: SyntheticRecord,
) -> None:
    """20th-percentile quiet should keep approximately the bottom 20%
    of windows by p-p. Validates the inverted comparison direction."""
    segs = extract_noise_segments(
        synthetic_record,
        strategy=PercentileQuietThreshold(20.0),
        channels=("CS12", "CS34"),
        window_ms=200.0,
        hop_ms=200.0,
    )
    # 40 windows total; keep bottom ~20% with ties = 8 +/- a few.
    assert 4 <= len(segs) <= 12


def test_extract_noise_empty_record_returns_empty() -> None:
    """An empty record (no channels in `channels` actually present)
    should return an empty list, not raise."""
    rec = SyntheticRecord(
        name="empty",
        patient="empty",
        fs=1000.0,
        signal=np.zeros((1000, 0)),
        channel_names=(),
    )
    segs = extract_noise_segments(
        rec,
        strategy=PercentileQuietThreshold(20.0),
        channels=("CS12",),
    )
    assert segs == []


def test_extract_noise_segment_length_matches_window(
    synthetic_record: SyntheticRecord,
) -> None:
    """Every returned segment's signal length should equal
    round(window_ms * 1e-3 * fs). Producers rely on this to assemble
    fixed-shape banks downstream."""
    segs = extract_noise_segments(
        synthetic_record,
        strategy=AbsoluteQuietThreshold(1.0),
        channels=("CS12", "CS34"),
        window_ms=200.0,
        hop_ms=200.0,
    )
    expected_len = round(200.0 * 1e-3 * synthetic_record.fs)
    assert all(s.signal.size == expected_len for s in segs)
    assert all(s.end_sample - s.start_sample == expected_len for s in segs)


def test_factory_can_customize_channels(
    synthetic_record_factory: Callable[..., SyntheticRecord],
) -> None:
    """When a record has different bipolar channel names, the
    extractor honors the caller-supplied `channels` list."""
    rec = synthetic_record_factory(bipolar_channels=("FOO", "BAR"))
    segs = extract_healthy_segments(
        rec,
        threshold=NoThreshold(),
        channels=("FOO", "BAR"),
        window_ms=200.0,
        hop_ms=200.0,
    )
    assert {s.channel for s in segs} == {"FOO", "BAR"}
