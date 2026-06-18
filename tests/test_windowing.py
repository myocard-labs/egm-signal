"""Tests for myocard_egm_signal.windowing."""

from __future__ import annotations

import numpy as np
import pytest

from myocard_egm_signal.windowing import sliding_window_peak_to_peak


def test_sliding_window_peak_to_peak_basic() -> None:
    """Constant signal -> p-p of zero per window. Three windows fit at
    hop 5 in a length-10 signal with window length 5."""
    sig = np.ones(10)
    starts, pps = sliding_window_peak_to_peak(sig, window_samples=5, hop_samples=5)
    assert starts.tolist() == [0, 5]
    assert np.all(pps == 0.0)


def test_sliding_window_peak_to_peak_matches_simple_calc() -> None:
    """For a step signal, each window's peak-to-peak should equal
    max - min within that window. The function is just a fast loop
    over windows."""
    sig = np.arange(10, dtype=float)  # 0, 1, ..., 9
    starts, pps = sliding_window_peak_to_peak(sig, window_samples=4, hop_samples=2)
    # Windows: [0,1,2,3], [2,3,4,5], [4,5,6,7], [6,7,8,9] — each has p-p == 3.
    assert starts.tolist() == [0, 2, 4, 6]
    assert np.allclose(pps, 3.0)


def test_sliding_window_peak_to_peak_handles_oversize_window() -> None:
    """Window larger than the signal -> empty output (not an error)."""
    sig = np.ones(5)
    starts, pps = sliding_window_peak_to_peak(sig, window_samples=10, hop_samples=1)
    assert starts.size == 0
    assert pps.size == 0


def test_sliding_window_peak_to_peak_handles_zero_window() -> None:
    """Non-positive window_samples returns empty arrays — a guarded
    contract that callers can rely on rather than receiving inf/NaN."""
    sig = np.ones(5)
    starts, pps = sliding_window_peak_to_peak(sig, window_samples=0, hop_samples=1)
    assert starts.size == 0
    assert pps.size == 0


def test_sliding_window_peak_to_peak_rejects_zero_hop() -> None:
    """hop_samples=0 would loop forever; raise instead."""
    sig = np.ones(5)
    with pytest.raises(ValueError, match="hop_samples must be positive"):
        sliding_window_peak_to_peak(sig, window_samples=3, hop_samples=0)
