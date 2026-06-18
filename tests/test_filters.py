"""Tests for myocard_egm_signal.filters."""

from __future__ import annotations

import numpy as np
import pytest

from myocard_egm_signal.filters import bandpass


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2)))


def test_bandpass_passes_in_band(time_axis: tuple[float, np.ndarray]) -> None:
    """A 100 Hz pure tone is inside the 30-300 Hz band; the filter
    should pass it through near-unchanged (modulo edge transients)."""
    fs, t = time_axis
    in_band = np.sin(2 * np.pi * 100 * t)
    y = bandpass(in_band, fs, 30, 300)
    inner = slice(int(0.2 * fs), -int(0.2 * fs))
    assert _rms(y[inner]) > 0.9 * _rms(in_band[inner])


def test_bandpass_attenuates_below_band(time_axis: tuple[float, np.ndarray]) -> None:
    """5 Hz is well below the 30 Hz cutoff; the filter should reduce
    its amplitude by more than 10x."""
    fs, t = time_axis
    low = np.sin(2 * np.pi * 5 * t)
    y = bandpass(low, fs, 30, 300)
    inner = slice(int(0.2 * fs), -int(0.2 * fs))
    assert _rms(y[inner]) < 0.1 * _rms(low[inner])


def test_bandpass_attenuates_above_band(time_axis: tuple[float, np.ndarray]) -> None:
    """450 Hz is well above the 300 Hz cutoff; same 10x attenuation
    expectation as the below-band case."""
    fs, t = time_axis
    high = np.sin(2 * np.pi * 450 * t)
    y = bandpass(high, fs, 30, 300)
    inner = slice(int(0.2 * fs), -int(0.2 * fs))
    assert _rms(y[inner]) < 0.1 * _rms(high[inner])


def test_bandpass_auto_caps_above_nyquist(time_axis: tuple[float, np.ndarray]) -> None:
    """high_hz=9999 is well above Nyquist; the filter should cap at
    0.99*fs/2 and still pass the in-band 100 Hz cleanly. Catches a
    regression where callers pass an unsafe high cutoff."""
    fs, t = time_axis
    sig = np.sin(2 * np.pi * 100 * t)
    y = bandpass(sig, fs, 30, 9999)
    assert y.shape == sig.shape
    inner = slice(int(0.2 * fs), -int(0.2 * fs))
    assert _rms(y[inner]) > 0.8 * _rms(sig[inner])


def test_bandpass_handles_2d_input(time_axis: tuple[float, np.ndarray]) -> None:
    """Filter applies independently along axis 0 for each channel.
    Confirms the 2-D path doesn't accidentally mix channels."""
    fs, t = time_axis
    # Two channels: 100 Hz (in-band) and 5 Hz (low).
    sig = np.column_stack([np.sin(2 * np.pi * 100 * t), np.sin(2 * np.pi * 5 * t)])
    y = bandpass(sig, fs, 30, 300)
    assert y.shape == sig.shape
    inner = slice(int(0.2 * fs), -int(0.2 * fs))
    # In-band channel preserved, low channel attenuated.
    assert _rms(y[inner, 0]) > 0.9 * _rms(sig[inner, 0])
    assert _rms(y[inner, 1]) < 0.1 * _rms(sig[inner, 1])


def test_bandpass_rejects_invalid_shape() -> None:
    """3-D input is not supported and should raise rather than silently
    do the wrong thing on the broadcasted output."""
    with pytest.raises(ValueError, match="1-D or 2-D"):
        bandpass(np.zeros((10, 10, 10)), fs=1000.0, low_hz=30, high_hz=300)


def test_bandpass_rejects_invalid_band() -> None:
    """Inverted or non-positive band edges should raise clearly."""
    sig = np.zeros(100)
    with pytest.raises(ValueError, match="Invalid band"):
        bandpass(sig, fs=1000.0, low_hz=0, high_hz=300)
    with pytest.raises(ValueError, match="Invalid band"):
        bandpass(sig, fs=1000.0, low_hz=400, high_hz=300)
