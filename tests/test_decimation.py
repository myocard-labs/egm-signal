"""Tests for decimation.

The property that matters is the one that is invisible when it fails:
content above the new Nyquist must be *removed*, not folded down onto
real signal. So most of these assert on the spectrum of the result
rather than on its shape.
"""

from __future__ import annotations

import numpy as np
import pytest

from myocard_egm_signal.filters import decimate
from myocard_egm_signal.filters.decimation import minimum_length_samples

FS = 1000.0


def _tone(freq_hz: float, n: int = 8000, fs: float = FS) -> np.ndarray:
    wave: np.ndarray = np.sin(2 * np.pi * freq_hz * np.arange(n) / fs)
    return wave


def _amplitude_at(signal: np.ndarray, freq_hz: float, fs: float) -> float:
    """Amplitude of ``signal`` at one frequency, via its DFT bin."""
    spectrum = np.abs(np.fft.rfft(signal)) * 2 / signal.size
    freqs = np.fft.rfftfreq(signal.size, d=1 / fs)
    return float(spectrum[int(np.argmin(np.abs(freqs - freq_hz)))])


def _steady_state_peak(signal: np.ndarray, trim: int = 100) -> float:
    """Largest excursion away from the filter's edge transients.

    A zero-phase filter pads and reverses, which leaves a transient at
    each end that has nothing to do with the passband. Measured at
    order 4 on an out-of-band tone, the whole-array maximum is 0.059
    while the steady-state leak is 0.0015 -- so an untrimmed maximum
    would report the padding rather than the filter, and would rate
    every order identically."""
    return float(np.abs(signal[trim:-trim]).max())


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n,factor", [(8000, 4), (8000, 2), (999, 4), (1000, 3), (17, 2)])
def test_output_length_is_ceil_n_over_factor(n: int, factor: int) -> None:
    assert decimate(_tone(50, n=n), factor).size == -(-n // factor)


def test_a_factor_of_one_is_an_exact_no_op() -> None:
    """Not merely 'close': with no aliasing to prevent, filtering would
    only distort a signal that needed nothing done to it."""
    x = _tone(50)
    np.testing.assert_array_equal(decimate(x, 1), x)


def test_channels_are_decimated_independently() -> None:
    a, b = _tone(30), _tone(80)
    stacked = decimate(np.column_stack([a, b]), 4)
    assert stacked.shape == (2000, 2)
    np.testing.assert_allclose(stacked[:, 0], decimate(a, 4), atol=1e-12)
    np.testing.assert_allclose(stacked[:, 1], decimate(b, 4), atol=1e-12)


# ---------------------------------------------------------------------------
# The point of the exercise
# ---------------------------------------------------------------------------


def test_content_below_the_new_nyquist_survives() -> None:
    """A tone well inside the retained band comes through essentially
    untouched — the anti-alias filter must not eat the signal."""
    decimated = decimate(_tone(50), 4)
    assert _amplitude_at(decimated, 50, FS / 4) > 0.99


def test_content_above_the_new_nyquist_is_removed_not_folded_down() -> None:
    """The failure this function exists to prevent.

    At fs=1000 decimated by 4, the new rate is 250 Hz and a 200 Hz tone
    aliases to |200 - 250| = 50 Hz — right inside the atrial band, and
    indistinguishable from real 50 Hz content once it lands there.
    Decimating without the filter is what produces that; the comparison
    is the assertion."""
    x = _tone(200)
    naive = x[::4]  # what "just take every 4th sample" does
    proper = decimate(x, 4)

    assert _amplitude_at(naive, 50, FS / 4) > 0.9, "naive decimation aliases it onto 50 Hz"
    assert _amplitude_at(proper, 50, FS / 4) < 0.01, "the filter removes it instead"


def test_the_stopband_holds_across_frequencies_and_factors() -> None:
    """Sweep rather than a single point, since one lucky frequency proves
    nothing about a filter."""
    for factor in (2, 4, 5):
        new_fs = FS / factor
        for freq in np.arange(new_fs / 2 * 1.6, FS / 2, 37.0):
            decimated = decimate(_tone(float(freq)), factor)
            leak = _steady_state_peak(decimated)
            assert leak < 0.01, (
                f"factor {factor}: a {freq:.0f} Hz tone leaked through at {leak:.4f}"
            )


def test_the_passband_holds_across_frequencies_and_factors() -> None:
    for factor in (2, 4, 5):
        new_nyquist = FS / factor / 2
        for freq in (new_nyquist * 0.1, new_nyquist * 0.3, new_nyquist * 0.5):
            decimated = decimate(_tone(float(freq)), factor)
            assert _amplitude_at(decimated, float(freq), FS / factor) > 0.95


def test_timing_is_preserved_because_the_filter_is_zero_phase() -> None:
    """The reason this uses a zero-phase filter rather than the causal one
    a resampler would normally use. A group delay would shift every
    detected activation time, and that shift would end up in the stored
    activation position."""
    n, centre = 8000, 4000
    u = np.arange(n, dtype=np.float64) - centre
    pulse = np.exp(-0.5 * (u / 40.0) ** 2)  # a broad, symmetric bump

    peak_after = int(np.argmax(decimate(pulse, 4))) * 4
    assert abs(peak_after - centre) <= 4, "the peak must not move by more than one output sample"


def test_a_higher_order_rejects_more() -> None:
    """Pins the measurement behind the order-4 default: order 2 leaks
    enough to matter, order 4 does not."""
    x = _tone(200)
    leak_2 = _steady_state_peak(decimate(x, 4, order=2))
    leak_4 = _steady_state_peak(decimate(x, 4, order=4))
    assert leak_2 > 10 * leak_4, f"order 2 leaked {leak_2:.4f}, order 4 leaked {leak_4:.4f}"
    assert leak_4 < 0.01


# ---------------------------------------------------------------------------
# Rejected input
# ---------------------------------------------------------------------------


def test_rejects_a_bad_factor() -> None:
    x = _tone(50)
    for bad in (0, -2):
        with pytest.raises(ValueError, match="at least 1"):
            decimate(x, bad)
    for bad_type in (2.0, "2", True):
        with pytest.raises(ValueError, match="must be an integer"):
            decimate(x, bad_type)  # type: ignore[arg-type]


def test_rejects_a_bad_cutoff_fraction() -> None:
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match=r"cutoff_fraction must be in \(0, 1\]"):
            decimate(_tone(50), 4, cutoff_fraction=bad)


def test_rejects_a_3d_signal() -> None:
    with pytest.raises(ValueError, match="1-D or 2-D"):
        decimate(np.zeros((10, 3, 2)), 2)


def test_a_signal_too_short_to_filter_says_so() -> None:
    """Rather than surfacing scipy's padlen error, which does not mention
    decimation, the order, or what to do about it."""
    assert minimum_length_samples(2) == 10
    assert minimum_length_samples(4) == 16
    assert minimum_length_samples(8) == 28

    with pytest.raises(ValueError, match="too short to anti-alias filter"):
        decimate(_tone(50, n=15), 2, order=4)
    # One sample longer and it works.
    assert decimate(_tone(50, n=16), 2, order=4).size == 8
    # A shorter filter can handle it.
    assert decimate(_tone(50, n=15), 2, order=2).size == 8


def test_a_factor_of_one_skips_the_length_check_too() -> None:
    """Nothing is filtered, so nothing needs padding -- a 3-sample trace
    passes straight through rather than being rejected."""
    x = _tone(50, n=3)
    np.testing.assert_array_equal(decimate(x, 1), x)
