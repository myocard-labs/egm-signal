"""Shared fixtures — synthetic Record builders for the test suite.

The :class:`SyntheticRecord` dataclass below is a concrete Record
Protocol implementation that the tests pass to the extractors and
calibration code. Tests build records with known peak-to-peak
amplitudes, known QRS positions, and known channel layouts so the
assertions can check specific numeric outcomes rather than only
smoke-test that nothing crashes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pytest


@dataclass(frozen=True)
class SyntheticRecord:
    """A concrete Record that satisfies the egm-signal Record Protocol.

    Tests construct one of these (typically via the
    :func:`build_synthetic_record` factory below) and pass it into the
    extractors / calibration code. Structural typing means we don't
    have to inherit anything.
    """

    name: str
    patient: str
    fs: float
    signal: np.ndarray
    channel_names: tuple[str, ...]
    qrs_samples: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))

    def channel_index(self, name: str) -> int:
        return self.channel_names.index(name)


def _synthetic_qrs_complex(fs: float, peak_to_peak: float, duration_ms: float = 80.0) -> np.ndarray:
    """Crude Q-R-S complex with the requested peak-to-peak amplitude.

    Triangular shape that gives a clear, easily-measured peak-to-peak.
    Not physiologically faithful — just deterministic for tests.
    """
    n = max(3, round(duration_ms * 1e-3 * fs))
    # Asymmetric: small Q dip, large R peak, small S dip.
    q_amp = -0.15 * peak_to_peak
    r_amp = 0.85 * peak_to_peak
    s_amp = -0.15 * peak_to_peak
    out = np.empty(n)
    third = n // 3
    out[:third] = np.linspace(q_amp, r_amp, third)
    out[third : 2 * third] = np.linspace(r_amp, s_amp, third)
    out[2 * third :] = np.linspace(s_amp, 0.0, n - 2 * third)
    return out


def build_synthetic_record(
    *,
    fs: float = 1000.0,
    n_seconds: float = 4.0,
    bipolar_pp_mv: float = 0.4,
    bipolar_channels: tuple[str, ...] = ("CS12", "CS34"),
    surface_leads: tuple[str, ...] = ("II", "V1", "aVF"),
    surface_pp_scaling: dict[str, float] | None = None,
    include_qrs: bool = True,
    qrs_period_ms: float = 800.0,
    name: str = "synthetic_001",
    patient: str = "syn1",
) -> SyntheticRecord:
    """Build a synthetic record with known properties.

    - Surface leads carry a periodic synthetic QRS complex. Lead II is
      scaled to ~1.0 mV peak-to-peak by default (the value tests
      anchor against). Other leads get a scaled-down copy.
    - Bipolar channels carry low-amplitude noise plus occasional
      'activations' (Gaussian bumps with peak-to-peak
      ``bipolar_pp_mv``).
    - ``qrs_samples`` is populated if ``include_qrs`` is True.

    Channel order in :attr:`signal` is: surface_leads then
    bipolar_channels.
    """
    n_samples = round(n_seconds * fs)
    rng = np.random.default_rng(0)
    surface_pp_scaling = surface_pp_scaling or {"II": 1.0, "V1": 0.6, "aVF": 0.4}

    channel_names = tuple(list(surface_leads) + list(bipolar_channels))
    signal = np.zeros((n_samples, len(channel_names)), dtype=np.float64)

    beat_samples: list[int] = []
    if include_qrs:
        period_samples = round(qrs_period_ms * 1e-3 * fs)
        offset = period_samples // 2
        beat_samples = list(range(offset, n_samples - period_samples, period_samples))
        complex_len_ms = 80.0
        for lead_idx, lead in enumerate(surface_leads):
            scaling = surface_pp_scaling.get(lead, 0.3)
            qrs = _synthetic_qrs_complex(fs, scaling, duration_ms=complex_len_ms)
            for b in beat_samples:
                stop = min(n_samples, b + len(qrs))
                width = stop - b
                signal[b:stop, lead_idx] += qrs[:width]

    # Bipolar channels: low background noise + three Gaussian bumps per
    # channel at fixed positions with peak-to-peak ~ bipolar_pp_mv.
    base_noise = 0.01 * bipolar_pp_mv
    for i, _chan in enumerate(bipolar_channels):
        col = len(surface_leads) + i
        signal[:, col] = base_noise * rng.standard_normal(n_samples)
        for offset_ms in (200.0, 1400.0, 2600.0):
            center = round(offset_ms * 1e-3 * fs)
            half = round(20.0 * 1e-3 * fs)
            lo = max(0, center - half)
            hi = min(n_samples, center + half + 1)
            t = np.linspace(-1, 1, hi - lo)
            bump = bipolar_pp_mv * 0.5 * np.exp(-(t**2) / 0.1)
            signal[lo:hi, col] += bump

    qrs_arr = np.asarray(beat_samples, dtype=np.int64)
    return SyntheticRecord(
        name=name,
        patient=patient,
        fs=fs,
        signal=signal,
        channel_names=channel_names,
        qrs_samples=qrs_arr,
    )


@pytest.fixture
def synthetic_record() -> SyntheticRecord:
    """Default record: 4 s @ 1 kHz, II/V1/aVF + CS12/CS34, with QRS."""
    return build_synthetic_record()


@pytest.fixture
def synthetic_record_no_qrs() -> SyntheticRecord:
    """Same shape as :func:`synthetic_record` but with no QRS annotations."""
    return build_synthetic_record(include_qrs=False)


@pytest.fixture
def synthetic_record_factory() -> Callable[..., SyntheticRecord]:
    """Return the builder so tests can construct custom records."""
    return build_synthetic_record


@pytest.fixture
def time_axis() -> tuple[float, np.ndarray]:
    """``(fs, t)`` tuple for a 1-second axis at 1 kHz — handy for filter tests."""
    fs = 1000.0
    t = np.linspace(0, 1, int(fs), endpoint=False)
    return fs, t
