"""Concrete detection preprocessors — the transforms ``x -> g``.

Each emphasises activations in a different way; **none of them detects
anything**. They produce the curve that the detection threshold
(``thresholds.detection``) then makes decisions on, and the two together
form the detection function (``detection.py``). See
:mod:`~.base` for the full vocabulary.

Three implementations, in increasing order of noise-robustness and cost:

- :class:`RectifiedDerivative` — ``|x[i] - x[i-1]|``. The default, and
  the direct expression of the ``dV/dt``-max activation convention.
  Cheapest; also the most sensitive to high-frequency noise, since
  differencing amplifies it.
- :class:`TeagerKaiser` — ``x[i]^2 - x[i-1]*x[i+1]``. Tracks
  instantaneous *energy* (amplitude times frequency), so it responds to
  a fast low-amplitude deflection that an amplitude-only measure would
  miss, at three multiplies per sample.
- :class:`BotteronEnvelope` — ``LP(|BP(x)|)``. Band-pass, rectify,
  smooth. The most robust, and the only one that presents a
  fractionated activation as a single event: the low-pass fills the
  dips *between* its deflections.

Which to use
------------
Start with :class:`RectifiedDerivative` — on clean or lightly-noisy
signal it agrees with the others and is free. Reach for
:class:`BotteronEnvelope` when the signal is fractionated or noisy
enough that peak *counting* goes wrong, which is the regime real AF
electrograms live in. :class:`TeagerKaiser` sits between them and is the
usual choice when amplitude alone is misleading.

These differ in what they are *shaped* like — a derivative curve is
spiky, an envelope is broad — so a threshold tuned against one does not
transfer to another. Thresholds are computed from the distribution of
``g`` itself precisely so that they adapt.

Math + primary sources: [`docs/theory.md`](../../../../docs/theory.md) §2.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from ...filters import bandpass, lowpass
from .base import DetectionPreprocessor

DEFAULT_BOTTERON_BAND_HZ: tuple[float, float] = (40.0, 250.0)
"""Band-pass edges for the Botteron envelope (Hz).

40-250 Hz follows Botteron & Smith 1995. Narrower than the 30-300 Hz
clinical extraction band: the point here is to isolate the *activation*
rather than to preserve waveform morphology, so trimming both ends
suppresses baseline wander and high-frequency noise before rectifying.
"""

DEFAULT_BOTTERON_LOWPASS_HZ: float = 20.0
"""Envelope smoothing cutoff for the Botteron envelope (Hz).

This is the knob that decides how wide a gap the envelope bridges, and
so how much fractionation is absorbed into a single complex rather than
read as several activations. Lower merges more aggressively — at the
cost of broadening the complex and pushing measured onset/offset
outward.
"""


class RectifiedDerivative(DetectionPreprocessor):
    """``g[i] = |x[i] - x[i-1]|`` — the ``dV/dt``-max convention.

    Activation is taken to occur at the steepest deflection: as the
    wavefront passes under the electrode pair the recorded field
    reverses fastest at that instant. This is the same convention
    ``egm-features.activation_position`` uses, so an anchor placed here
    and a position measured there agree by construction.

    ``g[0]`` is ``0.0``: a first difference is undefined at the first
    sample, and zero cannot become a spurious ``argmax``.
    """

    name: ClassVar[str] = "rectified_derivative"
    min_samples: ClassVar[int] = 2

    def _compute(self, signal: np.ndarray) -> np.ndarray:
        # g[0] = 0 (the difference is undefined there); g[i] = |x[i] - x[i-1]|
        # for i >= 1. Written as an explicit zero-fill rather than
        # `np.diff(signal, prepend=signal[0])` — that idiom is equivalent
        # and also length-preserving, but it reads as though g[0] were
        # being set to x[0], which is exactly the misreading to avoid in
        # code whose off-by-one behavior shifts every activation index.
        # This form also mirrors TeagerKaiser below.
        g = np.zeros(signal.shape, dtype=np.float64)
        g[1:] = np.abs(np.diff(signal))
        return g

    def __repr__(self) -> str:
        return "RectifiedDerivative()"


class TeagerKaiser(DetectionPreprocessor):
    """``g[i] = x[i]^2 - x[i-1]*x[i+1]`` — the Teager-Kaiser energy operator.

    Approximates instantaneous energy (amplitude^2 times frequency^2),
    so a fast deflection scores highly even when its amplitude is
    modest — useful where a low-voltage but sharp activation would be
    lost under an amplitude-only measure.

    The raw operator is returned **unclipped**. It can dip negative on
    signals that are not locally narrowband, and clipping would be a
    policy decision this primitive has no business making: ``argmax`` is
    unaffected, and the adaptive thresholds consume the distribution of
    ``g`` as it is. Both end samples are ``0.0``, where the three-point
    operator is undefined.
    """

    name: ClassVar[str] = "teager_kaiser"
    min_samples: ClassVar[int] = 3

    def _compute(self, signal: np.ndarray) -> np.ndarray:
        g = np.zeros(signal.shape, dtype=np.float64)
        g[1:-1] = signal[1:-1] ** 2 - signal[:-2] * signal[2:]
        return g

    def __repr__(self) -> str:
        return "TeagerKaiser()"


class BotteronEnvelope(DetectionPreprocessor):
    """``g = LP(|BP(x)|)`` — band-pass, rectify, low-pass.

    The robustness upgrade, and the only one of the three that keeps a
    **fractionated** activation in one piece. A fractionated complex
    dips between its deflections in the raw signal; after rectifying,
    the low-pass fills those dips, so the smoothed envelope makes a
    single up/down-crossing pair around the whole complex. That is what
    stops refractory suppression from having to clean up after an
    over-eager detector, and it is what makes envelope-based
    onset/offset measurable at all.

    The cost is deliberate smoothing bias: a heavier low-pass merges
    more but broadens the complex, pushing onset earlier and offset
    later. ``lowpass_hz`` is therefore a tuning knob, not an
    implementation detail.

    Rate-bound by construction
    --------------------------
    Alone among the preprocessors this one is specified in **Hz**, so it
    needs the sampling rate to turn its cutoffs into filter
    coefficients. ``fs`` is therefore taken here, with the band edges it
    belongs to, rather than at every ``compute()`` call — its two
    siblings are pure sample-domain arithmetic and would otherwise carry
    an argument they ignore.

    The consequence to know: **an instance is valid for one sampling
    rate.** A numpy array carries no rate, so handing a 500 Hz trace to
    an instance built for 1 kHz cannot be detected here — the filters
    would simply be designed for the wrong band. Build one preprocessor
    per rate; if you process mixed-rate sources, build it where you know
    the rate rather than hoisting it to module scope.

    Parameters
    ----------
    fs
        Sampling frequency in Hz of the signals this instance will be
        given. Required.
    band_hz
        Band-pass edges before rectification. Defaults to
        :data:`DEFAULT_BOTTERON_BAND_HZ` (40-250 Hz, Botteron 1995).
    lowpass_hz
        Envelope smoothing cutoff. Defaults to
        :data:`DEFAULT_BOTTERON_LOWPASS_HZ` (20 Hz).
    """

    name: ClassVar[str] = "botteron_envelope"

    #: Both filters are zero-phase, and ``sosfiltfilt`` requires more
    #: samples than its padding: ``padlen = 3 * (2 * len(sos) + 1)``. The
    #: 2nd-order band-pass gives ``len(sos) == 2`` -> padlen 15, the
    #: low-pass ``len(sos) == 1`` -> padlen 9, so the band-pass sets the
    #: floor at 16 samples. Below it this returns zeros like its
    #: siblings, rather than raising from inside scipy.
    min_samples: ClassVar[int] = 16

    def __init__(
        self,
        fs: float,
        band_hz: tuple[float, float] = DEFAULT_BOTTERON_BAND_HZ,
        lowpass_hz: float = DEFAULT_BOTTERON_LOWPASS_HZ,
    ) -> None:
        low_hz, high_hz = band_hz
        if fs <= 0:
            raise ValueError(f"fs must be positive, got {fs}.")
        if low_hz <= 0 or high_hz <= low_hz:
            raise ValueError(f"Invalid band: {band_hz}.")
        if lowpass_hz <= 0:
            raise ValueError(f"lowpass_hz must be positive, got {lowpass_hz}.")
        self.fs = float(fs)
        self.band_hz = (float(low_hz), float(high_hz))
        self.lowpass_hz = float(lowpass_hz)

    def _compute(self, signal: np.ndarray) -> np.ndarray:
        # Both filters are zero-phase, so the envelope's crossings stay
        # aligned with the deflections that produced them — the complex
        # bounds are read straight off this curve.
        band = bandpass(signal, self.fs, self.band_hz[0], self.band_hz[1])
        envelope: np.ndarray = lowpass(np.abs(band), self.fs, self.lowpass_hz)
        return envelope

    def __repr__(self) -> str:
        return (
            f"BotteronEnvelope(fs={self.fs}, band_hz={self.band_hz}, lowpass_hz={self.lowpass_hz})"
        )
