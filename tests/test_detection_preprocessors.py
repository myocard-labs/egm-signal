"""Tests for the detection preprocessors — the transforms ``x -> g``.

These do not detect anything on their own — they produce the curve the
detection threshold makes decisions on. The load-bearing property across
all three is that ``g`` peaks *at the activation*: every downstream step
(thresholding, complex bounds, anchor windowing) reads an index off
``g`` and treats it as a signal index, so a systematic offset here would
bias every window the splitter emits.
"""

from __future__ import annotations

import numpy as np
import pytest

from myocard_egm_signal.extraction.activation_based import (
    BotteronEnvelope,
    DetectionPreprocessor,
    RectifiedDerivative,
    TeagerKaiser,
)

FS = 1000.0

ALL_PREPROCESSORS: list[DetectionPreprocessor] = [
    RectifiedDerivative(),
    TeagerKaiser(),
    BotteronEnvelope(fs=FS),
]


def _biphasic_activation(n: int, centre: int, fs: float, width_ms: float = 8.0) -> np.ndarray:
    """One biphasic deflection centred at ``centre``.

    A Gabor pulse — a sine cycle under a Gaussian envelope. The sine's
    zero crossing and the envelope's peak coincide at ``centre``, so
    the steepest slope is there by construction and the true activation
    index is known exactly.

    The envelope is not cosmetic. A *truncated* sine (the obvious
    fixture) steps discontinuously to zero at its edges, and that step
    is as steep as the genuine upstroke — so ``argmax`` ties on the
    truncation and the fixture reports a detector bug that isn't one.
    Real deflections are amplitude-tapered; this matches that.
    """
    period = width_ms * 1e-3 * fs
    sigma = period / 3.0
    u = np.arange(n, dtype=np.float64) - centre
    pulse: np.ndarray = -np.sin(2 * np.pi * u / period) * np.exp(-0.5 * (u / sigma) ** 2)
    return pulse


def _fractionated_complex(n: int, centre: int, fs: float, n_deflections: int = 3) -> np.ndarray:
    """A complex of several closely-spaced deflections.

    Stands in for a fibrotic / fractionated activation: one *event*
    made of several sub-deflections a few ms apart.
    """
    x = np.zeros(n, dtype=np.float64)
    for k in range(-(n_deflections // 2), n_deflections // 2 + 1):
        sub = centre + int(k * 0.008 * fs)
        x += _biphasic_activation(n, sub, fs, width_ms=4.0)
    return x


# ---------------------------------------------------------------------------
# The shared contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("g_fn", ALL_PREPROCESSORS, ids=lambda f: f.name)
def test_peaks_at_the_activation(g_fn: DetectionPreprocessor) -> None:
    """Every preprocessor's argmax lands on the known activation.

    Tolerance is 3 ms: the envelope is deliberately smoothed, so it is
    allowed to be a little less sharp than the derivative — but not to
    be systematically displaced."""
    fs = 1000.0
    n, centre = 1000, 500
    x = _biphasic_activation(n, centre, fs)
    g = g_fn.compute(x)
    assert abs(int(np.argmax(g)) - centre) <= int(0.003 * fs)


@pytest.mark.parametrize("g_fn", ALL_PREPROCESSORS, ids=lambda f: f.name)
def test_output_length_matches_input(g_fn: DetectionPreprocessor) -> None:
    """An index into g must be directly an index into the signal —
    no offset bookkeeping at the call site."""
    fs = 1000.0
    x = _biphasic_activation(500, 250, fs)
    assert g_fn.compute(x).shape == x.shape


@pytest.mark.parametrize("g_fn", ALL_PREPROCESSORS, ids=lambda f: f.name)
def test_rejects_multichannel_input(g_fn: DetectionPreprocessor) -> None:
    """2-D input raises rather than guessing an axis — detection is
    per-channel and argmax across a block is meaningless. Enforced once
    in the base's template method, so this holds for every subclass."""
    with pytest.raises(ValueError, match="1-D"):
        g_fn.compute(np.zeros((100, 4)))


@pytest.mark.parametrize("g_fn", ALL_PREPROCESSORS, ids=lambda f: f.name)
def test_flat_signal_produces_no_evidence(g_fn: DetectionPreprocessor) -> None:
    """A flat channel yields an all-zero g, so nothing can cross a
    positive threshold — the fail-closed reading."""
    g = g_fn.compute(np.zeros(500))
    assert np.allclose(g, 0.0)


@pytest.mark.parametrize("g_fn", ALL_PREPROCESSORS, ids=lambda f: f.name)
def test_is_a_detection_preprocessor(g_fn: DetectionPreprocessor) -> None:
    """All three derive from the ABC, so the template method — and with
    it the shared input contract — cannot be bypassed."""
    assert isinstance(g_fn, DetectionPreprocessor)
    assert g_fn.name


def test_subclass_must_declare_a_name() -> None:
    """The ABC enforces the family's structure at class-definition
    time, which is the point of it being a base class rather than a
    Protocol: the failure is immediate and names the problem."""
    with pytest.raises(TypeError, match="must define a class-level `name`"):

        class Unnamed(DetectionPreprocessor):
            def _compute(self, signal: np.ndarray) -> np.ndarray:
                return signal


def test_subclass_must_implement_the_operator() -> None:
    """Instantiating a subclass that never implemented `_compute` fails
    rather than silently inheriting a no-op."""

    class Incomplete(DetectionPreprocessor):
        name = "incomplete"

    with pytest.raises(TypeError, match="abstract"):
        Incomplete()  # type: ignore[abstract]


def test_validation_cannot_be_bypassed_by_a_subclass() -> None:
    """A new preprocessor gets the 1-D contract for free — it is applied
    by the template method before `_compute` is ever reached, so a
    subclass cannot forget it. This is the concrete reason for the ABC."""

    class Careless(DetectionPreprocessor):
        name = "careless"

        def _compute(self, signal: np.ndarray) -> np.ndarray:
            return signal  # no checks of its own whatsoever

    careless = Careless()
    # The 1-D contract is enforced by the template method, not by the
    # subclass, which does no checking whatsoever.
    with pytest.raises(ValueError, match="1-D"):
        careless.compute(np.zeros((10, 3)))
    # ...and so is the fail-closed short-input rule, via the inherited
    # min_samples default.
    assert careless.compute(np.zeros(0)).shape == (0,)


# ---------------------------------------------------------------------------
# Per-implementation behavior
# ---------------------------------------------------------------------------


def test_rectified_derivative_is_the_exact_first_difference() -> None:
    """Pinned against a hand-computed case: this is the dV/dt-max
    convention egm-features also uses, so the two must not drift.

    ``x[0]`` is deliberately **non-zero**. An earlier version of this
    fixture started at 0.0, which meant the expected ``g[0] == 0.0``
    could not distinguish the correct behavior from the bug of setting
    ``g[0] = x[0]`` — both pass on that input. With a non-zero first
    sample the two are separable."""
    x = np.array([5.0, 7.0, 4.0, 4.0, 9.0])
    g = RectifiedDerivative().compute(x)
    assert np.allclose(g, [0.0, 2.0, 3.0, 0.0, 5.0])
    assert g[0] == 0.0 and x[0] != 0.0


def test_rectified_derivative_first_sample_is_zero_not_the_signal_value() -> None:
    """Guards the specific off-by-one this operator invites.

    The difference is undefined at sample 0, and the contract fills it
    with 0.0 so it can never win an ``argmax``. Leaking ``x[0]`` there
    instead would plant a peak at the trace edge — which, on a trace
    cropped so the activation sits near the front, competes with the
    real activation and silently shifts the anchor to sample 0."""
    rng = np.random.default_rng(1)
    for _ in range(20):
        x = rng.normal(loc=50.0, scale=5.0, size=rng.integers(2, 200))
        g = RectifiedDerivative().compute(x)
        assert g[0] == 0.0
        assert g.shape == x.shape


def test_teager_kaiser_is_the_exact_three_point_operator() -> None:
    """g[i] = x[i]^2 - x[i-1]*x[i+1], ends zeroed."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    g = TeagerKaiser().compute(x)
    # interior: 4-3=1, 9-8=1, 16-15=1
    assert np.allclose(g, [0.0, 1.0, 1.0, 1.0, 0.0])


def test_teager_kaiser_is_not_clipped() -> None:
    """The raw operator is returned even where it goes negative —
    clipping would be a policy call this primitive doesn't own."""
    # Alternating sign makes x[i-1]*x[i+1] exceed x[i]^2.
    x = np.array([0.0, 3.0, 1.0, 3.0, 0.0])
    g = TeagerKaiser().compute(x)
    assert g[2] < 0


@pytest.mark.parametrize("g_fn", ALL_PREPROCESSORS, ids=lambda f: f.name)
@pytest.mark.parametrize("length", [0, 1, 2, 3, 10, 15])
def test_short_inputs_return_zeros_not_errors(g_fn: DetectionPreprocessor, length: int) -> None:
    """Degenerate lengths detect nothing rather than raising — for
    *every* preprocessor.

    This is a regression test with history: BotteronEnvelope used to
    raise from inside `sosfiltfilt` here while its two siblings returned
    zeros, so the same input produced a different failure mode depending
    on which preprocessor you picked. Hoisting the rule into the base's
    template method, as one `min_samples` per subclass, fixed it."""
    x = np.zeros(length)
    g = g_fn.compute(x)
    assert g.shape == x.shape
    assert np.allclose(g, 0.0)


def test_min_samples_boundary_is_where_it_claims_to_be() -> None:
    """Pins the documented floor: 15 samples is below BotteronEnvelope's
    filter padding and 16 is not. If a scipy change moves the padlen
    rule, this fails instead of the failure surfacing as a mystery
    exception on a short trace."""
    fs = 1000.0
    botteron = BotteronEnvelope(fs=fs)
    assert botteron.min_samples == 16
    rng = np.random.default_rng(0)
    assert np.allclose(botteron.compute(rng.normal(size=15)), 0.0)
    assert not np.allclose(botteron.compute(rng.normal(size=16)), 0.0)


def test_botteron_bridges_fractionation_where_the_derivative_does_not() -> None:
    """The reason BotteronEnvelope exists.

    A fractionated complex is ONE activation made of several
    deflections. The envelope must read it as a single above-threshold
    run; the raw derivative fragments it into several, which is what
    would otherwise over-count activations and force the refractory NMS
    to clean up after the detector."""
    fs = 1000.0
    n, centre = 1000, 500
    x = _fractionated_complex(n, centre, fs)

    def _runs(g: np.ndarray) -> int:
        theta = 0.25 * float(g.max())
        above = g > theta
        return int(np.sum(np.diff(above.astype(int)) == 1)) + int(above[0])

    envelope_runs = _runs(BotteronEnvelope(fs=fs).compute(x))
    derivative_runs = _runs(RectifiedDerivative().compute(x))

    assert envelope_runs == 1, f"envelope should see one complex, saw {envelope_runs}"
    assert derivative_runs > envelope_runs, (
        f"derivative should fragment where the envelope does not "
        f"(derivative {derivative_runs}, envelope {envelope_runs})"
    )


def test_botteron_lowpass_controls_how_much_is_bridged() -> None:
    """lowpass_hz is a documented tuning knob, not an implementation
    detail: a higher cutoff smooths less and so bridges less."""
    fs = 1000.0
    x = _fractionated_complex(1000, 500, fs)

    def _runs(g: np.ndarray) -> int:
        theta = 0.25 * float(g.max())
        above = g > theta
        return int(np.sum(np.diff(above.astype(int)) == 1)) + int(above[0])

    merged = _runs(BotteronEnvelope(fs=fs, lowpass_hz=20.0).compute(x))
    barely = _runs(BotteronEnvelope(fs=fs, lowpass_hz=200.0).compute(x))
    assert merged <= barely


def test_botteron_rejects_bad_configuration() -> None:
    """Constructor validation mirrors bandpass's contract. `fs` is
    checked here too, now that it is configuration rather than a
    per-call argument — so a bad rate fails at construction, before any
    signal has been processed with it."""
    with pytest.raises(ValueError, match="fs must be positive"):
        BotteronEnvelope(fs=0.0)
    with pytest.raises(ValueError, match="fs must be positive"):
        BotteronEnvelope(fs=-1000.0)
    with pytest.raises(ValueError, match="Invalid band"):
        BotteronEnvelope(fs=FS, band_hz=(250.0, 40.0))
    with pytest.raises(ValueError, match="Invalid band"):
        BotteronEnvelope(fs=FS, band_hz=(0.0, 250.0))
    with pytest.raises(ValueError, match="lowpass_hz must be positive"):
        BotteronEnvelope(fs=FS, lowpass_hz=0.0)


def test_botteron_requires_a_rate() -> None:
    """`fs` has no default: the correct rate is a property of the data,
    and defaulting it would silently design the filters for the wrong
    band rather than fail."""
    with pytest.raises(TypeError):
        BotteronEnvelope()  # type: ignore[call-arg]


def test_rate_free_preprocessors_take_only_a_signal() -> None:
    """The point of moving `fs` onto Botteron's constructor: the two
    sample-domain operators no longer carry an argument they ignore, so
    `compute` cannot be handed a rate that would silently do nothing."""
    x = np.arange(10, dtype=np.float64)
    for g_fn in (RectifiedDerivative(), TeagerKaiser()):
        assert g_fn.compute(x).shape == x.shape
        with pytest.raises(TypeError):
            g_fn.compute(x, 1000.0)  # type: ignore[call-arg]
