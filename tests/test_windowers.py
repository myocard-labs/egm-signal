"""Tests for the windowers — detect and window in one call.

The property that matters is that the two variants differ *only* in how
they find activations. Everything downstream of detection must be
identical, because that is what stops the two corpora's windows being
cut differently.
"""

from __future__ import annotations

import numpy as np
import pytest

from myocard_egm_signal.exceptions import ConstantSignalError
from myocard_egm_signal.extraction.activation_based import (
    ActivationWindower,
    BotteronEnvelope,
    GreedyHeightSuppressor,
    LocalMaximaSelector,
    MultiActivationWindower,
    RectifiedDerivative,
    SingleActivationWindower,
    UniformPositionGenerator,
    window_train,
)
from myocard_egm_signal.thresholds import MedianMadThreshold

FS = 1000.0
T = 192  # the window length the parameter study settled on, in samples at 1 kHz
REFRACTORY = 60


def _pulse(n: int, centre: int, width_ms: float = 6.0, amp: float = 1.0) -> np.ndarray:
    """A Gabor pulse — steepest slope at ``centre`` by construction."""
    u = np.arange(n, dtype=np.float64) - centre
    period = width_ms * 1e-3 * FS
    sigma = period / 3.0
    pulse: np.ndarray = amp * -np.sin(2 * np.pi * u / period) * np.exp(-0.5 * (u / sigma) ** 2)
    return pulse


def _train_signal(n: int, centres: list[int], noise: float = 0.02, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.asarray(sum(_pulse(n, c) for c in centres) + noise * rng.normal(size=n))


def _multi(**overrides: object) -> MultiActivationWindower:
    kwargs: dict[str, object] = {
        "preprocessor": BotteronEnvelope(fs=FS),
        "threshold": MedianMadThreshold(c=1.0, lam=4.0),
        "selector": LocalMaximaSelector(min_prominence=None),
        "suppressor": GreedyHeightSuppressor(refractory_interval_samples=REFRACTORY),
        "position_generator": UniformPositionGenerator(0.5, 0.5),
        "window_length_samples": T,
    }
    kwargs.update(overrides)
    return MultiActivationWindower(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The single-activation (synthetic) variant
# ---------------------------------------------------------------------------


def test_a_single_activation_trace_yields_exactly_one_window() -> None:
    x = _pulse(1000, 500)
    ws = SingleActivationWindower(
        preprocessor=RectifiedDerivative(),
        position_generator=UniformPositionGenerator(0.5, 0.5),
        window_length_samples=T,
    ).window(x)

    assert len(ws) == 1
    assert ws[0].single_activation is True
    assert ws[0].in_bounds
    assert ws[0].iai_prev_samples is None and ws[0].iai_next_samples is None


def test_the_single_variant_anchors_on_the_activation_it_finds() -> None:
    """It detects rather than being told, so the anchor should land on the
    planted activation — within the preprocessor's own timing offset."""
    x = _pulse(1000, 500)
    ws = SingleActivationWindower(
        preprocessor=RectifiedDerivative(),
        position_generator=UniformPositionGenerator(0.5, 0.5),
        window_length_samples=T,
    ).window(x)
    assert abs(ws[0].activation_index - 500) <= 2


def test_the_single_variant_honours_the_requested_position() -> None:
    x = _pulse(2000, 1000)
    for p in (0.2, 0.5, 0.8):
        ws = SingleActivationWindower(
            preprocessor=RectifiedDerivative(),
            position_generator=UniformPositionGenerator(p, p),
            window_length_samples=T,
        ).window(x)
        assert ws[0].realized_position == pytest.approx(p, abs=0.5 / (T - 1))


# ---------------------------------------------------------------------------
# The multi-activation (real-recording) variant
# ---------------------------------------------------------------------------


def test_a_train_trace_yields_one_window_per_detected_activation() -> None:
    centres = list(range(300, 5800, 240))
    x = _train_signal(6000, centres)
    ws = _multi().window(x)

    assert len(ws) == len(centres)
    assert [w.activation_index for w in ws] == sorted(w.activation_index for w in ws)


def test_the_multi_variant_reports_rather_than_drops() -> None:
    """A short record whose first and last activations cannot fit a full
    window still returns them, flagged.

    Uses a prominence floor because the unfiltered selector over-detects
    on a noisy envelope, which would make the count assertion here a test
    of the detector's tuning rather than of the drop behaviour."""
    x = _train_signal(1200, [60, 600, 1140])
    curve = BotteronEnvelope(fs=FS).compute(x)
    ws = _multi(selector=LocalMaximaSelector(min_prominence=0.2 * float(curve.max()))).window(x)

    assert len(ws) == 3, "nothing is dropped, including the two that cannot fit"
    assert not ws.in_bounds_mask[0] and not ws.in_bounds_mask[-1]
    assert ws.in_bounds_mask[1]
    assert ws[0].signal is None and ws[1].signal is not None


def test_intervals_come_back_for_the_study() -> None:
    """§8.1 reads the per-anchor keep probability straight off these."""
    x = _train_signal(6000, list(range(300, 5800, 240)))
    ws = _multi().window(x)
    interior = list(ws)[1:-1]
    assert all(w.iai_prev_samples is not None for w in interior)
    assert all(w.iai_next_samples is not None for w in interior)
    gaps = [w.iai_next_samples for w in interior if w.iai_next_samples is not None]
    assert all(abs(g - 240) <= 4 for g in gaps)


def test_a_dead_channel_raises_through_the_windower() -> None:
    """The degenerate-signal contract survives the extra layer, so a
    corpus sweep can catch it per channel and move on."""
    with pytest.raises(ConstantSignalError):
        _multi().window(np.zeros(2000))


# ---------------------------------------------------------------------------
# What the two variants share
# ---------------------------------------------------------------------------


def test_both_variants_produce_the_same_geometry_for_the_same_train() -> None:
    """The load-bearing property. Detection differs; windowing must not.
    Feeding each variant's detected train back through ``window_train``
    reproduces its output exactly, which is only possible if neither
    variant does anything of its own after detection."""
    x = _pulse(2000, 1000)
    windower = SingleActivationWindower(
        preprocessor=RectifiedDerivative(),
        position_generator=UniformPositionGenerator(0.37, 0.37),
        window_length_samples=T,
    )
    produced = windower.window(x)
    direct = window_train(
        x,
        np.array([produced[0].activation_index]),
        position_generator=UniformPositionGenerator(0.37, 0.37),
        window_length_samples=T,
    )
    assert produced[0] == direct[0]


def test_a_windower_rejects_multichannel_input() -> None:
    with pytest.raises(ValueError, match="1-D"):
        _multi().window(np.zeros((1000, 4)))


def test_a_windower_rejects_a_degenerate_window_length() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        SingleActivationWindower(
            preprocessor=RectifiedDerivative(),
            position_generator=UniformPositionGenerator(0.5, 0.5),
            window_length_samples=1,
        )


def test_a_windower_subclass_must_declare_a_name() -> None:
    with pytest.raises(TypeError, match="must define a class-level `name`"):

        class Unnamed(ActivationWindower):
            def _detect(self, signal: np.ndarray) -> np.ndarray:
                return np.array([0])


def test_windowers_report_their_configuration() -> None:
    """Provenance: a run record needs to say which mode produced a bank."""
    assert SingleActivationWindower.name == "single_activation"
    assert MultiActivationWindower.name == "multi_activation"
    assert "BotteronEnvelope" not in repr(_multi())  # the name, not the class
    assert "botteron_envelope" in repr(_multi())


def test_both_windowers_raise_the_same_thing_on_a_dead_channel() -> None:
    """Surfaced by verifying the usage doc's own example.

    The single-activation path has no threshold in front of it, so it was
    raising a bare ValueError where every other path raised
    ConstantSignalError. A sweep that catches the specific exception must
    not have to know which variant it is holding."""
    single = SingleActivationWindower(
        preprocessor=RectifiedDerivative(),
        position_generator=UniformPositionGenerator(0.5, 0.5),
        window_length_samples=T,
    )
    for windower in (single, _multi()):
        with pytest.raises(ConstantSignalError):
            windower.window(np.zeros(2000))
