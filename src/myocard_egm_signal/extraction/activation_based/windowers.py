"""Windowers — detect and window one trace in a single call.

A windower is the whole activation-anchored path packaged behind one
method: hand it a trace, get back a :class:`~.anchoring.WindowSet`. It
holds the configuration — how to find activations, where to place the
window, how long the window is — so a producer wires it once and then
loops over traces.

Two variants, and the axis is how the train is obtained
-------------------------------------------------------
Only the detection step differs between the corpora; everything after it
is shared, which is why that is the abstract method and nothing else is.

- :class:`SingleActivationWindower` — one activation per trace, located
  as the global maximum of the detection curve. No threshold and no
  refractory suppression are needed or wanted: with exactly one
  activation present there is nothing to separate it from and nothing to
  merge. This is the synthetic case.
- :class:`MultiActivationWindower` — a continuous multi-activation
  stream, needing the full detection chain (preprocess → threshold →
  select → suppress). This is the real-recording case.

The result is the same type either way, and the geometry is computed by
the same :func:`~.anchoring.window_train`, so the two corpora cannot
drift apart in how their windows are cut.

When *not* to use a windower
----------------------------
Call :func:`~.anchoring.window_train` directly whenever the train is
already in hand and re-detecting would be wrong. The motivating case is
a probe that sweeps crop offsets across one simulation: it must detect
**once** on the source and then shift by exact integer offsets, so that
the resulting position axis is exact rather than carrying per-crop
detector jitter.

Nothing here drops anything — see :class:`~.anchoring.WindowSet` for why
that seam sits where it does.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np

from ...thresholds.base import SignalThreshold
from .anchoring import (
    ActivationPositionGenerator,
    WindowSet,
    _validate_window_length,
    window_train,
)
from .base import DetectionPreprocessor
from .candidates import CandidateSelector
from .detection import (
    TwoStageRefiner,
    detect_activation,
    detect_activation_train,
)
from .suppression import RefractorySuppressor


class ActivationWindower(ABC):
    """Detect activations in a trace, then window on them.

    Subclasses implement :meth:`_detect` and set :attr:`name`. The
    windowing half is deliberately *not* overridable: it is the part
    that must be identical across corpora.

    Parameters
    ----------
    position_generator
        Supplies each window's activation position, and owns its random
        stream. Configured once here rather than passed per trace.
    window_length_samples
        ``T``, shared across corpora and set by the parameter study.
    """

    name: ClassVar[str]

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "name", None):
            raise TypeError(
                f"{cls.__name__} must define a class-level `name`; it is recorded "
                "in provenance and identifies the windowing mode in run records."
            )

    def __init__(
        self,
        *,
        position_generator: ActivationPositionGenerator,
        window_length_samples: int,
    ) -> None:
        _validate_window_length(window_length_samples)
        self.position_generator = position_generator
        self.window_length_samples = window_length_samples

    def window(self, signal: np.ndarray) -> WindowSet:
        """Detect, then window. Template method; override :meth:`_detect`.

        Every window is returned, labelled. Filtering is the caller's —
        typically ``ws.select(ws.in_bounds_mask & ws.single_activation_mask)``
        on the real side, and nothing at all on the synthetic side, where
        the trace is sized so its one window always fits.
        """
        if signal.ndim != 1:
            raise ValueError(f"a signal is one channel (1-D), got {signal.ndim}-D.")
        return window_train(
            signal,
            self._detect(signal),
            position_generator=self.position_generator,
            window_length_samples=self.window_length_samples,
        )

    @abstractmethod
    def _detect(self, signal: np.ndarray) -> np.ndarray:
        """Return the ordered activation train for this trace."""


class SingleActivationWindower(ActivationWindower):
    """One activation per trace, at the detection curve's global maximum.

    The synthetic case. A generated trace contains a single activation,
    so detection reduces to ``argmax`` over the detection curve — no
    threshold (there is nothing to separate the activation from) and no
    refractory suppression (there is nothing to merge).

    **It detects rather than being told, and that is deliberate.** The
    stimulus time is a configured value, but the activation time *at a
    given electrode pair* is not: the stored trace is a pseudo-EGM, a
    distance-weighted sum of membrane current over the whole mesh, so
    its timing is set by when the wavefront passes that pair — a
    function of conduction velocity, the realized fibrosis draw, the
    electrode standoff and the pair's position. None of those is a
    number the configuration carries; they are the solver's output.

    A simulator *can* report an exact activation time by tracking when
    membrane potential crosses a threshold at a tissue node. Do not
    substitute it here. That is a **different measurand** from a bipolar
    EGM's steepest deflection — different signal, different physics,
    offset by the pseudo-EGM's spatial weighting — and the real corpus
    has no equivalent, since a recording offers nothing but the
    electrogram. Storing one in place of the other would put two
    different quantities under a single field name and flatter the very
    comparison the field exists to make. Such a tracker is worth having
    as a *cross-check* on this detector's bias and jitter, which is the
    one place ground truth exists; it is not a replacement for detecting.

    Parameters
    ----------
    preprocessor
        The detection-curve transform. Any of the three works here
        because there is only one activation to find; the sharp ones
        give the crispest timing.
    """

    name: ClassVar[str] = "single_activation"

    def __init__(
        self,
        *,
        preprocessor: DetectionPreprocessor,
        position_generator: ActivationPositionGenerator,
        window_length_samples: int,
    ) -> None:
        super().__init__(
            position_generator=position_generator,
            window_length_samples=window_length_samples,
        )
        self.preprocessor = preprocessor

    def _detect(self, signal: np.ndarray) -> np.ndarray:
        return np.array([detect_activation(signal, preprocessor=self.preprocessor)], dtype=np.int64)

    def __repr__(self) -> str:
        return (
            f"SingleActivationWindower(preprocessor={self.preprocessor.name!r}, "
            f"window_length_samples={self.window_length_samples})"
        )


class MultiActivationWindower(ActivationWindower):
    """A continuous stream of activations, via the full detection chain.

    The real-recording case: preprocess into a detection curve, threshold
    it, select candidate local maxima, then suppress within a refractory
    interval to decide which candidates are distinct activations.

    Every stage is injected rather than fixed, because the right choice
    is data-dependent and the escalation path — heavier preprocessing,
    time-varying thresholds, a smarter suppressor — is a known
    contingency if the baseline underperforms on real recordings.

    Parameters
    ----------
    preprocessor, threshold, selector, suppressor, refiner
        The detection chain. ``suppressor`` is required but may be
        ``None`` to skip suppression, which is meaningful only when the
        selector alone is known to yield distinct activations.
    """

    name: ClassVar[str] = "multi_activation"

    def __init__(
        self,
        *,
        preprocessor: DetectionPreprocessor,
        threshold: SignalThreshold,
        selector: CandidateSelector,
        suppressor: RefractorySuppressor | None,
        position_generator: ActivationPositionGenerator,
        window_length_samples: int,
        refiner: TwoStageRefiner | None = None,
    ) -> None:
        super().__init__(
            position_generator=position_generator,
            window_length_samples=window_length_samples,
        )
        self.preprocessor = preprocessor
        self.threshold = threshold
        self.selector = selector
        self.suppressor = suppressor
        self.refiner = refiner

    def _detect(self, signal: np.ndarray) -> np.ndarray:
        return detect_activation_train(
            signal,
            preprocessor=self.preprocessor,
            threshold=self.threshold,
            selector=self.selector,
            suppressor=self.suppressor,
            refiner=self.refiner,
        )

    def __repr__(self) -> str:
        return (
            f"MultiActivationWindower(preprocessor={self.preprocessor.name!r}, "
            f"threshold={self.threshold.name!r}, "
            f"window_length_samples={self.window_length_samples})"
        )
