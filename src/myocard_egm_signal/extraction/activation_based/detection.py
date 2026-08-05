"""The detection function — the whole chain.

This is the stage that actually answers "where are the activations?".
The preprocessor emphasised them, the threshold decided how prominent
counts, the selector listed the possibilities, and refractory
suppression decided which are distinct — only the composition detects.

Two entry points, because the two corpora need different things:

- :func:`detect_activation` — **synthetic**: one activation at a
  location the generator already knows, so ``argmax g`` suffices. No
  threshold, no suppression. Detection here is a cross-check, not a
  measurement.
- :func:`detect_activation_train` — **IAFDB**: a long, noisy,
  multi-activation stream, so the full chain runs per channel.

Every stage is a configured object rather than a bag of loose
parameters, so the chain reads as a composition and each stage's
settings live with the stage that uses them.

Sample-domain throughout
------------------------
Every interval is in **samples**, not milliseconds. This layer holds no
sampling rate — only :class:`~.preprocessors.BotteronEnvelope` does,
because its cutoffs are specified in Hz — so callers convert with
``round(ms * 1e-3 * fs)`` at the boundary where the rate is known.

Math + sources: [`docs/theory.md`](../../../../docs/theory.md) §3.
"""

from __future__ import annotations

import numpy as np

from ...thresholds.base import DetectionThreshold
from .base import DetectionPreprocessor
from .candidates import CandidateSelector
from .suppression import RefractorySuppressor


class TwoStageRefiner:
    """Snap activation times to a sharper curve's local maximum.

    The optional final stage. A smoothed envelope's maximum is pulled
    toward the heavier side of an asymmetric complex, so a long
    fractionated tail — the fibrotic case — drags the detected time
    late. This snaps each accepted activation to the nearest local
    maximum of a *sharp* preprocessor's curve, recovering the instant
    the smoothing blurred.

    A plain class rather than an interface: the method spec describes
    one refinement rule, not a family. If a second ever appears this
    grows an ABC like its neighbours.

    Parameters
    ----------
    preprocessor
        The sharp curve to refine against —
        :class:`~.preprocessors.RectifiedDerivative` is the natural
        choice, being the ``dV/dt``-max convention itself.
    radius_samples
        Search radius around each accepted time. Must be positive, and
        should stay **well under the refractory interval**: a radius
        able to reach a neighbouring activation would let refinement
        move a peak onto the wrong complex, and suppression has already
        run, so nothing downstream would catch it.
    """

    def __init__(self, preprocessor: DetectionPreprocessor, radius_samples: int) -> None:
        if radius_samples <= 0:
            raise ValueError(f"radius_samples must be positive, got {radius_samples}.")
        self.preprocessor = preprocessor
        self.radius_samples = int(radius_samples)

    def refine(self, signal: np.ndarray, times: np.ndarray) -> np.ndarray:
        """Return the refined activation times.

        Ordering is preserved but **not** re-enforced: if two refined
        times were to cross, that would mean the radius is too large
        relative to the refractory interval, and silently re-sorting
        would hide the misconfiguration rather than surface it.
        """
        if times.size == 0:
            return times
        curve = self.preprocessor.compute(signal)
        n = curve.size
        refined = np.empty_like(times)
        for k, t in enumerate(times):
            lo = max(0, int(t) - self.radius_samples)
            hi = min(n, int(t) + self.radius_samples + 1)
            refined[k] = lo + int(np.argmax(curve[lo:hi]))
        return refined

    def __repr__(self) -> str:
        return (
            f"TwoStageRefiner(preprocessor={self.preprocessor!r}, "
            f"radius_samples={self.radius_samples})"
        )


def detect_activation(signal: np.ndarray, *, preprocessor: DetectionPreprocessor) -> int:
    """Locate the single activation in a trace — the synthetic case.

    ``t_a = argmax g``. There is exactly one activation by construction,
    so no threshold or refractory logic applies; any preprocessor works,
    and :class:`~.preprocessors.RectifiedDerivative` gives the crispest
    timing.

    Raises
    ------
    ValueError
        If the detection curve is constant — a flat channel has no
        activation, and returning sample 0 would be a silent lie about
        a trace that contains nothing.
    """
    curve = preprocessor.compute(signal)
    if curve.size == 0 or float(curve.max()) == float(curve.min()):
        raise ValueError(
            "no activation: the detection curve is constant, so the trace "
            "contains nothing to anchor on."
        )
    return int(np.argmax(curve))


def detect_activation_train(
    signal: np.ndarray,
    *,
    preprocessor: DetectionPreprocessor,
    threshold: DetectionThreshold,
    selector: CandidateSelector,
    suppressor: RefractorySuppressor | None,
    refiner: TwoStageRefiner | None = None,
) -> np.ndarray:
    """Detect the ordered activation train in one channel.

    Runs the chain: preprocess -> threshold -> select candidates ->
    suppress -> (optionally) refine.

    Parameters
    ----------
    signal
        One channel, 1-D.
    preprocessor
        Produces the detection curve. The method spec's default for the
        IAFDB train is :class:`~.preprocessors.BotteronEnvelope`, whose
        smoothing merges fractionation before suppression ever sees it.
    threshold
        Computes ``tau`` from the curve.
    selector
        Turns the curve and ``tau`` into candidates —
        :class:`~.candidates.LocalMaximaSelector` is the spec's rule.
    suppressor
        Which candidates are *distinct* activations.

        **Required, and ``None`` means genuinely skip this step** — every
        candidate becomes an activation. That is occasionally what you
        want (inspecting raw candidates, or a curve you know yields one
        peak per activation), but on a fractionated complex it
        over-counts badly. There is no default, because silently
        supplying a suppressor the caller did not ask for would hide a
        step that changes the answer.
    refiner
        Optional :class:`TwoStageRefiner`. Defaults to ``None``, and
        here that default is the *specified* behaviour rather than a
        silent choice — the method spec has refinement off in the first
        pass.

    Returns
    -------
    np.ndarray
        Ordered ``int64`` sample indices. Empty if nothing is detected.
    """
    curve = preprocessor.compute(signal)
    tau = threshold.compute_threshold(curve)
    candidates = selector.select(curve, tau)
    accepted = suppressor.suppress(candidates) if suppressor is not None else candidates
    times = np.array(
        sorted(c.peak_sample for c in accepted),
        dtype=np.int64,
    )
    if refiner is not None:
        times = refiner.refine(signal, times)
    return times
