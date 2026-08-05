"""Activation-complex bounds — measuring onset, offset and duration.

An activation is not an instant but a *complex*: a rise, a peak, and a
fall, longer when the tissue is fibrotic and the deflection fractionates.
This measures that extent by walking outward from a detected activation
until the detection curve drops below a boundary level ``theta``::

    t_on  = the last sample before t_a where g falls below theta
    t_off = the first sample after  t_a where g falls below theta
    r_rise = t_a - t_on      r_fall = t_off - t_a      W_act = r_rise + r_fall

A study-time instrument, not a runtime filter
---------------------------------------------
This exists so a study can measure the distribution of ``r_rise`` and
``r_fall`` across a corpus — in particular its **long tail**, the
fibrotic complexes — and from that choose the activation-position range
so that windows almost never clip a complex. Once that range is chosen
the criterion stops binding, so the splitter must **not** re-test every
window against it.

That is a deliberate scope limit, not an omission. The method spec notes
that multi-beat work in a later phase will *want* a fraction of
edge-clipped windows so the model learns that case, so a hard clip
filter in the windowing step would be actively counterproductive.

Its relationship to the above-``tau`` segment
---------------------------------------------
Candidate selection already reports the contiguous above-``tau`` run
containing each peak, which is one notion of "how wide is this
complex". This is a **finer, different** notion, and neither bounds the
other: ``theta`` may sit above or below ``tau`` depending on how it is
configured. Measured on a four-activation train with
``tau = median + 4*MAD``:

===========================  =========  ==================================
theta                        vs tau     complex vs above-tau segment
===========================  =========  ==================================
0.50 * peak                  above      inside the segment
0.25 * peak                  above      inside the segment
0.10 * peak                  **below**  **extends beyond** the segment
===========================  =========  ==================================

So the segment is a useful reference, not a bounding box, and this
module measures against the curve rather than being confined to it.

Math + sources: [`docs/theory.md`](../../../../docs/theory.md) §4;
threshold-on-smoothed-rectified onset detection: Hodges & Bui 1996.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...exceptions import EmptySignalError
from ...thresholds.base import PositionAwareSignalThreshold, SignalThresholdLike


@dataclass(frozen=True)
class ActivationComplex:
    """The measured extent of one activation.

    ``onset_clamped`` / ``offset_clamped`` record whether that side
    terminated on a genuine ``theta`` crossing or simply ran out of room
    — hitting the array edge or the search radius. **A clamped side is
    not a measurement**, and pooling it into a duration distribution
    would bias the very tail the study is trying to characterise, so it
    is reported rather than silently indistinguishable. Use
    :attr:`is_complete` to filter.
    """

    activation_sample: int
    onset_sample: int
    offset_sample: int

    theta: float
    """The numeric level the edges were actually read at.

    Not the threshold *rule* that was passed in — the float it produced
    for **this** complex. The two are worth keeping distinct because
    :class:`~....thresholds.PeakFractionThreshold` yields a different
    value per activation (it scales with each peak), so recording only
    the rule would not let you reconstruct where a given edge came from.
    Kept on the result so an odd-looking duration can be traced back to
    the level that produced it.
    """

    onset_clamped: bool
    offset_clamped: bool

    @property
    def rise_samples(self) -> int:
        """``t_a - t_on`` — the rising portion, before the peak."""
        return self.activation_sample - self.onset_sample

    @property
    def fall_samples(self) -> int:
        """``t_off - t_a`` — the falling portion, after the peak."""
        return self.offset_sample - self.activation_sample

    @property
    def width_samples(self) -> int:
        """``W_act`` — the whole complex, rise plus fall."""
        return self.offset_sample - self.onset_sample

    @property
    def is_complete(self) -> bool:
        """True when *both* boundaries are genuine ``theta`` crossings."""
        return not (self.onset_clamped or self.offset_clamped)


def _resolve_search_radius(
    search_radius_samples: int | tuple[int, int] | None,
) -> tuple[int | None, int | None]:
    """Normalise the radius argument to ``(before, after)`` sample counts.

    A bare ``int`` is symmetric; a 2-tuple is ``(before, after)``.
    ``None`` on either side means "walk to the array edge".
    """
    if search_radius_samples is None:
        return None, None

    if isinstance(search_radius_samples, tuple):
        if len(search_radius_samples) != 2:
            raise ValueError(
                "search_radius_samples as a tuple must be (before, after), got "
                f"{len(search_radius_samples)} values."
            )
        before, after = search_radius_samples
    else:
        before = after = search_radius_samples

    for label, value in (("before", before), ("after", after)):
        if value <= 0:
            raise ValueError(
                f"search_radius_samples must be positive, got {value} for the {label} side."
            )
    return int(before), int(after)


def measure_complex(
    detection_curve: np.ndarray,
    activation_sample: int,
    *,
    boundary_threshold: SignalThresholdLike,
    search_radius_samples: int | tuple[int, int] | None = None,
) -> ActivationComplex:
    """Measure the complex around one detected activation.

    Walks outward from ``activation_sample`` until the curve drops below
    ``theta``, in both directions.

    Parameters
    ----------
    detection_curve
        The same curve the activation was detected on. A **smoothed**
        one (the Botteron envelope) is what makes this measurable at
        all: on a raw or sharp curve a fractionated complex dips below
        ``theta`` between its deflections, so the walk stops at the
        first dip and measures a fragment instead of the complex.
    activation_sample
        The detected activation, from the detection function.
    boundary_threshold
        How ``theta`` is chosen. Either family works and they answer
        different questions.
        :class:`~....thresholds.PeakFractionThreshold` (position-aware)
        scales with *this* complex's own peak, which suits comparing
        complex shapes across a corpus of varying amplitude.
        :class:`~....thresholds.MedianMadThreshold` with ``c = 1``
        (global) sits a fixed distance above the noise floor, which is
        the more meaningful question when complexes are weak — it cannot
        fall beneath the floor the way a small fraction of a small peak
        can, so the outward walk is guaranteed to terminate.
    search_radius_samples
        Optional cap on how far to walk. Without it, a low ``theta`` can
        walk into a neighbouring activation and report a "complex"
        spanning several beats. A sensible cap is somewhat under the
        refractory interval. A side stopped by the cap is flagged
        ``clamped``.

        Either an ``int`` (the same cap both ways) or a 2-tuple
        ``(before, after)``. **The asymmetric form is the one that fits
        the phenomenon**: a fibrotic complex is asymmetric by
        construction — the fractionated tail makes ``r_fall`` the long
        side, which is the whole reason §4 measures the two separately.
        A single symmetric cap therefore has to be set wide enough for
        the tail, which then lets the *onset* walk run further back
        toward the previous activation than it ever needs to. Splitting
        the cap lets the forward side stay generous without loosening
        the backward side.

    Returns
    -------
    ActivationComplex
        With ``onset_clamped`` / ``offset_clamped`` set when that side
        ran out of room rather than crossing ``theta``.

    Raises
    ------
    EmptySignalError, ConstantSignalError
        From the threshold. A flat curve has no complex to measure, and
        letting it through would report a zero-width one as a genuine
        measurement.
    """
    if detection_curve.ndim != 1:
        raise ValueError(f"a detection curve is one channel (1-D), got {detection_curve.ndim}-D.")
    # Signal before position, matching the threshold families: an absent
    # curve is the more fundamental problem, and reporting it as "sample
    # 0 is out of range" would send the reader looking in the wrong place.
    if detection_curve.size == 0:
        raise EmptySignalError("cannot measure a complex in a zero-length curve.")
    if not 0 <= activation_sample < detection_curve.size:
        raise ValueError(
            f"activation_sample {activation_sample} is outside the curve "
            f"(0..{detection_curve.size - 1})."
        )
    radius_before, radius_after = _resolve_search_radius(search_radius_samples)

    curve = detection_curve.astype(np.float64, copy=False)
    # The two threshold families take different arguments by design — a
    # peak-relative rule cannot be evaluated without knowing which peak,
    # a global one has no use for it. One `isinstance` is the whole cost
    # of accepting both.
    if isinstance(boundary_threshold, PositionAwareSignalThreshold):
        theta = boundary_threshold.compute_threshold(curve, activation_sample)
    else:
        theta = boundary_threshold.compute_threshold(curve)

    # How far the walk may go on each side: the array, narrowed by the
    # search radius if one was given for that side.
    lo_limit = 0 if radius_before is None else max(0, activation_sample - radius_before)
    hi_limit = (
        curve.size - 1
        if radius_after is None
        else min(curve.size - 1, activation_sample + radius_after)
    )

    # Backwards to the first sample below theta. Stopping *at* the limit
    # while still above theta means we ran out of room, not that we
    # found an edge.
    onset = activation_sample
    while onset > lo_limit and curve[onset] >= theta:
        onset -= 1
    onset_clamped = bool(curve[onset] >= theta)

    offset = activation_sample
    while offset < hi_limit and curve[offset] >= theta:
        offset += 1
    offset_clamped = bool(curve[offset] >= theta)

    return ActivationComplex(
        activation_sample=activation_sample,
        onset_sample=onset,
        offset_sample=offset,
        theta=theta,
        onset_clamped=onset_clamped,
        offset_clamped=offset_clamped,
    )


def measure_complexes(
    detection_curve: np.ndarray,
    activation_samples: np.ndarray,
    *,
    boundary_threshold: SignalThresholdLike,
    search_radius_samples: int | tuple[int, int] | None = None,
) -> list[ActivationComplex]:
    """Measure every complex in an activation train.

    Convenience over :func:`measure_complex`; the study use is always a
    whole train, and the per-complex results are what get pooled into
    the ``r_rise`` / ``r_fall`` distributions.
    """
    return [
        measure_complex(
            detection_curve,
            int(t),
            boundary_threshold=boundary_threshold,
            search_radius_samples=search_radius_samples,
        )
        for t in activation_samples
    ]
