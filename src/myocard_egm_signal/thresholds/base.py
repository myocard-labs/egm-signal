"""Threshold-strategy interfaces.

Three families, distinguished by *which array they consume* and *which
direction they keep*:

- :class:`ThresholdStrategy` — pooled peak-to-peak amplitudes; the
  keep-above (healthy-class) extractor.
- :class:`NoiseSegmentStrategy` — pooled peak-to-peak amplitudes; the
  keep-below (noise) extractor.
- :class:`DetectionThreshold` — a **detection curve** ``g`` from a
  detection preprocessor; the level a local maximum must reach to
  become a candidate activation.

Keeping them apart is the point. The first two have identical method
signatures and differ only in direction, so separate names are what stop
a keep-below strategy being passed where keep-above was meant. The third
consumes a completely different array — one channel's detection curve,
not a pooled amplitude distribution — so confusing it with either of the
others is a category error the type checker can now catch.

Interface style differs deliberately:

- **Protocol** where *someone else's* type must satisfy the interface
  structurally without inheriting from us. The two amplitude families
  are documented as user-extensible in ``docs/usage.md`` ("define your
  own threshold strategy"), so a caller's class qualifies by shape alone.
- **ABC** where the family is one we ship and extend in-repo *and* has
  shared behavior worth enforcing rather than restating.
  :class:`DetectionThreshold` is that case: its degenerate-input rules
  are easy to get subtly wrong and must not be re-derived per strategy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ThresholdStrategy(Protocol):
    """Plug-in interface for healthy-segment threshold computation.

    The extractor keeps windows whose peak-to-peak is **at or above**
    :meth:`compute_threshold`'s return value.
    """

    name: str

    def compute_threshold(self, pooled_peak_to_peaks: np.ndarray) -> float:
        """Return the effective threshold for this record."""
        ...


@runtime_checkable
class NoiseSegmentStrategy(Protocol):
    """Plug-in interface for noise-segment threshold computation.

    The extractor keeps windows whose peak-to-peak is **at or below**
    :meth:`compute_threshold`'s return value.
    """

    name: str

    def compute_threshold(self, pooled_peak_to_peaks: np.ndarray) -> float:
        """Return the upper-bound peak-to-peak for a window to count as noise."""
        ...


class DetectionThreshold(ABC):
    """The decision rule over a detection curve ``g``.

    Given the curve produced by a detection preprocessor, return the
    level a candidate must reach (``g[i] >= tau``) to count as a
    candidate activation. This is the *second* stage of the detection
    chain: the preprocessor decided what "prominent" looks like, this
    decides how prominent is prominent enough, and only the two together
    plus refractory suppression form a detection function.

    What the threshold is applied to
    --------------------------------
    **Local maxima, not every sample.** Candidate extraction takes the
    points where ``g[i-1] < g[i] > g[i+1]`` *and* ``g[i] >= tau``, then
    filters them by prominence. The threshold gates *how prominent*; it
    deliberately does **not** decide which candidates are distinct
    activations — that is the refractory interval's job, on the time
    axis. Keeping those two separate is what stops the threshold from
    silently merging two genuine activations that ride a single
    above-``tau`` run, which happens in fast AF when the envelope never
    fully returns to baseline between beats.

    *(Earlier revisions of this class specified a strictly-greater
    comparison. That mattered when the threshold was imagined as applied
    sample-wise: on a clean sparse curve ``tau`` is 0.0 and ``>=`` would
    admit all 1000 samples where ``>`` admitted 8. Under local-maxima
    extraction the two are identical — a flat baseline contains no
    strict local maximum — so the comparison no longer carries that
    weight, and ``>=`` is used to match the method spec and
    ``scipy.signal.find_peaks(height=...)``. Verified: 8 vs 8 on that
    same curve.)*

    Thresholds are computed **from the curve itself** rather than
    supplied as absolute numbers, because the three preprocessors
    produce differently-scaled curves — a derivative is spiky, an
    envelope is broad — and because amplitude varies across records,
    channels and patients. A fixed number would have to be retuned for
    every combination.

    Degenerate inputs, handled once here
    ------------------------------------
    Two cases never reach a subclass, because getting them wrong is
    silent and costly:

    - **Empty curve** -> ``+inf``. Nothing to detect, and no finite
      value reaches it.
    - **Constant curve** (``max == min``, which includes an all-zero
      curve from a dead channel) -> ``+inf``. A constant curve has no
      peaks at all, so there is nothing for a threshold to separate.
      ``+inf`` admits nothing under any comparison.

    Both are the fail-closed reading, matching the empty-pool sentinel
    convention the amplitude strategies use.
    """

    name: ClassVar[str]

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "name", None):
            raise TypeError(
                f"{cls.__name__} must define a class-level `name`; it is recorded "
                "in provenance and identifies the threshold rule in run records."
            )

    def compute_threshold(self, detection_curve: np.ndarray) -> float:
        """Return the level a candidate must reach (``g[i] >= tau``).

        Template method: applies the degenerate-input rules, then
        delegates to :meth:`_compute_threshold`. Subclasses override
        that, not this.
        """
        if detection_curve.ndim != 1:
            raise ValueError(
                f"a detection curve is one channel (1-D), got {detection_curve.ndim}-D."
            )
        if detection_curve.size == 0:
            return float("inf")
        curve = detection_curve.astype(np.float64, copy=False)
        if float(curve.max()) == float(curve.min()):
            return float("inf")
        return self._compute_threshold(curve)

    @abstractmethod
    def _compute_threshold(self, detection_curve: np.ndarray) -> float:
        """The rule itself.

        Called only with a non-empty, non-constant 1-D float64 curve, so
        implementations carry no defensive checks.
        """
