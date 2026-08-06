"""Base classes for activation-based extraction.

Vocabulary — three stages, three names
--------------------------------------
Activation detection is a *chain*, and conflating its stages makes the
code read as though one of them does the whole job:

1. **Detection preprocessing** (this module + ``preprocessors.py``) —
   transform the raw channel $x$ into a curve $g$ that *emphasises*
   activations. A preprocessor detects nothing on its own; it decides
   what "prominent" will mean to the next stage.
2. **Detection thresholding** (``thresholds.detection``) — the decision
   rule that turns $g$ into candidate activations.
3. **The detection function** (``detection.py``) — the whole chain,
   preprocessing plus thresholding plus refractory suppression, which
   is the thing that actually answers "where are the activations?"

Only stage 3 detects. Stages 1 and 2 are its parts.

Why an ABC rather than a Protocol
---------------------------------
The rest of this package uses Protocols (``ThresholdStrategy``,
``CalibrationStrategy``, ``Record``) because those describe shapes that
*other people's* types satisfy structurally — a producer's record type
must not have to import us to qualify.

Detection preprocessing is the opposite situation: a **closed family
with a known, fixed structure** that we ship and expect to be extended
here. An abstract base earns its keep by *enforcing* that structure
rather than documenting it:

- the shared input contract is validated in one place, in a template
  method, so a new subclass cannot forget to check it;
- the fail-closed short-input rule is applied uniformly, expressed as
  one number per subclass (``min_samples``) instead of hand-rolled in
  each;
- a subclass that forgets to name itself fails at *class-definition*
  time with a clear message, not at some later call site.

That last pair is not hypothetical. Before this base existed, each
preprocessor called the validator by convention and handled short input
itself — and :class:`~.preprocessors.BotteronEnvelope` silently didn't,
raising deep inside ``sosfiltfilt`` where the other two returned zeros.
The template method makes that class of divergence unrepresentable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np


class DetectionPreprocessor(ABC):
    """A transform from one raw channel to a detection curve ``g``.

    Subclasses implement :meth:`_compute` and set :attr:`name` and, if
    the operator needs more than one sample, :attr:`min_samples`. They
    do **not** override :meth:`compute` — it is the template method that
    enforces the shared contract.

    Contract guaranteed to every caller
    -----------------------------------
    - **Input is one channel.** 1-D only. Detection is inherently
      per-channel; ``argmax`` across a 2-D block would be meaningless,
      so a 2-D input raises rather than having an axis guessed for it.
    - **Output length equals input length**, so an index into ``g`` is
      directly an index into the signal — no offset bookkeeping at the
      call site. Positions where an operator is undefined (the first
      sample of a difference, both ends of a three-point operator) are
      filled with ``0.0``, which reads as "no evidence here" and cannot
      manufacture a spurious peak at a boundary.
    - **Too-short input returns all zeros** rather than raising: nothing
      crosses a positive threshold, so a degenerate channel detects
      nothing. This is the fail-closed reading, matching the package's
      empty-pool sentinel convention.
    """

    #: Stable identifier, recorded in provenance. Every subclass sets it.
    name: ClassVar[str]

    #: Fewest input samples the operator is defined for. Below this,
    #: :meth:`compute` short-circuits to zeros instead of letting the
    #: subclass fail in its own way.
    min_samples: ClassVar[int] = 1

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "name", None):
            raise TypeError(
                f"{cls.__name__} must define a class-level `name`; it is recorded "
                "in provenance and identifies the preprocessor in run records."
            )

    def compute(self, signal: np.ndarray) -> np.ndarray:
        """Return the detection curve ``g`` for one channel.

        Template method: validates the shared contract, applies the
        fail-closed short-input rule, then delegates to
        :meth:`_compute`. Subclasses override that, not this.

        Takes **only the signal**. Two of the three operators are pure
        sample-domain arithmetic and have no use for a sampling rate;
        the one that does (:class:`~.preprocessors.BotteronEnvelope`,
        whose filters are specified in Hz) takes ``fs`` at construction
        alongside the band edges it belongs with. Threading an ignored
        ``fs`` through the other two would imply a rate-dependence they
        do not have.
        """
        if signal.ndim != 1:
            raise ValueError(
                f"detection preprocessing takes a single channel (1-D), got "
                f"{signal.ndim}-D. Detection is per-channel; pass one channel at a time."
            )
        if signal.size < self.min_samples:
            return np.zeros(signal.shape, dtype=np.float64)
        return self._compute(signal.astype(np.float64, copy=False))

    @abstractmethod
    def _compute(self, signal: np.ndarray) -> np.ndarray:
        """The operator itself.

        Called only with a validated 1-D float64 array of at least
        :attr:`min_samples` samples, so implementations carry no
        defensive checks. Must return an array of the same length.
        """
