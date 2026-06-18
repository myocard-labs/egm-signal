"""Record-Protocol-driven segment extractors.

Public surface:

- :class:`HealthySegment`, :class:`NoiseSegment` — segment dataclasses.
- :func:`extract_healthy_segments`, :func:`extract_noise_segments` —
  the extractors themselves.
- :data:`DEFAULT_BIPOLAR_BAND_HZ` — clinical 30-300 Hz default band.
"""

from __future__ import annotations

from .extractors import (
    DEFAULT_BIPOLAR_BAND_HZ,
    extract_healthy_segments,
    extract_noise_segments,
)
from .segments import HealthySegment, NoiseSegment

__all__ = [
    "DEFAULT_BIPOLAR_BAND_HZ",
    "HealthySegment",
    "NoiseSegment",
    "extract_healthy_segments",
    "extract_noise_segments",
]
