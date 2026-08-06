"""Signal-level filters.

Today: band-pass, low-pass, and decimation. Future additions (notch,
smoothing) drop into their own submodules here and are re-exported below
so consumers keep one import path:
``from myocard_egm_signal.filters import bandpass``.
"""

from __future__ import annotations

from .bandpass import bandpass
from .decimation import (
    DEFAULT_ANTIALIAS_CUTOFF_FRACTION,
    DEFAULT_ANTIALIAS_ORDER,
    decimate,
)
from .lowpass import lowpass

__all__ = [
    "DEFAULT_ANTIALIAS_CUTOFF_FRACTION",
    "DEFAULT_ANTIALIAS_ORDER",
    "bandpass",
    "decimate",
    "lowpass",
]
