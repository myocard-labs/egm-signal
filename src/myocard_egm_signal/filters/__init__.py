"""Signal-level filters.

Today: band-pass. Future additions (notch, smoothing, decimation) drop
into their own submodules here and are re-exported below so consumers
keep one import path: ``from myocard_egm_signal.filters import bandpass``.
"""

from __future__ import annotations

from .bandpass import bandpass

__all__ = ["bandpass"]
