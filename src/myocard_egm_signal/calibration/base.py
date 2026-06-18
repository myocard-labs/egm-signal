"""Calibration dataclass + CalibrationStrategy Protocol.

The result type (``Calibration``) and the strategy interface live
here. Concrete strategies (R-wave anchoring; future percentile-based,
fixed-gain, etc.) live in their own sibling modules and are
re-exported from this subpackage's ``__init__.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..records import Record


@dataclass(frozen=True)
class Calibration:
    """Result of computing a per-record calibration.

    Attributes
    ----------
    scalar
        Multiply the recorded signal by this scalar to obtain
        calibrated amplitudes: ``calibrated = scalar * raw``.
    method
        Identifier of the strategy that produced this calibration
        (e.g. ``"r_wave_anchoring"``).
    metadata
        Strategy-specific diagnostic information (lead used, beat
        count, measured raw amplitude, etc.). The dataclass is frozen
        but the dict object itself is mutable — treat as read-only.
    """

    scalar: float
    method: str
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class CalibrationStrategy(Protocol):
    """Plug-in interface for record-level amplitude calibration."""

    name: str

    def compute(self, record: Record) -> Calibration:
        """Compute a per-record calibration."""
        ...
