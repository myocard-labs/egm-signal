"""Convenience helpers that don't fit elsewhere.

The leading underscore in the filename signals "implementation detail" —
helpers re-exported via the subpackage ``__init__.py`` are part of the
public surface, but consumers should import them from
``myocard_egm_signal.calibration`` (not directly from this file).
"""

from __future__ import annotations

from ..records import Record
from .base import Calibration, CalibrationStrategy
from .r_wave_anchoring import DEFAULT_TARGET_QRS_PP_MV, RWaveAnchoring


def compute_calibration(
    record: Record,
    strategy: CalibrationStrategy | None = None,
    *,
    target_qrs_pp_mv: float | None = None,
) -> Calibration:
    """Compute a per-record calibration.

    Convenience: if ``strategy`` is None, construct an
    :class:`RWaveAnchoring` with the supplied ``target_qrs_pp_mv``
    (or its default). Pass either a ready-made strategy OR a
    ``target_qrs_pp_mv`` keyword, not both.
    """
    if strategy is not None and target_qrs_pp_mv is not None:
        raise TypeError("Pass strategy OR keyword arguments, not both.")
    if strategy is None:
        target = target_qrs_pp_mv if target_qrs_pp_mv is not None else DEFAULT_TARGET_QRS_PP_MV
        strategy = RWaveAnchoring(target_qrs_pp_mv=target)
    return strategy.compute(record)
