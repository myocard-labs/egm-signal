"""Convenience helpers that don't fit elsewhere.

The leading underscore in the filename signals "implementation detail" —
helpers re-exported via the subpackage ``__init__.py`` are part of the
public surface, but consumers should import them from
``myocard_egm_signal.calibration`` (not directly from this file).
"""

from __future__ import annotations

from ..records import Record
from .base import Calibration, CalibrationStrategy
from .r_wave_anchoring import RWaveAnchoring


def compute_calibration(
    record: Record,
    strategy: CalibrationStrategy | None = None,
    *,
    target_qrs_pp_mv: float | None = None,
) -> Calibration:
    """Compute a per-record calibration.

    Convenience: if ``strategy`` is None, construct an
    :class:`RWaveAnchoring` with the supplied ``target_qrs_pp_mv``.
    Pass either a ready-made strategy OR a ``target_qrs_pp_mv``
    keyword — exactly one, never both and never neither.

    There is no fallback when both are omitted: the target amplitude is
    a policy value this library refuses to choose (see
    :mod:`.r_wave_anchoring`), so omitting it is an error rather than a
    silent default.

    Raises
    ------
    TypeError
        If both ``strategy`` and ``target_qrs_pp_mv`` are given, or if
        neither is.
    """
    if strategy is not None and target_qrs_pp_mv is not None:
        raise TypeError("Pass strategy OR keyword arguments, not both.")
    if strategy is None:
        if target_qrs_pp_mv is None:
            raise TypeError(
                "Pass either a strategy or target_qrs_pp_mv. There is no default "
                "target amplitude: it is a policy value the caller owns."
            )
        strategy = RWaveAnchoring(target_qrs_pp_mv=target_qrs_pp_mv)
    return strategy.compute(record)
