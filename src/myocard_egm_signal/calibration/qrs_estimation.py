"""QRS peak-to-peak measurement on the surface ECG.

Used by :class:`RWaveAnchoring` and importable standalone for
diagnostic plots and the inspect CLI.
"""

from __future__ import annotations

import numpy as np

from ..records import Record

DEFAULT_PREFERRED_LEADS: tuple[str, ...] = (
    "II",
    "I",
    "V1",
    "aVF",
    "aVL",
    "III",
    "aVR",
    "V5",
)
"""Default surface-ECG leads to try, in priority order.

Lead II usually has the largest, most consistently detectable QRS.
Override at strategy construction time if your dataset uses different
or fewer leads.
"""


def estimate_qrs_peak_to_peak(
    record: Record,
    *,
    window_ms: float = 100.0,
    preferred_leads: tuple[str, ...] = DEFAULT_PREFERRED_LEADS,
) -> tuple[str, float, int]:
    """Estimate the median QRS peak-to-peak on the first available lead.

    Returns ``(lead_used, median_peak_to_peak, n_beats)``.

    Reads ``record.qrs_samples`` as the source of beat annotations.
    The :class:`Record` Protocol does not require this attribute; if
    missing or empty, raises ``ValueError``.
    """
    qrs_samples = getattr(record, "qrs_samples", None)
    if qrs_samples is None or len(qrs_samples) == 0:
        raise ValueError(
            "Record carries no QRS annotations; cannot measure QRS "
            "peak-to-peak. Use a different calibration strategy or "
            "annotate the record first."
        )
    lead = None
    for name in preferred_leads:
        if name in record.channel_names:
            lead = name
            break
    if lead is None:
        raise ValueError(
            f"None of the preferred leads {preferred_leads} are present "
            f"in record.channel_names. Override preferred_leads at "
            "strategy construction time to match this dataset."
        )

    col = record.channel_index(lead)
    half = round(window_ms * 1e-3 * record.fs / 2)
    pps: list[float] = []
    n_total = record.signal.shape[0]
    for s in qrs_samples:
        lo = max(0, int(s) - half)
        hi = min(n_total, int(s) + half + 1)
        if hi - lo < 3:
            continue
        window = record.signal[lo:hi, col]
        pps.append(float(np.max(window) - np.min(window)))
    if not pps:
        return lead, 0.0, 0
    return lead, float(np.median(pps)), len(pps)
