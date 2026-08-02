"""R-wave-anchored calibration strategy.

Background
----------
For many intracardiac recordings, the recorded "mV" values do not
correspond to true physiological amplitudes. Common reasons:

- ADC gain fields in the header carry a nominal fallback value rather
  than the calibrated patient-specific value (the IAFDB case: 7 of 8
  patients).
- Calibration steps were skipped or the calibration channel is
  unavailable.

The surface ECG and intracardiac channels typically share the same
amplifier/digitizer chain, so a single per-record scalar can recover
physiological scale. :class:`RWaveAnchoring` measures the median
surface-ECG QRS peak-to-peak amplitude on a preferred lead, then
solves ``scalar = target / measured`` against a chosen target value.

Caveat — engineering choice, not literature method
--------------------------------------------------
R-wave anchoring of intracardiac amplitudes is NOT an established
literature method. It is an engineering response to the calibration
gap. Document transparently in downstream writeups.

No default target amplitude
---------------------------
``target_qrs_pp_mv`` is **required**. This library deliberately ships no
default for it: the target is a *policy* choice about what amplitude
scale a corpus is calibrated to, and policy defaults belong in the
executable consumer's configuration (or a JSON Schema), not in a
foundation library. A library-side default previously coexisted with a
different CLI-side default, so the same code calibrated to two
different scales depending on which entry point you came through.
Requiring the argument makes that impossible to reintroduce.
"""

from __future__ import annotations

from ..records import Record
from .base import Calibration
from .qrs_estimation import DEFAULT_PREFERRED_LEADS, estimate_qrs_peak_to_peak


class RWaveAnchoring:
    """Calibration via median surface-ECG QRS peak-to-peak amplitude.

    For each beat annotated in ``record.qrs_samples`` (if exposed by
    the record; this is an optional attribute that the IAFDB record
    carries but the Protocol does not require), measure the
    peak-to-peak amplitude on the first available preferred lead in a
    window centered on the QRS sample. The median across beats becomes
    the measured amplitude; the calibration scalar is
    ``target_qrs_pp_mv / measured``.

    If the record does NOT carry QRS annotations as
    ``record.qrs_samples``, this strategy raises ``ValueError`` — pass
    a different strategy in that case.

    Parameters
    ----------
    target_qrs_pp_mv
        Required. Target peak-to-peak QRS amplitude in mV — the scale
        every record in the corpus is calibrated to. No default: see
        the module docstring for why the library refuses to pick one.
    window_ms
        Half-window in milliseconds either side of each QRS annotation
        sample; the peak-to-peak is computed inside this window.
    preferred_leads
        Surface ECG lead names to try, in priority order. The first
        lead found in ``record.channel_names`` wins.
    """

    name: str = "r_wave_anchoring"

    def __init__(
        self,
        target_qrs_pp_mv: float,
        window_ms: float = 100.0,
        preferred_leads: tuple[str, ...] = DEFAULT_PREFERRED_LEADS,
    ) -> None:
        if target_qrs_pp_mv <= 0:
            raise ValueError("target_qrs_pp_mv must be positive.")
        if window_ms <= 0:
            raise ValueError("window_ms must be positive.")
        if not preferred_leads:
            raise ValueError("preferred_leads must not be empty.")
        self.target_qrs_pp_mv = float(target_qrs_pp_mv)
        self.window_ms = float(window_ms)
        self.preferred_leads = tuple(preferred_leads)

    def compute(self, record: Record) -> Calibration:
        lead, pp, n_beats = estimate_qrs_peak_to_peak(
            record,
            window_ms=self.window_ms,
            preferred_leads=self.preferred_leads,
        )
        if pp <= 0:
            raise ValueError(
                f"Measured QRS peak-to-peak on lead {lead!r} is {pp}; cannot anchor calibration."
            )
        scalar = self.target_qrs_pp_mv / pp
        return Calibration(
            scalar=scalar,
            method=self.name,
            metadata={
                "lead": lead,
                "target_qrs_pp_mv": self.target_qrs_pp_mv,
                "window_ms": self.window_ms,
                "measured_qrs_pp": pp,
                "n_beats": n_beats,
            },
        )
