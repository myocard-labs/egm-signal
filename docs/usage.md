# Using myocard-egm-signal

`myocard-egm-signal` is a small library of signal-processing primitives for intracardiac EGM data. Filters, windowing math, threshold strategies, calibration scaffolding, and Record-Protocol-driven segment extractors. Pure numpy + scipy. No other myocard-* repos as dependencies.

Reach for it when you want to:

- Band-pass an EGM signal in a notebook without pulling in producer-side baggage.
- Write a new producer pipeline that needs segment extraction.
- Implement a new threshold strategy or calibration scheme that any producer can pick up.

The package is dataset-agnostic. The extractors take any object satisfying the `Record` Protocol; `iafdb-pipeline`'s `IAFDBRecord` satisfies it structurally today, future producers' record types satisfy it the same way.

## Install

During pre-1.0 iteration:

```bash
pip install "myocard-egm-signal @ git+https://github.com/myocard-labs/egm-signal.git"
```

There are no optional extras — the runtime is numpy + scipy, period.

## Common workflows

### Band-pass an EGM signal

```python
import numpy as np
from myocard_egm_signal import bandpass

# A 1-D or 2-D signal at any sample rate. The high cutoff is auto-capped
# at 0.99 * fs/2 so you never have to hand-check Nyquist.
filtered = bandpass(my_signal, fs=1000.0, low_hz=30, high_hz=300)

# 2-D inputs are filtered independently along axis 0 per channel.
multichannel = bandpass(np.column_stack([sig_a, sig_b]), fs=1000.0, low_hz=30, high_hz=300)
```

### Compute a per-record calibration

R-wave anchoring measures the median QRS peak-to-peak on a chosen surface ECG lead and solves for the scalar that brings it to a chosen target:

```python
from myocard_egm_signal import RWaveAnchoring, compute_calibration

# `target_qrs_pp_mv` is required — there is no default. The target is
# the amplitude scale your whole corpus is calibrated to, so it is a
# policy value your config owns, not one this library picks for you.
cal = compute_calibration(my_record, target_qrs_pp_mv=1.0)

# Or pass a fully-constructed strategy with custom preferred leads:
strat = RWaveAnchoring(
    target_qrs_pp_mv=1.0,
    preferred_leads=("II", "V1", "aVF"),
)
cal = strat.compute(my_record)

print(cal.scalar, cal.metadata["lead"], cal.metadata["n_beats"])
# calibrated_signal = my_record.signal * cal.scalar
```

### Extract high-voltage (healthy) segments

```python
from myocard_egm_signal import (
    extract_healthy_segments,
    AbsoluteThreshold,
    PercentileThreshold,
    NoThreshold,
)

# Absolute mV threshold (Sánchez sinus rhythm convention is 0.5 mV).
segs = extract_healthy_segments(
    my_record,
    threshold=AbsoluteThreshold(0.5),
    channels=("CS12", "CS34", "CS56", "CS78", "CS90"),
    calibration=cal,
    window_ms=512.0,
    hop_ms=256.0,
)

# Per-record percentile (top 30%), scale-invariant — no calibration needed.
segs = extract_healthy_segments(
    my_record,
    threshold=PercentileThreshold(70.0),
    channels=("CS12", "CS34"),
)

# Pass-through (every windowed segment) for unsupervised pretraining banks.
segs = extract_healthy_segments(
    my_record,
    threshold=NoThreshold(),
    channels=("CS12", "CS34"),
)
```

### Extract low-amplitude (noise) segments

```python
from myocard_egm_signal import (
    extract_noise_segments,
    AbsoluteQuietThreshold,
    PercentileQuietThreshold,
)

# Conservative absolute threshold (Sanders 2003 "electrically silent" tier).
segs = extract_noise_segments(
    my_record,
    strategy=AbsoluteQuietThreshold(0.05),
    channels=("CS12", "CS34"),
)

# Per-record percentile (bottom 20%), scale-invariant.
segs = extract_noise_segments(
    my_record,
    strategy=PercentileQuietThreshold(20.0),
    channels=("CS12", "CS34"),
)
```

### Define your own threshold strategy

The healthy- and noise-side strategies are plain Python classes that satisfy a Protocol — no inheritance required:

```python
import numpy as np
from myocard_egm_signal.thresholds import ThresholdStrategy

class RelativeVoltageIndex:
    """Hypothetical RVI-style strategy: cutoff = mean + 2 * std."""

    name: str = "rvi"

    def compute_threshold(self, pooled_peak_to_peaks: np.ndarray) -> float:
        if pooled_peak_to_peaks.size == 0:
            return float("inf")
        return float(np.mean(pooled_peak_to_peaks) + 2 * np.std(pooled_peak_to_peaks))


# Confirm structural compatibility:
assert isinstance(RelativeVoltageIndex(), ThresholdStrategy)

segs = extract_healthy_segments(
    my_record,
    threshold=RelativeVoltageIndex(),
    channels=("CS12", "CS34"),
)
```

The noise-side analog satisfies `NoiseSegmentStrategy` with the same signature — same Protocol shape, distinct name to make the direction visible at call sites.

### Calibrate classifier probabilities (temperature scaling)

Distinct from R-wave anchoring (which calibrates raw EGM amplitudes against a reference). Temperature scaling calibrates the *output probabilities* of a trained classifier — useful after training so that "the model says P = 0.8" really means "this case will land in the positive class ~80% of the time" (Guo et al. 2017, *On Calibration of Modern Neural Networks*).

```python
import numpy as np
from myocard_egm_signal import apply_temperature, fit_temperature

# logits: pre-sigmoid model output for a labeled calibration set.
# labels: ground-truth 0/1 for the same samples.
logits = np.array([2.4, -3.1, 5.0, -0.2, 1.8, -2.7])
labels = np.array([1, 0, 1, 0, 1, 0])

T = fit_temperature(logits, labels)
print(f"fitted T = {T:.3f}")
# T > 1: model is overconfident, calibration softens predictions.
# T < 1: model is underconfident, calibration sharpens predictions.
# T = 1: no change.

# Apply T to any logits — same model, same T — to get calibrated logits.
# Take sigmoid downstream when you actually need probabilities.
calibrated_logits = apply_temperature(logits, T)
calibrated_probs = 1.0 / (1.0 + np.exp(-calibrated_logits))
```

Temperature scaling is monotonic in logits, so AUROC, accuracy at threshold 0.5, and any other rank-based metric are unchanged after calibration. ECE, the reliability diagram, and threshold-tuned decisions (e.g. ablation-flagging at `P > 0.7`) do shift.

This module ships only the math — pure numpy/scipy, no torch, no I/O. The consumer (the eval CLI, the viewer's analysis tab, paper-figure notebooks, the C++ deployment runtime) decides where to fit `T` from, what to do with the fitted value, and when to apply it.

### Define your own Record (write a new producer)

Any object with `name`, `patient`, `fs`, `signal`, `channel_names`, and a `channel_index(name)` method satisfies the Record Protocol. The simplest implementation is a frozen dataclass:

```python
from dataclasses import dataclass, field
import numpy as np

@dataclass(frozen=True)
class MyRecord:
    name: str
    patient: str
    fs: float
    signal: np.ndarray            # (n_samples, n_channels)
    channel_names: tuple[str, ...]
    # Optional: QRS samples for the R-wave-anchoring calibration path.
    qrs_samples: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))

    def channel_index(self, name: str) -> int:
        return self.channel_names.index(name)

# Pass an instance straight to any egm-signal function. Structural typing
# means the Protocol matches without isinstance registration.
```

If your producer doesn't have QRS annotations, use a different calibration strategy (or skip calibration entirely — the noise extractor and the percentile threshold strategies are scale-invariant).

## Module map

| Module | What's in it |
|---|---|
| `myocard_egm_signal.records` | `Record` Protocol — the structural type the extractors consume. |
| `myocard_egm_signal.filters` | `bandpass` (zero-phase Butterworth). Subpackage; future: notch, smoothing. |
| `myocard_egm_signal.windowing` | `sliding_window_peak_to_peak` and future sliding-window primitives. |
| `myocard_egm_signal.thresholds` | Keep-above (`ThresholdStrategy` + Absolute/Percentile/None) and keep-below (`NoiseSegmentStrategy` + AbsoluteQuiet/PercentileQuiet) strategy hierarchies. |
| `myocard_egm_signal.calibration` | Signal-amplitude calibration: `Calibration` + `CalibrationStrategy` + `RWaveAnchoring` + `compute_calibration` + `estimate_qrs_peak_to_peak`. Calibrates raw EGM signals against a QRS-derived reference. |
| `myocard_egm_signal.extraction` | `HealthySegment` + `NoiseSegment` + `extract_healthy_segments` + `extract_noise_segments` + `DEFAULT_BIPOLAR_BAND_HZ`. |
| `myocard_egm_signal.model` | ML model pre/post-processing math. Today: `fit_temperature` + `apply_temperature` + `DEFAULT_TEMPERATURE_BOUNDS` for probability calibration (temperature scaling). Future: Platt scaling, isotonic regression, multi-class softmax calibration. |

Everything in `__all__` is also re-exported from the top-level `myocard_egm_signal` for convenience — most callers will write `from myocard_egm_signal import bandpass` rather than reach into a submodule.

## Where to read more

- For the design rationale (why these are Protocols, why two parallel threshold hierarchies, why preferred_leads is a kwarg): `project/architecture.md`.
- For what's planned but not in v0.1.0: `project/roadmap.md`.
