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

### Low-pass a signal

`lowpass` is the same filter family with a single corner, and the same auto-capping and zero-phase guarantees:

```python
from myocard_egm_signal import lowpass

smoothed = lowpass(my_signal, fs=1000.0, cutoff_hz=20)
```

Its main use here is the rectify-then-smooth step of an activation envelope: low-passing a *rectified* signal fills the dips between the deflections of a fractionated complex, so the complex reads as one event rather than several. Because the filter is zero-phase, the smoothed envelope's threshold crossings stay aligned with the features that produced them.

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

### Cut a record into activation-anchored windows

The segment extractors above cut a record at a fixed stride. This cuts it **relative to the activations it contains**: find where the tissue activates, then place a fixed-length window at a chosen position relative to each activation. That position is varied deliberately, so a downstream model cannot learn "the interesting part is always in the middle" instead of learning morphology.

The whole flow is one object. Configure it once, then loop over traces:

```python
import numpy as np
from myocard_egm_signal import (
    BotteronEnvelope,
    GreedyHeightSuppressor,
    LocalMaximaSelector,
    MedianMadThreshold,
    MultiActivationWindower,
    UniformPositionGenerator,
)

fs = 1000.0

windower = MultiActivationWindower(
    # How to turn the raw signal into a curve that peaks at activations.
    preprocessor=BotteronEnvelope(fs=fs),
    # How prominent a peak must be to count. Read off the curve itself,
    # so one configuration works across records of differing amplitude.
    threshold=MedianMadThreshold(c=1.0, lam=4.0),
    # Which local maxima survive. The prominence floor drops noise blips;
    # scale it to your curve (20% of its maximum is a reasonable start).
    selector=LocalMaximaSelector(min_prominence=None),
    # Minimum spacing between distinct activations, in samples.
    suppressor=GreedyHeightSuppressor(refractory_interval_samples=60),
    # Where in the window each activation lands, as a [0, 1] fraction.
    position_generator=UniformPositionGenerator(0.35, 0.65, seed=0),
    # Window length in samples. One value, shared with whatever consumes
    # these windows.
    window_length_samples=192,
)

windows = windower.window(my_channel)      # my_channel is 1-D
```

For a trace holding a **single** activation — a simulated beat, say — swap in `SingleActivationWindower`, which locates the activation as the maximum of the detection curve and needs no threshold or refractory interval:

```python
from myocard_egm_signal import RectifiedDerivative, SingleActivationWindower

windower = SingleActivationWindower(
    preprocessor=RectifiedDerivative(),
    position_generator=UniformPositionGenerator(0.35, 0.65, seed=0),
    window_length_samples=192,
)
```

Both return the same type, and both compute the window geometry with the same code, so windows cut from different sources are directly comparable.

#### Nothing is dropped — you decide

`window(...)` returns a `WindowSet` holding **one record per activation**, each labelled. It never discards anything, because what counts as unusable differs by dataset, and because the *rate* at which windows fail is usually a number you want to measure rather than lose:

```python
for w in windows:
    w.realized_position     # [0, 1] — where the activation actually landed
    w.in_bounds             # False if the window ran off the end of the signal
    w.single_activation     # False if a neighbouring activation is inside it
    w.iai_prev_samples      # interval to the previous activation, or None
    w.iai_next_samples      # interval to the next, or None
    w.signal                # the cropped window — None when not in_bounds

# Filter explicitly. This is the usual real-recording rule:
usable = windows.select(windows.in_bounds_mask & windows.single_activation_mask)
print(f"kept {len(usable)} of {len(windows)}")

# The masks are plain boolean arrays, so any other policy is one line.
long_gap = np.array([(w.iai_prev_samples or 0) > 200 for w in windows])
sparse = windows.select(windows.in_bounds_mask & long_gap)
```

`single_activation` counts members of the **detected** train. A fractionated complex that the detector splits into several peaks will make a genuinely single-beat window read `False` — that is a statement about detector tuning, not about the tissue.

#### Pooling across channels and records

Build a corpus by pooling. `concat` takes any iterable, so filtering and pooling compose:

```python
from myocard_egm_signal import WindowSet

per_channel = [windower.window(ch) for ch in my_channels]
corpus = WindowSet.concat(
    ws.select(ws.in_bounds_mask & ws.single_activation_mask) for ws in per_channel
)
```

It refuses to mix window lengths, since a pooled set of ragged windows cannot be stacked into an array and the failure would otherwise surface far from its cause.

**One caveat.** `activation_index`, `start_index` and `end_index` are positions *within their own source signal*. Once windows from several channels share a set, two of them can carry identical indices while pointing at unrelated places. Positions, flags, intervals and crops all survive pooling; the indices only identify a window inside the set it came from. If you need to know which channel a window came from, keep your own `(source, WindowSet)` pairs and pool last — this library deliberately carries no source identity, since that scheme belongs to your pipeline.

#### Choosing where the activation lands

`UniformPositionGenerator(low, high)` draws a fraction per window. Collapse it to a point for a fixed position — that is the A/B baseline against varied positions, and it is a value change rather than a different code path:

```python
varied = UniformPositionGenerator(0.35, 0.65, seed=0)   # activation moves around
fixed = UniformPositionGenerator(0.5, 0.5)              # always centred
```

The generator owns its random stream, seeded once at construction, so a corpus build is reproducible from the seed alone. Repeated calls continue the stream rather than repeating it. A collapsed generator never touches the stream at all, so the fixed arm is unaffected by how much was drawn elsewhere.

A range need not be symmetric about 0.5 — a generated trace with no signal before its upstroke wants a back-bounded range, so nothing forces symmetry.

#### When you already have the activations

If the train is in hand — or re-detecting would be wrong, as when sweeping crop offsets across one trace and needing an exact offset axis — call the primitive directly:

```python
from myocard_egm_signal import detect_activation_train, window_train

train = detect_activation_train(
    my_channel,
    preprocessor=BotteronEnvelope(fs=fs),
    threshold=MedianMadThreshold(c=1.0, lam=4.0),
    selector=LocalMaximaSelector(min_prominence=None),
    suppressor=GreedyHeightSuppressor(refractory_interval_samples=60),
)

windows = window_train(
    my_channel,
    train,
    position_generator=UniformPositionGenerator(0.5, 0.5),
    window_length_samples=192,
)
```

#### Measuring how wide an activation is

Separately from windowing, you can measure each activation's extent — the rise before the peak, the fall after it. Fibrotic complexes are longer and markedly asymmetric, and these numbers are what tell you how much margin a window needs:

```python
from myocard_egm_signal import measure_complexes

curve = BotteronEnvelope(fs=fs).compute(my_channel)
complexes = measure_complexes(
    curve,
    train,
    # Where to read the edges: a fraction of each peak's own height.
    boundary_threshold=PeakFractionThreshold(fraction=0.25),
    # Optional cap on the outward walk, (before, after) in samples. The
    # asymmetric form suits the phenomenon: the tail is the long side.
    search_radius_samples=(60, 120),
)

for c in complexes:
    if c.is_complete:            # both edges are real crossings, not run-outs
        c.rise_samples, c.fall_samples, c.width_samples
```

A side that hit the array edge or the search radius is flagged `onset_clamped` / `offset_clamped` rather than reported as a measurement — pooling those would bias exactly the long tail you are trying to characterise.

#### Signals that cannot be processed

A zero-length signal raises `EmptySignalError` (a bug in the caller — nothing legitimately produces one) and a completely flat signal raises `ConstantSignalError` (a dead or disconnected electrode, or a channel clipped to a rail). Both derive from `DegenerateSignalError`, so a sweep can catch the pair or tell them apart:

```python
from myocard_egm_signal import ConstantSignalError

kept, dead = [], 0
for ch in my_channels:
    try:
        kept.append(windower.window(ch))
    except ConstantSignalError:
        dead += 1               # flat channel: count it and move on
```

The math behind all of this — the detection curve, the threshold rules, the windowing arithmetic, and why the position distribution needs watching — is in [`theory.md`](theory.md) §2–§5.

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
| `myocard_egm_signal.filters` | `bandpass` + `lowpass` (zero-phase Butterworth). Subpackage; future: notch, smoothing, decimation. |
| `myocard_egm_signal.windowing` | `sliding_window_peak_to_peak` and future sliding-window primitives. |
| `myocard_egm_signal.thresholds` | Four threshold families: keep-above over pooled amplitudes (`ThresholdStrategy` + Absolute/Percentile/None), keep-below over the same (`NoiseSegmentStrategy` + AbsoluteQuiet/PercentileQuiet), and two over a 1-D signal — `SignalThreshold` (`MedianMadThreshold`, `PercentileSignalThreshold`) reading the whole array, and `PositionAwareSignalThreshold` (`PeakFractionThreshold`) reading it at a given sample. |
| `myocard_egm_signal.exceptions` | `DegenerateSignalError` and its two causes — `EmptySignalError` (a caller bug) and `ConstantSignalError` (a dead or rail-clipped channel). |
| `myocard_egm_signal.calibration` | Signal-amplitude calibration: `Calibration` + `CalibrationStrategy` + `RWaveAnchoring` + `compute_calibration` + `estimate_qrs_peak_to_peak`. Calibrates raw EGM signals against a QRS-derived reference. |
| `myocard_egm_signal.extraction` | Fixed-stride segment extraction: `HealthySegment` + `NoiseSegment` + `extract_healthy_segments` + `extract_noise_segments` + `DEFAULT_BIPOLAR_BAND_HZ`. |
| `myocard_egm_signal.extraction.activation_based` | Cutting a record **relative to its activations** rather than at a fixed stride. Detection curves (`RectifiedDerivative`, `TeagerKaiser`, `BotteronEnvelope`), the detection chain (`detect_activation`, `detect_activation_train`, `LocalMaximaSelector`, `GreedyHeightSuppressor`, `TwoStageRefiner`), complex bounds (`measure_complex`, `measure_complexes`), and the windowing path (`UniformPositionGenerator`, `window_train`, `WindowSet`, `SingleActivationWindower`, `MultiActivationWindower`). |
| `myocard_egm_signal.model` | ML model pre/post-processing math. Today: `fit_temperature` + `apply_temperature` + `DEFAULT_TEMPERATURE_BOUNDS` for probability calibration (temperature scaling). Future: Platt scaling, isotonic regression, multi-class softmax calibration. |

Everything in `__all__` is also re-exported from the top-level `myocard_egm_signal` for convenience — most callers will write `from myocard_egm_signal import bandpass` rather than reach into a submodule.

## Where to read more

- For the design rationale (why these are Protocols, why two parallel threshold hierarchies, why preferred_leads is a kwarg): `project/architecture.md`.
- For what's planned but not in v0.1.0: `project/roadmap.md`.
