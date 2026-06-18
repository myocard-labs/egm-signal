# egm-signal — roadmap

What's planned for future releases. Internal doc — public users see the
README and `docs/usage.md`.

## v0.1.0 — initial release (shipped)

Scope (recap, see `architecture.md` for the design rationale):

- `Record` Protocol.
- `bandpass` (zero-phase Butterworth, axis-0).
- `sliding_window_peak_to_peak`.
- `ThresholdStrategy` + `NoiseSegmentStrategy` Protocols.
- `AbsoluteThreshold`, `PercentileThreshold`, `NoThreshold`.
- `AbsoluteQuietThreshold`, `PercentileQuietThreshold`.
- `Calibration` + `CalibrationStrategy` Protocol + `RWaveAnchoring`
  (with `preferred_leads` kwarg) + `compute_calibration` +
  `estimate_qrs_peak_to_peak`.
- `HealthySegment` + `NoiseSegment` + `extract_healthy_segments` +
  `extract_noise_segments`.
- 48 unit tests against synthetic signals with known peak-to-peak
  amplitudes and known QRS positions.

## v0.2.0+ — concrete next steps

These are sized for "could land in one focused PR each." Order is
suggestive; pick by which producer needs it first.

### Additional filters

- **`filters.notch`** — single-frequency notch (50/60 Hz powerline
  rejection). Producers today rely on the band-pass to attenuate
  powerline; that works because 50/60 Hz is below the 30 Hz lower
  cutoff, but a dedicated notch is cleaner and lets the band-pass
  start lower (e.g. 10 Hz for atrial activation studies).
- **`filters.smoothing`** — Savitzky-Golay or moving-average for
  baseline-wander removal. Currently producers just trust the
  band-pass; explicit smoothing helps with high-amplitude artifact
  rejection upstream of threshold selection.
- **`filters.decimation`** — anti-alias + downsample. The synthetic
  pipeline today simulates at higher rates than the classifier
  consumes; a decimation primitive here avoids a duplicate
  implementation.

### Additional window primitives

- **`windowing.sliding_window_rms`** — for energy-based thresholding,
  which some literature uses instead of peak-to-peak.
- **`windowing.sliding_window_zero_crossings`** — for activation
  detection without amplitude calibration.
- **`windowing.sliding_window_dominant_frequency`** — short-window
  spectral peak. Borders on feature-engineering (which is
  egm-features's job) but is useful for online filtering.

### Additional calibration strategies

- **`calibration.percentile_calibration`** — calibrate so a chosen
  percentile of the *intracardiac* p-p distribution matches a target.
  Useful when no surface ECG is available.
- **`calibration.fixed_gain`** — pull the ADC gain from the file
  header. Cleanest for datasets that calibrate at acquisition time.
- **`calibration.manual`** — caller-supplied scalar. For one-off
  diagnostic work or when the calibration is computed externally.

### Additional threshold strategies

- **`thresholds.healthy.RelativeVoltageIndex`** — `mean + k * std` of
  the pooled distribution. Common in the morphology literature.
- **`thresholds.healthy.MaxPercentile`** — per-channel percentile
  followed by a max across channels (rather than the current pooled
  percentile). Less sensitive to one noisy channel.

### Extractor variants

- **`extraction.activation_based`** — segment around detected
  activations rather than at fixed sliding-window stride. The current
  fixed-window approach misses sub-window-scale activation alignment.
- **`extraction.multi_beat`** — segment N consecutive beats per
  extraction unit. Needed for the egm-classifier Phase 2 multi-beat
  steady-state pacing work.

### Per-channel calibration

The current `Calibration` is a single scalar per record. Some
datasets need per-channel scaling (different bipolar pairs have
different effective gains). The model change: `Calibration.scalar`
becomes `float | dict[str, float]`, or a parallel
`Calibration.per_channel: dict[str, float]` ships beside `scalar`.

### Versioning + py.typed

- Already shipping `py.typed` in v0.1.0.
- Treat each minor bump as additive (new strategies, new helpers).
- Treat each major bump as a Protocol signature change. The first such
  change will be needed when one of the deferred items above
  requires it — likely "multi-beat extraction needs more from Record."

## Won't-do (out of scope, but documented to save the question)

- **No on-disk format ownership.** Producers serialize via
  `myocard-egm-data`; egm-signal stays in-memory only.
- **No CLI.** This is a library. The producers that consume it have
  CLIs.
- **No feature engineering.** Zero-crossings as a windowing primitive
  is reasonable; computing spectral entropy as a per-trace feature
  belongs in egm-features.
- **No model code.** No torch dependency, no nn.Module. Stays at the
  numpy + scipy level.

## Open architectural questions for later

These don't need decisions for v0.1.0 but are worth thinking about
when the second or third consumer arrives:

- **Should the Record Protocol carry a `dtype` attribute?** Today
  everything is implicit float64 → float32 in the producer's writer.
  A Protocol-level dtype would let calibration math respect the input
  precision.
- **Should there be a `MultiSegmentRecord` Protocol** for sources that
  expose multiple recording sessions per patient? Today the record
  is one continuous signal; multi-segment would need either splitting
  upstream or a richer Protocol here.
- **Per-call vs per-strategy `band_hz`?** Currently the extractors
  take `band_hz` as a kwarg with a clinical default. The bandpass is
  shared across both extractors and the synthetic mixer; centralizing
  the default (a module-level `CLINICAL_BIPOLAR_BAND_HZ` constant
  importable from the top level) would reduce repeat-default-typing
  in the producers.
