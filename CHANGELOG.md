# Changelog

All notable changes to `myocard-egm-signal` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project aims to follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] — 2026-08-06

Activation-aware signal processing: find where the tissue activates, measure how wide each
activation is, and cut fixed-length windows anchored on them. Purely **additive** — nothing
that worked against 0.3.0 behaves differently.

### Added

- **`extraction.activation_based`** — the activation-anchored path, end to end.
  - *Detection curves*: `RectifiedDerivative`, `TeagerKaiser`, `BotteronEnvelope`, behind a
    `DetectionPreprocessor` ABC. The first two are sharp and spike at every steep deflection;
    the envelope is smoothed and merges a fractionated complex into one bump before any
    later stage sees it.
  - *The detection chain*: `detect_activation_train` composes preprocessing → thresholding →
    `LocalMaximaSelector` → `GreedyHeightSuppressor`, with an optional `TwoStageRefiner`.
    `detect_activation` is the single-activation case. Threshold and refractory interval do
    **different** jobs — amplitude versus time — and are deliberately not interchangeable.
  - *Complex bounds*: `measure_complex` / `measure_complexes` walk outward from an activation
    to report onset, offset, rise and fall. A side that ran out of room is flagged `clamped`
    rather than reported as a measurement, since pooling those would bias the long tail the
    measurement exists to characterise.
  - *Windowing*: `window_train` cuts one window per activation and **classifies** each —
    `in_bounds`, `single_activation`, and the neighbouring intervals — without dropping any.
    `WindowSet` carries the results with masks, `select`, and `concat` for corpus building.
  - *Position policy*: `UniformPositionGenerator` (under an `ActivationPositionGenerator` ABC)
    supplies each window's position and owns its random stream. Collapsing the range to a
    point gives a fixed position, so the varied-versus-fixed comparison is a value change
    rather than a separate code path.
  - *One-call path*: `SingleActivationWindower` and `MultiActivationWindower` package
    detect-and-window. Detection is the only step that differs between them; everything after
    it is shared, so windows cut from different sources stay comparable.
- **`filters.lowpass`** — zero-phase Butterworth low-pass, mirroring `bandpass`.
- **`filters.decimate`** — anti-alias then downsample by an integer factor. The low-pass is
  half the operation, not preparation for it: without it a component above the new Nyquist
  folds into the retained band and is unrecoverable. Zero-phase, so activation timing is not
  shifted.
- **`thresholds.detection`** — two new threshold families over a 1-D signal, split by how much
  context the rule needs rather than by what it is used for: `SignalThreshold`
  (`MedianMadThreshold`, `PercentileSignalThreshold`) reads the whole array;
  `PositionAwareSignalThreshold` (`PeakFractionThreshold`) reads it at a given sample.
- **`exceptions`** — `DegenerateSignalError` with `EmptySignalError` (a caller bug) and
  `ConstantSignalError` (a dead or rail-clipped channel). Raised rather than returning a
  sentinel, because a sentinel threshold propagates into later stages and re-emerges looking
  like a real result. A batch caller can catch the pair or tell them apart.
- **`docs/theory.md`** — the repo's canonical math home: every operator's discrete form, why
  it is defined that way rather than an obvious alternative, and a linked primary source per
  technique. Also absorbs the older primitives' math from `iafdb-pipeline`, on the rule that
  the repo owning a primitive owns its math.

### Notes

- No policy defaults are shipped. Threshold multipliers, prominence floors, refractory
  intervals, position ranges and window lengths are all required arguments — they decide what
  the science is, and belong to the pipeline that knows its data. The anti-alias filter's
  order and cutoff fraction *do* carry defaults, because they describe how faithfully the
  function resamples rather than anything about the data.

## [0.3.0] — 2026-08-01

### Removed

- **BREAKING — `DEFAULT_TARGET_QRS_PP_MV`.** `RWaveAnchoring`'s `target_qrs_pp_mv` is now a
  **required** argument, and `compute_calibration` raises `TypeError` when given neither a
  strategy nor a target (it previously fell back to the library default).

  *Why:* a foundation library ships no policy defaults. The target amplitude decides what scale
  a whole corpus is calibrated to, so it belongs in the consuming executable's config. The
  library default (1.5 mV) had drifted from iafdb-pipeline's CLI default (1.0 mV), so the same
  code calibrated to two different scales depending on the entry point. Requiring the argument
  makes that class of drift impossible rather than merely fixed.

  *Migration:* pass the value explicitly — `RWaveAnchoring(target_qrs_pp_mv=1.0)`. No stored
  data changes: every bank on disk was written through a CLI that already passed its own target.

### Changed

- `.gitignore` output-dir patterns are root-anchored (`/data/` rather than `data/`) so a
  same-named source package can never be silently untracked.

## [0.2.0] — 2026-06-22

### Added

- `model.temperature_scaling` — a temperature-scaling calibration primitive, consumed by
  egm-classifier's export/calibration step.

## [0.1.0] — 2026-06-18

First release: the shared DSP-primitives library extracted from iafdb-pipeline —
filtering, windowing, threshold / noise-segment strategies, calibration, and segment
extraction, all in-memory over numpy arrays.

### Added

- **Filtering** — `bandpass` (zero-phase Butterworth, axis-0).
- **Windowing** — `sliding_window_peak_to_peak`.
- **Threshold / noise strategies** — `ThresholdStrategy` / `NoiseSegmentStrategy`
  Protocols + `AbsoluteThreshold`, `PercentileThreshold`, `NoThreshold`,
  `AbsoluteQuietThreshold`, `PercentileQuietThreshold`.
- **Calibration** — `Calibration` + `CalibrationStrategy` Protocol + `RWaveAnchoring`
  (with `preferred_leads`), `compute_calibration`, `estimate_qrs_peak_to_peak`.
- **Extraction** — the `Record` Protocol, `HealthySegment` / `NoiseSegment`, and
  `extract_healthy_segments` / `extract_noise_segments`.
- `py.typed` marker + 48 unit tests against synthetic signals with known peak-to-peak
  amplitudes and QRS positions.

### Dependencies

Lean: numpy + scipy. No torch, no h5py, no internal myocard- dependencies.

[0.3.0]: https://github.com/myocard-labs/egm-signal/releases/tag/v0.3.0
[0.2.0]: https://github.com/myocard-labs/egm-signal/releases/tag/v0.2.0
[0.1.0]: https://github.com/myocard-labs/egm-signal/releases/tag/v0.1.0
