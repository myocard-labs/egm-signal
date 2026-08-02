# Changelog

All notable changes to `myocard-egm-signal` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project aims to follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
