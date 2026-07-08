# Changelog

All notable changes to `myocard-egm-signal` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project aims to follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.2.0]: https://github.com/myocard-labs/egm-signal/releases/tag/v0.2.0
[0.1.0]: https://github.com/myocard-labs/egm-signal/releases/tag/v0.1.0
