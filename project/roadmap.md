# egm-signal — roadmap

Future work only — shipped history lives in [`CHANGELOG.md`](../CHANGELOG.md). Internal
doc; public users read the README + `docs/usage.md`.

Work lands here as it's identified, sits in the **Backlog** until a phase-planning session
promotes it into a **Phase** cluster, then moves to the CHANGELOG once shipped. Phase
clusters mirror the science Project Phases in
`intracardiac-platform/project/project_plan.md`. Items scheduled into cross-cutting Phase
work carry a `→ tracked at intracardiac-platform Phase X` annotation; the rest are
component-internal — add when a consumer needs them.

## Phase 1.5 — sim-realism

**Shipped in v0.4.0** (SIG1 `extraction.activation_based`, `filters.lowpass`,
`filters.decimation`, `docs/theory.md`) and **v0.3.0** (B22, the QRS-default removal). See
[`CHANGELOG.md`](../CHANGELOG.md) for what landed and
[`phase_1_5_plan.md`](phase_1_5_plan.md) for how.

Nothing outstanding from this phase.

## Phase 4 — multi-beat

- **`extraction.multi_beat`** — segment N consecutive beats per extraction unit; needed for
  multi-beat sequence classification. The first Protocol-signature *major* bump likely lands
  here ("multi-beat extraction needs more from `Record`").

> → Tracked at `intracardiac-platform/project/project_plan.md` Phase 4 (multi-beat
> sequence classification), per [[reference-multi-beat-consensus]].

## Backlog (unscheduled — promoted into a phase at a planning session)

Each sized for "one focused PR"; add when a consumer needs it.

- **Additional filters** — `filters.notch` (50/60 Hz powerline rejection; lets the band-pass
  start lower, e.g. 10 Hz for atrial-activation studies) and `filters.smoothing`
  (Savitzky-Golay / moving-average for baseline-wander removal upstream of threshold
  selection).
- **Additional window primitives** — `sliding_window_rms` (energy-based thresholding) and
  `sliding_window_zero_crossings` (amplitude-free activation detection).
  (`sliding_window_dominant_frequency` borders on feature engineering — likely belongs in
  egm-features, not here.)
- **Additional calibration strategies** — `percentile_calibration` (match a percentile of
  the intracardiac p-p distribution to a target; no surface ECG needed), `fixed_gain` (ADC
  gain from the file header), and `manual` (caller-supplied scalar).
- **Additional threshold strategies** — `RelativeVoltageIndex` (`mean + k·std` of the pooled
  distribution) and `MaxPercentile` (per-channel percentile then max-across-channels, less
  sensitive to one noisy channel).
- **Per-channel calibration** — today `Calibration` is one scalar per record; some datasets
  need per-channel scaling (`Calibration.scalar → float | dict[str, float]`, or a parallel
  `Calibration.per_channel` map).

## Known issues

None open.

## Open architectural questions for later

Worth thinking about as the second/third consumer arrives; no decision needed yet:

- **A `dtype` attribute on the `Record` Protocol** so calibration math respects input
  precision (today: implicit float64 → float32 in the producer's writer).
- **A `MultiSegmentRecord` Protocol** for sources exposing multiple recording sessions per
  patient (today the record is one continuous signal).
- **Per-call vs per-strategy `band_hz`** — the original idea was to centralize the clinical
  default in a module-level `CLINICAL_BIPOLAR_BAND_HZ` constant importable from the top level,
  to reduce repeat-default-typing in the producers. **B22 puts that in question:** the
  library-defaults rule says a foundation library ships no policy defaults, and centralizing a
  band default is more of the pattern B22 exists to remove — as is the
  `DEFAULT_BIPOLAR_BAND_HZ` (30–300 Hz) constant `extraction/extractors.py` already ships. The
  counter-argument is that 30–300 Hz is a *literature* constant (Sánchez 2021 / Unger 2019 /
  Deno 2017), not a project policy choice — but "a reasonable median in healthy adults" was the
  same defence offered for the QRS target, and it lost. Unresolved on purpose: it spans the
  producers that pass `band_hz`, so it needs the project-lead, not this repo. Raise it if a
  second band ever appears.

## Won't-do (out of scope, but documented to save the question)

- **No on-disk format ownership.** Producers serialize via `myocard-egm-data`; egm-signal
  stays in-memory only.
- **No CLI.** This is a library; the producers that consume it have CLIs.
- **No feature engineering.** Zero-crossings as a windowing primitive is reasonable;
  computing spectral entropy as a per-trace feature belongs in egm-features.
- **No model code.** No torch dependency, no `nn.Module` — stays at the numpy + scipy level.
