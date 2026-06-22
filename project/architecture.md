# egm-signal — architecture and design rationale

Internal design doc for people building / maintaining the library. The
public surface is documented in `docs/usage.md`; this doc explains the
why.

## Where the package sits

```
┌───────────────────────┐   ┌────────────────────────┐
│ myocard-egm-contracts │   │   myocard-egm-data     │
│  (format schemas)     │   │   (HDF5 I/O, datasets)  │
└───────────────────────┘   └────────────────────────┘
                                       ▲
                                       │
                                       │  consumed by
                                       │
        ┌──────────────────────────────┴──────────────────────────────┐
        │                                                              │
┌───────┴──────────┐    ┌──────────────────┐    ┌──────────────────┐  │
│ iafdb-pipeline   │    │ synthetic-egm-   │    │  egm-features    │  │
│ (real bank prod) │    │ pipeline (mixer) │    │  (feature vecs)  │  │
└───────┬──────────┘    └─────────┬────────┘    └─────────┬────────┘  │
        │                         │                       │           │
        │       all three import primitives from          │           │
        └─────────────────┬───────┴───────────────────────┘           │
                          ▼                                            │
                ┌──────────────────────┐                                │
                │  myocard-egm-signal  │ ◄──── no internal deps        │
                │   (this repo)         │       pure numpy + scipy     │
                └──────────────────────┘                                │
                                                                        │
                            (egm-classifier consumes ClassifierBank      ┘
                             from egm-data, not raw signals — but
                             starting with the model/ subpackage,
                             egm-classifier and downstream analysis
                             tools also import from egm-signal for
                             ML pre/post-processing math)
```

Three signal-side consumers share one implementation of bandpass,
sliding-window peak-to-peak, threshold strategies, and the
signal-amplitude calibration scaffolding. The legacy
`synthetic-egm-pipeline` had inlined copies of `_bandpass` in two places
with a docstring explicitly saying "matches iafdb-pipeline's
filters.bandpass" — that smell is what this package exists to resolve.

A fourth consumer family — `egm-classifier`'s export CLI, the
`egm-viewer` analysis tab, paper-figure notebooks — pulls in the
ML-side primitives in `model/` (temperature scaling today, more
classifier-output utilities later). These never look at raw EGM
signals; they operate on classifier logits/probabilities. Keeping
both sides in egm-signal avoids a separate "ML utilities" package
for what's currently a small surface.

## Folder layout

Light grouping: a folder per domain that already has multiple files or
is clearly going to grow. Single-purpose files stay at the top level.

```
src/myocard_egm_signal/
├── records.py             ← Record Protocol; ~1 class, unlikely to grow
├── windowing.py           ← grows to sliding_window_rms etc. later
├── filters/               ← grows to notch, smoothing, decimation
├── calibration/           ← signal-amplitude calibration; grows to new strategies
│   ├── base.py            ← Calibration + CalibrationStrategy Protocol
│   ├── qrs_estimation.py  ← estimate_qrs_peak_to_peak helper
│   ├── r_wave_anchoring.py
│   └── _helpers.py        ← compute_calibration convenience
├── thresholds/            ← grows to new strategies in either direction
│   ├── base.py            ← Both Protocols
│   ├── healthy.py         ← Absolute + Percentile + NoThreshold
│   └── noise.py           ← AbsoluteQuiet + PercentileQuiet
├── extraction/            ← grows to future extractors
│   ├── segments.py        ← HealthySegment + NoiseSegment dataclasses
│   └── extractors.py      ← The two functions
└── model/                 ← ML model pre/post-processing math
    └── temperature_scaling.py   ← fit_temperature + apply_temperature
```

Each subpackage `__init__.py` re-exports its public names so consumers
write `from myocard_egm_signal.calibration import RWaveAnchoring` rather
than reaching into `r_wave_anchoring.py` directly. The top-level
`__init__.py` re-exports everything again so the common case is just
`from myocard_egm_signal import RWaveAnchoring`.

## Why the Record Protocol

The extractors operate on "a record" — some object with a multi-channel
signal, a sample rate, channel names, and a channel-index lookup.
Several options were on the table:

- **A concrete `Record` base class** that producer record types inherit
  from. Rejected: forces a runtime import of `myocard_egm_signal` from
  every producer-side record definition, and demands explicit
  inheritance from each. Doesn't actually buy us anything over a
  Protocol.
- **A type-erased dict argument** (`record: dict[str, Any]`). Rejected:
  loses static type-checking, makes the docstring contract weaker
  ("must have these keys") than the equivalent Protocol contract
  ("must have these attributes").
- **A Protocol** (the chosen design). Producer record types satisfy it
  structurally — no inheritance, no isinstance registration, no runtime
  dependency. mypy checks the satisfaction; `runtime_checkable` lets
  tests sanity-check it at runtime too.

The Protocol declares attributes as read-only `@property` so frozen
dataclasses satisfy it cleanly. A bare `var: T` declaration in a
Protocol means "must be a mutable attribute," which frozen dataclasses
can't satisfy. The `@property` declaration means "must be readable,"
which any reasonable record satisfies.

## Why two parallel threshold hierarchies

Healthy and noise extraction differ in one thing: the comparison
direction (`pp >= threshold` vs `pp <= threshold`). At first glance
that's a single-bit flag, so it's tempting to unify the threshold
strategies into one hierarchy with a `direction` parameter.

We didn't, for two reasons:

1. **Type signatures communicate intent.** A function annotated
   `strategy: ThresholdStrategy` accepts strategies meant for keep-above
   selection. A function annotated `strategy: NoiseSegmentStrategy`
   accepts strategies meant for keep-below selection. mypy refuses to
   pass a `PercentileQuietThreshold` to `extract_healthy_segments`
   without an explicit cast. This catches the "I meant noise, I wrote
   healthy" bug at the type check, not at runtime against real data.

2. **The empty-pool sentinel differs by direction.** When the pooled
   p-p distribution is empty (e.g. a record with no matching
   channels), the right "reject everything" return value is `+inf` on
   the healthy side (no window meets `pp >= +inf`) and `-inf` on the
   noise side (no window meets `pp <= -inf`). A unified hierarchy
   would have to know the direction at compute time, which means
   passing it through — at which point we're back to two hierarchies
   in everything but name.

The two hierarchies share the same Protocol shape (a `compute_threshold`
method on a pooled-p-p numpy array). They live in
`thresholds/healthy.py` and `thresholds/noise.py` respectively; the
Protocols themselves live in `thresholds/base.py`.

`NoThreshold` lives in `healthy.py` only — there is no
`NoQuietThreshold`. "Every window is noise" is not a meaningful
operation; the producer should write a regular (non-noise) bank if it
wants every window.

## Why `preferred_leads` is a kwarg, not a constant

The R-wave anchoring strategy needs a list of surface ECG leads to try
in priority order. IAFDB happens to expose II, I, V1, aVF, aVL, III,
aVR, V5 — but that list is dataset-specific (e.g. a 12-lead system
might use a richer order, a system without V1 would skip it). Three
options were considered:

- **A module-level constant.** Rejected: callers can't override per
  call without monkey-patching.
- **A class attribute on `RWaveAnchoring`.** Rejected: same problem;
  also leaks IAFDB lead names into the class definition forever.
- **A kwarg on `RWaveAnchoring.__init__`** with a documented default
  (the chosen design). The default ships in `DEFAULT_PREFERRED_LEADS`
  and the doc warns "Override at strategy construction time if your
  dataset uses different or fewer leads."

The Calibration dataclass records the actually-used lead in its
`metadata` dict so a downstream report can answer "which lead did the
calibration use?" without re-reading the original record.

## The dataclass + Protocol + concrete-strategy split

`calibration.base.py` exposes three things:

- `Calibration` — a frozen dataclass that carries the result of a
  calibration computation. Data, not behavior.
- `CalibrationStrategy` — a Protocol that says "I can compute a
  Calibration from a Record." Interface, not implementation.
- (Empty) — concrete strategies live in their own files.

`r_wave_anchoring.py` ships the only concrete strategy today
(`RWaveAnchoring`). Future strategies (percentile-based, fixed-gain,
manual, etc.) drop into their own sibling files and get re-exported
from `calibration/__init__.py`. The convenience function
`compute_calibration` lives in `_helpers.py` — the leading underscore
signals "module-internal," and it's re-exported via the subpackage
`__init__.py` so consumers don't have to know which file it's in.

This same shape is mirrored by the threshold strategies (Protocol in
`base.py`, concretes in `healthy.py` / `noise.py`) and is the pattern
to follow for any new strategy family.

## Empty-pool sentinel convention

Both `PercentileThreshold` and `PercentileQuietThreshold` need to
return something when the pooled p-p distribution is empty. We chose
opposite sentinels deliberately:

- `PercentileThreshold` (keep-above) returns `+inf`. The orchestrator's
  `pp >= +inf` is never true; the bank is empty.
- `PercentileQuietThreshold` (keep-below) returns `-inf`. The
  orchestrator's `pp <= -inf` is never true; the bank is empty.

Both mean "reject everything when there's no data" but the numerical
sentinel is opposite per side. Returning the wrong sentinel would
silently produce a full bank when the input was actually empty —
preferable to fail closed.

`NoThreshold` returns `-inf` (always-keep on the healthy side). It must
not be used on the noise side — the docstring warns about this. There
is no `NoQuietThreshold` to enforce the rule with the type system.

## Why two subpackages with "calibration" in their description

Two distinct concepts unfortunately share the English word "calibration":

- **`calibration/`** — *signal-amplitude* calibration. Solves for the
  scalar that brings a raw EGM record's QRS peak-to-peak to a chosen
  reference (R-wave anchoring against surface ECG leads). Inputs:
  multi-channel signals + QRS sample indices. Outputs: a
  `Calibration` object whose `.scalar` multiplies the signal.
- **`model/`** — *probability* calibration (today: temperature
  scaling). Solves for the scalar that makes a trained classifier's
  predicted probabilities match empirical class frequencies. Inputs:
  pre-sigmoid logits + binary labels. Outputs: a fitted `T` and an
  apply step that divides logits by it.

The two are unrelated and combining them under one folder would
overload the namespace. We keep them as separate subpackages and use
explicit names everywhere (`compute_calibration` for signal-amplitude,
`fit_temperature` for probability) so the call site disambiguates
which "calibration" is being performed.

Future additions follow the same split: signal-side calibration
strategies (percentile-based, fixed-gain, manual) drop into
`calibration/`; classifier-output calibration strategies (Platt
scaling, isotonic regression, multi-class softmax variants) drop
into `model/`.

## How a new producer pulls this in

Concretely, in `iafdb-pipeline`'s `bank_export.py`:

```python
from myocard_egm_signal import (
    AbsoluteThreshold, NoThreshold, PercentileThreshold,
    RWaveAnchoring, extract_healthy_segments,
)
from myocard_iafdb_pipeline.constants import BIPOLAR_CHANNELS

cal = RWaveAnchoring(target_qrs_pp_mv=1.0).compute(record)
segments = extract_healthy_segments(
    record,
    threshold=AbsoluteThreshold(0.5),
    channels=BIPOLAR_CHANNELS,  # IAFDB-specific
    calibration=cal,
)
# segments is list[HealthySegment]; producer assembles a Pydantic
# IafdbBank from these and hands to myocard-egm-data's writer.
```

The producer keeps its dataset-specific knowledge (the bipolar
channel set, the PhysioNet download, the record loader) and pulls in
the generic primitives via imports. No producer keeps a copy of
`bandpass` anymore.

## How an ML-side consumer pulls this in

Concretely, in `egm-classifier`'s export CLI (a future PR):

```python
from myocard_egm_signal import apply_temperature, fit_temperature

# Pulled from the eval pipeline: per-trace logits + ground-truth labels.
T = fit_temperature(eval_logits, eval_labels)

# Either bake T into the ONNX graph at export time, or persist T in
# egm_class_model_metadata.json so the deployment runtime applies it.
calibrated_logits = apply_temperature(eval_logits, T)
```

Same import shape as the signal-side consumers; same "egm-signal is
where polyrepo-shared pure functions live" principle. The classifier
keeps its model-specific knowledge (architecture, training loop, ONNX
exporter) and pulls in the math via imports.
