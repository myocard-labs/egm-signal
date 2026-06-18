# myocard-egm-signal

> Generic signal-processing primitives for intracardiac EGM data: filters, windowing, threshold strategies, calibration, and Record-Protocol-driven segment extractors.

Part of the [myocard-labs](https://github.com/myocard-labs) cardiac signal processing toolkit.

---

## Why

This package owns the signal-processing primitives that are shared across the producer pipelines and the feature library — the math that doesn't know or care which dataset the signal came from. Filters, sliding-window math, threshold strategies, calibration scaffolding, and the segment extractors all live here so they have **one implementation** rather than three slightly-different copies in each producer.

It is positioned between the format schemas (`myocard-egm-contracts`) and the producers that build banks against those schemas (`iafdb-pipeline`, `synthetic-egm-pipeline`). The package has **no internal dependencies** — pure numpy + scipy. Anyone who wants to band-pass an EGM in a notebook can `pip install myocard-egm-signal` and skip everything else.

The package is **dataset-agnostic**. The extractors take any object satisfying the `Record` Protocol — a multi-channel signal with a sample rate, channel names, and a channel-index lookup. A concrete `IAFDBRecord` from iafdb-pipeline satisfies it structurally today; a future producer's record type satisfies it the same way.

---

## Install

From PyPI (when published):

```bash
pip install myocard-egm-signal
```

From source (during pre-1.0 iteration):

```bash
pip install git+https://github.com/myocard-labs/egm-signal.git
```

Editable install for development:

```bash
git clone https://github.com/myocard-labs/egm-signal.git
cd egm-signal
pip install -e ".[dev]"
pre-commit install
```

---

## Programmatic usage

```python
import numpy as np
from myocard_egm_signal import (
    bandpass,
    extract_healthy_segments,
    AbsoluteThreshold,
    RWaveAnchoring,
)

# A 1-D or 2-D signal at any sample rate; band-pass it in the clinical
# bipolar EGM band.
filtered = bandpass(my_signal, fs=1000.0, low_hz=30, high_hz=300)

# Calibrate a record by R-wave anchoring on the surface ECG, then
# extract high-voltage bipolar segments. `record` is any object that
# satisfies the Record Protocol (name, patient, fs, signal,
# channel_names, channel_index).
cal = RWaveAnchoring(target_qrs_pp_mv=1.5).compute(record)
segments = extract_healthy_segments(
    record,
    threshold=AbsoluteThreshold(0.5),
    channels=("CS12", "CS34", "CS56", "CS78", "CS90"),
    calibration=cal,
)
```

For the keep-below companion (noise extraction), see
`extract_noise_segments` + the `*QuietThreshold` strategies.

---

## Module map

| Module | What's in it |
|---|---|
| `myocard_egm_signal.records` | `Record` Protocol — the structural type the extractors consume. |
| `myocard_egm_signal.filters` | `bandpass` (zero-phase Butterworth). Future: notch, smoothing. |
| `myocard_egm_signal.windowing` | `sliding_window_peak_to_peak` and future sliding-window primitives. |
| `myocard_egm_signal.thresholds` | `ThresholdStrategy` + `AbsoluteThreshold` + `PercentileThreshold` + `NoThreshold` — keep-above selection. |
| `myocard_egm_signal.noise_thresholds` | `NoiseSegmentStrategy` + `AbsoluteQuietThreshold` + `PercentileQuietThreshold` — keep-below selection. |
| `myocard_egm_signal.calibration` | `Calibration` + `CalibrationStrategy` + `RWaveAnchoring` + `compute_calibration` + `estimate_qrs_peak_to_peak`. |
| `myocard_egm_signal.extraction` | `HealthySegment` + `NoiseSegment` + `extract_healthy_segments` + `extract_noise_segments`. |

---

## Tests

```bash
pytest                  # full suite
pytest --cov            # with coverage
ruff check .            # lint
ruff format --check .   # format check
mypy                    # type check
```

CI runs the same checks on Python 3.10, 3.11, and 3.12 — see `.github/workflows/ci.yml`.

---

## Project status

Pre-1.0; expect breaking changes across minor versions until the API stabilizes.

- For end-user usage examples + a "define your own threshold strategy" walkthrough, see [`docs/usage.md`](docs/usage.md).
- For design rationale (why Protocols, why two parallel threshold hierarchies, why `preferred_leads` is a kwarg), see [`project/architecture.md`](project/architecture.md).
- For the v0.2.0+ plan, see [`project/roadmap.md`](project/roadmap.md).
- For the broader refactor context, see `intracardiac-platform/project/project_plan.md`.

---

## Citation

If you use this software in academic work, please cite:

```bibtex
@software{klein_myocard_egm_signal_2026,
  author  = {Klein, Daniel},
  title   = {myocard-egm-signal: signal-processing primitives for intracardiac EGM data},
  year    = {2026},
  url     = {https://github.com/myocard-labs/egm-signal},
}
```

---

## License

MIT — see [LICENSE](LICENSE). Attribution requirements for data sources used by this package are listed in [NOTICE](NOTICE).
