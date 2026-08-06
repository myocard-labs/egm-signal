# egm-signal — Phase 1.5 implementation plan

**Repo:** egm-signal · **Phase:** 1.5
**Phase design doc:** `intracardiac-platform/phases/phase_1_5/design.md`
**Status:** in progress · **Progress:** 10/13 steps done (S0 ✅ · S1 ✅ · S2 ✅ · S3 ✅ · S4 ✅ · S5 ✅ · S6 ✅ · S6b ✅ · S6c ✅ · S7 ✅)

**Release model (corrected 2026-08-01, Daniel).** Supersedes S0's "ships alone as v0.3.0 ahead of
SIG1": egm-signal appears **once** in the Wave-1 order, so **all** of this plan's code lands before a
single PR → merge to `release` → tag. One commit per step, **no push until just before the PR**. B22
and SIG1 therefore ship as **one release, v0.3.0** — the existing CHANGELOG entry gains an `### Added`
section for SIG1 at S9 rather than a second version that never gets tagged. Consequence for
iafdb-pipeline: the v0.3.0 tag they adopt (CL-028 / CL-033) now arrives at the *end* of this repo's
work rather than ahead of SIG1 — harmless, since IAF3 follows SIG1 in the serial order anyway, but
the log post promised in CL-030 lands later than implied.
**Repo estimate:** **12–24 h active** (SIG1, incl. the `docs/theory.md` graduation and the 2026-08-04 S4 re-scope) · **+0.5–1 h**
(B22) · **+0.5–1.5 h** if B9 is promoted from backlog. **Local only** — design §6 effort estimation +
tracking is **skipped for Phase 1.5** (Daniel, 2026-07-29): no §6 roll-up, no `Actual`/`Elapsed`, no
`estimation_ledger.csv` rows. These ranges stay here as rough planning aids and go nowhere.

**Wave placement (updated 2026-08-01).** Superseded: the earlier "parallel to Wave 1" note. §7 now runs
Wave 1 **serially, one repo per focused review** — egm-contracts ✅ → egm-data ✅ → SEP12 ✅ → CLF5 ✅ →
**SIG1 (this repo, step 5)** → IAF3 → FEA1 → STU6. Still true that SIG1 has no schema and no sibling
deps, so the order is an execution choice, not a dependency; SIG1 sits here because both Wave-2
producers (SEP2, IAF1) import it, so it is the deepest critical-path piece to de-risk early. **The
Wave-1 gate requires both pure libraries tagged**, so this session ends with **v0.3.0** (B22) and
**v0.4.0** (SIG1) tagged.

---

## Scope — what this plan covers

| Phase item | What it needs from this repo | Steps |
|---|---|---|
| **SIG1** | `extraction.activation_based` — detection function `g` (rectified `dV/dt` · Teager–Kaiser · Botteron envelope), adaptive-threshold + refractory-NMS train detection, envelope onset/offset, **the shared anchor-window helper** (`window_from_anchor`) + the **boundary + multi-beat predicates**; plus **`docs/theory.md`**, the repo's canonical math home (project-lead, 2026-07-28) | S1–S7a, S9 |
| **B9** *(conditional)* | `filters.decimation` — anti-alias + downsample. **Only if** the §8.1 `T`/rate decision needs resampling; otherwise stays in `roadmap.md`. | S8 |
| **B22** | **QRS-calibration default removal** — delete `DEFAULT_TARGET_QRS_PP_MV`, make the argument required, leaving iafdb-pipeline's CLI config the single source. Resolves the two-layer default split iafdb-pipeline documents in its `docs/theory.md`. Arrived via the iafdb-pipeline chat; value ruled by research, structural fix ruled by Daniel 2026-07-28; id assigned by the project-lead in CL-024 §5a (design §4). **Breaking → ships alone as v0.3.0.** | S0 |

Consumers: **SEP2** (synthetic-egm-pipeline) and **IAF1** (iafdb-pipeline) both depend on SIG1 and
own only their *response* — SEP2 sizes the sim so the crop never overhangs; IAF1 drops boundary +
multi-beat windows and may stride. Neither re-implements the crop math.

Math reference: `intracardiac-platform/project/investigations/activation_splitting_method.md`.

## Design notes

Local decisions taken before coding. Anything long-lived is folded into `project/architecture.md` at
S9.

- **Subpackage, not a single module.** §3 names the module `extraction.activation_based`; it lands as
  a **subpackage** `extraction/activation_based/` because it carries four separable concerns
  (detection functions, train detection, complex bounds, anchor windowing) and one flat module would
  run past 400 lines — well beyond this repo's largest file (209). This follows the architecture doc's
  "a folder per domain that … is clearly going to grow." Public names re-export up through
  `extraction/__init__.py` and the top-level `__init__.py`, so consumers still write
  `from myocard_egm_signal import detect_activation_train`.

- **Detection functions are a strategy family, not a `method=` string.** The three options carry
  different parameters (Botteron needs a band + a low-pass cutoff; the other two are parameterless),
  so they follow the repo's established Protocol + concrete-strategy pattern (`CalibrationStrategy` /
  `ThresholdStrategy`) rather than egm-features' `method=` literal. A parameterless string argument
  would have nowhere to put the Botteron cutoffs.

- **The detection threshold is a third threshold family.** *(Built at S3; the plan originally said
  "Protocol" — it shipped as an **ABC**, `DetectionThreshold`, per the S2 review rule; renamed `SignalThreshold` at the S5 review.)* `τ` is a
  threshold on the detection curve `g`, not on pooled peak-to-peak, so it gets its own interface in
  `thresholds/base.py` with concretes in `thresholds/detection.py`, alongside the existing keep-above /
  keep-below pair. Same reasoning the architecture doc gives for keeping two hierarchies: **the type
  signature communicates which array you're allowed to hand it**.

- **~~Fail closed when `MAD(g) = 0`~~ — wrong, corrected at S3 by measurement.** The plan assumed
  `MAD = 0` meant a dead channel and should return `+inf`. It doesn't: a **clean synthetic trace**,
  exactly flat between activations, has the same signature, and failing closed there would make the
  synthetic path silently detect nothing. What actually ships: `+inf` only for an **empty** or
  **constant** curve (no peaks to separate), and for `MAD = 0` the formula degrades to
  `τ = c·median(g)`. *(The original note here justified this with a **strictly-greater** contract;
  superseded 2026-08-04 — under local-maxima extraction `≥` and `>` are identical, so the comparison
  isn't what makes it work. A flat baseline has no strict local maximum. See S4.)*

- **`AnchoredWindow` reports the *realized* position.** After `s = round(t_a − p(T−1))` the actual
  fractional position is `(t_a − s)/(T−1)`, which differs from the requested `p` by up to half a
  sample. The result dataclass carries both, so SEP2 and IAF1 can report the position statistic they
  actually produced — that statistic is what STU5/STU4 compare, and it is exactly the drift this
  helper exists to prevent.

- **Bounds checking is the caller's move.** `window_from_anchor` raises on an out-of-range crop rather
  than returning `None`; callers test `window_is_within_bounds` first. This keeps the return type
  clean and matches the two consumers' actual behavior — SEP2 guarantees the crop fits by sizing the
  sim, IAF1 tests and drops.

- **Plain frozen dataclasses, not egm-contracts models** — preempting the pr_checklist §2 line
  "cross-module handoffs use typed egm-contracts models." That rule governs handoffs of *persisted,
  cross-repo data formats*; egm-signal sits **below** contracts by charter (no `myocard-*` sibling
  deps at all), and `AnchoredWindow` / `ActivationComplex` are in-memory DSP results, not artifacts.
  Same precedent the repo already set with `HealthySegment`, `NoiseSegment`, and `Calibration`. Noted
  here so the placement audit doesn't re-open it at PR time.

- **~~No `docs/theory.md` in this repo~~ — reversed 2026-07-28 by the project-lead.** My first draft
  argued the platform method spec should stay the single source and that mirroring it here would
  drift. The project-lead overruled on the ownership rule: *the repo that owns a primitive owns its
  math theory*, and SIG1's primitives have **two** consumers (SEP2, IAF1), so the shared math cannot
  live in either consumer's doc. The resolution avoids the drift I was worried about — this is a
  **graduation, not a mirror**: the as-built derivations move out of
  `activation_splitting_method.md` into `docs/theory.md`, and the investigation stays behind as the
  design/research record, cross-linking forward. One canonical home either way. **S7a** does it.

- **Botteron needs a low-pass we don't have.** `LP₂₀(|BP₄₀₋₂₅₀(x)|)` requires a low-pass primitive;
  the repo ships `bandpass` only. S1 adds `filters.lowpass` as a sibling of `bandpass` rather than
  faking it with a near-zero low edge.

## How this repo is verified in-sandbox

**Always run the gate inside a venv built from this repo's own `pyproject.toml`** — never with
ad-hoc `pip install --break-system-packages` against the system interpreter:

```
python3 -m venv .venv-check
.venv-check/bin/pip install -e ".[dev]"
.venv-check/bin/python -m mypy                       # bare, per CL-098
.venv-check/bin/python -m ruff format --check src tests
.venv-check/bin/python -m ruff check src tests
.venv-check/bin/python -m pytest -q
```

`.venv*/` is gitignored, so the venv can persist across steps.

> **⚠ The sandbox interpreter changed under us (2026-08-06), and this recipe now half-works.** The
> sandbox went from **Python 3.12 → 3.10** (coincided with a Claude desktop update, so most likely a
> new sandbox image; Daniel's own machine has no 3.10, so **local runs are unaffected** — this is a
> sandbox-only problem). It **cannot be fixed from inside**: `uv python install 3.12` reaches
> `github.com` but the binary lives on `release-assets.githubusercontent.com`, which the proxy refuses
> (403 after CONNECT); there is no sudo, and jammy's apt has no 3.12. The downgrade breaks the
> persisted venv two ways and quietly degrades mypy:
>
> - **The stale venv is unusable and undeletable.** Its interpreter is now 3.10 while its
>   `site-packages` is still `lib/python3.12/`, so `pip`, `mypy` and everything else vanish (`ruff`
>   survives only because it is a standalone binary in `bin/`). The mount refuses `rm` on it
>   (`Operation not permitted`), so it cannot simply be rebuilt in place.
> - **Python 3.10 caps numpy at 2.2.x** (2.3+ needs ≥3.11), and numpy 2.2's stubs make mypy emit
>   ~120 spurious `type-arg` errors — spread across files nobody touched (`test_filters.py`,
>   `thresholds/base.py`, …), which is how you tell it apart from a real regression.
>
> **Workaround until the interpreter is back:** build the gate venv in the writable scratch mount and
> point it at the repo, with no editable install (so nothing is written into the repo):
>
> ```
> python3 -m venv <scratch>/.venv-gate
> <scratch>/.venv-gate/bin/pip install "numpy>=1.26,<2.5" "scipy>=1.10" "pytest>=8.0" \
>     "ruff==0.15.17" "mypy==2.1.0"
> PYTHONPATH=src <scratch>/.venv-gate/bin/pytest -q
> PYTHONPATH=src <scratch>/.venv-gate/bin/mypy --cache-dir=/tmp/mypy-cache src tests
> ```
>
> Clear the mypy cache when switching environments — a stale cache from the other interpreter makes
> mypy die with `INTERNAL ERROR` rather than a useful message. **`ruff` and `pytest` are trustworthy
> under this workaround; mypy is not** — filter `type-arg` and read what remains. Everything else
> (`attr-defined`, `call-arg`, …) is still real, and did catch a genuine missing `__all__` entry at
> S6c. A clean mypy run must happen on a 3.12 interpreter before the PR.

**Why it matters (learned the hard way at S0/S1).** The gate's result depends on the *numpy* version,
and only `pip install -e ".[dev]"` resolves the one this repo declares (`numpy>=1.26,<2.5` → 2.4.6):

| numpy | mypy result | why |
|---|---|---|
| newest (ignores the `<2.5` cap) | fails parsing `numpy/__init__.pyi` | its stubs use PEP-695 `type` statements, which `python_version = "3.10"` rejects |
| 1.26 (the floor) | 43 spurious `type-arg` errors | pre-PEP-696 stubs make `ndarray` generic with no defaults, so `strict`'s `disallow_any_generics` flags every bare `np.ndarray` |
| **2.4.6 (what `[dev]` resolves)** | **clean** | stubs carry TypeVar defaults, so bare `np.ndarray` is legal |

Both failure modes are *environment* artifacts that look exactly like real type errors in files the
change never touched — the tell is errors appearing in untouched modules.

## Steps

Each step is one focused commit, ends green (`ruff format` + `ruff check` + `mypy src` + `pytest`),
and states its verification. ☐ todo · 🔨 wip · ✅ done

### S0 — Remove the QRS-calibration default (make it required) ✅ (0.5–1 h) — **breaking · ships first as v0.3.0**
- **Done 2026-08-01.** Constant deleted and dropped from both `__all__`s; `target_qrs_pp_mv` required
  on `RWaveAnchoring`; `compute_calibration` raises `TypeError` when given neither a strategy nor a
  target. Four test call sites updated (`:95`, `:123`, `:135`, `:140` — one more than the three
  predicted: the `preferred_leads=()` check at `:135` also constructed bare). Three tests added: the
  constructor raise, the neither-argument raise, and a guard asserting the constant is absent from
  both namespaces. Docs de-defaulted (`docs/usage.md`, `README.md`) and the module docstring now
  carries the *why*. **Verified:** 64 passed · `ruff format` 27 files unchanged · `ruff check` clean ·
  `.gitignore` root-anchoring checked **both** directions with `git check-ignore`.
  **mypy — clean** (re-verified 2026-08-01 in a proper project venv; the earlier "34 pre-existing
  errors" report was my own environment mistake, see the verification note below).
- **Fold into this session (fleet housekeeping, both from the log):** **CL-099** — root-anchor the
  `.gitignore` output-dir patterns (`data/` `banks/` `checkpoints/` `runs/` `logs/` `mlruns/` `wandb/`
  `artifacts/` are all still unanchored here, so any same-named source package would be silently
  swallowed — green locally, `ModuleNotFoundError` in CI), verifying both directions with
  `git check-ignore`. **CL-117** — `__version__` already uses the `importlib.metadata` pattern
  (`src/myocard_egm_signal/__init__.py` L74), so that one is verify-and-report only. Already satisfied
  and needing nothing: `mypy files = ["src","tests"]` (CL-098/CL-100) and exact `[dev]` pins
  `ruff==0.15.17` / `mypy==2.1.0` (CL-085).
- **Change:** `calibration/r_wave_anchoring.py` — **delete `DEFAULT_TARGET_QRS_PP_MV`** and make
  `RWaveAnchoring(target_qrs_pp_mv=...)` a **required** argument; drop the constant from
  `calibration/__init__.py` and the top-level `__init__.py` (`__all__` in both).
  `calibration/_helpers.py` — `compute_calibration` keeps its either/or ergonomics but now raises
  `TypeError` when **neither** `strategy` nor `target_qrs_pp_mv` is supplied, instead of silently
  falling back. Docs: `docs/usage.md:46–47` ("the default 1.5 mV target") and `README.md:63` are
  rewritten to show the value being passed explicitly.
- **Why required rather than 1.5 → 1.0:** the **library-defaults rule** — a foundation library ships
  no policy defaults; they live in the JSON Schema or the executable consumer's config. Flipping the
  number makes the two copies agree today but keeps the structure that let them diverge. After this,
  the constellation has exactly **one** target-amplitude default: iafdb-pipeline's
  `cli/_config.py:189` (1.0), which is also where the research chat's rationale for the value should
  be recorded. egm-signal stops having an opinion on the value.
- **Verify:** `pytest` green — **three tests do rely on the default and must be updated**:
  `test_calibration.py:95` (`RWaveAnchoring(preferred_leads=("V1",))`), `:123` (`RWaveAnchoring()`),
  and `:140` (`isinstance(RWaveAnchoring(), CalibrationStrategy)`). Add a test asserting
  `RWaveAnchoring()` raises `TypeError`, and one asserting `compute_calibration(record)` with neither
  argument raises. `mypy src` clean.
- **Blast radius (checked, 2026-07-28):** **no bank changes and no regeneration.** Every produced
  bank came through the iafdb-pipeline CLI, which already passes its own config default of 1.0
  (`cli/_config.py:189`, and all nine `examples/*.yaml`); the egm-contracts and egm-data fixtures and
  `egm-data/docs/usage.md` all use 1.0 too. **The one hard break is iafdb-pipeline**, the sole
  consumer: `export/bank_export.py:50` imports the constant by name (→ `ImportError`, not a silent
  behavior change) and `:116` uses it as its own signature default. Editable sibling installs make
  that live the moment this commits — hence the isolated release below.
- **Tag timing: unblocked** — iafdb-pipeline said tag whenever (CL-033); don't hold v0.3.0 for their
  adoption, since nothing in their plan depends on it and holding it would block SIG1, which *is* on
  the early critical path. Commitment made in exchange: **post to the coordination log immediately
  after tagging** so their red suite is scheduled, not a surprise. B22's honest end-to-end cost is the
  pair — egm-signal 0.5–1 h **+** iafdb-pipeline 1–2 h — if the project-lead wants it costed whole.
- **Release:** ships **alone** as **v0.3.0** — its own PR + tag, before SIG1 starts, so
  iafdb-pipeline adopts one small breaking change and verifies green rather than debugging it tangled
  up with a large feature add. Same de-risking logic as the schema-migration-wave rule. SIG1 then
  lands as **v0.4.0** (additive), which iafdb-pipeline picks up when it starts IAF1. This makes S9's
  version bump **0.4.0**, not 0.3.0.
- **Cross-repo follow-up (Daniel is notifying the iafdb-pipeline chat):** drop the
  `DEFAULT_TARGET_QRS_PP_MV` import, make `bank_export`'s `target_qrs_pp_mv` required so the CLI
  config stays the single source, re-pin egm-signal to v0.3.0, and refresh the three now-resolved
  `docs/theory.md` passages (~216, ~384, ~426–429) — recording the research rationale for 1.0 there,
  since that repo now owns the value.
- **Depends on:** none — independent of S1–S9.

### S1 — `filters.lowpass` ✅ (0.5–1 h)
- **Done 2026-08-01.** `filters/lowpass.py` mirroring `bandpass`'s conventions (axis-0, 1-D/2-D,
  `0.99·Nyquist` auto-cap, `sosfiltfilt`, order 2); re-exported from `filters/__init__.py` and the
  top-level `__init__.py`; `docs/usage.md` gained a section + the module-table row. **8 tests added**
  (72 total, all passing) — beyond the planned pass/attenuate/cap/2-D/raise set, two pin the
  properties later steps depend on: **zero-phase** (a Gaussian bump's argmax does not move, so S5's
  onset/offset crossings aren't biased) and **fractionation bridging** (a rectified three-deflection
  complex low-passes into exactly **one** above-threshold run while the raw rectified signal
  fragments — the mechanism S2's `BotteronEnvelope` and S5's bounds both rest on).
- **Note:** the module docstring records *why* this isn't `bandpass(low_hz=ε)` — that would put a
  spurious high-pass corner near DC, exactly where a rectified envelope carries most of its energy.
- **Change:** `filters/lowpass.py` — zero-phase Butterworth low-pass along axis 0, mirroring
  `bandpass`'s signature, Nyquist capping, and error contract. Re-export from `filters/__init__.py`.
- **Verify:** unit tests — a two-tone signal loses the high tone and keeps the low one within
  tolerance; shape preserved for 1-D and 2-D input; the same `ValueError` cases as `bandpass`.
- **Depends on:** none.

### S2 — Detection-function family ✅ (1–2 h)
- **Done 2026-08-01; revised 2026-08-02 after review.** New subpackage
  `extraction/activation_based/` with `base.py` (`DetectionPreprocessor` **ABC**) +
  `preprocessors.py` (`RectifiedDerivative`, `TeagerKaiser`, `BotteronEnvelope` + the two Botteron
  default constants). 46 tests in `tests/test_detection_preprocessors.py`, 118 total; full gate green.
- **Review change 1 — vocabulary (Daniel).** The three transforms are *not* detection functions;
  they detect nothing. Detection is a three-stage chain and each stage now has its own name:
  **detection preprocessing** (`DetectionPreprocessor`, the transform `x → g`) → **detection
  thresholding** (S3, the decision rule) → **the detection function** (S4, the whole chain, which is
  the only thing that actually detects). Renamed module + Protocol + test module accordingly.
  **Cross-repo:** `activation_splitting_method.md` and design §3 both use "detection function" for
  the *transform* — the usage this corrects. Flagged to the project-lead; SEP2/IAF1 import these
  names, so it is worth aligning before Wave 2.
- **Review change 2 — ABC instead of Protocol (Daniel).** A closed family with known structure
  should *enforce* it, not document it. `compute()` is now a **template method** on the ABC:
  validate → apply the short-input rule → delegate to the subclass's `_compute()`. A subclass that
  omits `name` fails at class-definition time; one that omits `_compute` fails at instantiation.
  `architecture.md` needs the ABC-vs-Protocol rule recorded at S9 (Protocol where *other repos'*
  types must conform structurally — `Record`, `ThresholdStrategy`; ABC for families we ship and
  extend in-repo).
- **Review change 3 — `np.diff(prepend=)` readability, and a non-discriminating test (Daniel).**
  Daniel queried whether `np.abs(np.diff(signal, prepend=signal[0]))` set `g[0]` to `x[0]` instead of
  `0`, and whether `prepend` grew the output. **Both verified: the code was correct** — `prepend`
  forms `[x[0], x[0], x[1], …]` *before* differencing, so `g[0] = x[0] − x[0] = 0` exactly, and
  because `diff` then removes one element the length is preserved (checked at n = 2, 5, 100, 1000).
  Two changes anyway:
  1. **Rewritten as an explicit zero-fill** (`g = zeros(...); g[1:] = abs(diff(signal))`). Identical
     output over random arrays, but it cannot be misread — and in code whose off-by-one shifts every
     activation index, a reviewer losing time to the idiom *is* a defect. It also now mirrors
     `TeagerKaiser`'s shape.
  2. **The unit test could not have caught the bug he suspected.** Its fixture started at
     `x[0] = 0.0`, so the expected `g[0] == 0.0` passes under *both* the correct behavior and the
     `g[0] = x[0]` bug — demonstrated by running a deliberately buggy implementation against it.
     Fixture changed to a non-zero first sample (verified: the buggy version now fails), plus a
     dedicated randomized test asserting `g[0] == 0` where `x[0] ≠ 0`. **Right answer, weak test** —
     the question was worth asking even though the code was fine.
- **Review change 4 — `fs` moved onto `BotteronEnvelope.__init__` (Daniel).** Only the envelope needs
  a sampling rate; the other two are pure sample-domain arithmetic and were carrying an argument they
  ignored. `compute(signal)` is now the whole ABC signature, and `fs` sits with the band edges it
  belongs to. It is validated at construction, so a bad rate fails before any signal is processed.
  Two tests added: the rate-free pair now **reject** a second argument, and the envelope refuses to be
  built without a rate (no default — guessing one would silently design the filters for the wrong
  band rather than fail).
  - **Trade-off accepted, documented in the class docstring and theory §2:** an envelope instance is
    now **bound to one sampling rate**, and since a numpy array carries no rate, feeding it a
    differently-sampled trace cannot be detected here. Mitigation is placement — construct it where
    the rate is known, not at module scope. Worth stating because it is a *new* failure mode: with
    `fs` per call the rate always travelled with the data.
  - Timing note: done now specifically because S4 and the Wave-2 consumers (SEP2, IAF1) have not yet
    bound to the old signature. After that this is a three-repo change.
- **The ABC caught a real bug, not just a style point.** Hoisting the fail-closed short-input rule
  into the template method (one `min_samples` per subclass) exposed that `BotteronEnvelope` **raised
  from inside `sosfiltfilt`** on short input while its two siblings returned zeros — the same input,
  a different failure mode depending on which preprocessor you picked. My original S2 test only
  covered the other two, so it passed. Now: `min_samples = 16`, **derived** from scipy's
  `padlen = 3·(2·len(sos)+1)` rule (2nd-order band-pass → `len(sos)=2` → padlen 15) and verified
  empirically, with the boundary pinned by a test and the parametrized short-input test extended to
  all three.
- **Two deviations from the plan text, both toward house convention:**
  1. **Protocol in `base.py`, not alongside the concretes.** `architecture.md` states the
     Protocol-in-`base`/concretes-in-siblings split is "the pattern to follow for any new strategy
     family", and `thresholds/` and `calibration/` both do it.
  2. **Plain classes, not frozen dataclasses.** The house strategy pattern is a plain class with a
     `name: str` class attribute (`AbsoluteThreshold`, `RWaveAnchoring`). It also sidesteps a real
     trap: on a frozen dataclass, `name: str = "…"` becomes an *init field* callers could override,
     and the `ClassVar` fix wouldn't satisfy the Protocol's mutable-attribute declaration.
- **Contract decisions worth knowing downstream:** 1-D input only (detection is per-channel; a 2-D
  `argmax` is meaningless) · output length == input length, so an index into `g` *is* a signal index ·
  undefined positions filled with `0.0`, never a spurious boundary peak · degenerate-length input
  returns zeros rather than raising, so a dead channel detects nothing (fail-closed) · Teager–Kaiser
  returned **unclipped** — it can dip negative, and clipping is a policy call this layer doesn't own.
- **Test-fixture bug caught by the suite, worth recording.** The first `_biphasic_activation` was a
  *truncated* sine; its edge discontinuity is exactly as steep as the genuine upstroke, so
  `RectifiedDerivative`'s argmax tied on the truncation and landed 4 samples late — a fixture
  artifact reported as a detector error. Replaced with a **Gabor pulse** (sine under a Gaussian
  envelope), which is also the more faithful model of a real tapered deflection. Measured offsets on
  the corrected fixture: derivative **+0**, Botteron **+0**, Teager–Kaiser **−1** samples.
- **Change:** `extraction/activation_based/detection_functions.py` — `DetectionFunction` Protocol
  (`compute(signal, fs) -> np.ndarray`, `name: str`) plus three frozen-dataclass concretes:
  `RectifiedDerivative` (`g[i] = |x[i] − x[i−1]|`, the `dV/dt`-max default), `TeagerKaiser`
  (`g[i] = x[i]² − x[i−1]·x[i+1]`), `BotteronEnvelope` (`LP₂₀(|BP₄₀₋₂₅₀(x)|)`, reusing `bandpass` +
  S1's `lowpass`). All return an array the same length as the input, edges handled explicitly.
- **Verify:** unit tests on a synthetic biphasic activation at a known sample — each `g` peaks within
  a small tolerance of the true activation index; Botteron bridges a fractionated three-deflection
  complex into one above-threshold run while the rectified derivative does not (the smoothing
  mechanism the method spec relies on).
- **Depends on:** S1.

### S3 — Detection-threshold strategies ✅ (0.5–1.5 h)
- **Done 2026-08-02.** `thresholds/detection.py` — `MedianMadThreshold` (`τ = c·median(g) + λ·MAD(g)`)
  and `PercentileDetectionThreshold` *(both renamed at the S5 review)*, plus the `median_absolute_deviation` helper; `DetectionThreshold`
  **ABC** in `thresholds/base.py` alongside the two existing Protocols. 27 tests, 145 total, gate green.
  Theory §3 written (3.1–3.4) with Hampel 1974 + Leys 2013, both Crossref-verified.
- **ABC here too, and the base doc now states the rule:** Protocol where *another repo's* type must
  conform structurally (the two amplitude families are advertised as user-extensible in `usage.md`);
  ABC where we ship and extend the family in-repo and there is shared behavior worth enforcing. The
  degenerate-input handling is exactly that. The two older Protocol families are deliberately left
  alone — changing them would break the documented "define your own strategy" story.
- **Naming:** only `PercentileSignalThreshold` is qualified, because `healthy.py` already owns
  `PercentileThreshold` and everything is re-exported flat from the package root. `MedianMadThreshold`
  has no clash, so qualifying it would be noise.
- **No default `c` / `λ` / `q`** — how aggressive detection is, is policy the calling pipeline owns
  (the B22 rule again). §8.1 sets them for this project.
- **~~A failing test overturned a contract decision~~ — and the ruling later overturned it back.**
  A failing test showed that on a **clean sparse curve** (`MAD = 0`, `median = 0`, so `τ = 0`) a
  sample-wise `>=` admits all 1000 samples versus 8 under `>`, so I made the contract strictly
  greater. **Superseded 2026-08-04:** research ruled that the threshold gates **local maxima**, not
  samples — and a flat baseline has no strict local maximum, so `≥` and `>` are identical (8 and 8;
  4 and 4 on a Gabor pulse). Contract is now `≥`, matching the spec and `find_peaks(height=…)`. The
  investigation was still worth it: it is what established that the comparison *isn't* where the
  selectivity comes from.
- **A limitation found by measurement, now largely dissolved by the S4 ruling.** On an
  exactly-zero-baseline curve the median/MAD rule gives no *discrimination* — it degrades to "any
  non-zero sample". Measured on a clean Gabor trace through `RectifiedDerivative`: 59% of the curve is
  exactly 0.0 (the Gaussian tail underflows), `τ = 0`, and 206 of 500 samples sit above it — as **one
  contiguous run containing the true peak**. *(2026-08-04: under local-maxima extraction that run
  yields just **4 candidates**, which refractory suppression then reduces to one. The threshold was
  never meant to be the selective step.)* Acceptable, because S4's refractory suppression picks the time within
  that run and the method spec treats synthetic detection as a cross-check; and it does not arise on
  real signal, which has a noise floor everywhere. Pinned by a test asserting the run structure, and
  written into both the class docstring and theory §3.4 so SEP2 meets it before the behavior surprises
  them.
- **Change:** `thresholds/detection.py` — `DetectionThresholdStrategy` Protocol +
  `MedianMadThreshold(c, lambda_)` (`τ = c·median(g) + λ·MAD(g)`, `+inf` when MAD is 0) and
  `PercentileDetectionThreshold(q)`. Re-export from `thresholds/__init__.py`.
- **Verify:** unit tests — known median/MAD input returns the hand-computed τ; a constant array
  returns `+inf`; the percentile strategy matches `np.percentile` on a known distribution.
- **Depends on:** none (parallel with S1/S2).

### S4 — The detection function: candidates → suppression → train ✅ (3–5 h) — **re-scoped 2026-08-04**
> _The method spec was fleshed out by Daniel + research on 2026-08-04; the 5-step IAFDB algorithm, the
> candidate-extraction rule and the suppressor-as-strategy decision are all new since this plan was
> written. Old scope (1.5–3 h) was "threshold `g`, then NMS"._

- **Change A — candidate extraction** (`activation_based/candidates.py`): the spec's **step 3**. Not
  "every sample above `τ`" but **one candidate per contiguous above-`τ` segment**, taken at that
  segment's `argmax`, with segments below a **minimum width** or **peak prominence** dropped as noise
  blips / far-field. Returns candidate indices + their segment extents (the extent *is* the
  `W_act` that S5 measures, so it is worth returning rather than recomputing).
- **Change B — suppression strategy family** (`activation_based/suppression.py`): the spec makes this
  a **run-time-selectable strategy**, like the preprocessors. `RefractorySuppressor` ABC +
  **`GreedyHeightSuppressor`** (the Phase-1.5 default: sort tall→short, accept if nothing already kept
  lies within `Δ_refr`). The spec's other four — causal blanking, sliding-window max, segment-merge,
  DP-optimal — are documented in the ABC's menu and go to `roadmap.md`; building one is enough to
  prove the seam. *Complexity is explicitly not the deciding factor (tens–hundreds of candidates,
  offline), so this is a correctness/simplicity choice.*
- **Change C — the chain** (`activation_based/detection.py`): `detect_activation(x, *, preprocessor)`
  → `int` for the **synthetic** case (global `argmax g`; no threshold, no suppression — the generator
  knows the location, so detection is a cross-check) and `detect_activation_train(...)` for the
  **IAFDB** case, running preprocess → threshold → candidates → suppress. This function *is* "the
  detection function" in the corrected vocabulary.
- **Change D — optional two-stage refinement** (spec step 5, **off by default**): snap each accepted
  `t_a` to the local `argmax |dV/dt|` within a small neighbourhood, recovering the sharp instant the
  envelope's smoothing blurred. Exists because the Botteron default carries a *late* timing bias on
  asymmetric (fibrotic) complexes.
- **Verify:** a synthetic train of N activations at known spacing returns exactly N indices · a
  fractionated complex returns **one** index under the envelope *and* under a sharp preprocessor
  (the first by merging at the `g` level, the second by suppression — the two routes the spec
  distinguishes) · two genuine activations just beyond `Δ_refr` return **two** · sub-width /
  low-prominence blips are dropped · a flat channel returns empty · refinement moves the time toward
  the `|dV/dt|` maximum on an asymmetric complex and is inert when off.
- **✅ SETTLED 2026-08-04 by research — local maxima, not segment-argmax.** The spec's step 3 now
  reads: all local maxima of `g` above `τ` (`g[i-1] < g[i] > g[i+1]` and `g[i] ≥ τ`), filtered by
  **prominence**; a fractionated complex may contribute several and that is fine, because deciding
  which are *distinct activations* is the refractory step's job. The spec adds an explicit **"don't"**
  for segment-argmax, on exactly the failure I measured: two genuine activations riding one above-`τ`
  run — common in fast AF, when the envelope doesn't fully return to baseline between beats — would
  collapse to one peak, a silent miss. Cited to **Pan & Tompkins 1985** as the standard
  local-peaks → threshold → refractory chain.
  - **The above-`τ` segment is still used** — as `W_act` for onset/offset (S5) — just not as the
    merge rule. S5's "consume S4's segment extents" note stands.
  - **`scipy.signal.find_peaks` is back on the table**, since it implements exactly this: local
    maxima, `height=τ`, `prominence=`, `distance=Δ_refr`. Worth using rather than hand-rolling, with
    the caveat that its `distance` is greedy-by-height — which *is* the chosen suppressor, so the
    library call and the spec's step 4 agree. The suppression-strategy seam still matters for the
    other four algorithms in the menu.
- **↩ Knock-on: the S3 comparison contract flips `>` → `≥`.** My S3 contract said strictly-greater,
  justified by a measurement (a clean sparse curve gives `τ = 0`, so `≥` admits all 1000 baseline
  samples where `>` admits 8). That reasoning was sound **for sample-wise thresholding**, which is now
  not the extraction rule. Verified under local-maxima extraction: `≥` and `>` are **identical** —
  8 and 8 on that curve, 4 and 4 on a clean Gabor pulse — because a flat baseline contains no strict
  local maximum. So the comparison is not load-bearing, and `≥` is adopted to match the spec and
  `find_peaks(height=...)`. Updated in `thresholds/base.py`, `thresholds/detection.py`, theory §3.4
  (retitled "What the threshold is applied to"), the notation entry for `τ`, and the two tests that
  asserted the old contrast. **The S3 "no discrimination on a clean curve" caveat also softens**: the
  206-sample region becomes **4 candidates** under local-maxima extraction, not one undifferentiated
  run.
- **Depends on:** S2, S3.
- **Done 2026-08-04.** Three modules — `candidates.py` (local maxima at or above `τ`, prominence /
  width filters, enclosing above-`τ` segment returned for S5), `suppression.py` (`RefractorySuppressor`
  ABC + `GreedyHeightSuppressor`, with the spec's other four documented in the menu and deferred to
  `roadmap.md`), `detection.py` (`detect_activation` for synthetic, `detect_activation_train` for the
  chain, `refine_activation_times` for the optional second stage). 26 tests, 170 total, gate green.
  Theory §3.5 written.
- **`scipy.signal.find_peaks` used rather than hand-rolled**, and the reason is not brevity: verified
  that its `height` comparison is **inclusive**, matching the spec's `g[i] ≥ τ`, and that it resolves
  a **flat-topped** peak to its midpoint. The naive `g[i-1] < g[i] > g[i+1]` test finds *nothing* on a
  plateau — on quantized or synthetic data that silently loses activations. Pinned by a test.
- **The prominence filter turned out to be effectively required, not optional.** On a single
  fractionated complex with realistic noise, running the chain with no prominence floor gives **3**
  activations under the Botteron envelope and **9** under the rectified derivative; with a floor at
  20% of the curve maximum, **1** and **1**. Cause: an adaptive threshold a few MAD above a quiet
  baseline is low in absolute terms, so filter ringing and noise bumps clear it — and being spaced
  further apart than `Δ_refr`, suppression *keeps* them. The failure is silent, because spurious
  detections look like activations. Documented in `find_candidates`, theory §3.5, and pinned by
  `test_without_prominence_the_chain_over_detects`. Signature keeps it `None`-able (the right value is
  data-dependent — the B22 rule) but the docstring now says plainly that omitting it over-detects.
- **Suppression is height-ordered, and that is a design choice worth knowing.** A causal
  (Pan–Tompkins-style) scan accepts whichever peak of a cluster arrives *first*, so a small precursor
  deflection masks the genuine activation behind it. Height-ordering makes the result independent of
  which end of the record you start from — right for an offline splitter with no causality constraint.
- **Refinement guard:** the refiner's `radius_samples` must stay well under the suppressor's
  `refractory_interval_samples`. A radius able to reach a neighbouring activation lets refinement move
  a peak onto the wrong complex, and suppression has already run, so nothing downstream catches it.
- **Review pass 2026-08-04 (Daniel) — six changes, one of which removed a feature:**
  1. **`refractory_samples` → `refractory_interval_samples`, and moved onto the suppressor's
     constructor.** It is configuration, like Botteron's band edges or the threshold's multipliers, so
     one instance means one interval; `suppress(candidates)` now takes only candidates.
  2. **`_above_threshold_segments` rewritten** with a worked example and step-by-step comments — the
     `np.diff`/`np.split` idiom is compact but opaque on first read.
  3. **`min_width_samples` removed.** Asked for a first-hand source and I have none. Checking the
     *current* spec settled it: the rewritten step 3 filters on **prominence alone** — "minimum width"
     was in the earlier draft and is gone. Worse, I had mapped it to `find_peaks(width=)`, which
     measures width *at half prominence*, so it partly restates the prominence test rather than
     complementing it — exactly the overlapping-filter interaction Daniel flagged. Untested surface
     with no spec backing and no measurement; removed rather than defended.
  4. **The dense `start, end = next(...)` generator replaced** by a named `_enclosing_segment` helper
     with an explicit loop — and, on a follow-up question from Daniel, its silent fallback **removed**.
     It returned a fabricated single-sample segment when no enclosing run was found. That case is in
     fact **unreachable**: a peak exists only because `curve[peak] ≥ τ`, and the segments are the runs
     where `curve ≥ τ` — same comparison, same `τ`. Verified over 3000 randomised curve/threshold
     combinations plus plateau and NaN curves: zero violations. So it now raises `RuntimeError` naming
     the broken invariant. The fabricated extent was the worse failure mode: it would have flowed a
     made-up complex width into S5's onset/offset measurement, corrupting a downstream number instead
     of failing where the fault is.
  5. **Candidate selection is now a strategy family** — `CandidateSelector` ABC + `LocalMaximaSelector`
     — matching the preprocessor / threshold / suppressor shape, and moving `min_prominence` off
     `detect_activation_train`'s signature onto the object that uses it.
  6. **`suppressor=None` now genuinely skips suppression** instead of silently substituting a greedy
     default. It is a **required** keyword that accepts `None`, so the caller says "no suppression" on
     purpose rather than reaching it by omission. `min_prominence` is required in the same way and for
     the same reason — both silently change the activation count. `refiner=None` keeps its default,
     because *that* default is the spec's documented behaviour rather than a silent choice.
  7. **`find_peaks(prominence=0.0)` when the caller passes `None` — kept, with the reasoning inline.**
     Daniel queried the substitution, I replaced it with a pass-through plus an explicit
     `peak_prominences` call, and he preferred the original as more elegant. Kept, but the *why* is
     now in the code: `find_peaks` only populates `props["prominences"]` when asked to filter on them,
     and we report prominence on every candidate regardless. `0.0` is a true no-op filter rather than
     an approximation — the comparison is **inclusive** — verified over 4000 curves (quantized and
     flat-heavy, where plateaus are likeliest) that `None` and `0.0` select identical peaks, with zero
     prominence-exactly-0 peaks ever occurring. The equivalence is now a *guarded assumption*:
     `test_no_prominence_floor_and_a_zero_floor_agree` fails loudly if a future scipy makes that
     comparison strict, rather than the selector quietly dropping peaks.
  - Also folded in: `refine_with` + `refine_radius_samples` became a `TwoStageRefiner` object, which
     removes the invalid combination of a refiner with no radius. A plain class, not an ABC — the spec
     describes one refinement rule, not a family.

### S5 — Activation-complex bounds (onset / offset) ✅ (1.5–3 h)
- **Change:** `extraction/activation_based/complex_bounds.py` — `ActivationComplex` frozen dataclass
  (`onset_sample`, `activation_sample`, `offset_sample`, `rise_samples`, `fall_samples`) +
  `activation_bounds(g, t_a, *, theta)` walking outward from `t_a` to the last sub-θ sample before
  and the first after. `theta` accepts either a fraction of the local peak or a multiple of the
  baseline MAD (the two forms the method spec allows), with the choice explicit at the call site.
- **Verify:** unit tests — a synthetic complex with known onset/offset recovers both within
  tolerance; an asymmetric complex (long fibrotic tail) yields `fall > rise`; a complex running off
  the array end clamps to the array bounds instead of raising.
- **Scope clarified by the 2026-08-04 spec update — this is a *study-time* instrument, not a runtime
  filter.** §8.1 uses `r_rise` / `r_fall` to *choose the position range* so complexes are almost never
  clipped; once chosen the bound stops binding, so the splitter must **not** re-test every window. Two
  consequences: (a) keep it a standalone measurement function that IAF1 can call in a study without
  invoking it per window; (b) do **not** add a clip-filter to S6 — the spec notes that multi-beat work
  in a later phase will deliberately *want* some edge-clipped windows, so filtering here would be
  counterproductive.
- **Feeds from S4:** candidate extraction already computes each above-`τ` segment's extent, which is
  the same `W_act` this step measures. S5 takes those extents rather than rediscovering the segment,
  and refines them to `θ`-crossings.
- **Practical fallback the spec now records:** where per-complex boundaries can't be measured
  reliably on real AF signal, the answer is *not* to force them — fall back to fixed **generous
  asymmetric margins** (healthy front, healthier back, for the long fibrotic tail). Worth surfacing in
  the docstring so a caller reads it as "informs margins", not "gates windows".
- **Depends on:** S2, S4.
- **Done 2026-08-04.** `complex_bounds.py` — `ActivationComplex` (onset/offset/rise/fall/width plus
  per-side **clamped** flags), ~~a `BoundaryLevel` ABC with `PeakFractionLevel` and `BaselineMadLevel`~~,
  and `measure_complex` / `measure_complexes`. Theory §4 written.
- **S5 review, 2026-08-05 — three changes plus a threshold-family restructure.** 204 tests, gate green.
  1. `ActivationComplex.level` → `theta`, and the `level=` parameter → `boundary_threshold=`. The
     ambiguity was a *collision*: a parameter holding a rule object and a field holding a float shared
     one name.
  2. `search_radius_samples` accepts a `(before, after)` tuple. Symmetric was the wrong shape for the
     phenomenon — §4.1 splits `r_rise` from `r_fall` precisely because fibrotic complexes are
     asymmetric, so one cap sized for the tail also licenses the backward walk to run just as far.
  3. **`BoundaryLevel` and `BaselineMadLevel` deleted.** `BaselineMadLevel(k)` was exactly
     `MedianMadThreshold(c=1, lam=k)` — the same arithmetic written twice.
- **Threshold families are now split by context needed, not by use.** `DetectionThreshold` →
  `SignalThreshold` (whole array) plus a **sibling** `PositionAwareSignalThreshold` (array + index);
  `PercentileDetectionThreshold` → `PercentileSignalThreshold`; `PeakFractionLevel` →
  `PeakFractionThreshold`, moved into `thresholds/detection.py`. Naming a type after its *use* was the
  original error — a median/MAD rule is the same computation whether it decides "is this an activation?"
  or "where does this complex end?", so `DetectionThreshold` was a lie in the second case. Siblings
  rather than parent/child because a subclass that *requires* an argument the base lacks breaks Liskov.
  A `SignalThresholdLike` alias covers callers accepting either. **A dispatch helper was considered and
  rejected** — one caller, an unsubtle check; revisit at a second consumer or when
  `RangeAwareSignalThreshold` lands. All of this was free now and breaking after the tag: none of these
  names are in `v0.2.0`.
- **The `+inf` sentinel is gone; degenerate signals raise.** New `exceptions.py`:
  `EmptySignalError` (a programming error — nothing legitimately produces a zero-length array) and
  `ConstantSignalError` (a *data* condition — dead electrode, or a channel clipped to a rail), under a
  shared `DegenerateSignalError(ValueError)` so a batch caller can catch the pair or distinguish them.
  The sentinel was not merely inelegant, it was wrong: measured, a `+inf` θ made the §4 walk terminate
  immediately and report `width=0, is_complete=True, clamped=False` — a fabricated complex flagged as a
  genuine measurement, headed for the §8.1 duration distribution. Divergence from the pooled-amplitude
  empty-pool sentinel is deliberate: an empty *pool* is a legitimate filtering outcome, an empty
  *signal* is not. Knock-on: `detect_activation_train` on a flat channel now raises instead of returning
  an empty array — which is right, since an empty array could not be distinguished from a healthy
  channel that simply had no activations in the window.
- **Known limitation, recorded not fixed:** the constancy test is **whole-array**. A channel that
  flatlines *intermittently* passes it, and every threshold from that trace is then silently biased —
  median and MAD both pull toward the flat value, lowering `τ` and admitting noise elsewhere. Detecting
  bad *sections* needs segment-wise analysis this library does not do, and the case has not been
  characterised on IAFDB. Theory §3.1 + `exceptions.py`; raised to the fleet backlog (CL-124).
- **Correction to this plan's own premise: S4's segment is *not* a bounding box.** The plan said S5
  "consumes S4's segment extents". Measured: `θ` may sit either side of `τ`, so at `θ = 0.10·peak` the
  complex **extends beyond** the above-`τ` segment (471–526 vs segment 474–524), while at 0.25 and 0.50
  it sits inside. So the segment is a *reference*, not a bound, and `measure_complex` works against the
  curve rather than being confined to the segment. Recorded in the module docstring and theory §4.3.
- **The runaway walk, and the condition that causes it.** An outward walk with no bound can run into a
  neighbouring activation. Measured on a four-activation train: at `θ = 0.02·peak` **with realistic
  noise**, the "complex" spans 597 samples and reaches past the previous activation — because `θ` falls
  **below the noise floor**, so the curve never dips under it between beats. Without noise the same
  fraction sits above the floor and behaves normally. Hence an optional `search_radius_samples`, and a
  test pinning both the runaway and the counterpart (a median/MAD level, defined relative to the floor,
  cannot fall beneath it).
- **`clamped` flags are the load-bearing detail.** A side that ran out of room — array edge or search
  radius — is *not a measurement*. Since §8.1 pools these to characterise the long tail of `r_fall`,
  silently including clamped sides would bias exactly the statistic the study exists to produce.
  Reported per side, with `is_complete` to filter.
- **The baseline level is `median + λ·MAD`, not `λ·MAD`.** The spec says "a small multiple of the
  baseline-noise MAD", but a MAD multiple alone is a *spread*, not a level: on a curve with a raised
  baseline it sits below the noise floor and the walk never terminates. Adding the median makes it a
  level above the baseline, which is what the phrase means operationally — and, once written that way,
  it *is* `MedianMadThreshold(c=1, lam=λ)`, which is how the duplicate came to light.
- **Scope held to the spec:** this is a study-time instrument. No clip-filter is added here or in S6 —
  the spec notes later multi-beat work will *want* some edge-clipped windows. The practical fallback
  (fixed generous asymmetric margins when boundaries can't be measured on real AF) is documented.

### S6 — Anchor-window kernel + predicates ✅ (1–2 h) — **re-scoped 2026-08-05 (CL-125)**
> **What this step is now.** S6 shipped the **single-anchor kernel**. The design has since moved the
> per-anchor *loop*, the `PositionRange` type and its *draw* into this repo (CL-125), so the
> caller-facing primitive is **S6b**'s `window_train`. S6's functions stay as the kernel it is built
> from; `window_from_anchor` is no longer the API a producer reaches for. The original spec below is
> unchanged; the review findings follow it.

- **Change:** `extraction/activation_based/anchoring.py` — `AnchoredWindow` frozen dataclass
  (`start_sample`, `end_sample`, `activation_sample`, `requested_position`, `realized_position`,
  `signal`); `anchor_window_start(t_a, p, T) -> int` (`s = round(t_a − p(T−1))`, pure integer math);
  `window_from_anchor(x, t_a, p, T) -> AnchoredWindow` (raises on out-of-range);
  **`realized_position` is now a fleet contract** — this design note is what drove egm-contracts to add
  a per-trace `activation_position` column to **both** banks (CL-052 · CL-060 · CL-062), `$ref`'d from
  one shared `common.schema.json#/$defs/ActivationPosition` so the corpora cannot drift. Match its
  wording exactly: `[0,1]` where **0.0 = first sample, 1.0 = last sample**, `idx = round(frac·(T−1))`,
  and it is the position the crop **produced**, never one measured back off the waveform (a consumer
  wanting the measured dV/dt-max position uses egm-features). `0.0` is a legitimate value, so nothing
  downstream may treat absent as zero;
  `window_is_within_bounds(s, T, n_samples) -> bool`; `window_is_single_beat(train, s, T) -> bool`
  (exactly one member of `train` falls in `[s, s+T)`). `T < 2` raises — `realized_position` divides
  by `T − 1`, and a one-sample window has no meaningful position.
- **Verify:** unit tests — `p = 0.5` centres a known activation; `p = 0` / `p = 1` place it at the
  first / last sample; `realized_position` is within half a sample of `requested_position` across a
  sweep of `p` and `T`; `T < 2` raises; the boundary predicate is false exactly when the crop would
  run off either end; the multi-beat predicate is true for an isolated activation and false when a
  neighbour falls inside the window, including at the `[s, s+T)` half-open edges.
- **Depends on:** S4 (the train type the multi-beat predicate consumes).
- **Done 2026-08-05.** `anchoring.py` — `AnchoredWindow`, `anchor_window_start`,
  `window_from_anchor`, `window_is_within_bounds`, `window_is_single_beat`. 22 tests, 226 total, gate
  green. Theory §5 written (5.1 placement · 5.2 fraction-vs-offset · 5.3 requested-vs-realized ·
  5.4 the three enforced rules).
- **`realized_position` matches the shipped contract wording, checked against the schema.** Re-read
  `common.schema.json#/$defs/ActivationPosition` rather than working from this plan's paraphrase:
  `[0,1]`, 0.0 = first sample, `idx = round(frac·(T−1))`, produced-not-measured, and absence ≠ zero.
  A test applies the contract's *own* conversion formula to our output and asserts it recovers the
  anchor exactly, at every `T` swept — so if the stored value ever stopped meaning what the schema
  says, that test fails rather than the two corpora quietly disagreeing.
- **Rounding is bounded, and the bound is measured not asserted.** Over 200 000 random `(p, T)` pairs
  with `T ∈ [2, 1000)`: **max 0.49999 samples, mean 0.250**. The bound is attained exactly on a tie —
  `T=100, p=0.5` gives `p(T−1) = 49.5` → realized 0.505051, error 0.500 samples; `T=101` is exact.
  Ties round to even (Python's `round`); the direction is arbitrary and documented as not
  load-bearing, but fixed so a request is reproducible.
- **Quantization is worth knowing before reading a position histogram.** Only `T` realized values are
  reachable, spaced `1/(T−1)`: 0.0101 at `T=100`, 0.111 at `T=10`. The comb in a stored-position
  distribution at short window lengths is an artifact of the crop, not physiology. Recorded in §5.3.
- **Out-of-bounds raises rather than sliding.** Clamping to the array edge would change the realized
  position without saying so — the activation would no longer sit where it was asked to, and the
  stored value would record the slide as intentional. `window_is_within_bounds` is the predicate for
  callers that want to skip instead of handling an exception per trace.
- **`AnchoredWindow.signal` is `compare=False, repr=False`.** A generated `__eq__` over an ndarray
  raises on the ambiguous truth value, and the array would make the frozen dataclass unhashable. The
  index fields identify a window on their own; pinned by a test.
- **S6 review (Daniel, 2026-08-05) — three findings, two fixed here, one escalated.**
  1. *Naming.* `position` → `activation_position`, matching the schema field and the `$def` exactly, so
     one word spans code, bank column and contract. Repo-wide finding: **counts and indices shared the
     `_sample(s)` suffix** — `window_samples` is a length, `start_sample` an index. Going forward
     `_samples` marks a count/duration and `_index` an array position. **Scoped to `anchoring.py`**
     (Daniel): committed S1–S5 code keeps its names rather than taking a rename diff.
  2. *Redundant validation + a dead predicate.* `_validate_window_samples` fired twice per
     `window_from_anchor` (measured), and `window_is_single_beat` had **no caller in `src/`** — it was
     written to a spec that had no loop to use it in. Both dissolve in S6b: the batch primitive
     validates once and the predicate becomes load-bearing.
  3. *The real gap — no train, no `𝒫`, no selection algorithm.* Escalated as **CL-125** (scope) and
     **CL-126** (a measurement). Both resolved 2026-08-05 → **S6b**.

### S6b — `window_train` + `PositionRange` ✅ (3–5 h) — **new 2026-08-05 (CL-125 · CL-126 · CL-127)**

> **Why this step exists.** S6 implemented design §3's SIG1 row as it stood, which named only the
> single-anchor helper. Reviewing it surfaced that the *selection algorithm* — loop the train, draw
> `p ∼ 𝒫`, judge each window — lived nowhere: design §3 had left it split across SEP2 and IAF1 as
> "response only" work. **I had already flagged this in CL-016 and declined to escalate it**, judging
> it lower-stakes than the crop-rounding case. That was backwards: a one-sample crop disagreement is
> invisible in a histogram, whereas a differently-implemented draw or drop rule moves the
> *distribution* — and a distribution match is precisely what T1 tests. Re-raised as CL-125 and
> adopted.

- **Change:** `extraction/activation_based/anchoring.py` (or a sibling module) —
  - **`PositionRange`** — the `𝒫` type. A `(lo, hi)` fraction pair, **point-collapsible** (`(x, x)` is
    the fixed-position baseline arm), mirroring the mixer's `snr_db_range=(X,X)` idiom per design §3
    **A4**. Carries the draw (`rng`-driven) and the shared `idx = round(frac·(T−1))` convention.
    **Ships no default range** — the values are the producer's (library-defaults rule).
  - **`window_train(signal, activation_train, *, position_range, window_length_samples, rng)
    -> WindowSet`** — the single windowing primitive. One window per anchor; per window it reports the
    crop, the realized position, the start index, `single_beat`, `in_bounds`, and `iai_prev` /
    `iai_next`.
  - **`WindowSet`** — the result container: the per-window records plus columnar accessors for the
    flags and realized positions (masks and arrays, *not* a `.keep()` — see the seam below).
- **The classify-not-drop seam (the load-bearing decision).** `window_train` **does not drop**. It
  labels each window and hands the whole set back; **keep / drop / stride / pool is the producer's**
  (IAF1 drops boundary + multi-beat; SEP2's train-of-one is always in bounds by sim-sizing). Two
  reasons this seam sits here and not one function deeper: dropping is *policy* and the two corpora
  have deliberately different policies, and §8.1 needs to *measure* the drop rate — which is impossible
  if the drops happen invisibly inside the library.
- **Synthetic is a train of one.** Both corpora go through `window_train`; the synthetic side passes a
  single known activation. One code path, so the geometry cannot drift between corpora — the same
  argument that put the crop math here, applied one level up.
- **`iai_prev` / `iai_next` are reported because §8.1 needs them** (CL-127): the per-anchor
  keep-probability reads straight off the observed intervals, with no need to model the survival
  function `S`. First and last anchors have no neighbour on one side — report `None` there rather than
  a sentinel, since "no neighbour" is a different statement from "an infinitely long interval", and a
  train of one has `None` on both sides.
- **Out-of-bounds windows carry no crop.** `in_bounds=False` means there is no valid length-`T` array
  to return, so the record's signal is `None` while `s`, the realized position and the flags remain
  defined (all are pure arithmetic). The single-anchor `window_from_anchor` keeps **raising** — it is
  the kernel, and a caller naming one anchor has made a specific request that cannot be honoured.
- **Verify:** `window_train` over a synthetic train of one reproduces `window_from_anchor` exactly;
  over a detected train it returns one record per anchor in train order; `single_beat` is false exactly
  when a neighbour lands in `[s, s+T)`; `in_bounds` is false exactly at the record edges and the record
  still carries `s` + realized position; `iai_prev`/`iai_next` are `None` only at the train ends;
  a point-collapsed `PositionRange` yields a constant realized position (up to the half-sample rounding
  bound); a seeded `rng` is reproducible; the realized-position distribution over a wide train tracks
  the requested range.
- **Also verify the bias, since we now have the machinery:** a regression asserting the CL-126 numbers
  — on a synthetic regular train at `T ≈ cycle` the *surviving* (single-beat) positions are
  centre-biased, and on a memoryless train they are not. This is the check §8.1 will run for real, and
  pinning it here means the study inherits a tested measurement rather than writing its own.
- **Depends on:** S4 (the train), S6 (the kernel).
- **Done 2026-08-05.** `PositionRange`, `AnchoredWindow` (now carrying the classification),
  `WindowSet`, `window_train`. 41 tests in `test_anchoring.py`, 241 total, gate green. Theory
  §5.4–§5.7 written ahead of the code and reconciled after.
- **One type, not two.** `window_from_anchor` and `window_train` both return `AnchoredWindow`;
  `single_beat` is `bool | None` where **`None` means *not assessed*** — the kernel has no train to
  judge against — which is a different statement from `False` (a neighbour was found). Considered a
  second record type for the train path; rejected as two near-identical dataclasses for one field's
  worth of difference.
- **A collapsed `PositionRange` does not consume the rng.** The fixed arm returns its constant without
  drawing, so it is reproducible however the generator is seeded or shared, and the A/B arms cannot
  desynchronise through draw-order. Pinned by a test.
- **The measurement sharpened the theory — "regular vs memoryless" is the wrong axis.** Writing the
  regression revealed that what decides the bias is whether **`T/2` fits under the shortest intervals**:
  a central anchor needs ~`T/2` margin on *each* side, an edge anchor needs nearly all of `T` on *one*.
  Measured at `T=192`, uniform `p`, keeping single-beat windows (5-bin shape of survivors, edge/centre):

  | intervals | `T/2` vs floor | edge/centre |
  |---|---|---|
  | exponential μ=200, **no floor** | — | **0.98** (flat — the analytic cancellation, confirmed) |
  | exponential μ=200, floor 40 | 96 > 40 | **1.08** |
  | exponential μ=200, floor 100 | 96 < 100 | **0.62** |
  | regular 180±15 (fast flutter) | 96 < ~150 | **0.64** |

  The two exponential rows differ *only* in their floor and land on opposite sides, which isolates the
  mechanism. A regular rhythm near its cycle is just the commonest way to get a floor above `T/2` — and
  the relevant one, since `T=192` is fixed and flutter cycles near 180 ms are in the envelope. The
  practical §8.1 check is therefore **`T/2` against a low percentile of the measured intervals**, per
  record, computable from `iai_prev_samples` / `iai_next_samples` alone. Theory §5.7 rewritten to match;
  worth a line back to research since CL-127's framing was rhythm-based.
- **`WindowSet` exposes masks and `select(mask)`, never a `keep()`.** The conjunction
  `in_bounds_mask & single_beat_mask` is spelled at the call site rather than assumed here — naming it
  would be asserting the producer's policy from inside the library, which is exactly the seam CL-125
  drew.

- **S6b review (Daniel, 2026-08-05/06) — six comments; five applied here, one became S6c.**
  1. **`PositionRange` → `ActivationPositionGenerator` (ABC) + `UniformPositionGenerator`.** It is a
     *generator of positions under a distribution*, and uniform-over-an-interval is one policy, not the
     definition. A4's semantics are untouched: point-collapsible `(lo, hi)`, `(x, x)` = the fixed arm.
  2. **`draw(rng, count)` → `generate(count)`, with the stream owned at init.** Counted first: *sample*
     appears ~50 times across this library's names and **always means a signal sample** —
     `window_length_samples`, `hop_samples`, `qrs_samples`, `n_samples`. A method named `sample()`
     returning position fractions would be the one statistical use among fifty signal ones, and the
     proposed `gen_samples` keeps the overloaded word while reading as "generate signal samples" — the
     exact wrong meaning. `generate` avoids it entirely. Seeding at construction also drops `rng` from
     `window_train`'s signature. **Makes the object stateful — a first for this repo → CL-128.**
  3. **`WindowSet` carries no configuration.** Dropped `position_range` *and* `window_length_samples`:
     both are already on every record, so a copy on the set could only ever disagree with them.
  4. **`single_beat` → `single_activation`.** What the code can assert is that one member of the
     *detected train* is inside; a fractionated complex the detector splits makes a genuinely
     single-beat window read `False`. Naming it `single_beat` promised a physiological fact the
     detector cannot deliver. Pinned by a test using a 4-sample split. Diverges from design §3 → CL-128.
  5. **Public surface cut to what callers need.** `anchor_window_start`, `window_is_within_bounds`,
     `window_is_single_beat` each had exactly **one internal caller** and were validate-then-delegate
     wrappers; `window_from_anchor` was fully subsumed by `window_train` with a train of one. All four
     are now private — leaving a raising variant public invites a producer to reach for the wrong one
     and lose the classification.
  6. The windower idea → **S6c**.
- **`WindowSet.concat` added 2026-08-06 (Daniel).** A **classmethod**, not a staticmethod: it is an
  alternate constructor, so returning `cls(...)` gives a subclass its own type back, matching the
  `dict.fromkeys` idiom. Takes an iterable (composes with a generator expression, so the real-side
  idiom `concat(s.select(...) for s in sets)` reads in one line). **Rejects mixed window lengths** —
  `T` is one value coupled across the simulator, the splitter and the classifier input, so a
  mixed-length pool is a wiring mistake rather than a choice, and catching it here beats discovering
  it when the windows fail to stack with nothing left to say which source disagreed.
- **Pooling costs the index fields their meaning — documented, not fixed.** `activation_index`,
  `start_index` and `end_index` are positions in *their own source signal*, so once windows from
  different channels or records share a set, two can carry identical indices pointing at unrelated
  places. Positions, flags, intervals and crops all survive pooling; the indices do not. A producer
  needing per-window provenance (IAF1's patient-aware split is the obvious one) should keep its own
  `(source, WindowSet)` pairs and pool last, or not pool at all. Deliberately **no source-identity
  field here** — inventing one would be guessing at a provenance scheme the producer owns.

### S6c — `ActivationWindower` ✅ (1.5–3 h) — **new 2026-08-06 (Daniel's S6b review)**
- **Change:** `extraction/activation_based/windowers.py` — `ActivationWindower` ABC holding the
  position generator and `T`, with `window(signal) -> WindowSet` as a template method and `_detect`
  as the single abstract step; `SingleActivationWindower` (argmax of the detection curve) and
  `MultiActivationWindower` (the full chain). One configured object, one call per trace.
- **The axis is how the train is obtained, not beat count.** Only detection differs between the
  corpora; everything after it is shared, so that is the only overridable step — which is what
  prevents the two corpora's windows being cut differently. Pinned by a test that feeds a windower's
  own detected train back through `window_train` and asserts the record is identical.
- **No third "known-anchor" variant — settled by CL-130, and for a better reason than we asked about.**
  The method spec's *"the generator knows the activation location"* is **wrong**: it is true of the
  **stimulus** and false of the **electrode**. synthetic-egm confirmed the stored trace is a pseudo-EGM
  (a distance-weighted sum of membrane current over the whole mesh), so its timing depends on
  conduction velocity, the *realized* fibrosis draw, electrode standoff and pair position — solver
  outputs, none of them config values. Daniel's read was right.
  **The part neither of us had:** Finitewave *does* ship `ActivationTimeTracker`, so an exact time **is**
  obtainable — and it still must not be used here, because a `V_m` node-threshold crossing is a
  **different measurand** from a bipolar EGM's steepest deflection. Substituting it would put two
  different quantities under one field name across the corpora and flatter the T1 comparison the field
  exists to make. That argument is stronger than "we don't have the number", so it is now recorded in
  `SingleActivationWindower`'s docstring — the earlier text advised the opposite and was **wrong**.
  The tracker is backlogged instead as a *detector cross-check*: synthetic is the only place ground
  truth exists, so it is the only place this detector's bias and jitter can be measured rather than
  trusted.
- **`window_train` stays public** for the case where re-detecting would be wrong: a probe sweeping crop
  offsets over one simulation must detect once on the source and shift by exact integers, or the
  position axis carries per-crop detector jitter.
- **Done 2026-08-06.** 20 windower tests; 246 total; ruff clean. mypy caught a real defect the tests
  could not — four names imported in tests but missing from `__all__`, which `attr-defined` flagged
  while the runtime import worked fine.

### S7 — Package wiring + usage docs ✅ (1–2 h)
- **Change:** `extraction/activation_based/__init__.py` re-exports; `extraction/__init__.py` and the
  top-level `__init__.py` re-export the public names; a new `docs/usage.md` section covering the
  detect → bound → anchor flow end to end — call signatures, argument meanings, and a runnable
  example. The *math* stays out: formulas, derivations, and the citations go in `docs/theory.md`
  (S7a), and usage links to it. Same usage-vs-theory split as egm-features.
- **Verify:** a doctest-style example in `docs/usage.md` runs end to end on a synthetic record;
  `from myocard_egm_signal import ...` resolves every new public name; `mypy src` clean.
- **Depends on:** S4, S5, S6.
- **Done 2026-08-06.** All 26 `activation_based` names plus the three exceptions re-export from
  `extraction/__init__.py` and the package root (62 top-level exports; every one verified to resolve).
  New `docs/usage.md` section — the one-call windower path, the classify-then-filter idiom, pooling,
  choosing positions, the lower-level `window_train`, complex bounds, and the degenerate-signal
  handling. Math stays in `theory.md`, linked once at the end. Module map gained the
  `activation_based` and `exceptions` rows and a corrected `thresholds` row (four families, not two).
- **Verifying the doc found a real inconsistency in committed S4 code.** The plan asks for a runnable
  example, so I extracted every python block from the new section and executed them in one shared
  namespace, as a reader following top to bottom. The dead-channel block failed: `detect_activation`
  raised a **bare `ValueError`** where every other path raises `ConstantSignalError`. It is the one
  function in the chain with no threshold in front of it, so a sweep catching the specific exception
  would have crashed on a flat channel *only when using the single-activation windower* — precisely
  the case the doc was telling people to write. Now raises `ConstantSignalError` / `EmptySignalError`
  like everything else, with a test asserting **both** windowers agree on a dead channel. 252 tests.
- **Doc examples are executed, not eyeballed.** Worth keeping as the S9 exit check: the 8 blocks run
  against a synthetic record and the last one is deliberately fed a flat channel to exercise the
  `except` path it demonstrates.

### S7a — `docs/theory.md` — the repo's canonical math home 🔨 (2.5–5 h) — **now incremental**
- **Restructured 2026-08-01 (Daniel):** the theory doc is written **as the math lands**, not in one
  pass at the end, so each step's math is reviewable while its code is fresh. **Every technique
  carries a clickable primary-source link** (DOI where one exists, else PubMed/PMC/publisher) so a
  reviewer can spin up on an unfamiliar method and check the implementation against its source.
- **Created 2026-08-01** with §1 Filtering (zero-phase · band-pass · low-pass) and §2 Detection
  functions (all three + a selection table). §3–§5 are stubbed with the step that fills them.
- **Citation discipline adopted:** every DOI **verified against Crossref** before it goes in — title,
  journal, volume, pages. This caught one of my own errors: I first cited Marchlinski 2000 for the
  `dV/dt`-max *timing* convention, but that paper establishes **voltage-amplitude tiers**, not
  activation timing. Replaced with **Steinhaus 1989** (Circ Res 64(3):449–462), which actually proves
  the max-slope ↔ depolarization correspondence — and, usefully, bounds its error: >1.8 ms under
  non-uniform coupling, i.e. **worst exactly on fibrotic substrate**. Marchlinski stays, scoped
  correctly. Anything unverifiable is labelled (the 1930 Butterworth paper predates DOIs; its link is
  marked an unofficial scan).
- **Remaining for this step:** §3–§5 as S3–S6 land, then the graduation of the older primitives'
  math from `iafdb-pipeline/docs/theory.md` (§1.2 band-pass is already done as part of this).
- **Change:** create `docs/theory.md`, matching the `egm-features/docs/theory.md` house style
  (front matter → rendering note → table of contents → notation → numbered sections → references).
  Two graduations in one pass, both **moves + polish, not re-derivations**:
  - **SIG1's math**, graduated from
    `intracardiac-platform/project/investigations/activation_splitting_method.md`: the detection
    function `g` and its three variants (rectified `dV/dt` as default — the
    `egm-features.activation_position` convention — Teager–Kaiser, Botteron envelope) with *when each
    is worth its cost*; train detection (adaptive `τ_det` + refractory NMS) and envelope
    onset/offset, including the smoothing mechanism that keeps a fractionated complex in one piece;
    `window_from_anchor` (`s = round(t_a − p(T−1))`, `W = x[s:s+T]`) on the locked `[0,1]`-fraction
    convention; and the boundary (`s < 0 or s+T > L`) + multi-beat predicates.
  - **The already-shipped primitives**, graduated from `iafdb-pipeline/docs/theory.md` §1.1–1.3 and
    §2.1: band-pass, sliding-window peak-to-peak, the threshold strategies, and R-wave-anchoring
    calibration. Same ownership rule, applied to the primitives that predate it.
- **Explicitly out of scope** (consumer-owned, they cross-link back): SEP2's sim-sizing response;
  IAF1's drop / stride / per-record-filtering response; iafdb's §0 ADC-counts→mV input scaling, §4
  sim-vs-real divergence map, §5 parameters, §6 provenance (boundary confirmed by iafdb-pipeline,
  CL-029).
- **Do not name a default calibration target** anywhere in the anchoring section — graduate the math
  and point at the consumer's config for the value (iafdb-pipeline, CL-029). Naming it here would
  rebuild in prose the two-sources-of-truth problem **S0** removes from code.
- **Pool assembly comes with the threshold strategies** — settled in egm-signal's favour (CL-031 →
  CL-034). I document pooling across the supplied channel list, the skip-absent behavior, and the ±inf
  fail-closed sentinels; iafdb keeps *which* set it passes (`BIPOLAR_CHANNELS`, distal-to-proximal)
  and its channel-layout story, with a cross-link from my §1.3.
- **Measured channel facts for §1.3 + §2.1** (verified against all 32 IAFDB headers, 2026-07-29 —
  iafdb's CL-034 correction confirmed and sharpened; state these as measured, not assumed):
  - **All 5 CS bipolar pairs are present in all 32 records**, so `skip-absent` **never fires on the
    bipolar path**. Document it as a general defensive guard with no exercising consumer today —
    honest, and it stops a future reader inferring IAFDB motivation that isn't there.
  - **Surface leads are exactly 3 per record and the set varies** — four distinct combinations:
    `{I,II,V1}` ×12, `{aVF,II,V1}` ×8, `{aVF,I,II}` ×8, `{aVF,I,V1}` ×4. **No record carries all
    eight.** So `RWaveAnchoring`'s priority walk over `preferred_leads` is **load-bearing, not
    defensive** — that's the real per-record variation, and it lives in calibration (§2.1), not pooling.
  - Under the shipped `DEFAULT_PREFERRED_LEADS`, lead **II is selected for 28/32 records and I for
    4/32**; `aVL`, `III`, `aVR`, `V5` never occur in IAFDB at all, so the tail of the default priority
    list is inert on this dataset. Useful for the paper's methods section ("which lead calibrated
    which record") and a concrete argument for why the list is a kwarg rather than a constant.
- **Verify:** every formula matches the implementation as built in S1–S6 — this is written **after**
  the code so it documents as-built behavior, not intent; the `[[terminology-activation-peak-vs-rwave-anchoring]]`
  distinction is stated explicitly, since this doc is now the one place both anchoring concepts are
  described side by side; links resolve in both directions (investigation → theory, iafdb-pipeline →
  theory).
- **Coordination:** the iafdb-pipeline chat is trimming its `theory.md` to consumption-only and
  linking here — its trim and my absorption must land together or the math is briefly orphaned. Its
  §2.1 / §7 passages on the QRS-target default are the same ones **S0** already obliges them to
  refresh, so both edits are one touch on their side.
- **Not gold-plated:** per the project-lead, this is a permanent shared home, not tutorial exposition
  — activation detection and windowing are familiar ground here, unlike egm-features' entropy /
  fractal math, which is why that doc runs to 950 lines and this one shouldn't.
- **Depends on:** S1–S6 (documents as-built), S7.

### S8 — `filters.decimation` (B9) ☐ (0.5–1.5 h) — **conditional**
- **Change:** `filters/decimation.py` — anti-alias low-pass + integer-factor downsample, reusing S1's
  `lowpass`. **Only build this if §8.1 concludes the `T`/rate decision needs resampling**; otherwise
  delete this step and leave B9 in `roadmap.md`.
- **Verify:** unit tests — output length is `ceil(n/factor)`; a tone above the new Nyquist is
  attenuated rather than aliased down; a tone below it survives.
- **Depends on:** S1, and the §8.1 outcome.

### S9 — Docs + phase-exit ☐ (0.5–1.5 h)
- **Change:** `project/architecture.md` gains the `extraction/activation_based/` subpackage in the
  folder-layout tree plus a short section on the third threshold hierarchy and the fail-closed
  detection sentinel; `roadmap.md` drops the now-shipped Phase 1.5 items (and B9 if S8 ran);
  `CHANGELOG.md` gets the `[Unreleased]` → **0.4.0** entry; version bump in `pyproject.toml`
  (**0.4.0** — S0 already consumed 0.3.0 as its isolated breaking release).
- **Verify:** the full `intracardiac-platform/project/pr_checklist.md` run passes, including the
  §2 code-placement audit (nothing here reaches for a bank, an artifact, or a sibling package).
- **Depends on:** all prior steps.

## Complexity + estimate

Scored with the rubric in
`intracardiac-platform/project/investigations/estimate_vs_actual_tracking.md` §8.

| Item | Cx | Size | Estimate | Driver notes |
|---|---|---|---|---|
| SIG1 | **5** | L | 12–24 h | *Change size:* several new modules + a new filter + a third threshold family + a **suppression-strategy family** (added 2026-08-04) + the repo's first `docs/theory.md`. *Novelty:* **high** — net-new detection algorithm, and the method spec itself warns that clean detection on real EGM "is hard in the best of cases." *Surface:* one repo, now with a cross-repo doc handoff to iafdb-pipeline. *Verification:* unit tests on synthetic signals with known activation positions — low burden. |
| B9 | **1** | XS | 0.5–1.5 h | Mechanical; one module reusing S1. Conditional on §8.1. |
| QRS-default removal | **2** | S | 0.5–1 h | Small diff, but **breaking**: a public name leaves `__all__`, `compute_calibration`'s fallback goes, three tests need updating, and it forces a coordinated adoption + re-pin in iafdb-pipeline plus its own release cycle. That coordination — not the code — is what lifts it off XS. No behavior change to any produced artifact. |

**The L-vs-M call — now settled at L.** The first draft flagged a tension: zero coordination and cheap
verification argued **M(3)**, algorithmic novelty argued **L(5)**. The `docs/theory.md` deliverable
resolves it — it adds real surface and a cross-repo doc handoff without adding novelty, so **L(5)**
now holds on two drivers rather than one. The **estimate** absorbed the graduation (8–16 h → 10–21 h);
the **points** didn't move, which is the rubric working as intended: a doc move is hours, not
complexity. It stays below the XL anchor (STU4) by a wide margin.

## Effort tracking

Mechanism: `intracardiac-platform/project/investigations/estimate_vs_actual_tracking.md` §7. Daniel
speaks the markers; this chat stamps the time from `date`. **Active = marked span − breaks.**
**Backstop:** unmarked silence > **2 h** = away.

> **⚠ SUPERSEDED — effort tracking is skipped for Phase 1.5** (Daniel, 2026-07-29; design §6). Flow-down
> proved a much bigger lift than planned, so this phase produces **no §6 roll-up, no `Actual` /
> `Elapsed`, and no `estimation_ledger.csv` rows** — the ledger stays cold and the methodology is
> redesigned in a later phase. Nothing below rolls up anywhere; **no markers are needed this phase**.
> The rows are kept only as a record of what was measured before the call, and they answer my CL-032
> question (whether to flag the reconstructed row) by removing the question.
>
> _Prior rule, now moot: CL-024 §5b had made a repo chat's flow-down session count toward its issues'
> `Actual`. It landed after this repo's planning session, so the rows below were reconstructed from
> file timestamps rather than markers._

### Session log

| Timestamp (local) | Event | Focus (issue) | Note |
|---|---|---|---|
| 2026-07-28 ~14:1x | start *(reconstructed)* | SIG1 | **Estimated, not measured** — session opened some time before the first file write; Daniel to confirm. |
| 2026-07-28 14:54 | *(anchor)* | SIG1 | Measured: `phase_1_5_plan.md` last write of the first drafting pass. |
| 2026-07-28 18:36 | stop *(reconstructed)* | SIG1 | Measured: last `date` stamp of the evening's work. |
| 2026-07-28 18:36 → 2026-07-29 10:0x | away | — | Overnight; excluded by the >2 h backstop. |
| 2026-07-29 10:0x | resume | SIG1 | Coordination-log inbox pass: CL-024 / CL-028 / CL-029. |
| 2026-07-29 *(open)* | — | SIG1 | Session still active. |

### Effort by issue

| Issue | Task-type | Estimate | Active | Elapsed | Sessions |
|---|---|---|---|---|---|
| SIG1 | pipeline (DSP primitive) | 10–21 h *(incl. 2.5–5 h docs)* | **~4.5 h so far** *(reconstructed — planning only, no code yet)* | 2 d | 1 |
| B22 (QRS-default removal) | API change / release | 0.5–1 h | — | — | 0 |
| B9 | pipeline (DSP primitive) | 0.5–1.5 h *(conditional)* | — | — | 0 |
| **Repo total** | | **10.5–22 h** *(+0.5–1.5 h if B9 runs)* | **~4.5 h** | | |

> _The ~4.5 h is **planning + coordination only** — no SIG1 code exists yet. It is bracketed from real
> file timestamps (≈14:1x–18:36 on 2026-07-28, plus the 2026-07-29 inbox pass), **not** from spoken
> markers, so treat it as ±1 h and flag the ledger row `reconstructed`. Its main use is as a data point
> on what flow-down itself costs, which is exactly what CL-024 §5b decided to start capturing._

## Notes / decisions log

- **2026-07-28** — Escalated the anchor-window placement question (shared crop math in SIG1 vs.
  duplicated in SEP2 + IAF1). Project-lead ruled it into egm-signal: `repo_charters` line 84 routes
  segmentation here, the "keep it local" caveat applies only to single-consumer code, and one shared
  helper is what stops SEP2 and IAF1 rounding a sample apart into a false position bias in STU5/STU4.
  §3 updated; SEP2/IAF1 reworded to "response only."
- **2026-07-28** — Project-lead confirmed SIG1 runs **parallel to Wave 1**, not in Wave 2.
- **2026-07-28** — Added **S0** on Daniel's request; originated in the iafdb-pipeline chat, value
  ruled by research. Scoped first as a 1.5 → 1.0 flip, then **escalated to the structural fix** —
  Daniel ruled: **delete the constant, make the argument required**, per the library-defaults rule
  (flipping the number would have left two copies of a policy default free to drift again). Verified
  it changes no produced artifact; the CLI path already used 1.0.
- **2026-07-28** — Release cadence for S0 settled (Daniel): **alone as v0.3.0**, before SIG1, so
  iafdb-pipeline adopts one isolated breaking change and verifies green. SIG1 becomes **v0.4.0**.
  Correction worth recording: the first draft of S0 claimed no tests relied on the default — true for
  a value flip, **false** for the required-argument change, which breaks three
  (`test_calibration.py:95`, `:123`, `:140`).
- **2026-07-28** — **`docs/theory.md` added to SIG1** (project-lead), reversing my earlier design note
  against it. Scope then widened by Daniel: the doc also **absorbs the already-shipped primitives'
  math** — band-pass, sliding-window p-p, threshold strategies, R-wave anchoring — currently sitting
  in `iafdb-pipeline/docs/theory.md` §1.1–1.3 / §2.1. Reason: iafdb-pipeline is trimming to
  consumption-only *now*, so taking those sections in the same pass costs one coordination round
  instead of two and applies the ownership rule completely rather than half. Estimate 8–16 h → 10–21 h;
  Cx unchanged at L(5).
- **2026-08-04** — **Method spec substantially expanded** (Daniel + research) — `activation_splitting_method.md`
  now fully defines what was previously sketched. Six things land on this repo:
  1. **Sharp vs smoothed families.** (a)/(b) spike at every deflection so a fractionated complex gives
     several peaks; only (c) merges them at the `g` level. **Botteron is the default for the IAFDB
     train**, `dV/dt` fine for synthetic. My S2 docs called RectifiedDerivative "the default" — wrong,
     corrected in `preprocessors.py` and theory §2.4.
  2. **The envelope's timing bias** — smoothing drags the peak toward the heavier side of an
     asymmetric complex, i.e. *late* on a fibrotic tail. Now documented; motivates item 5.
  3. **Candidate extraction is one-per-segment** (step 3), with min-width / min-prominence filtering.
     This **bounds the S3 limitation I documented**: the 206-sample above-`τ` run collapses to one
     candidate at its argmax. Theory §3.4 updated to say so rather than leaving it open.
  4. **Suppression is a run-time-selectable strategy family**, with a five-algorithm menu and
     greedy-by-height as the Phase-1.5 default. New ABC family — the biggest S4 change.
  5. **Optional two-stage refinement** (step 5, off by default): snap `t_a` to the local `|dV/dt|`
     maximum to undo the envelope's smoothing bias.
  6. **`τ_det` and `Δ_refr` do different jobs and must not be traded** — amplitude vs time. Raising
     `τ` to suppress fractionation also removes genuine **low-voltage** activations, which are exactly
     the fibrotic regions of interest. Belongs in the theory §3.5 prose.
  **S4 re-scoped** 1.5–3 h → 3–5 h; **S5 clarified** as a study-time instrument, not a runtime filter,
  and now consumes S4's segment extents. Repo estimate 10–21 h → 12–24 h; **Cx stays L(5)** — the
  surface grew but the novelty (the dominant variance driver) did not, and it remains well below the
  XL anchor.
  **One open question raised, not decided** — see S4: the spec calls "local maxima above `τ`" and
  "argmax per contiguous above-`τ` segment" equivalent, and they are not, for a *sharp* `g`. Needs a
  ruling before S4 is built.
- **2026-08-04** — **CL-120 resolved in egm-signal's favour** (project-lead): the three-name split is
  adopted, **design §3 SIG1 reworded** to detection preprocessing → detection curve `g` → detection
  function (the whole chain). The method spec is research-owned, so its rewording is routed as
  **CL-121** (open, research), to land before SEP2/IAF1 adopt in Wave 2. No code change here —
  egm-signal already ships the vocabulary.
- **2026-08-04** — **Candidate extraction settled by research: local maxima + prominence**, with an
  explicit "don't" for segment-argmax, matching the fast-AF failure I measured. Knock-on: the S3
  comparison contract flips `>` → `≥` (the two are identical under local-maxima extraction, so the
  earlier justification no longer applies). Both recorded under S4.
- **2026-08-01** — Caught up on end-of-planning + Wave 1. Four things landed on this repo: (1) **effort
  tracking skipped** phase-wide (§6) — Effort section marked superseded; (2) **Wave 1 is serial** and
  SIG1 is **step 5**, up now that CLF5 is done (CL-116) — supersedes "parallel to Wave 1"; (3) my
  `realized_position` design note propagated into egm-contracts as the shared
  **`ActivationPosition`** `$def` `$ref`'d by both banks (CL-052 / CL-060 / CL-062 / CL-064), so S6 is
  now writing to a fleet contract rather than a local convention; (4) two fleet chores folded into the
  S0 session — CL-099 `.gitignore` root-anchoring (outstanding) and CL-117 `__version__` (already
  compliant). Context for later: §8.1 settled **T = 192 samples at 1 kHz** (`T ≡ 0 mod 64`, CLF3
  MobileViT), which is what my `T` arguments will carry downstream.
- **2026-07-29** — CL-024 §5 closed both of my flow-down loose ends: the QRS-default removal is
  **B22** (design §4), and **flow-down planning counts toward a repo's `Actual`**. SIG1-parallel-to-Wave-1
  confirmed. Applied above; the retroactive effort rows are reconstructed, not marked.
- **2026-07-29** — iafdb-pipeline replied on both handoffs. **CL-028** (S0): confirmed, planned as their
  S0/B22, gated only on my v0.3.0 tag — with the correction that *their* adoption is 1–2 h, not
  trivial, because 14 `export_bank(...)` test call sites rely on the signature default. My 0.5–1 h was
  always repo-local, so my estimate stands. **CL-029** (theory): scope + timing agreed, they annotate
  now and delete at my v0.4.0; two boundary details — don't carry the target *value* across (agreed,
  folded into S7a) and leave pool assembly with them (**disagreeing**, see S7a).
- **2026-07-29** — Both iafdb threads closed. **CL-033:** tag v0.3.0 whenever — don't hold it; S0 is
  unblocked. **CL-034:** pool assembly conceded to egm-signal, *and* they corrected their own CL-029
  claim: present-ness reflects **surface-ECG** variation, not bipolar, so `skip-absent` never fires on
  the bipolar path. I verified across all 32 IAFDB headers before adopting it — confirmed, and
  sharpened (3 surface leads per record, four distinct sets, no record with all eight). Facts folded
  into S7a. No escalation needed.
- **2026-07-28** — *Open, not escalated yet:* drawing `p ∼ 𝒫` currently sits with SEP2 and IAF1, so
  the **representation of `𝒫`** (a `(lo, hi)` fraction pair, collapsed to a point for the fixed-position
  baseline — the A4 decision, mirroring the mixer's `snr_db_range=(X,X)` idiom) is implemented twice.
  Same drift-class argument that moved the crop math here, but far lower stakes than the rounding
  case. Raise with the project-lead only if a second inconsistency shows up.
