# Theory — myocard-egm-signal

The math behind the primitives this library ships: the exact discrete
form of each operator, why it is defined that way rather than an
obvious alternative, and a linked primary source for every technique.

**Who this is for.** The person reviewing an implementation and asking
"is this computing the right thing?" Each section therefore gives the
formula *as implemented*, names the module it lives in, and points at
the test that pins it — so a claim here can be checked against code in
one hop, in either direction.

For call signatures and worked examples see [`usage.md`](usage.md); for
why the package is shaped the way it is (Protocols, strategy families,
sentinel conventions) see
[`project/architecture.md`](../project/architecture.md).

> **Status: written incrementally.** Sections land as their code does,
> alongside Phase-1.5 SIG1. §1–§2 are complete; §3–§5 arrive with the
> steps named in their stubs. This doc is also the destination for the
> egm-signal math currently parked in `iafdb-pipeline/docs/theory.md`
> §1.1–1.3 / §2.1 — the repo that owns a primitive owns its math — so
> some sections graduate content rather than deriving it fresh.

> **Rendering note.** Equations are LaTeX — `$$…$$` display, `$…$`
> inline. GitHub and VS Code typeset these; a plain-text viewer shows
> the source. Backticked names (`fs`, `sosfiltfilt`) are code
> identifiers, not math symbols.

## Table of contents

- [Notation](#notation)
- [1. Filtering](#1-filtering)
  - [1.1 Zero-phase filtering](#11-zero-phase-filtering)
  - [1.2 Band-pass](#12-band-pass)
  - [1.3 Low-pass](#13-low-pass)
- [2. Detection preprocessing](#2-detection-preprocessing)
  - [2.1 Rectified derivative](#21-rectified-derivative)
  - [2.2 Teager–Kaiser energy](#22-teagerkaiser-energy)
  - [2.3 Botteron envelope](#23-botteron-envelope)
  - [2.4 Choosing between them](#24-choosing-between-them)
- [3. Thresholding and the detection function](#3-thresholding-and-the-detection-function) *(S3–S4)*
- [4. Activation-complex bounds](#4-activation-complex-bounds) *(S5)*
- [5. Anchor windowing](#5-anchor-windowing) *(S6)*
- [6. References](#6-references)

## Notation

- $x[i]$, $i = 0,\dots,n-1$ — one channel of bipolar EGM, $n$ samples at
  sampling frequency $f_s$ Hz. **One channel**: every operator below is
  defined per-channel.
- $f_\text{Nyq} = f_s/2$ — the Nyquist frequency.
- $g[i]$ — the **detection curve**: a same-length transform of $x$
  produced by a detection preprocessor (§2), whose local maxima sit at
  activations.
- $t_a$ — an **activation time**, as a sample index.
- $\lfloor\cdot\rceil$ — round to nearest integer.

Amplitudes are whatever unit the caller supplies. Nothing here assumes
mV: the operators are scale-equivariant, and the thresholds that consume
them (§3) are computed from the distribution of $g$ itself.

## 1. Filtering

### 1.1 Zero-phase filtering

Both filters are applied with `scipy.signal.sosfiltfilt`, which runs the
filter **forward, then backward** over the reversed signal. If the
one-directional filter has frequency response $H(\omega)$, the
two-pass response is

$$
H_\text{fb}(\omega) = H(\omega)\,\overline{H(\omega)} = |H(\omega)|^2 ,
$$

which is **real and non-negative** — so its phase is identically zero
and no feature is displaced in time. Two consequences we depend on:

1. **Magnitude is squared.** A two-pass order-2 Butterworth attenuates
   like a one-pass order-4. The effective roll-off is twice the nominal
   order; do not read the `order` argument as the achieved slope.
2. **The result is non-causal.** Output at sample $i$ depends on future
   samples. Irrelevant offline, disqualifying for real-time use.

Zero phase is not a nicety here — it is load-bearing. §4 reads onset and
offset off a smoothed envelope as threshold crossings, and §5 places a
window relative to a detected peak. Any group delay would bias *every*
activation boundary and *every* window position by the same amount, in a
way no downstream step could detect.

Edge behavior is `sosfiltfilt`'s Gustafsson initial-state method, which
chooses initial conditions so the forward and backward passes agree at
the boundaries.

> **Implementation** — `filters/bandpass.py`, `filters/lowpass.py`.
> **Pinned by** `tests/test_filters.py::test_lowpass_is_zero_phase`
> (a Gaussian bump's argmax must not move).
> **Sources** — [Gustafsson 1996](https://doi.org/10.1109/78.492552) ·
> [`scipy.signal.sosfiltfilt`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html).

### 1.2 Band-pass

A Butterworth band-pass of order $N$ with edges $(f_\text{lo},
f_\text{hi})$, realized as second-order sections. The Butterworth
magnitude response is maximally flat in-band:

$$
|H(f)|^2 = \frac{1}{1 + \left(f/f_c\right)^{2N}} .
$$

Maximal flatness is why it is the right family here: an EGM's diagnostic
content is its *morphology*, and a filter with in-band ripple (Chebyshev)
or a non-monotonic magnitude would distort the very deflection shapes the
downstream features measure.

The high edge is clamped to $0.99\,f_\text{Nyq}$ before design. This is a
guard against a real and easy caller error — a band copied from a
1 kHz-rate context applied to a lower-rate signal — which would otherwise
produce a normalized frequency $\ge 1$ and a raised exception deep in
`butter`.

The **30–300 Hz** default used for bipolar EGM extraction is a clinical
convention, not a property of this library; it lives in
`extraction.DEFAULT_BIPOLAR_BAND_HZ`. The function itself is
band-agnostic.

> **Implementation** — `filters/bandpass.py`.
> **Sources** — [Butterworth 1930](https://www.changpuak.ch/electronics/downloads/On_the_Theory_of_Filter_Amplifiers.pdf) (the original derivation) ·
> [`scipy.signal.butter`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html).
> Clinical band: Sánchez et al. 2021 / Unger et al. 2019 / Deno et al. 2017.

### 1.3 Low-pass

Same family, one corner:

$$
|H(f)|^2 = \frac{1}{1 + \left(f/f_c\right)^{2N}} ,
$$

with the cutoff clamped to $0.99\,f_\text{Nyq}$ as above.

**Why it is a separate primitive** and not `bandpass` with a tiny low
edge: the low-pass exists to smooth a **rectified** signal (§2.3), and a
rectified signal's energy is concentrated at and near DC — that is the
whole point of rectifying. A band-pass with $f_\text{lo} = \varepsilon$
would place a high-pass corner exactly there and remove the mean the
envelope is built from.

**What the smoothing does.** For a rectified signal, low-passing is
approximately a local average over a window of width $\sim f_s/f_c$
samples. This is the mechanism that makes a **fractionated** activation
readable as one event: the raw rectified signal dips to near zero between
sub-deflections, but if those dips are narrower than the averaging width
they are filled in, and the smoothed curve stays above a threshold across
the whole complex. Consequently $f_c$ is the knob that decides *how much
fractionation counts as one activation* — lower merges more aggressively,
at the cost of broadening the complex and pushing measured onset earlier
and offset later (§4).

> **Implementation** — `filters/lowpass.py`.
> **Pinned by** `tests/test_filters.py::test_lowpass_bridges_a_fractionated_complex`
> (three deflections → exactly one above-threshold run; the raw rectified
> signal fragments).

## 2. Detection preprocessing

### Vocabulary: what actually detects

Activation detection is a **chain**, and it is worth being strict about
which stage is which — the three are routinely conflated, including in
this project's own method spec:

| stage | what it does | where |
|---|---|---|
| **Detection preprocessing** | transform $x \to g$ so activations are emphasised | §2.2–§2.4 |
| **Detection thresholding** | the decision rule that turns $g$ into candidates | §3 |
| **The detection function** | the whole chain, including refractory suppression | §3 |

A preprocessor **detects nothing**. It decides what "prominent" will
mean to the next stage, and that is all. Only the full chain answers
"where are the activations?"

> The platform method spec
> (`activation_splitting_method.md`) and design §3 use "detection
> function" for the *transform*. That usage is being corrected; the term
> is kept for the whole chain, which is what it properly names.

### The shared contract

All three preprocessors guarantee: 1-D input; output length equal to
input length (so an index into $g$ *is* an index into $x$); positions
where the operator is undefined filled with $0.0$; and an all-zero
result for input shorter than the operator needs. That last one is
fail-closed — nothing crosses a positive threshold, so a degenerate
channel detects nothing.

The contract is enforced in an abstract base class rather than
documented in a Protocol, because it is a closed family we ship and
expect to extend. `compute()` is a **template method** — it validates,
applies the short-input rule from each subclass's `min_samples`, then
delegates to `_compute()`. A new preprocessor therefore cannot forget
the contract, and the failure mode this replaced was real: the envelope
used to raise from inside `sosfiltfilt` on short input where the other
two returned zeros.

**`compute()` takes only the signal.** Two of the three operators are
pure sample-domain arithmetic — $|x[i]-x[i-1]|$ and
$x[i]^2 - x[i-1]x[i+1]$ contain no time constant — so they are
*rate-independent*, and the same input gives the same $g$ at any
sampling rate. Only the Botteron envelope is specified in Hz, and it
takes $f_s$ at construction alongside the band edges it belongs with.
Passing a rate to the other two would imply a dependence the math does
not have. The trade-off: an envelope instance is bound to one sampling
rate, and since a numpy array carries no rate, a mismatch cannot be
detected — construct it where the rate is known.

> **Implementation** — `extraction/activation_based/preprocessors.py`;
> ABC in `extraction/activation_based/base.py`.
> **Pinned by** `tests/test_detection_preprocessors.py`.

### 2.1 Rectified derivative

$$
g[i] = \bigl|\, x[i] - x[i-1] \,\bigr|, \qquad g[0] = 0 .
$$

The discrete $|dV/dt|$, and the default. It expresses the convention
that local activation occurs at the **steepest deflection**. That
convention has a real theoretical basis, but read the fine print:
Steinhaus 1989 shows for a **unipolar** electrogram that the intrinsic
deflection — the time of maximum *negative* slope of the QRS complex —
coincides with the maximum positive $dV/dt$ of the underlying
transmembrane action potential, i.e. with depolarization. Under uniform
propagation the two agree to well under a millisecond.

Two caveats that matter for us, both from the same paper:

- **Our signals are bipolar, not unipolar.** The theorem is unipolar.
  Steinhaus found bipolar timing errors comparable to unipolar in
  general, and *smaller* when distant events contribute — reassuring,
  but applying the max-slope rule to a bipolar trace remains a
  convention rather than a result.
- **Non-ideal propagation breaks the correspondence.** Activation-sequence
  changes and non-uniform coupling produced timing errors in excess of
  1.8 ms. Fibrotic substrate is exactly non-uniform coupling — so on the
  tissue we care most about, the anchor is least theoretically anchored.

Peak *amplitude* is the common alternative and is cheaper still, but it
tracks the post-activation extremum rather than the activation event.

$g[0] = 0$ rather than a replicated value: a first difference is

$g[0] = 0$ rather than a replicated value: a first difference is
undefined at the first sample, and zero cannot become a spurious
$\arg\max$ at the boundary.

This is the same convention `egm-features.activation_position` uses, so
an anchor placed here and a position measured there agree by
construction — deliberate, since Phase 1.5 compares stored anchor
positions against feature-derived ones.

**Failure mode to know:** differencing is a high-pass with gain rising
linearly in frequency, so broadband noise is amplified. On noisy or
fractionated signal this is the least reliable of the three, which is
exactly why the other two exist.

> **Pinned by** `test_rectified_derivative_is_the_exact_first_difference`
> (hand-computed case).
> **Source** — [Steinhaus 1989, Circ Res 64(3):449–462](https://doi.org/10.1161/01.RES.64.3.449)
> for the max-slope ↔ depolarization correspondence and its error bounds.

### 2.2 Teager–Kaiser energy

$$
g[i] = x[i]^2 - x[i-1]\,x[i+1], \qquad g[0] = g[n-1] = 0 .
$$

The discrete Teager–Kaiser energy operator (TKEO). For a sampled
sinusoid $x[i] = A\cos(\Omega i + \phi)$ it evaluates to

$$
g[i] = A^2 \sin^2(\Omega) \;\approx\; A^2\Omega^2 \quad (\Omega \ll 1),
$$

i.e. it tracks the product of amplitude **and** frequency, unlike a
plain squaring which sees amplitude only. That is the property worth
having on EGM: a low-voltage but *sharp* deflection — the fibrotic case —
scores comparably to a large slow one, so it is not lost under an
amplitude-only measure. Cost is three multiplies per sample and a
three-sample support, so it stays local.

**Returned unclipped.** The operator can go negative where the signal is
not locally narrowband (the derivation above assumes a single component).
Many implementations clip at zero; this one does not, because clipping is
a policy decision belonging to whatever consumes $g$, $\arg\max$ is
unaffected, and the median/MAD thresholds of §3 take the distribution of
$g$ as it is. If you want a rectified variant, clip at the call site.

> **Pinned by** `test_teager_kaiser_is_the_exact_three_point_operator`
> and `test_teager_kaiser_is_not_clipped`.
> **Sources** — [Kaiser 1990, ICASSP, pp. 381–384](https://doi.org/10.1109/ICASSP.1990.115702)
> (the original algorithm) ·
> [Maragos, Kaiser & Quatieri 1993, IEEE TSP 41(4):1532–1550](https://doi.org/10.1109/78.212729)
> (the AM–FM demodulation theory that explains *why* it measures $A^2\Omega^2$) ·
> [Non-linear energy operator for intracardial electrograms, IFMBE 2009](https://link.springer.com/chapter/10.1007/978-3-642-03882-2_233)
> (TKEO applied to EGM / CFAE detection specifically).

### 2.3 Botteron envelope

$$
g = \mathrm{LP}_{20}\Bigl(\bigl|\,\mathrm{BP}_{40\text{–}250}(x)\,\bigr|\Bigr) .
$$

Three stages, each doing one job:

1. **Band-pass 40–250 Hz** — isolate the activation. Deliberately
   narrower than the 30–300 Hz morphology band: here we are detecting an
   *event*, not preserving a waveform, so trimming both ends removes
   baseline wander below and noise above.
2. **Rectify** — $|\cdot|$ discards polarity. An activation is biphasic
   or polyphasic and its polarity depends on wavefront direction
   relative to the electrode pair, which carries no information about
   *whether* activation happened.
3. **Low-pass 20 Hz** — smooth the rectified signal into an envelope
   (§1.3).

**The property that matters.** A fractionated complex is *one*
activation composed of several deflections. In the raw signal it dips to
baseline between them; a naive threshold therefore starts and stops
several times and the detector reports several activations where there is
one. The low-pass bridges those dips, so the envelope makes a single
up/down-crossing pair around the whole complex. This is what makes the
refractory suppression of §3 a safety net rather than the primary defence,
and it is what makes onset/offset (§4) measurable at all.

**The cost is a bias, and it is directional.** Smoothing broadens: the
envelope rises before the first deflection and decays after the last, so
measured onset is early and offset is late, by roughly the averaging
width. The alternative design — threshold a lightly-smoothed curve, then
*merge* segments closer than some gap — trades this bias for an extra
parameter. We take the smoothing, because $f_c$ is one knob and the
envelope is needed anyway.

> **Pinned by** `test_botteron_bridges_fractionation_where_the_derivative_does_not`
> and `test_botteron_lowpass_controls_how_much_is_bridged`.
> **Sources** — [Botteron & Smith 1995, IEEE TBME 42(6):579–586](https://doi.org/10.1109/10.387197)
> (the band-pass/rectify/low-pass envelope) ·
> [Botteron & Smith 1996, Circulation 93(3):513–518](https://pubmed.ncbi.nlm.nih.gov/8565169/)
> (the clinical companion) ·
> [Hodges & Bui 1996](https://pubmed.ncbi.nlm.nih.gov/9020824/)
> (threshold-on-smoothed-rectified-signal as a general onset test, and
> its smoothing-induced bias; open-access re-derivation in the
> [EMG-onset review, J NeuroEng Rehabil 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10594734/)).

### 2.4 Choosing between them

| | cost | noise robustness | fractionation | use when |
|---|---|---|---|---|
| Rectified derivative | 1 subtract | lowest — differencing amplifies HF | fragments it | clean signal; agreement with `egm-features` positions matters |
| Teager–Kaiser | 3 multiplies | middling | fragments it | amplitude alone misleads (sharp low-voltage deflections) |
| Botteron envelope | 2 filter passes | highest | **merges it** | real AF electrograms; whenever peak *counting* must be right |

The three are **not interchangeable at a fixed threshold** — a
derivative curve is spiky and an envelope is broad, so the same numeric
cutoff means different things. This is precisely why §3's thresholds are
computed from the distribution of $g$ rather than supplied as absolute
numbers.

## 3. Thresholding and the detection function

*Lands with S3 (adaptive thresholds) and S4 (refractory non-maximum
suppression, completing the chain).* Will cover: the median/MAD adaptive threshold
$\tau = c\cdot\mathrm{median}(g) + \lambda\cdot\mathrm{MAD}(g)$ and why
MAD rather than standard deviation; the fail-closed $+\infty$ sentinel
when $\mathrm{MAD}(g) = 0$; and refractory non-maximum suppression with
its two-sided failure mode (splitting one fractionated activation vs
merging two genuine ones).

## 4. Activation-complex bounds

*Lands with S5.* Will cover: onset/offset as outward threshold crossings
on the envelope, the choice of $\theta$ (fraction-of-peak vs a multiple
of baseline MAD), the resulting rise/fall durations, and the smoothing
bias inherited from §1.3.

## 5. Anchor windowing

*Lands with S6.* Will cover: $s = \lfloor t_a - p\,(T-1) \rceil$, the
$[0,1]$ fractional-position convention shared fleet-wide as
`ActivationPosition`, requested-vs-realized position under rounding, and
the boundary and multi-beat predicates.

## 6. References

Primary sources for every technique above, linked. Where a DOI exists it
is the link; otherwise PubMed/PMC or the publisher page.

### Filtering

- **Butterworth S.** *On the theory of filter amplifiers.* Experimental
  Wireless & the Wireless Engineer, 7:536–541, 1930.
  [[PDF — third-party scan]](https://www.changpuak.ch/electronics/downloads/On_the_Theory_of_Filter_Amplifiers.pdf)
  — the maximally-flat design used for both filters. The 1930 journal
  predates DOIs; this link is an unofficial scan, unverified.
- **Gustafsson F.** *Determining the initial states in forward-backward
  filtering.* IEEE Trans Signal Processing 44(4):988–992, 1996.
  [[DOI]](https://doi.org/10.1109/78.492552) — the edge-handling method
  behind `sosfiltfilt`'s zero-phase result.

### Activation detection

- **Kaiser JF.** *On a simple algorithm to calculate the 'energy' of a
  signal.* ICASSP 1990, pp. 381–384.
  [[DOI]](https://doi.org/10.1109/ICASSP.1990.115702) — the TKEO.
- **Maragos P, Kaiser JF, Quatieri TF.** *On amplitude and frequency
  demodulation using energy operators.* IEEE Trans Signal Processing
  41(4):1532–1550, 1993. [[DOI]](https://doi.org/10.1109/78.212729) —
  why TKEO tracks $A^2\Omega^2$.
- **Botteron GW, Smith JM.** *A technique for measurement of the extent
  of spatial organization of atrial activation during atrial
  fibrillation in the intact human heart.* IEEE Trans Biomed Eng
  42(6):579–586, 1995. [[DOI]](https://doi.org/10.1109/10.387197) — the
  band-pass/rectify/low-pass envelope.
- **Botteron GW, Smith JM.** *Quantitative Assessment of the Spatial
  Organization of Atrial Fibrillation in the Intact Human Heart.*
  Circulation 93(3):513–518, 1996.
  [[DOI]](https://doi.org/10.1161/01.CIR.93.3.513)
  [[PubMed]](https://pubmed.ncbi.nlm.nih.gov/8565169/) — clinical
  companion to the above.
- **Hodges PW, Bui BH.** *A comparison of computer-based methods for the
  determination of onset of muscle contraction using electromyography.*
  Electroencephalogr Clin Neurophysiol 101:511–519, 1996.
  [[PubMed]](https://pubmed.ncbi.nlm.nih.gov/9020824/) — the canonical
  statement of threshold-on-smoothed-rectified onset detection, and it
  quantifies the smoothing-induced onset bias. Paywalled; the method is
  re-derived in the open-access
  [EMG-onset review, J NeuroEng Rehabil 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10594734/).
- **Non-Linear Energy Operator for the Analysis of Intracardial
  Electrograms.** IFMBE Proceedings, 2009.
  [[Springer]](https://link.springer.com/chapter/10.1007/978-3-642-03882-2_233)
  — TKEO applied to EGM / CFAE detection, i.e. this domain.

### Activation timing

- **Steinhaus BM.** *Estimating cardiac transmembrane activation and
  recovery times from unipolar and bipolar extracellular electrograms: a
  simulation study.* Circ Res 64(3):449–462, 1989.
  [[DOI]](https://doi.org/10.1161/01.RES.64.3.449) — the basis for
  max-slope-as-activation-time, **and** the error bounds under
  non-ideal propagation that §2.1 warns about.

### Clinical conventions

- **Marchlinski FE, Callans DJ, Gottlieb CD, Zado E.** *Linear Ablation
  Lesions for Control of Unmappable Ventricular Tachycardia in Patients
  With Ischemic and Nonischemic Cardiomyopathy.* Circulation
  101(11):1288–1296, 2000.
  [[DOI]](https://doi.org/10.1161/01.CIR.101.11.1288) — the bipolar
  **voltage-amplitude** tiers (>1.5 mV normal, <0.5 mV dense scar). Cited
  fleet-wide for thresholds; *not* a source for activation timing.
- The 30–300 Hz bipolar extraction band (Sánchez 2021 / Unger 2019 /
  Deno 2017) is recorded in the project bibliography at
  `intracardiac-platform/references/`; it is a consumer-side convention,
  not a property of these filters.

### Implementation references

- [`scipy.signal.butter`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html)
  · [`scipy.signal.sosfiltfilt`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html)

### Project context

- Method spec driving §2–§5:
  `intracardiac-platform/project/investigations/activation_splitting_method.md`.
- Sibling theory docs: `egm-features/docs/theory.md` (per-trace feature
  math, including the `activation_position` convention §2.1 matches);
  `egm-classifier/docs/theory.md` (preprocessing, metrics, calibration).
