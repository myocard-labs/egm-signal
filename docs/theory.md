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
> alongside Phase-1.5 SIG1. §1–§3.4 are complete; §3.5–§5 arrive with
> the steps named in their stubs. This doc is also the destination for the
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
- [3. Thresholding and the detection function](#3-thresholding-and-the-detection-function)
  - [3.1 Why the threshold is read off the curve](#31-why-the-threshold-is-read-off-the-curve)
  - [3.2 Median/MAD threshold](#32-medianmad-threshold)
  - [3.3 Percentile threshold](#33-percentile-threshold)
  - [3.4 What the threshold is applied to](#34-what-the-threshold-is-applied-to)
  - [3.5 Refractory suppression](#35-refractory-suppression) *(S4)*
- [4. Activation-complex bounds](#4-activation-complex-bounds) *(S5)*
- [5. Anchor windowing](#5-anchor-windowing) *(S6)*
- [6. References](#6-references)

## Notation

Every symbol used anywhere below, grouped by where it appears. Symbols
introduced by a section that has not landed yet are marked with the step
that brings them.

**Signal and indexing**

- $x[i]$, $i = 0,\dots,n-1$ — one channel of bipolar EGM, $n$ samples at
  sampling frequency $f_s$ Hz. **One channel**: every operator here is
  defined per-channel.
- $f_\text{Nyq} = f_s/2$ — the Nyquist frequency.
- $dV/dt$ — the signal's time derivative; $\lvert dV/dt\rvert$ its
  magnitude, approximated by the first difference.

**Filtering (§1)**

- $H(\omega)$ — a filter's frequency response at angular frequency
  $\omega$; $\overline{H(\omega)}$ its complex conjugate.
- $H_\text{fb}(\omega)$ — the *forward-backward* (two-pass) response,
  $\lvert H(\omega)\rvert^2$.
- $\lvert H(f)\rvert^2$ — magnitude-squared response at frequency $f$.
- $N$ — Butterworth filter order.
- $f_c$ — a filter's cutoff (corner) frequency.
- $f_\text{lo}$, $f_\text{hi}$ — band-pass lower and upper edges.
- $\varepsilon$ — an arbitrarily small positive number, used only to
  describe the degenerate band-pass we *reject* in §1.3.
- $\mathrm{BP}_{40\text{–}250}$, $\mathrm{LP}_{20}$ — a band-pass and a
  low-pass, subscripted with their cutoffs in Hz.

**Detection preprocessing (§2)**

- $g$, $g[i]$ — the **detection curve**: a same-length transform of $x$
  whose local maxima sit at activations.
- $A$, $\Omega$, $\phi$ — amplitude, digital angular frequency and phase
  of the test sinusoid used to derive what Teager–Kaiser measures
  (§2.2). $\Omega \ll 1$ denotes the low-frequency limit.
- $t_a$ — an **activation time**, as a sample index.
- $\arg\max$ — the index at which a curve attains its maximum.

**Thresholding (§3)**

- $\tau$ — the detection threshold. A **local maximum** of $g$ becomes a
  candidate when $g[i] \ge \tau$ (§3.4); it is not applied sample-wise.
- $c$ — multiplier on the baseline median.
- $\lambda$ — multiplier on the MAD.
- $\operatorname{median}(\cdot)$ — the median.
- $\operatorname{MAD}(g) = \operatorname{median}(\lvert g - \operatorname{median}(g)\rvert)$
  — the median absolute deviation, a robust scale estimate in raw units.
- $\sigma$ — the standard deviation, and $k$ its multiplier in the
  mean-plus-$k\sigma$ rule we *reject* in §3.2.
- $P_q(g)$ — the $q$-th percentile of $g$; $q \in (0, 100)$.
- $+\infty$ — the fail-closed sentinel returned for an empty or constant
  curve.
- $\Delta_\text{refr}$ — the refractory interval, the minimum spacing
  between distinct activations. *(§3.5, S4.)*

**Complex bounds (§4, S5)**

- $\theta$ — the envelope threshold at which onset and offset are read,
  either a fraction of the local peak or a multiple of the baseline MAD.
- $t_\text{on}$, $t_\text{off}$ — onset and offset sample indices.
- $r_\text{rise} = t_a - t_\text{on}$,
  $r_\text{fall} = t_\text{off} - t_a$ — the rising and falling
  durations of an activation complex.

**Anchor windowing (§5, S6)**

- $T$ — window length in samples.
- $p \in [0,1]$ — the activation's fractional position within the
  window; $0.0$ is the first sample, $1.0$ the last.
- $s = \lfloor t_a - p\,(T-1) \rceil$ — the window's start sample.
- $\lfloor\cdot\rceil$ — round to nearest integer.

**Conventions**

Amplitudes are whatever unit the caller supplies — nothing here assumes
mV. The operators are scale-equivariant and the thresholds are computed
from the distribution of $g$ itself, so a change of units passes through
without retuning.

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

The split that decides behaviour is **sharp vs smoothed**, because it
sets whether a *fractionated* activation reaches the next stage as one
peak or several.

| | family | cost | noise robustness | fractionation | timing |
|---|---|---|---|---|---|
| Rectified derivative | sharp | 1 subtract | lowest — differencing amplifies HF | **fragments** it | crispest |
| Teager–Kaiser | sharp | 3 multiplies | middling | **fragments** it | crisp |
| Botteron envelope | smoothed | 2 filter passes | highest | **merges** it | biased late on asymmetric complexes |

All three **gate on amplitude**; only the envelope also **merges
fractionation**, at the $g$ level, before any refractory logic runs.

**The method spec's defaults.** For the **IAFDB train** — real, noisy,
fractionated — the **Botteron envelope**, because the smoothing
pre-solves fractionation and leaves refractory suppression (§3.5) as a
safety net rather than the primary defence. For **synthetic**, where one
activation sits at a location the generator already knows, any of them
works and the rectified derivative gives the crispest timing; detection
there is a cross-check, not a measurement.

**The envelope's timing bias.** Smoothing pulls the envelope's maximum
toward the *heavier* side of an asymmetric complex, so a long
fractionated tail — the fibrotic case — drags the detected time late.
The crop absorbs a few milliseconds of anchor error, so a first pass can
live with it; the fix when precise timing matters is the optional
two-stage refinement, which snaps the accepted time back to the local
$\lvert dV/dt\rvert$ maximum (§3.5).

The three are **not interchangeable at a fixed threshold** — a
derivative curve is spiky and an envelope is broad, so the same numeric
cutoff means different things. This is precisely why §3's thresholds are
computed from the distribution of $g$ rather than supplied as absolute
numbers.

## 3. Thresholding and the detection function

Stage two: given the detection curve $g$, decide how prominent a sample
must be to count as a candidate. Stage three — refractory suppression,
which turns candidates into an activation train — lands with S4.

### 3.1 Why the threshold is read off the curve

A fixed number cannot work. The three preprocessors produce
differently-scaled curves, and EGM amplitude varies by an order of
magnitude across records, channels and patients, so any absolute level
would need retuning per combination. Both rules below are therefore
*adaptive*: they read their level from the curve they are given. Same
idea as constant-false-alarm-rate detection in radar — estimate the
background from the data, then sit a fixed distance above it.

### 3.2 Median/MAD threshold

$$
\tau = c\,\operatorname{median}(g) + \lambda\,\operatorname{MAD}(g),
\qquad
\operatorname{MAD}(g) = \operatorname{median}\bigl(\lvert g - \operatorname{median}(g)\rvert\bigr).
$$

A robust analogue of "mean plus $k$ standard deviations": the median
locates the baseline, the MAD measures its spread, $\lambda$ sets how far
above it to sit.

**Why MAD and not $\sigma$** — the load-bearing choice. The curve
*contains the activations being detected*, and in it they are large
outliers. The standard deviation has a **breakdown point of zero**: one
arbitrarily large sample moves it arbitrarily far. So a $\sigma$-based
threshold rises with the very events it is meant to find — a channel
with more or larger activations gets a higher bar and detects a smaller
fraction of them, which is exactly backwards. The MAD has a breakdown
point of **50%**: up to half the samples may be arbitrary without moving
it. Activations occupy a few percent of a record, so the MAD measures
the baseline, which is what a detection threshold should reference.

Measured on a 2000-sample curve, going from 5 activations to 200:

| rule | $\tau$ at 5 | $\tau$ at 200 | drift |
|---|---|---|---|
| $c\cdot\mathrm{median} + \lambda\cdot\mathrm{MAD}$ | 0.53 | 0.62 | **x1.16** |
| $\mathrm{mean} + k\,\sigma$ | 12.7 | 79.9 | x6.3 |

$\lambda$ is in **raw MAD units**. For Gaussian data
$\sigma \approx 1.4826\,\mathrm{MAD}$, so multiply by that to read it as a
sigma multiple. Neither $c$ nor $\lambda$ has a default: how aggressive
detection should be is a policy the calling pipeline owns.

**$\mathrm{MAD} = 0$ is not an error.** It means over half the samples
share one value — equally true of a dead channel *and* of a clean
synthetic trace that is exactly flat between activations. The two are
indistinguishable from $g$ alone, so it is deliberately not
special-cased: the formula degrades to $\tau = c\cdot\mathrm{median}(g)$,
which for a zero baseline is $\tau = 0$. That is where the comparison
convention earns its keep (§3.4).

### 3.3 Percentile threshold

$$
\tau = P_q(g).
$$

Fixes the candidate *rate* rather than their prominence: at $q = 99$ at
most 1% of samples clear the bar, whatever the curve looks like.
Predictable and shape-independent, which suits a first pass on an
unfamiliar record. Its weakness mirrors that strength — it always
promotes something, so a channel with no activations still yields its
top 1%. An exploration tool, not a production splitter.

### 3.4 What the threshold is applied to

**Local maxima, not every sample.** Candidate extraction takes the points
where

$$
g[i-1] < g[i] > g[i+1] \quad\text{and}\quad g[i] \ge \tau ,
$$

then filters them by a minimum **prominence** to drop noise blips and
far-field. A fractionated complex may contribute several candidates, and
that is intended — deciding which are *distinct activations* is the
refractory step's job (§3.5), on the time axis.

**Why not one candidate per above-$\tau$ segment.** The tempting
simplification is to collapse each contiguous above-$\tau$ run to its
single $\arg\max$. The method spec explicitly rules this out, and
measurement shows why: it lets the *threshold* do the merging, so two
genuine activations riding **one** above-$\tau$ run — common in fast AF,
when the envelope never fully returns to baseline between beats —
collapse into one peak, a silent miss that no $\Delta_\text{refr}$
tuning can recover.

Measured, two genuine activations 80–90 ms apart:

| $f_c$ (envelope) | above-$\tau$ segments | widest | segment-argmax | local maxima |
|---|---|---|---|---|
| 20 Hz | 2 | 49 | 2 activations | 2 activations |
| 10 Hz | 1 | **168** | **1 — one lost** | 2 activations |
| 6 Hz | 1 | **232** | **1 — one lost** | 2 activations |

With $\Delta_\text{refr} = 60$ samples, those runs span three to four
refractory intervals. The above-$\tau$ segment *is* still used — it is
the $W_\text{act}$ that §4 measures onset and offset within — just not
as the merge rule.

**On $\ge$ versus $>$.** The comparison is at-or-above, matching the
method spec and `scipy.signal.find_peaks(height=...)`. An earlier
revision of this document argued for strictly-greater, on the grounds
that a clean sparse curve has $\tau = 0$ and $\ge$ would admit all 1000
baseline samples where $>$ admitted 8. That reasoning was sound *for
sample-wise thresholding*, which is not the extraction rule: a flat
baseline contains no strict local maximum, so under local-maxima
extraction the two comparisons give **identical** results (verified: 8
and 8 on that curve; 4 and 4 on a clean Gabor pulse). The comparison
simply is not load-bearing here.

Two degenerate curves never reach a rule at all, both returning
$+\infty$ (fail-closed, matching the empty-pool sentinel convention): an
**empty** curve, and a **constant** one, which has no peaks to separate.

> **Implementation** — `thresholds/detection.py`; ABC in
> `thresholds/base.py`; candidate extraction in
> `extraction/activation_based/` (S4).
> **Pinned by** `tests/test_detection_thresholds.py`, including the
> breakdown-point contrast and the local-maxima equivalence.
> **Sources** — [Hampel 1974, JASA 69(346):383-393](https://doi.org/10.1080/01621459.1974.10482962)
> (the influence curve and breakdown point, the formal basis for
> preferring MAD) ·
> [Leys et al. 2013, J Exp Soc Psychol 49(4):764-766](https://doi.org/10.1016/j.jesp.2013.03.013)
> (a short, readable argument for median/MAD over mean/SD, with the
> 1.4826 consistency constant) ·
> [Pan & Tompkins 1985, IEEE TBME BME-32(3):230-236](https://doi.org/10.1109/TBME.1985.325532)
> (the local-peaks → threshold → refractory chain this follows).

### 3.5 Refractory suppression

*Lands with S4, completing the chain.* Will cover non-maximum
suppression over a refractory interval $\Delta_\text{refr}$, and its
two-sided failure mode: splitting one fractionated activation into
several, versus merging two genuine ones.

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

### Event detection

- **Pan J, Tompkins WJ.** *A Real-Time QRS Detection Algorithm.* IEEE
  Transactions on Biomedical Engineering BME-32(3):230–236, 1985.
  [[DOI]](https://doi.org/10.1109/TBME.1985.325532) — the
  local-peaks → threshold → refractory chain this detection function
  follows, and the origin of refractory blanking as a suppression rule.

### Robust statistics

- **Hampel FR.** *The Influence Curve and its Role in Robust
  Estimation.* Journal of the American Statistical Association
  69(346):383–393, 1974.
  [[DOI]](https://doi.org/10.1080/01621459.1974.10482962) — the
  influence curve and breakdown point; the formal basis for preferring
  the MAD over the standard deviation on data containing the outliers
  you are trying to find.
- **Leys C, Ley C, Klein O, Bernard P, Licata L.** *Detecting outliers:
  Do not use standard deviation around the mean, use absolute deviation
  around the median.* Journal of Experimental Social Psychology
  49(4):764–766, 2013.
  [[DOI]](https://doi.org/10.1016/j.jesp.2013.03.013) — short and
  readable, and the source of the 1.4826 consistency constant.

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
