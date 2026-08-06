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
> alongside Phase-1.5 SIG1. §1–§4 and §5.1–§5.3 are implemented;
> **§5.4–§5.7 are written ahead of the code** and specify S6b. This doc
> is also the destination for the egm-signal math currently parked in
> `iafdb-pipeline/docs/theory.md` §1.1–1.3 / §2.1 — the repo that owns a
> primitive owns its math — so some sections graduate content rather
> than deriving it fresh.

> **Rendering note.** Equations are LaTeX — `$$…$$` display, `$…$`
> inline. GitHub and VS Code typeset these; a plain-text viewer shows
> the source. Backticked names (`fs`, `sosfiltfilt`) are code
> identifiers, not math symbols. A symbol rendered as a **link** jumps
> to its entry in [Notation](#notation) — the first use in each section
> is linked, so you never have to scroll up guessing.

## Table of contents

- [Notation](#notation)
- [1. Filtering](#1-filtering)
  - [1.1 Zero-phase filtering](#11-zero-phase-filtering)
  - [1.2 Band-pass](#12-band-pass)
  - [1.3 Low-pass](#13-low-pass)
  - [1.4 Decimation](#14-decimation)
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
  - [3.5 Candidate extraction, suppression, and the chain](#35-candidate-extraction-suppression-and-the-chain)
- [4. Activation-complex bounds](#4-activation-complex-bounds)
  - [4.1 Onset, offset, and the durations](#41-onset-offset-and-the-durations)
  - [4.2 Choosing theta](#42-choosing-theta)
  - [4.3 Relationship to the above-tau segment](#43-relationship-to-the-above-tau-segment)
  - [4.4 A study-time criterion, not a runtime filter](#44-a-study-time-criterion-not-a-runtime-filter)
- [5. Anchor windowing](#5-anchor-windowing)
  - [5.1 Placing the window](#51-placing-the-window)
  - [5.2 Why a fraction and not a sample offset](#52-why-a-fraction-and-not-a-sample-offset)
  - [5.3 Requested versus realized position](#53-requested-versus-realized-position)
  - [5.4 The position range](#54-the-position-range)
  - [5.5 Windowing a train](#55-windowing-a-train)
  - [5.6 Classification, not dropping](#56-classification-not-dropping)
  - [5.7 The drop is not position-neutral](#57-the-drop-is-not-position-neutral)
  - [5.8 Three rules this module exists to enforce](#58-three-rules-this-module-exists-to-enforce)
- [6. References](#6-references)

## Notation

Every symbol used anywhere below, grouped by where it appears.

**Signal and indexing**

- <a id="sym-x"></a>$x[i]$, $i = 0,\dots,n-1$ — one channel of bipolar EGM, $n$ samples at
  sampling frequency $f_s$ Hz. **One channel**: every operator here is
  defined per-channel.
- <a id="sym-fnyq"></a>$f_\text{Nyq} = f_s/2$ — the Nyquist frequency.
- <a id="sym-dvdt"></a>$dV/dt$ — the signal's time derivative; $\lvert dV/dt\rvert$ its
  magnitude, approximated by the first difference.

**Filtering (§1)**

- <a id="sym-H"></a>$H(\omega)$ — a filter's frequency response at angular frequency
  $\omega$; $\overline{H(\omega)}$ its complex conjugate.
- <a id="sym-Hfb"></a>$H_\text{fb}(\omega)$ — the *forward-backward* (two-pass) response,
  $\lvert H(\omega)\rvert^2$.
- $\lvert H(f)\rvert^2$ — magnitude-squared response at frequency $f$.
- <a id="sym-N"></a>$N$ — Butterworth filter order.
- <a id="sym-fc"></a>$f_c$ — a filter's cutoff (corner) frequency.
- <a id="sym-band"></a>$f_\text{lo}$, $f_\text{hi}$ — band-pass lower and upper edges.
- <a id="sym-eps"></a>$\varepsilon$ — an arbitrarily small positive number, used only to
  describe the degenerate band-pass we *reject* in §1.3.
- <a id="sym-filters"></a>$\mathrm{BP}_{40\text{–}250}$, $\mathrm{LP}_{20}$ — a band-pass and a
  low-pass, subscripted with their cutoffs in Hz.
- <a id="sym-M"></a>$M$ — the integer decimation factor (§1.4); the rate becomes
  $f_s/M$ and $\alpha$ is the anti-alias cutoff as a fraction of the new
  Nyquist.

**Detection preprocessing (§2)**

- <a id="sym-g"></a>$g$, $g[i]$ — the **detection curve**: a same-length transform of $x$
  whose local maxima sit at activations.
- <a id="sym-AOmega"></a>$A$, $\Omega$, $\phi$ — amplitude, digital angular frequency and phase
  of the test sinusoid used to derive what Teager–Kaiser measures
  (§2.2). $\Omega \ll 1$ denotes the low-frequency limit.
- <a id="sym-ta"></a>$t_a$ — an **activation time**, as a sample index.
- <a id="sym-argmax"></a>$\arg\max$ — the index at which a curve attains its maximum.

**Thresholding (§3)**

- <a id="sym-tau"></a>$\tau$ — the detection threshold. A **local maximum** of $g$ becomes a
  candidate when $g[i] \ge \tau$ (§3.4); it is not applied sample-wise.
- <a id="sym-c"></a>$c$ — multiplier on the baseline median.
- <a id="sym-lambda"></a>$\lambda$ — multiplier on the MAD.
- <a id="sym-median"></a>$\operatorname{median}(\cdot)$ — the median.
- <a id="sym-mad"></a>$\operatorname{MAD}(g) = \operatorname{median}(\lvert g - \operatorname{median}(g)\rvert)$
  — the median absolute deviation, a robust scale estimate in raw units.
- <a id="sym-sigma"></a>$\sigma$ — the standard deviation, appearing in the
  mean-plus-$k\sigma$ rule we *reject* in §3.2.
- <a id="sym-k"></a>$k$ — the multiplier on $\sigma$ in the mean-plus-$k\sigma$ rule we
  *reject* in §3.2. Used only there; the MAD multiplier everywhere else,
  including the §4.2 boundary level, is $\lambda$, because both are the
  same `MedianMadThreshold(c, lam)`.
- <a id="sym-Pq"></a>$P_q(g)$ — the $q$-th percentile of $g$; $q \in (0, 100)$.
- <a id="sym-refr"></a>$\Delta_\text{refr}$ — the refractory interval, the minimum spacing
  between distinct activations. *(§3.5, S4.)*

**Complex bounds (§4, S5)**

- <a id="sym-theta"></a>$\theta$ — the envelope threshold at which onset and offset are read,
  either a fraction of the local peak or a level above the baseline MAD.
- <a id="sym-f"></a>$f \in (0,1)$ — the peak fraction in
  $\theta = f \cdot g[t_a]$.
- <a id="sym-onoff"></a>$t_\text{on}$, $t_\text{off}$ — onset and offset sample indices.
- <a id="sym-rise"></a>$r_\text{rise} = t_a - t_\text{on}$,
  $r_\text{fall} = t_\text{off} - t_a$ — the rising and falling
  durations of an activation complex.
- <a id="sym-Wact"></a>$W_\text{act} = r_\text{rise} + r_\text{fall}$ — the total width of
  an activation complex.
- <a id="sym-radius"></a>$R_\text{before}$, $R_\text{after}$ — the per-side search radius
  bounding the outward walk (§4.2).

**Anchor windowing (§5)**

- <a id="sym-T"></a>$T$ — window length in samples.
- <a id="sym-p"></a>$p \in [0,1]$ — the activation's fractional position within the
  window; $0.0$ is the first sample, $1.0$ the last.
- <a id="sym-s"></a>$s = \lfloor t_a - p\,(T-1) \rceil$ — the window's start sample; the
  window is the half-open interval $[s,\, s+T)$.
- <a id="sym-prealized"></a>$p_\text{realized} = (t_a - s)/(T-1)$ — the position the crop
  actually produced, as opposed to the one requested. This is the value
  stored as `ActivationPosition`.
- <a id="sym-Pcal"></a>$\mathcal{P}$ — the **position range** $p$ is drawn from; a
  $(\text{lo}, \text{hi})$ fraction pair, point-collapsible.
  $\mathcal{P}_\text{synth} \supseteq \mathcal{P}_\text{iafdb}$ by design.
- <a id="sym-train"></a>$\{t_a^{(k)}\}_k$ — the ordered **activation train**; $k$ indexes
  activations, so $W_k$, $p_k$, $s_k$ are that activation's window,
  position and start.
- <a id="sym-iai"></a>$\mathrm{IAI}_\text{prev}$, $\mathrm{IAI}_\text{next}$ — the intervals
  from an anchor to its previous / next neighbour;
  $\Delta t^{(k)} = t_a^{(k+1)} - t_a^{(k)}$.
- <a id="sym-surv"></a>$S(u) = \Pr[\mathrm{IAI} \ge u]$ — the interval **survival function**.
- <a id="sym-round"></a>$\lfloor\cdot\rceil$ — round to nearest integer.

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

A Butterworth band-pass of order [$N$](#sym-N) with edges $(f_\text{lo},
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
the whole complex. Consequently [$f_c$](#sym-fc) is the knob that decides *how much
fractionation counts as one activation* — lower merges more aggressively,
at the cost of broadening the complex and pushing measured onset earlier
and offset later (§4).

> **Implementation** — `filters/lowpass.py`.
> **Pinned by** `tests/test_filters.py::test_lowpass_bridges_a_fractionated_complex`
> (three deflections → exactly one above-threshold run; the raw rectified
> signal fragments).

### 1.4 Decimation

Keeping every $M$-th sample lowers the rate to $f_s/M$ and the Nyquist
frequency to $f_s/2M$. Anything above that new limit does not disappear:
it **folds back** into the retained band. A component at frequency $f$
reappears at

$$
f_\text{alias} = \min_{k \in \mathbb{Z}} \left| f - k\,\frac{f_s}{M} \right| ,
$$

and once it lands there it is indistinguishable from signal that was
genuinely at $f_\text{alias}$ — no later processing can separate them.
At $f_s = 1000$ and $M = 4$, a 200 Hz component reappears at
$|200 - 250| = 50$ Hz, inside the band the atrial signal occupies. So
the anti-alias low-pass is not preparation for decimation; it is half of
what decimation *is*.

**Where to put the cutoff.** Filtering exactly at the new Nyquist is not
enough, because a Butterworth is only $-3$ dB down at its own corner and
its whole transition band would fold. The cutoff is therefore set to a
fraction of the new limit,

$$
f_c = \alpha \cdot \frac{f_s}{2M}, \qquad \alpha = 0.8 ,
$$

which puts the transition inside the discarded region. The same fraction
`scipy.signal.decimate` uses.

**Why order 4 rather than the order 2 used elsewhere.** Measured at
$f_s = 1000$, $M = 4$, $\alpha = 0.8$ — the amplitude an out-of-band
200 Hz tone retains in the decimated result, and the amplitude a 50 Hz
tone keeps:

| $N$ | 200 Hz leak | 50 Hz retained | 110 Hz retained |
|---|---|---|---|
| 2 | 0.0385 | 0.947 | 0.399 |
| **4** | **0.0017** | **0.997** | 0.306 |
| 8 | 0.0002 | 1.000 | 0.162 |

Order 4 buys a factor of 23 in rejection over order 2 for 0.2% of
passband. Order 8 gains another factor of ten but collapses the
transition band, taking legitimate content near the edge with it, so 4
is the balance point. Recall from §1.1 that the filter is applied
forward and backward, so the achieved roll-off is that of a one-pass
filter of order $2N$.

**Zero-phase, which is not the usual choice for a resampler.** A causal
anti-alias filter delays the signal by its group delay, shifting every
event in the trace. Here the quantity everything downstream is built on
is *when* an activation occurs (§3, §5), so a uniform shift would move
every detected activation time and flow directly into the stored
activation position. Zero-phase costs nothing in an offline pipeline and
keeps the timing exact.

**A practical caution.** The forward-backward padding leaves a transient
at each end that has nothing to do with the passband. Measured on an
out-of-band tone at order 4: the largest excursion anywhere in the
result is $0.059$, while the steady-state leak is $0.0015$ — a factor of
39. Any assessment of filter quality that takes a maximum over the whole
array is therefore measuring the padding, and will rate every order
alike. Trim the edges before measuring, or measure spectrally.

> **Implementation** — `filters/decimation.py`; reuses §1.3's low-pass,
> passing $f_s = 1$ so the cutoff is expressed purely as a function of
> $M$ and the caller cannot supply a rate inconsistent with the data.
> **Pinned by** `tests/test_decimation.py`, which compares against naive
> `signal[::M]` to show the 50 Hz alias appearing only without the
> filter.

**Sources.** The sampling theorem and the folding relation are standard;
see Oppenheim & Schafer, *Discrete-Time Signal Processing*, on sampling
rate reduction, and Crochiere & Rabiner, *Multirate Digital Signal
Processing* (1983), for decimation specifically.

## 2. Detection preprocessing

### Vocabulary: what actually detects

Activation detection is a **chain**, and it is worth being strict about
which stage is which — the three are routinely conflated, including in
this project's own method spec:

| stage | what it does | where |
|---|---|---|
| **Detection preprocessing** | transform $x \to g$ so activations are emphasised | §2.2–§2.4 |
| **Detection thresholding** | the decision rule that turns [$g$](#sym-g) into candidates | §3 |
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
input length (so an index into [$g$](#sym-g) *is* an index into $x$); positions
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
a policy decision belonging to whatever consumes [$g$](#sym-g), $\arg\max$ is
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
parameter. We take the smoothing, because [$f_c$](#sym-fc) is one knob and the
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
fractionation**, at the [$g$](#sym-g) level, before any refractory logic runs.

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

Stage two: given the detection curve [$g$](#sym-g), decide how prominent a sample
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

A robust analogue of "mean plus [$k$](#sym-k) standard deviations": the median
locates the baseline, the MAD measures its spread, [$\lambda$](#sym-lambda) sets how far
above it to sit.

**Why MAD and not [$\sigma$](#sym-sigma)** — the load-bearing choice. The curve
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

| rule | [$\tau$](#sym-tau) at 5 | $\tau$ at 200 | drift |
|---|---|---|---|
| $c\cdot\mathrm{median} + \lambda\cdot\mathrm{MAD}$ | 0.53 | 0.62 | **x1.16** |
| $\mathrm{mean} + k\,\sigma$ | 12.7 | 79.9 | x6.3 |

$\lambda$ is in **raw MAD units**. For Gaussian data
$\sigma \approx 1.4826\,\mathrm{MAD}$, so multiply by that to read it as a
sigma multiple. Neither [$c$](#sym-c) nor $\lambda$ has a default: how aggressive
detection should be is a policy the calling pipeline owns.

**$\mathrm{MAD} = 0$ is not an error.** It means over half the samples
share one value — equally true of a dead channel *and* of a clean
synthetic trace that is exactly flat between activations. The two are
indistinguishable from [$g$](#sym-g) alone, so it is deliberately not
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

**Why not one candidate per above-[$\tau$](#sym-tau) segment.** The tempting
simplification is to collapse each contiguous above-$\tau$ run to its
single $\arg\max$. The method spec explicitly rules this out, and
measurement shows why: it lets the *threshold* do the merging, so two
genuine activations riding **one** above-$\tau$ run — common in fast AF,
when the envelope never fully returns to baseline between beats —
collapse into one peak, a silent miss that no [$\Delta_\text{refr}$](#sym-refr)
tuning can recover.

Measured, two genuine activations 80–90 ms apart:

| [$f_c$](#sym-fc) (envelope) | above-$\tau$ segments | widest | segment-argmax | local maxima |
|---|---|---|---|---|
| 20 Hz | 2 | 49 | 2 activations | 2 activations |
| 10 Hz | 1 | **168** | **1 — one lost** | 2 activations |
| 6 Hz | 1 | **232** | **1 — one lost** | 2 activations |

With $\Delta_\text{refr} = 60$ samples, those runs span three to four
refractory intervals. The above-$\tau$ segment *is* still used — it is
the [$W_\text{act}$](#sym-Wact) that §4 measures onset and offset within — just not
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

Two degenerate curves never reach a rule at all, and both **raise**: an
**empty** curve, and a **constant** one, which has no peaks to separate.

They raise rather than returning a sentinel, and they raise *different*
exceptions, for reasons worth stating because an earlier revision did
neither. A sentinel $+\infty$ propagates: measured downstream, it made
the §4 outward walk terminate on its first comparison and report a
zero-width complex flagged as a **complete measurement** — indistinguishable
from a genuine instantaneous one, and destined for the duration
distribution the study rests on. Failing where the condition arises is
the only version that stays visible.

The two conditions are then separated because their causes are
categorically different. An empty array is a **programming error** —
nothing legitimately produces one, so it means a bad slice upstream. A
constant array is a **data condition**: a dead or disconnected electrode
records a flat line, and so does a channel clipped to a rail. Sweeping
thousands of channels, the second is expected and should be caught,
counted and skipped; the first should not be swallowed by that same
handler.

**Known limitation — this test is whole-array only.** A channel that
flatlines *intermittently*, dropping out for a few seconds mid-record,
passes it. Every threshold computed from such a trace is then biased by
the dead stretch, silently: the median and the MAD are both pulled toward
the flat value in proportion to how much of the record it occupies, which
*lowers* $\tau$ and admits noise elsewhere in the trace. Detecting bad
*sections* needs a segment-wise analysis this library does not yet do,
and the case has not been characterised on IAFDB.

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

### 3.5 Candidate extraction, suppression, and the chain

The last two stages, and then the composition that is the detection
function.

**Candidate selection.** A candidate is a local maximum of [$g$](#sym-g) at or
above [$\tau$](#sym-tau) (§3.4), filtered by a minimum **prominence** — how far a
peak rises above the higher of the two saddles bounding it.

Prominence is the only candidate filter. An earlier implementation also
exposed a minimum peak *width*, which was dropped: the method spec's
step 3 filters on prominence alone, and a width measured at half
prominence (what the library primitive provides) is partly a restatement
of the prominence test rather than an independent criterion. Two
overlapping shape filters invite interactions that neither the spec nor
any measurement here justifies.

The prominence filter is optional in the signature and **effectively
required in practice**. An adaptive threshold sitting a few MAD above a
quiet baseline is low in absolute terms, so filter ringing around a
strong activation and ordinary noise bumps both clear it, and — being
spaced further apart than [$\Delta_\text{refr}$](#sym-refr) — suppression keeps them.
Measured on one fractionated complex with realistic noise:

| preprocessor | no prominence floor | floor at $0.2\max g$ |
|---|---|---|
| Botteron envelope | 3 activations | **1** |
| rectified derivative | 9 activations | **1** |

The failure is silent, because spurious detections look exactly like
activations.

**Refractory suppression.** Enforce a minimum spacing
$\Delta_\text{refr}$ between distinct activations. The Phase-1.5 default
is **greedy by height**: sort candidates tallest to shortest, accept
each only if no already-accepted peak lies within $\Delta_\text{refr}$.
Every survivor is therefore the tallest peak in its own refractory
neighbourhood.

Height-ordered rather than time-ordered matters. A causal scan
(Pan–Tompkins blanking) accepts whichever peak of a cluster arrives
*first* and blanks the rest, so a small precursor deflection masks the
genuine activation behind it. Ordering by height makes the result
independent of which end of the record you start from — appropriate for
an offline splitter, which has no causality constraint to respect. The
method spec keeps four alternatives on a menu (causal blanking,
sliding-window max, segment-merge, DP-optimal); complexity is not the
deciding factor, since there are tens to hundreds of candidates per
channel and this is offline work.

**The two knobs do different jobs and must not be traded.** $\tau$
separates activation from noise and far-field, on *amplitude*;
$\Delta_\text{refr}$ collapses one activation's several deflections, on
*time*. Raising $\tau$ to suppress fractionation also removes genuine
**low-voltage** activations — which are exactly the fibrotic regions
this project exists to find.

**Two-stage refinement (optional, off by default).** The smoothed
envelope's maximum is pulled toward the heavier side of an asymmetric
complex, so a long fractionated tail drags the detected time late.
Passing a *sharp* preprocessor as the refiner snaps each accepted
activation to the nearest local maximum of that curve, within a small
radius, recovering the instant the smoothing blurred. Keep the radius
well under $\Delta_\text{refr}$: a radius able to reach a neighbouring
activation lets refinement move a peak onto the wrong complex, and
suppression has already run, so nothing downstream would catch it.

**The chain.** For the IAFDB train: preprocess $\to$ threshold $\to$
candidates $\to$ suppress $\to$ (optionally) refine. For synthetic,
where the generator knows where it put the single activation,
$t_a = \arg\max g$ and none of the rest applies.

Each stage is a configured object — preprocessor, threshold, selector,
suppressor, refiner — so the chain reads as a composition and each
stage's parameters live with the stage that uses them. Two of those
choices are deliberately unavailable by omission: the prominence floor
and the suppressor must both be passed explicitly, because omitting
either silently changes the activation count.

> **Implementation** — `extraction/activation_based/candidates.py`,
> `suppression.py`, `detection.py`.
> **Pinned by** `tests/test_activation_detection.py`, including the
> merged-above-$\tau$-run case, the fractionation-under-both-families
> case, and the over-detection measurement above.
> **Source** — [Pan & Tompkins 1985, IEEE TBME BME-32(3):230-236](https://doi.org/10.1109/TBME.1985.325532).

## 4. Activation-complex bounds

An activation is not an instant but a *complex* — a rise, a peak, a fall
— and a fibrotic one is longer, with a trailing fractionated tail. This
section measures that extent, because the windowing margins in §5 are
chosen from its distribution.

### 4.1 Onset, offset, and the durations

Walk outward from the detected activation until the curve drops below a
boundary level [$\theta$](#sym-theta):

$$
t_\text{on} = \max\{\, i < t_a : g[i] < \theta \,\}, \qquad
t_\text{off} = \min\{\, i > t_a : g[i] < \theta \,\},
$$

$$
r_\text{rise} = t_a - t_\text{on}, \qquad
r_\text{fall} = t_\text{off} - t_a, \qquad
W_\text{act} = r_\text{rise} + r_\text{fall}.
$$

**The curve must be the smoothed one.** On a sharp [$g$](#sym-g) a fractionated
complex dips below $\theta$ *between* its deflections, so the walk stops
at the first dip and measures one deflection rather than the complex.
The envelope's low-pass bridges those dips — the same mechanism as
§1.3 — which is what makes the measurement meaningful at all. Pinned by
a test comparing the two.

**Fibrotic complexes are asymmetric**, with $r_\text{fall} \gg
r_\text{rise}$, which is why the method spec calls for a *healthier back
margin* than front margin when the fallback fixed margins are used.

### 4.2 Choosing $\theta$

Two forms, answering different questions:

- **Fraction of the local peak**, $\theta = f \cdot g[t_a]$. Scales with
  each complex individually, so a low-voltage fibrotic activation is
  measured against its own amplitude rather than the record's largest —
  usually right when comparing complex *shapes* across a corpus whose
  amplitudes vary.
- **Above the noise floor**, $\theta = \operatorname{median}(g) +
  \lambda\,\operatorname{MAD}(g)$. Answers "where does this stop being
  distinguishable from baseline?", and is robust for the same reason the
  detection threshold is (§3.2).

The spec phrases the second as "a small multiple of the baseline-noise
MAD". The median is added deliberately: a MAD multiple alone is a
*spread*, not a level, and on a curve whose baseline sits above zero it
would fall beneath the noise floor.

**The second form is literally §3.2's rule** — it is
$\tau = c\operatorname{median}(g) + \lambda\operatorname{MAD}(g)$ with
$c = 1$, and the code uses that one class rather than restating the
arithmetic. This is the clearest case for why the threshold families are
named after *how much context they need* rather than what they are used
for: the same computation serves detection and boundary measurement, so
a type called "detection threshold" would have been a lie in one of the
two places. What genuinely differs is that the peak-fraction form cannot
be evaluated without knowing **which** peak, and the median/MAD form has
no use for that argument — so they are separate interfaces, and a caller
accepting either distinguishes them with one `isinstance`.

**Which matters, because [$\theta$](#sym-theta) below the noise floor never
terminates.** The walk continues until it happens to dip — in practice,
until it reaches a neighbouring activation. Measured on a
four-activation train with realistic noise:

| $\theta$ | vs baseline median | measured width |
|---|---|---|
| $0.02 \cdot g[t_a]$, no noise | above | 54 samples |
| $0.02 \cdot g[t_a]$, noise 0.02 | **below** | **597 samples** — reaches past the previous activation |
| $\operatorname{median} + 3\operatorname{MAD}$, noise 0.02 | above by construction | terminates normally |

Hence two safeguards: an optional **search radius** bounding the walk,
and a **clamped** flag on each side recording whether it terminated on a
genuine crossing or merely ran out of room. A clamped side is not a
measurement, and pooling it into a duration distribution would bias
exactly the long tail the study exists to characterise.

The radius may be set **per side**, and for this measurement that is the
form that fits the phenomenon. §4.1 defines $r_\text{rise}$ and
$r_\text{fall}$ separately precisely because a fibrotic complex is
*asymmetric* — the fractionated tail makes the falling side the long one.
A single symmetric radius must therefore be set wide enough for that
tail, which simultaneously licenses the backward walk to run just as far
toward the **previous** activation, where no such length is expected. Two
radii let the forward side stay generous while the backward side stays
tight, so the safeguard bounds the direction that actually runs away:

$$
t_\text{on} \ge t_a - R_\text{before},
\qquad
t_\text{off} \le t_a + R_\text{after}.
$$

A reasonable starting point is $R_\text{before}$ a little under the
refractory interval and $R_\text{after}$ larger, sized from the
$r_\text{fall}$ distribution the study is measuring — which makes the
choice mildly circular on the first pass, so start loose and tighten.

### 4.3 Relationship to the above-$\tau$ segment

Candidate selection (§3.5) already reports the contiguous above-[$\tau$](#sym-tau)
run containing each peak. That is *a* notion of complex width; this is a
finer one, and **neither bounds the other**, because [$\theta$](#sym-theta) may sit
either side of $\tau$. Measured with $\tau = \operatorname{median} + 4
\operatorname{MAD}$:

| $\theta$ | vs $\tau$ | complex vs segment |
|---|---|---|
| $0.50 \cdot g[t_a]$ | above | inside the segment |
| $0.25 \cdot g[t_a]$ | above | inside the segment |
| $0.10 \cdot g[t_a]$ | **below** | **extends beyond** it |

So the segment is a reference, not a bounding box.

### 4.4 A study-time criterion, not a runtime filter

This measures [$r_\text{rise}$](#sym-rise) and [$r_\text{fall}$](#sym-rise) across a corpus so
that §8.1 can choose an activation-position range where complexes are
almost never clipped:

$$
p \in \left[\, r_\text{rise}/T,\; 1 - r_\text{fall}/T \,\right].
$$

Once that range is chosen the bound stops binding, so the splitter does
**not** re-test each window. That is deliberate: multi-beat work in a
later phase will *want* a fraction of edge-clipped windows so the model
learns that case, making a hard clip filter counterproductive.

The spec is also candid that this may not survive contact with real AF
electrograms, where clean rise/fall boundaries often do not exist. The
documented fallback is fixed **generous asymmetric margins** — a healthy
front, a healthier back — with this measurement's job being to *inform*
those margins rather than to run per window.

> **Implementation** — `extraction/activation_based/complex_bounds.py`.
> **Pinned by** `tests/test_complex_bounds.py`, including the runaway
> walk, the sharp-versus-smoothed comparison, and the clamped-boundary
> reporting.
> **Source** — [Hodges & Bui 1996](https://pubmed.ncbi.nlm.nih.gov/9020824/)
> for threshold-on-smoothed-rectified onset detection and its
> smoothing-induced bias; open-access re-derivation in the
> [EMG-onset review](https://pmc.ncbi.nlm.nih.gov/articles/PMC10594734/).

## 5. Anchor windowing

### 5.1 Placing the window

Given an activation at [$t_a$](#sym-ta), a requested fractional position
[$p$](#sym-p) and a window length [$T$](#sym-T), the window is
$[s,\, s+T)$ with

$$
s = \lfloor t_a - p\,(T-1) \rceil ,
$$

so $p = 0$ puts the activation on the window's **first** sample, $p = 1$
on its **last**, and $p = 0.5$ centres it. The $T-1$ rather than $T$ is
the whole content of the convention: there are $T$ samples but only
$T-1$ intervals between them, so $p$ interpolates between sample $0$ and
sample $T-1$ inclusive. Using $T$ would make $p = 1$ land one sample past
the end.

The window is **half-open**, matching the crop: sample $s$ is inside,
sample $s+T$ is not. The multi-beat predicate uses the same interval, so
a neighbouring activation exactly at $s+T$ is correctly judged to be
outside the window it is not in.

### 5.2 Why a fraction and not a sample offset

$p$ is **rate- and length-independent by construction**. The same
$p = 0.35$ means the same thing on a 500-sample window at 1 kHz and a
250-sample window at 500 Hz. That is what lets a corpus assembled from
sources with different sampling rates and window lengths have a single
comparable position distribution — which is the point, since §8.1
compares the synthetic and IAFDB distributions against each other. A
sample offset would have to be re-derived for every rate/length
combination, and the derivation would end up living at each call site.

This is a fleet contract, not a local convention:
`common.schema.json#/$defs/ActivationPosition` in egm-contracts defines
it once and both `iafdb_bank` and `synthetic_bank` `$ref` it, so the two
corpora cannot drift.

### 5.3 Requested versus realized position

$s$ is an integer, so the position a window *achieves* is generally not
the one requested. Both are kept, and the **realized** one is what gets
stored:

$$
p_\text{realized} = \frac{t_a - s}{T - 1}.
$$

The error is bounded by half a sample, because integer rounding of $s$
is its only source. Measured over 200 000 random $(p, T)$ pairs with
$T \in [2, 1000)$: **max 0.49999 samples, mean 0.250**. The bound is
attained exactly when $p\,(T-1)$ lands on a tie:

| $T$ | $p\,(T-1)$ at $p = 0.5$ | $s$ | $p_\text{realized}$ | error |
|---|---|---|---|---|
| 100 | 49.5 — a tie | $t_a - 50$ | 0.505051 | 0.500 samples |
| 101 | 50.0 — exact | $t_a - 50$ | 0.500000 | 0.000 samples |

Ties round to even, which is Python's `round`. Which way a tie breaks is
arbitrary and the choice is not load-bearing — either direction moves the
window one sample and the realized position half a sample, inside the
error the caller has already accepted — but it is fixed, so the same
request always yields the same crop.

The realized position is also **quantized**: only $T$ values are
reachable, spaced $1/(T-1)$ apart. At $T = 100$ that is 0.0101, at
$T = 10$ it is 0.111. Worth knowing before reading structure into a
histogram of stored positions at short window lengths — the comb is an
artifact of the crop, not of the physiology.

The conversion back is the contract's own formula, and it is exact:

$$
\lfloor p_\text{realized}\,(T-1) \rceil = t_a - s ,
$$

since $t_a - s$ is already an integer. A consumer therefore recovers the
anchor sample exactly, at any $T$.

### 5.4 The position range

$p$ is not chosen per window by hand; it is drawn from a range
$\mathcal{P}$, and drawing it is what **blocks the positional shortcut**
— if every window put the activation in the same place, a classifier
could key on position instead of morphology, which is the T1 hypothesis
this whole apparatus exists to test.

In code $\mathcal{P}$ is a *generator* rather than a bare interval —
uniform-over-$(\text{lo}, \text{hi})$ is the policy this phase uses, but
it is a policy, and a shaped distribution is a plausible later need (if
the §5.7 residual ever wanted correcting, that is where it would go).

The uniform form is a $(\text{lo}, \text{hi})$ fraction pair
and is **point-collapsible**: $(x, x)$ is a fixed position, which is the
baseline arm of the anchored-versus-varied A/B. One representation
covers both arms, so switching between them is a value change rather
than a code path — the same idiom the noise mixer uses for
`snr_db_range=(X, X)`.

**The range is shared as a mechanism, not as a value.** The two corpora
deliberately use *different* ranges: $\mathcal{P}_\text{synth} \supseteq
\mathcal{P}_\text{iafdb}$, because a single-beat synthetic trace has no
signal before the upstroke, so a far-back position fills the window with
flat pre-activation that is both uninformative and unlike real EGM. The
synthetic range is therefore **back-bounded** — which also means the
range is *not* required to be symmetric about $0.5$, and this library
does not impose that. A central band is a common choice (§5.6), not a
constraint of the type.

The consequence is recorded as a project limitation rather than fixed:
because the ranges differ by design, the stored `activation_position`
coordinate reads **non-zero in the corpus-comparison distance by
construction**, and carries no realism information.

### 5.5 Windowing a train

The caller-facing operation is not "window this activation" but
**"window this train"**: given a signal and the ordered activation train
$\{t_a^{(k)}\}$ from §3.5, produce one window per anchor,

$$
p_k \sim \mathcal{P},
\qquad
s_k = \lfloor t_a^{(k)} - p_k\,(T-1) \rceil,
\qquad
W_k = x[s_k : s_k + T].
$$

**Synthetic traces go through the same operation as a train of one.**
They have a single known activation, so the "train" has one member. That
is not a special case bolted on — it is the reason the operation is
shaped this way. Running both corpora through one code path is what
guarantees the window geometry cannot drift between them, which is the
same argument that put the crop arithmetic in this library, applied one
level up.

### 5.6 Classification, not dropping

Not every anchor yields a usable window. Two conditions matter:

- **In bounds** — $[s_k,\, s_k+T)$ lies inside the signal. The first and
  last activations of a record often fail this.
- **Single beat** — exactly one train member lies in $[s_k,\, s_k+T)$. A
  window containing a neighbouring activation is multi-beat, and no
  single position describes it.

This library **reports both and drops neither.** It returns every window
labelled, along with each anchor's neighbouring intervals
$\mathrm{IAI}_\text{prev}$ and $\mathrm{IAI}_\text{next}$; deciding what
to keep belongs to the producer. Two reasons the seam sits here:

1. **The policies genuinely differ.** The IAFDB side drops boundary and
   multi-beat windows; the synthetic side sizes its simulation so the
   single window is always in bounds and multi-beat is impossible. A
   library that dropped would be imposing one corpus's policy on both.
2. **The drop rate is a measured quantity.** Choosing $T$ and
   $\mathcal{P}$ is a yield-versus-morphology trade-off, and yield is
   exactly the fraction dropped. Drops that happen invisibly inside a
   library cannot be counted by the study that exists to count them.

### 5.7 The drop is not position-neutral

This is the subtle one, and it is why the intervals are reported rather
than merely used.

A window survives the multi-beat test when the previous activation is at
least $pT$ behind and the next at least $(1-p)T$ ahead. With
$S(u) = \Pr[\mathrm{IAI} \ge u]$ the interval survival function and
neighbours roughly independent,

$$
\Pr[\text{single beat} \mid p] = S(pT)\,S\big((1-p)T\big).
$$

That is **a function of $p$**. So dropping multi-beat windows does not
merely reduce the count — it *reshapes the realized position
distribution*, on the IAFDB side only, since the synthetic side drops
nothing. The distribution actually stored is therefore not the
$\mathcal{P}$ that was requested.

*(The analysis uses $pT$ where the crop uses $p(T-1)$; the one-sample
difference is immaterial to the yield model and is kept here to match
the method spec's form.)*

**Memoryless intervals cancel exactly.** For $S(u) = e^{-u/\mu}$,

$$
S(pT)\,S\big((1-p)T\big) = e^{-pT/\mu}\,e^{-(1-p)T/\mu} = e^{-T/\mu},
$$

independent of $p$. So an exponential interval distribution loses
windows without reshaping the positions of the survivors at all.

**What actually decides it is whether $T/2$ fits under the shortest
intervals.** "Regular versus memoryless" is the wrong axis, and
measuring it through the implementation makes that clear. A **central**
anchor needs about $T/2$ of margin on *each* side; an **edge** anchor
needs nearly all of $T$ on *one* side. So when the interval distribution
has substantial mass below $T$ but little below $T/2$, central anchors
are protected and edge anchors are not, and the survivors skew central.
Measured at $T = 192$, drawing $p$ uniformly over $[0,1]$ and keeping
the single-beat windows — the shape of the surviving positions in five
bins, normalised to its own peak:

| interval distribution | $T/2$ vs floor | surviving shape (5 bins over $p$) | edge/centre |
|---|---|---|---|
| exponential $\mu{=}200$, no floor | — | 0.97 · 0.94 · 0.99 · 1.00 · 0.99 | **0.98** |
| exponential $\mu{=}200$, floor 40 | 96 > 40 | 1.00 · 0.84 · 0.91 · 0.91 · 0.98 | **1.08** |
| exponential $\mu{=}200$, floor 100 | 96 < 100 | 0.62 · 0.72 · 1.00 · 0.74 · 0.62 | **0.62** |
| regular 180 ± 15 (fast flutter) | 96 < ~150 | 0.62 · 0.94 · 1.00 · 0.98 · 0.65 | **0.64** |

The two exponential rows differ *only* in their floor and land on
opposite sides of the effect, which isolates the mechanism: it is the
floor relative to $T/2$, not the regularity, that matters. A regular
rhythm near its cycle length is simply the most common way to get a
floor above $T/2$ — and it is the relevant one here, since $T = 192$ ms
is fixed by an unrelated architectural constraint and atrial flutter
cycles near 180 ms sit inside the design envelope.

The practical form of the check: compare $T/2$ against a **low
percentile** of the measured interval distribution, per record. That is
computable from the reported intervals alone.

**The resolution is to choose $\mathcal{P}$, not to correct the output.**
The keep weight is symmetric in $p$ and flat near $0.5$, steepening only
toward the edges, so a **central band** makes the reshaping negligible by
construction. It is also the yield-maximising choice exactly where the
bias appears: a central $p$ needs symmetric margins of about $T/2$ on
each side, which a 180 ms cycle comfortably provides, whereas an edge $p$
needs more than $T$ on one side, which it barely does.

The two corrections one would otherwise reach for were both considered
and rejected for this phase: **acceptance-rejection thinning** costs
yield, which is the scarce quantity; **inverse-probability weighting**
keeps every window but requires weighted distances downstream and weights
persisted alongside the data. Neither is worth it for an effect that is
flutter-only and disappears once the simulator becomes multi-beat. If a
correction is ever needed, thinning is the one to reach for, because it
composes with the existing drop step and changes nothing downstream.

What this library owes that decision is the **measurement**: reporting
$\mathrm{IAI}_\text{prev}$ and $\mathrm{IAI}_\text{next}$ per anchor lets
the per-anchor keep probability be read straight off observed data, with
no need to fit $S$.

### 5.8 Three rules this module exists to enforce

**$T \ge 2$.** $p_\text{realized}$ divides by $T-1$, and a one-sample
window has no meaningful position — its single sample is simultaneously
the first and the last. Rejected rather than special-cased, because any
convention picked for the degenerate case would flow straight into the
stored distribution.

**A window that does not fit raises; it does not slide.** Clamping to
the array edge would change the realized position without saying so: the
activation would no longer sit where it was asked to, and the stored
value would record the slide as though it were intended. Callers that
want to skip such activations test `window_is_within_bounds`
first.

**Produced, never measured.** $p_\text{realized}$ is the anchor the crop
*placed*. It is not re-derived from the waveform afterwards — a consumer
wanting the measured dV/dt-max position computes it with egm-features.
Conflating the two would quietly convert a comparison of *where we put
the activation* into a comparison of *where a detector later found it*,
which is a different quantity with different failure modes (§3.5). For
the same reason $p = 0$ is a **legitimate stored value**, meaning the
activation sits on the first sample, so absence must never be read as
zero.

> **Implementation** — `extraction/activation_based/anchoring.py`.
> **Pinned by** `tests/test_anchoring.py`, including the round-trip
> through the contract's conversion formula and the half-sample bound.

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
