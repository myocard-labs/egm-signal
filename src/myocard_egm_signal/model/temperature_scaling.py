"""Probability calibration via temperature scaling.

Per Guo et al. 2017, *On Calibration of Modern Neural Networks*
(arXiv 1706.04599), modern deep classifiers are typically
overconfident: the probability they assign to their predicted class
is systematically higher than the empirical frequency of being
correct at that probability level. Temperature scaling is the
simplest post-hoc fix: divide all logits by a single scalar ``T > 0``
before applying sigmoid (binary) or softmax (multiclass), and pick
``T`` to minimize the NLL on a held-out labeled set.

Properties:

- **Monotonic in logits.** ``sigmoid(z/T)`` preserves the ordering of
  samples by logit, so AUROC and rank-based metrics are unchanged.
  Only calibration-quality metrics (ECE, reliability) and any
  threshold-dependent decision at thresholds ≠ 0.5 actually shift.
- **One parameter.** ``T > 1`` softens overconfident predictions;
  ``T < 1`` sharpens underconfident ones; ``T = 1`` is identity.
- **Convex-ish NLL.** The NLL surface in ``T`` is smooth and
  unimodal for typical neural-network logit distributions, so
  bounded scalar minimization finds the optimum reliably.

This module ships only the math: pure numpy in, pure numpy out, no
model state, no torch. The consumer (egm-classifier's export CLI,
egm-viewer's analysis tab, paper-figure notebooks) decides what to
do with the fitted ``T`` — bake it into an ONNX graph, persist it
in `egm_class_model_metadata.json`, or just report it.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

DEFAULT_TEMPERATURE_BOUNDS = (0.05, 20.0)
"""Bracket for the scalar minimization in :func:`fit_temperature`.

Lower bound 0.05 ≈ "extreme sharpening" — any tighter optimum
indicates a numerical pathology rather than a genuine fit. Upper
bound 20.0 ≈ "extreme softening" — useful headroom for very
overconfident models. The optimum for typical trained classifiers
lands in ``[1.0, 3.0]``.
"""


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    bounds: tuple[float, float] = DEFAULT_TEMPERATURE_BOUNDS,
) -> float:
    """Fit the single scalar ``T`` minimizing binary NLL on (logits, labels).

    Solves ``argmin_T mean( log(1 + exp(z/T)) - y * (z/T) )``, the
    standard binary cross-entropy loss with temperature scaling, via
    :func:`scipy.optimize.minimize_scalar` with a bounded method.

    Parameters
    ----------
    logits
        ``[N]`` or ``[N, 1]`` array of pre-sigmoid logits. Flattened
        to ``[N]`` before fitting.
    labels
        ``[N]`` array of binary labels (0 or 1). Integer or float
        accepted; cast to float64 for the loss computation.
    bounds
        ``(low, high)`` search interval for ``T``. Defaults to
        :data:`DEFAULT_TEMPERATURE_BOUNDS`.

    Returns
    -------
    float
        The fitted ``T``. Always strictly positive.

    Raises
    ------
    ValueError
        On empty input, shape mismatch, or invalid ``bounds``.
    RuntimeError
        If the bounded scalar minimizer reports failure (extremely
        rare on well-formed inputs).
    """
    z = np.asarray(logits).reshape(-1).astype(np.float64)
    y = np.asarray(labels).reshape(-1).astype(np.float64)
    if z.size == 0:
        raise ValueError("fit_temperature: logits/labels are empty.")
    if z.shape != y.shape:
        raise ValueError(f"fit_temperature: shape mismatch; logits {z.shape} vs labels {y.shape}.")
    low, high = bounds
    if not (low > 0 and high > low):
        raise ValueError(
            f"fit_temperature: bounds must satisfy 0 < low < high; got ({low}, {high})."
        )

    def nll(t: float) -> float:
        # Numerically stable binary cross-entropy with logits-over-T:
        #   loss = log(1 + exp(z/T)) - y * z/T
        # logaddexp(0, u) computes log(1 + exp(u)) without overflow.
        u = z / t
        return float(np.mean(np.logaddexp(0.0, u) - y * u))

    result = minimize_scalar(nll, bounds=(low, high), method="bounded")
    if not result.success:
        raise RuntimeError(f"fit_temperature: minimize_scalar failed: {result.message}")
    return float(result.x)


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Scale logits by ``1 / T``: return ``logits / temperature``.

    Returns logits, not probabilities — the caller decides when to
    sigmoid (or whether to persist scaled logits + ``T`` separately
    for downstream re-analysis). The thin wrapper exists for symmetry
    with :func:`fit_temperature` and so call sites document intent.

    Parameters
    ----------
    logits
        Input logits, any shape.
    temperature
        ``T > 0``. ``T > 1`` softens, ``T < 1`` sharpens, ``T = 1``
        is identity.

    Returns
    -------
    np.ndarray
        Scaled logits with the same shape as ``logits``.

    Raises
    ------
    ValueError
        On ``temperature <= 0``.
    """
    if temperature <= 0:
        raise ValueError(f"apply_temperature: temperature must be > 0; got {temperature}.")
    return logits / temperature
