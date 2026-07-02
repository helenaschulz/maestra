"""Temperature scaling for class probabilities.

A model's raw probabilities can be over- or under-confident; under a calibration-sensitive
metric like log_loss a few overconfident-wrong rows dominate the score (on leaf-classification
15 of 891 out-of-fold rows give the true class ~0 probability → −log(eps) ≈ 34.5 each). Temperature
scaling fits a single scalar ``T`` to the out-of-fold probabilities — minimising their log_loss —
and reshapes every probability vector as ``p**(1/T)`` renormalised to sum to 1. ``T > 1`` softens
(less confident), ``T < 1`` sharpens. It never changes the arg-max, so labels and accuracy are
untouched; only probability metrics move.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss


def apply_temperature(proba: pd.DataFrame, temperature: float) -> pd.DataFrame:
    """Reshape a full probability distribution by ``temperature`` (``p**(1/T)``, row-renormalised).

    ``proba`` must be the complete per-class distribution (>= 2 columns summing to 1).
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    scaled = np.power(proba.to_numpy(), 1.0 / temperature)
    scaled = scaled / scaled.sum(axis=1, keepdims=True)
    return pd.DataFrame(scaled, index=proba.index, columns=proba.columns)


def fit_temperature(y_true, proba: pd.DataFrame, *, bounds: tuple[float, float] = (0.05, 20.0)) -> float:
    """Return the temperature that minimises the log_loss of ``proba`` on ``y_true``.

    A 1-D bounded search; the optimum is unique (log_loss is convex in ``1/T``). ``y_true`` and
    ``proba`` are the pooled out-of-fold labels and probabilities.
    """
    labels = list(proba.columns)

    def loss(temperature: float) -> float:
        return log_loss(y_true, apply_temperature(proba, temperature).to_numpy(), labels=labels)

    return float(minimize_scalar(loss, bounds=bounds, method="bounded").x)
