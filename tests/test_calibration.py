"""Tests for temperature scaling. Pure numpy/sklearn — fast and offline."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import log_loss

from maestra.calibration import apply_temperature, fit_temperature


def test_temperature_one_is_identity():
    proba = pd.DataFrame({"a": [0.7, 0.2], "b": [0.3, 0.8]})
    out = apply_temperature(proba, 1.0)
    assert np.allclose(out.to_numpy(), proba.to_numpy())


def test_temperature_preserves_argmax_and_normalisation():
    proba = pd.DataFrame({"a": [0.7, 0.1, 0.5], "b": [0.2, 0.8, 0.5], "c": [0.1, 0.1, 0.0]})
    out = apply_temperature(proba, 3.0)  # soften
    assert out.sum(axis=1).round(6).eq(1.0).all()                 # still a distribution
    assert out.idxmax(axis=1).tolist() == proba.idxmax(axis=1).tolist()  # labels unchanged
    assert out.iloc[0].max() < proba.iloc[0].max()                # T>1 lowers the peak


def test_fit_temperature_reduces_log_loss_on_overconfident_model():
    # 8 confident-correct rows + 2 confident-WRONG rows: log_loss is dominated by the wrong ones,
    # so the optimal temperature softens (> 1) and lowers the loss.
    y = [0] * 5 + [1] * 5
    p_pos = [0.01] * 4 + [0.99] + [0.99] * 4 + [0.01]  # rows 4 and 9 are confidently wrong
    proba = pd.DataFrame({0: [1 - p for p in p_pos], 1: p_pos})
    labels = [0, 1]

    T = fit_temperature(y, proba)
    assert T > 1.0  # overconfident -> soften
    raw = log_loss(y, proba.to_numpy(), labels=labels)
    cal = log_loss(y, apply_temperature(proba, T).to_numpy(), labels=labels)
    assert cal < raw


def test_invalid_temperature_raises():
    with pytest.raises(ValueError):
        apply_temperature(pd.DataFrame({"a": [1.0], "b": [0.0]}), 0.0)
