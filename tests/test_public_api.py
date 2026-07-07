"""Smoke test for the public `maestra` package surface (P3) -- exercises the exact import path
`docs/examples/compare_quickstart.ipynb` uses (`from maestra import compare`), not run in CI
itself. Two sklearn dummies stand in for the notebook's real pipelines."""
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression


def test_compare_is_reachable_as_the_public_maestra_import():
    from maestra import CompareResult, compare

    rng = np.random.default_rng(0)
    x = np.arange(30, dtype=float)
    df = pd.DataFrame({"x": x, "target": 3.0 * x + rng.normal(0, 0.5, 30)})
    result = compare(DummyRegressor(strategy="mean"), LinearRegression(), df, "target", cv=3)
    assert isinstance(result, CompareResult)
    assert result.verdict in ("improved", "no_improvement", "underpowered")
    assert isinstance(result.summary(), str)
