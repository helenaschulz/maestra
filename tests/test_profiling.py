import pandas as pd

from automl_agent.profiling import profile_dataframe


def _df():
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "city": ["Berlin", "Berlin", None, "Hamburg"],
            "target": [0, 1, 0, 1],
        }
    )


def test_profile_shape_and_target_flag():
    prof = profile_dataframe(_df(), "target")
    assert prof["n_rows"] == 4
    assert prof["target"] == "target"
    by_name = {c["name"]: c for c in prof["columns"]}
    assert by_name["target"]["is_target"] is True
    assert by_name["id"]["is_target"] is False


def test_missing_and_cardinality_fractions():
    prof = profile_dataframe(_df(), "target")
    city = next(c for c in prof["columns"] if c["name"] == "city")
    assert city["n_missing"] == 1
    assert city["missing_frac"] == 0.25
    assert city["n_unique"] == 2  # Berlin, Hamburg (NaN excluded)


def test_id_like_flag_distinguishes_ids_from_continuous():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],                  # int, unique per row -> id_like
            "code": ["a", "b", "c", "d"],        # object, unique per row -> id_like
            "measure": [1.1, 2.2, 3.3, 4.4],     # float, unique per row -> NOT id_like
            "cat": [1, 1, 2, 2],                 # int, low cardinality -> NOT id_like
            "target": [0, 1, 0, 1],
        }
    )
    by_name = {c["name"]: c for c in profile_dataframe(df, "target")["columns"]}
    assert by_name["id"]["id_like"] is True
    assert by_name["code"]["id_like"] is True
    assert by_name["measure"]["id_like"] is False  # the bug class: continuous floats
    assert by_name["cat"]["id_like"] is False


def test_examples_exclude_nan_and_truncate():
    df = pd.DataFrame({"x": ["a" * 100, None], "target": [0, 1]})
    prof = profile_dataframe(df, "target")
    x = next(c for c in prof["columns"] if c["name"] == "x")
    assert len(x["examples"]) == 1  # NaN dropped
    assert len(x["examples"][0]) == 40 and x["examples"][0].endswith("...")
