import pandas as pd

from maestra.profiling import profile_dataframe


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


def test_period_candidates_on_a_real_datetime64_column():
    df = pd.DataFrame({
        "ts": pd.date_range("2024-01-01", periods=10, freq="D"),
        "target": range(10),
    })
    prof = profile_dataframe(df, "target")
    ts = next(c for c in prof["columns"] if c["name"] == "ts")
    assert ts["period_candidates"] == ["month_of:ts", "week_of:ts", "dayofweek_of:ts"]


def test_period_candidates_on_a_parseable_date_string_column():
    df = pd.DataFrame({
        "datetime": [f"2024-01-{d:02d} 00:00:00" for d in range(1, 11)],
        "target": range(10),
    })
    prof = profile_dataframe(df, "target")
    dt = next(c for c in prof["columns"] if c["name"] == "datetime")
    assert dt["period_candidates"] == [
        "month_of:datetime", "week_of:datetime", "dayofweek_of:datetime"]
    assert list(df.columns) == ["datetime", "target"]  # profiling never mutates df


def test_period_candidates_empty_for_non_datetime_columns():
    df = pd.DataFrame({
        "amount": [1.0, 2.0, 3.0],
        "year": [2020, 2021, 2022],  # numeric time axis -> 'time' strategy, not period slicing
        "city": ["Berlin", "Hamburg", "Munich"],
        "target": [0, 1, 0],
    })
    prof = profile_dataframe(df, "target")
    by_name = {c["name"]: c for c in prof["columns"]}
    assert by_name["amount"]["period_candidates"] == []
    assert by_name["year"]["period_candidates"] == []
    assert by_name["city"]["period_candidates"] == []


def test_period_candidates_empty_for_mostly_unparseable_text():
    df = pd.DataFrame({"notes": ["hello", "world", "??", "x"], "target": [0, 1, 0, 1]})
    prof = profile_dataframe(df, "target")
    notes = next(c for c in prof["columns"] if c["name"] == "notes")
    assert notes["period_candidates"] == []


def test_period_candidates_samples_large_free_text_columns_without_erroring():
    """A long free-text column (bigger than the parse-probe sample) must not be flagged
    datetime-like, and profiling it must stay cheap (bounded sample, not the whole column)."""
    n = 500
    df = pd.DataFrame({
        "description": [f"lorem ipsum dolor sit amet number {i} et cetera" for i in range(n)],
        "target": list(range(n)),
    })
    prof = profile_dataframe(df, "target")
    desc = next(c for c in prof["columns"] if c["name"] == "description")
    assert desc["period_candidates"] == []


def test_description_context_wraps_and_truncates():
    from maestra.profiling import description_context

    assert description_context(None) is None
    assert description_context("   ") is None
    out = description_context("KitchenQual: Ex > Gd > TA")
    assert "KitchenQual: Ex > Gd > TA" in out
    assert "Dataset description" in out          # prompt-ready header
    long = description_context("x" * 10_000, max_chars=100)
    assert "[... truncated]" in long and len(long) < 400
