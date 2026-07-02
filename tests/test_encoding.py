"""Tests for ordinal encoding (M2). The LLM is never called — proposals are dicts; only the
deterministic fit/transform and its guards are exercised."""
import pandas as pd

from maestra.encoding import fit_ordinal_encodings


def _df():
    return pd.DataFrame({
        "qual": ["Gd", "Ex", "TA", "Po", "Fa", "Gd"],       # ordinal, order Po<Fa<TA<Gd<Ex
        "colour": ["red", "blue", "red", "green", "blue", "red"],  # nominal, no order
        "price": [200, 300, 150, 100, 120, 210],
    })


def test_ordinal_maps_to_worst_to_best_rank():
    enc = fit_ordinal_encodings(_df(), [
        {"column": "qual", "order": ["Po", "Fa", "TA", "Gd", "Ex"], "reason": "quality rating"}
    ], target="price")
    out = enc.transform(_df())
    assert out["qual"].tolist() == [3, 4, 2, 0, 1, 3]        # ranks, not strings
    assert enc.records[0]["column"] == "qual" and enc.records[0]["coverage"] == 1.0
    assert any("ORDINAL 'qual'" in line for line in enc.log)


def test_unseen_value_becomes_nan_not_an_error():
    df = pd.DataFrame({"size": ["S", "M", "XL"], "y": [1, 2, 3]})  # 'XL' not in the order
    enc = fit_ordinal_encodings(df, [
        {"column": "size", "order": ["S", "M", "L"], "reason": "size"}], target="y")
    # coverage 2/3 >= 0.5 so the map is kept; the unseen 'XL' maps to NaN
    out = enc.transform(df)
    assert out["size"].tolist()[:2] == [0, 1] and pd.isna(out["size"].iloc[2])


def test_hallucinated_order_is_skipped():
    # order matches none of the observed values -> the LLM invented categories
    enc = fit_ordinal_encodings(_df(), [
        {"column": "qual", "order": ["awful", "meh", "great"], "reason": "guessed"}], target="price")
    assert enc.maps == {}
    assert any("matches only" in line for line in enc.log)


def test_numeric_and_missing_and_target_columns_are_skipped():
    df = _df().assign(y=[0, 1, 0, 1, 0, 1])  # separate target so 'price' tests the numeric guard
    enc = fit_ordinal_encodings(df, [
        {"column": "price", "order": ["a", "b"], "reason": "numeric"},
        {"column": "ghost", "order": ["a", "b"], "reason": "absent"},
        {"column": "y", "order": ["a", "b"], "reason": "target"},
    ], target="y")
    assert enc.maps == {}
    joined = " ".join(enc.log)
    assert "already numeric" in joined and "not present" in joined and "is the target" in joined


def test_degenerate_order_is_skipped():
    enc = fit_ordinal_encodings(_df(), [
        {"column": "qual", "order": ["Gd"], "reason": "single"},
        {"column": "colour", "order": ["red", "red"], "reason": "dup"},
    ], target="price")
    assert enc.maps == {}


def test_transform_is_a_pure_replacement_leaving_other_columns_untouched():
    enc = fit_ordinal_encodings(_df(), [
        {"column": "qual", "order": ["Po", "Fa", "TA", "Gd", "Ex"], "reason": "r"}], target="price")
    out = enc.transform(_df())
    assert out["colour"].tolist() == _df()["colour"].tolist()  # nominal column unchanged
    assert out["price"].tolist() == _df()["price"].tolist()


# --- pipeline wiring (LLM + engine mocked) ------------------------------------------

def test_pipeline_applies_ordinal_and_reports_it(monkeypatch):
    from maestra import pipeline
    from maestra.engine import TrainingResult
    from maestra.validation import CVResult

    df = pd.DataFrame({"qual": ["Po", "Ex", "TA", "Gd"] * 3, "y": [0, 1, 0, 1] * 3})
    seen = {}

    def fake_cross_validate(df_, target, **kwargs):
        seen["qual_dtype"] = str(df_["qual"].dtype)  # numeric here means ordinal ran first
        return CVResult("accuracy", "binary", [0.8, 0.8], 0.8, 0.0, 2, False, True)

    monkeypatch.setattr(pipeline, "propose_ordinal_encodings",
                        lambda *a, **k: {"encodings": [
                            {"column": "qual", "order": ["Po", "Fa", "TA", "Gd", "Ex"], "reason": "rating"}]})
    monkeypatch.setattr(pipeline, "cross_validate", fake_cross_validate)
    monkeypatch.setattr(pipeline, "fit_predictor",
                        lambda *a, **k: TrainingResult("binary", "accuracy", pd.DataFrame(), {}))

    result = pipeline.run_pipeline(df, "y", model="m", test_size=0.2, time_limit=1, seed=0,
                                   model_dir="x", cv_folds=2, ordinal=True, use_llm=False, use_fe=False)

    assert seen["qual_dtype"].startswith(("int", "float"))     # the CV saw ranks, not strings
    assert result.ordinal["encodings"][0]["column"] == "qual"  # provenance reported
    assert any("ORDINAL 'qual'" in line for line in result.ordinal["log"])
