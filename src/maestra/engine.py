"""AutoGluon engine — the only place numbers are crunched.

The LLM never computes. Train/holdout split, model search, hyperparameter tuning and
metric calculation all happen here, inside AutoGluon. This module returns plain data
(leaderboard, metrics) and prints nothing, so it is easy to test and reuse.

Also defines the minimal ``Engine`` fit/predict protocol (P3): ``SklearnEngine`` lets
``compare()`` (see ``maestra.__init__``) run the arbiter over ANY sklearn-compatible
estimator, not just AutoGluon. ``LightGBMEngine`` is a fast proxy engine (LightGBM is
already a transitive AutoGluon dependency) for quick checks where a full AutoGluon fit
would be overkill. ``AutoGluonEngine`` exists for API symmetry with ``SklearnEngine``
and is usable standalone, but ``validation.py::cross_validate``'s existing AutoGluon path
(presets, sample_weight, custom Scorer objects, target transforms) is untouched code, not
routed through it — those are AutoGluon-native concepts with no sklearn equivalent, so
unifying them into one fit/predict interface would either lose functionality or bloat the
interface. ``cross_validate`` special-cases ``engine is None`` (and ``AutoGluonEngine``
instances) to keep that path byte-identical.

``TabularPredictor`` is imported LAZILY (inside the functions that construct one), not at
module level: this lets ``import maestra`` (hence ``SklearnEngine``/``compare()``) succeed
even where AutoGluon isn't installed (P3's Colab notebook uses a ``--no-deps`` install for
exactly this). ``autogluon.tabular[all]`` stays a required ``pyproject.toml`` dependency —
this is an import-time nicety, not a packaging change (see P3's docs/RESULTS.md note on why
making the *dependency* itself optional was investigated and NOT done).

N6 (AutoGluon >= 1.6): :func:`resolve_inner_validation` translates the OUTER fold strategy into
AutoGluon's ``validation_structure`` so its INTERNAL model selection respects groups/time too;
``train_and_evaluate``/``fit_predictor`` declare it on ignored carrier columns (feature set
unchanged). ``None`` keeps every AutoGluon call byte-identical to pre-N6.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass
class TrainingResult:
    """Outcome of a single train+evaluate run on a holdout set."""

    problem_type: str
    eval_metric: str
    leaderboard: pd.DataFrame
    metrics: dict[str, float]
    predictor: object | None = None  # the fitted TabularPredictor, for downstream prediction
    val_score: float | None = None   # AutoGluon's best internal validation score (NOT the holdout)


def autogluon_version() -> str | None:
    """Installed ``autogluon.tabular`` version (for run records, N6), or ``None`` if absent.
    Read from package metadata, so it never imports AutoGluon itself."""
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover
        return None
    try:
        return version("autogluon.tabular")
    except PackageNotFoundError:
        return None


# Fallback when AutoGluon isn't importable: preset names whose config turns on bagging +
# stacking (auto_stack / dynamic_stacking). ``temporal_forward_only`` raises with stacking in
# AutoGluon 1.6, so this must err on the side of "stacks".
_STACKING_PRESETS = ("best", "best_quality", "bq", "high", "high_quality", "hq", "good",
                     "good_quality", "gq", "extreme", "extreme_quality", "eq", "tabarena")


def presets_stack(presets) -> bool:
    """True if ``presets`` (a name, a list of names, or ``None``) may enable stacking.

    Read from AutoGluon's own preset table when installed (``auto_stack``, a truthy
    ``dynamic_stacking`` incl. ``"auto"``, or ``num_stack_levels > 0`` all count — conservative),
    else from :data:`_STACKING_PRESETS`. ``None`` = AutoGluon's default ``medium_quality``: no.
    """
    if presets is None:
        return False
    names = [presets] if isinstance(presets, str) else list(presets)
    try:
        from autogluon.tabular.configs.presets_configs import (
            tabular_presets_alias,
            tabular_presets_dict,
        )
    except ImportError:
        return any(str(n) in _STACKING_PRESETS for n in names)
    for name in names:
        cfg = tabular_presets_dict.get(tabular_presets_alias.get(name, name), {})
        if (cfg.get("auto_stack") or cfg.get("dynamic_stacking")
                or (cfg.get("num_stack_levels") or 0) > 0):
            return True
    return False


def inner_validation_structure(strategy: str | None, *, group_column: str | None = None,
                               time_column: str | None = None, stacking: bool = False
                               ) -> tuple[dict | None, str]:
    """Map the OUTER fold strategy onto AutoGluon 1.6's ``validation_structure`` (N6).

    Deterministic, no own inference: the caller passes the strategy its outer folds actually
    use. ``group`` -> ``{"group_on": col}``; ``time``/``time_local``/rolling-origin ->
    ``{"time_on": col, "temporal_forward_only": True}`` (without ``temporal_forward_only`` when
    stacking is active — AutoGluon then uses leave-one-block-out, noted, never silent);
    ``random``/``None`` -> ``None`` (AutoGluon is called exactly as before N6).

    Returns ``(structure_or_None, note)``; the note is the human-readable record line.
    """
    if strategy == "group":
        if not group_column:
            raise ValueError("strategy 'group' needs a group_column")
        return {"group_on": group_column}, f"group-disjoint on {group_column}"
    if strategy in ("time", "time_local", "rolling_origin"):
        if not time_column:
            raise ValueError(f"strategy {strategy!r} needs a time_column")
        if stacking:
            return ({"time_on": time_column},
                    f"time-ordered on {time_column} (stacking active: temporal_forward_only "
                    f"dropped, AutoGluon uses leave-one-block-out)")
        return ({"time_on": time_column, "temporal_forward_only": True},
                f"forward in time on {time_column}")
    return None, "random"


def outer_strategy(*, group_column: str | None = None, time_column: str | None = None,
                   period_column: str | None = None, splitter=None) -> str:
    """The outer fold strategy ``validation.py::_make_folds`` actually builds from these
    arguments — same precedence (splitter > group > time_local > time > random), so the inner
    structure can never disagree with the outer folds."""
    if splitter is not None:
        return "rolling_origin" if time_column is not None else "custom_splitter"
    if group_column is not None:
        return "group"
    if time_column is not None and period_column is not None:
        return "time_local"
    if period_column is not None:
        raise ValueError("period_column without time_column: inner validation structure "
                         "undefined (N6 stop condition)")
    if time_column is not None:
        return "time"
    return "random"


def resolve_inner_validation(columns, *, group_column: str | None = None,
                             time_column: str | None = None, period_column: str | None = None,
                             splitter=None, presets=None, enabled: bool = True
                             ) -> tuple[dict | None, dict]:
    """One call for every AutoGluon caller (N6): outer strategy -> ``validation_structure``
    (real column names), checked against ``columns`` -- the columns of the frame the structure
    values will be read from (the caller's ``structure_source``, normally the raw rows).

    Returns ``(structure_or_None, record)``; ``record`` = ``{"strategy", "structure", "note"}``
    is what goes into run records and the dossier. A missing structure column drops the
    structure and the note says so (never a silent fallback). ``enabled=False`` is the
    explicit pre-N6 arm for A/B measurement.
    """
    strategy = outer_strategy(group_column=group_column, time_column=time_column,
                              period_column=period_column, splitter=splitter)
    if not enabled:
        return None, {"strategy": strategy, "structure": None,
                      "note": "random (inner structure disabled)"}
    structure, note = inner_validation_structure(
        strategy, group_column=group_column, time_column=time_column,
        stacking=presets_stack(presets))
    if strategy == "custom_splitter":
        note = "random (custom splitter without time_column: no column to declare)"
    if structure is not None:
        missing = [c for k, c in structure.items()
                   if k in ("group_on", "time_on") and c not in set(columns)]
        if missing:
            structure = None
            note = f"random (structure column {missing[0]} not available to the fit)"
    return structure, {"strategy": strategy, "structure": structure, "note": note}


def inner_validation_label(record: dict | None) -> str:
    """Dossier wording for a ``resolve_inner_validation`` record (German, like the spec)."""
    struct = (record or {}).get("structure") or {}
    if "group_on" in struct:
        return f"gruppen-disjunkt auf {struct['group_on']}"
    if "time_on" in struct:
        if struct.get("temporal_forward_only"):
            return f"zeitlich vorwärts auf {struct['time_on']}"
        return f"zeitlich geblockt auf {struct['time_on']} (leave-one-block-out, Stacking aktiv)"
    return "zufällig"


def split(df: pd.DataFrame, test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ``df`` into ``(train, holdout)`` by random row sampling.

    A simple random split is enough for this project; AutoGluon does its own internal
    validation split on the training portion for model selection. N6 open point: this
    outer holdout split knows no groups/time — deliberately unchanged (see docs/RESULTS.md).
    """
    holdout = df.sample(frac=test_size, random_state=seed)
    train = df.drop(index=holdout.index)
    return train, holdout


# Hidden carrier columns for the structure (N6). AutoGluon resolves ``validation_structure`` on
# the raw frame, so the structure is declared on these copies -- listed in the learner's
# ``ignored_columns`` -- instead of on the real columns. Effects: (1) the feature set is exactly
# what it was before N6 (the real column stays whatever feature it was; the carrier is never a
# feature); (2) a string date column works (AutoGluon's ``time_on`` needs numeric/datetime, and
# converting the REAL column would turn it into datetime features); (3) a structure column that
# cleaning/FE removed can still be carried in from the raw rows (``structure_source``).
_CARRIER = {"group_on": "__maestra_group_on__", "time_on": "__maestra_time_on__"}


def _time_as_int(values: pd.Series, name: str) -> pd.Series:
    if values.dtype.kind in "iuf":
        return values
    if values.dtype.kind != "M":
        values = pd.to_datetime(values, errors="coerce", format="mixed")
    if values.isna().any():
        raise ValueError(f"time column {name!r} has values that don't parse as time; "
                         f"cannot declare it as AutoGluon's time_on")
    return values.astype("int64")


def _fit_kwargs_with_structure(train: pd.DataFrame, validation_structure: dict | None,
                               structure_source: pd.DataFrame | None):
    """``(train, fit_kwargs, init_kwargs)``. ``None`` -> ``(train, {}, {})``: AutoGluon is
    constructed and fitted exactly as before N6 (no new keyword, no new column)."""
    if validation_structure is None:
        return train, {}, {}
    source = train if structure_source is None else structure_source
    ag_structure, carriers = {}, {}
    for key, value in validation_structure.items():
        if key in _CARRIER:
            col = source.loc[train.index, value]
            carriers[_CARRIER[key]] = (_time_as_int(col, value) if key == "time_on"
                                       else col).to_numpy()
            ag_structure[key] = _CARRIER[key]
        else:
            ag_structure[key] = value
    train = train.assign(**carriers)
    return (train, {"validation_structure": ag_structure},
            {"learner_kwargs": {"ignored_columns": list(carriers)}})


def train_and_evaluate(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    target: str,
    time_limit: int,
    model_dir: str,
    eval_metric: str | None = None,
    presets: str | None = None,
    sample_weight: str | None = None,
    validation_structure: dict | None = None,
    structure_source: pd.DataFrame | None = None,
) -> TrainingResult:
    """Fit an AutoGluon predictor and evaluate it on the holdout set.

    Args:
        train: Training rows (features + target).
        holdout: Held-out rows for honest evaluation.
        target: Name of the label column.
        time_limit: Training budget in seconds.
        model_dir: Where AutoGluon persists the trained models.
        presets: AutoGluon quality preset (e.g. ``"best_quality"`` for multi-layer
            stacking + bagging). ``None`` uses AutoGluon's default (``"medium_quality"``,
            fast). Whatever is chosen must be identical on the CV folds and the final model,
            or the CV↔LB gap compares two different model configurations.
        sample_weight: Name of a per-row weight column (excluded from features by AutoGluon)
            used in BOTH training and metric evaluation — e.g. Walmart's holiday-week ×5
            weighting. Must be present in ``train``/``holdout``. ``None`` = unweighted.
        validation_structure: N6. ``{"group_on": col}`` / ``{"time_on": col, ...}`` in REAL column
            names (from :func:`resolve_inner_validation`), so AutoGluon 1.6's INTERNAL splits
            (early stopping, ensemble weights, ``score_val``) respect the outer group/time
            structure. Declared on ignored carrier copies of the columns (see ``_CARRIER``), so
            the feature set never changes. ``None`` = AutoGluon is called exactly as pre-N6.
        structure_source: Frame indexed like ``train`` holding the structure column(s) when
            ``train`` itself no longer has them (e.g. cleaning dropped an id column). Default:
            ``train``.

    Returns:
        A :class:`TrainingResult`. The problem type and evaluation metric are inferred
        by AutoGluon from the target column.
    """
    from autogluon.tabular import TabularPredictor

    train, fit_kw, init_kw = _fit_kwargs_with_structure(train, validation_structure,
                                                        structure_source)
    predictor = TabularPredictor(label=target, path=model_dir, eval_metric=eval_metric,
                                 sample_weight=sample_weight, **init_kw).fit(
        train, time_limit=time_limit, presets=presets, **fit_kw)
    leaderboard = predictor.leaderboard(holdout, silent=True)
    # Best internal validation score (AutoGluon's own train/val split) — used by the
    # quality gate so the holdout is never consulted for that decision.
    val_score = float(leaderboard["score_val"].max()) if "score_val" in leaderboard.columns else None
    return TrainingResult(
        problem_type=predictor.problem_type,
        eval_metric=predictor.eval_metric.name,
        leaderboard=leaderboard,
        metrics=predictor.evaluate(holdout, silent=True),
        predictor=predictor,
        val_score=val_score,
    )


def predict(predictor, X: pd.DataFrame) -> pd.Series:
    """Predict labels for ``X`` with a fitted predictor (delegates to AutoGluon)."""
    return predictor.predict(X)


def predict_proba(predictor, X: pd.DataFrame) -> pd.DataFrame:
    """Predict class probabilities for ``X`` (delegates to AutoGluon).

    Returns a DataFrame with one column per class (columns are the predictor's class
    labels, in its own order); for binary problems both classes are present. This is the
    format-neutral primitive — reshaping into a competition's submission format happens in
    the pipeline, where the sample submission's columns are known.
    """
    return predictor.predict_proba(X)


def fit_predictor(train: pd.DataFrame, target: str, time_limit: int, model_dir: str,
                  eval_metric: str | None = None, presets: str | None = None,
                  sample_weight: str | None = None,
                  validation_structure: dict | None = None,
                  structure_source: pd.DataFrame | None = None) -> TrainingResult:
    """Fit a predictor on ALL labeled rows, with no holdout evaluation.

    Used by the cross-validation path, where the honest estimate is the CV score and this
    final model exists only for prediction (submission/report). ``metrics`` is therefore
    empty; ``val_score`` is AutoGluon's internal validation score. ``presets`` /
    ``sample_weight`` / ``validation_structure`` / ``structure_source`` — see
    :func:`train_and_evaluate` (must match what the CV folds used).
    """
    from autogluon.tabular import TabularPredictor

    train, fit_kw, init_kw = _fit_kwargs_with_structure(train, validation_structure,
                                                        structure_source)
    predictor = TabularPredictor(label=target, path=model_dir, eval_metric=eval_metric,
                                 sample_weight=sample_weight, **init_kw).fit(
        train, time_limit=time_limit, presets=presets, **fit_kw)
    leaderboard = predictor.leaderboard(silent=True)
    val_score = float(leaderboard["score_val"].max()) if "score_val" in leaderboard.columns else None
    return TrainingResult(
        problem_type=predictor.problem_type,
        eval_metric=predictor.eval_metric.name,
        leaderboard=leaderboard,
        metrics={},
        predictor=predictor,
        val_score=val_score,
    )


def _looks_like_classification(y: pd.Series) -> bool:
    """Same heuristic as ``validation.py::_is_classification`` (duplicated, not imported, to
    avoid a cycle: ``validation.py`` imports FROM this module). sklearn's own
    ``type_of_target`` calls any integer-VALUED column with >2 distinct values "multiclass" —
    including plain regression targets like a rounded price or a count — so numeric targets
    additionally need a small distinct-value count to actually be a class label."""
    from sklearn.utils.multiclass import type_of_target

    y = y.dropna()
    if type_of_target(y) not in ("binary", "multiclass"):
        return False
    if y.dtype.kind in "iuf":
        return y.nunique() <= 20
    return True


class Engine(ABC):
    """Minimal fit/predict/score protocol so a cross-validation loop can run over ANY model,
    not just AutoGluon. One instance is used per fold: ``fit`` receives already-cleaned
    feature/target frames (cleaning and feature engineering stay outside the engine, in
    ``validation.py``) and returns a fitted engine; ``score`` evaluates it on a held-out fold.
    """

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Engine":
        """Fit on training rows. Returns ``self`` (fitted), so calls can be chained."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict labels for ``X``."""

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame | None:
        """Predict class probabilities, or ``None`` if the wrapped model can't (default)."""
        return None

    @abstractmethod
    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Score a fitted engine on held-out rows. Higher must always mean better — unlike
        AutoGluon's metrics (which can be true errors, e.g. RMSE), this generic path fixes
        the sklearn convention (accuracy, R², sklearn's ``neg_*`` losses) so CV means compare
        cleanly across engines without a separate ``greater_is_better`` flag.
        """


class SklearnEngine(Engine):
    """Wraps any sklearn-compatible estimator (fit/predict, optionally predict_proba). Cloned
    fresh per fold via :func:`sklearn.base.clone` so folds never share fitted state."""

    def __init__(self, estimator, *, scoring: str | None = None):
        self._template = estimator
        self._scoring = scoring
        self._fitted = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SklearnEngine":
        from sklearn.base import clone

        self._fitted = clone(self._template).fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(self._fitted.predict(X), index=X.index)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame | None:
        if not hasattr(self._fitted, "predict_proba"):
            return None
        proba = self._fitted.predict_proba(X)
        return pd.DataFrame(proba, index=X.index, columns=self._fitted.classes_)

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        if self._scoring is not None:
            from sklearn.metrics import get_scorer

            return float(get_scorer(self._scoring)(self._fitted, X, y))
        return float(self._fitted.score(X, y))  # sklearn default: accuracy / R^2, both gib=True


class LightGBMEngine(SklearnEngine):
    """A fast, cheap-to-fit proxy engine (P3) for quick checks (e.g. a future light-weight
    validation gate) where a full AutoGluon fit would be overkill. LightGBM is already a
    transitive dependency of ``autogluon.tabular[all]`` — no new install, no optional dep-group
    needed (checked 2026-07-07: ``import lightgbm`` succeeds out of the box).

    Classifier vs. regressor is auto-selected from the target's type at ``fit`` time (unknown
    at construction), matching this project's own ``_is_classification`` convention elsewhere.
    Everything else (predict/predict_proba/score) is inherited from :class:`SklearnEngine`.
    """

    def __init__(self, *, scoring: str | None = None, **params):
        super().__init__(estimator=None, scoring=scoring)
        self._params = params

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LightGBMEngine":
        from lightgbm import LGBMClassifier, LGBMRegressor

        cls = LGBMClassifier if _looks_like_classification(y) else LGBMRegressor
        self._template = cls(verbose=-1, **self._params)
        return super().fit(X, y)


class AutoGluonEngine(Engine):
    """Adapter over AutoGluon, for API symmetry with :class:`SklearnEngine` and standalone use
    (e.g. directly via ``compare()``). NOT what ``validation.py::cross_validate`` calls for its
    own default AutoGluon path — that path's presets/sample_weight/custom-Scorer/target-transform
    support has no equivalent in this minimal interface, so ``cross_validate`` keeps its
    existing, untouched code for ``engine=None``/an ``AutoGluonEngine`` instance (see module
    docstring) rather than routing through ``fit``/``predict`` here.
    """

    def __init__(self, target: str, *, model_dir: str, time_limit: int,
                eval_metric: str | None = None, presets: str | None = None):
        self._target = target
        self._model_dir = model_dir
        self._time_limit = time_limit
        self._eval_metric = eval_metric
        self._presets = presets
        self._result: TrainingResult | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "AutoGluonEngine":
        train = X.assign(**{self._target: y.to_numpy()})
        self._result = fit_predictor(train, self._target, self._time_limit, self._model_dir,
                                     eval_metric=self._eval_metric, presets=self._presets)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return predict(self._result.predictor, X)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame | None:
        if self._result.problem_type not in ("binary", "multiclass"):
            return None
        return predict_proba(self._result.predictor, X)

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        gib = getattr(self._result.predictor.eval_metric, "greater_is_better", True)
        val = X.assign(**{self._target: y.to_numpy()})
        metrics = self._result.predictor.evaluate(val, silent=True)
        raw = metrics.get(self._result.eval_metric) or next(iter(metrics.values()))
        return float(raw) if gib else -float(raw)  # normalise to this Engine's gib=True contract
