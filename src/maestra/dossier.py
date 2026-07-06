"""HTML dossier: a run's full evidence — what was tried, measured, rejected, and why — as one
clickable, dependency-free static file. Verdict first (a traffic light + one stakeholder
sentence), the data-science evidence collapsible (`<details>`) beneath it: two reading depths in
one document. The same renderer feeds the pre-modelling audit report (:func:`render_audit`).

Division of labour, matching the project invariant: the LLM writes ONLY the prose — the
stakeholder verdict sentence and the metric-in-target-units notes (see :func:`dossier_narrative`,
mocked in tests). It explains; it never decides. The traffic-light colour and every number are
derived deterministically here, so :func:`render_dossier` is pure and offline-testable — pass the
sentences in, or omit them for deterministic fallbacks.

One static HTML file, inline CSS, no external assets, no JS framework (`<details>` is the only
interactivity), and no new dependency (f-strings + :mod:`html` escaping).
"""
from __future__ import annotations

import html

_LIGHT = {  # (background, label) per traffic-light colour
    "green": ("#1a7f37", "GREEN"),
    "yellow": ("#9a6700", "YELLOW"),
    "red": ("#cf222e", "RED"),
}


def _esc(value) -> str:
    """HTML-escape any value (numbers/None included) for safe interpolation."""
    return html.escape("" if value is None else str(value))


def _verdict_light(result) -> tuple[str, str]:
    """Deterministic traffic light for a completed run + a default one-line verdict. The LLM may
    replace the SENTENCE, never the colour — it explains, it does not decide."""
    if getattr(result, "training", None) is None:
        return "red", "The run did not produce a model — there is no result to trust."
    auc = getattr(result, "adversarial_auc", None)
    if auc is not None and auc >= 0.8:
        return "red", (f"Train and test are easily told apart (adversarial AUC {auc:.2f}) — this "
                       "estimate will not hold on the real test set.")
    if auc is not None and auc >= 0.6:
        return "yellow", (f"A mild train/test shift (adversarial AUC {auc:.2f}) means the estimate "
                          "may read optimistic — see the limitations below.")
    if getattr(result, "cv", None) is None:
        return "yellow", ("The estimate comes from a single holdout, not cross-validation — treat "
                          "it as noisier than a CV number.")
    return "green", ("A leakage-free cross-validation produced this estimate and train/test look "
                     "alike — the number is trustworthy as far as this data goes.")


def collect_interventions(result) -> list[dict]:
    """Normalise every measured intervention (generated features, Skeptic keeps, target framing)
    into one shape: ``{name, kind, proposed_by, delta, reason, accepted}``. Rejected interventions
    are included on equal footing with accepted ones — that visibility is the point of the dossier.
    ``delta`` is the paired CV improvement (``None`` when the arbiter never measured it)."""
    rows: list[dict] = []
    for h in getattr(result, "hybrid", None) or []:
        rows.append({"name": h.get("name"), "kind": "generated_feature",
                     "proposed_by": f"codegen:{h.get('source')}", "delta": h.get("cv_delta"),
                     "reason": h.get("reason"), "accepted": bool(h.get("kept"))})
    for s in getattr(result, "skeptic", None) or []:
        rows.append({"name": f"keep:{s.get('column')}", "kind": "skeptic_keep",
                     "proposed_by": "skeptic", "delta": s.get("cv_delta"),
                     "reason": s.get("reason"), "accepted": bool(s.get("vetoed"))})
    tf = getattr(result, "target_framing", None)
    if tf and tf.get("transform") and tf.get("transform") != "none":
        rows.append({"name": f"target:{tf.get('transform')}", "kind": "target_framing",
                     "proposed_by": "target_framing", "delta": tf.get("cv_delta"),
                     "reason": "improved" if tf.get("accepted") else "no_improvement",
                     "accepted": bool(tf.get("accepted"))})
    return rows


def _metric_note(metric: str, greater_is_better: bool, notes: dict | None) -> str:
    """The LLM's target-units translation for ``metric`` if given, else a deterministic fallback —
    so no raw metric name is ever shown without a plain-language note."""
    if notes and metric in notes:
        return notes[metric]
    direction = "higher is better" if greater_is_better else "lower is better"
    return f"{direction} — the model's score in the target's own units."


def _limitations(result) -> list[str]:
    """Auto-derived caveats: what was switched off, and what the numbers cannot support."""
    out: list[str] = []
    if getattr(result, "cv", None) is None:
        out.append("No cross-validation — the estimate is a single holdout split (noisier).")
    if getattr(result, "fold_strategy", None) is None:
        out.append("The Validation Strategist did not run — folds are random/stratified, which "
                   "lies if the data is grouped or temporal.")
    if getattr(result, "target_framing", None) is None:
        out.append("Target framing was not evaluated (a skewed target may want a log transform).")
    auc = getattr(result, "adversarial_auc", None)
    if auc is not None and auc >= 0.6:
        out.append(f"Train and test differ (adversarial AUC {auc:.2f}) — the estimate may not "
                   "transfer to the real test set.")
    return out


def _section(title: str, body: str, *, open_: bool = False) -> str:
    return (f'<details{" open" if open_ else ""}><summary>{_esc(title)}</summary>'
            f'<div class="body">{body}</div></details>')


def _kv_table(rows: list[tuple[str, str]]) -> str:
    cells = "".join(f"<tr><th>{_esc(k)}</th><td>{v}</td></tr>" for k, v in rows)
    return f"<table class='kv'>{cells}</table>"


def _interventions_html(rows: list[dict]) -> str:
    if not rows:
        return "<p class='muted'>No interventions were proposed on this run.</p>"
    head = ("<tr><th>Intervention</th><th>Kind</th><th>Proposed by</th><th>Δ CV</th>"
            "<th>Reason</th><th>Adopted</th></tr>")
    body = ""
    for r in rows:
        delta = "—" if r["delta"] is None else f"{r['delta']:+.4g}"
        mark = "<span class='yes'>✓ kept</span>" if r["accepted"] else "<span class='no'>✗ dropped</span>"
        body += (f"<tr><td>{_esc(r['name'])}</td><td>{_esc(r['kind'])}</td>"
                 f"<td>{_esc(r['proposed_by'])}</td><td>{_esc(delta)}</td>"
                 f"<td>{_esc(r['reason'])}</td><td>{mark}</td></tr>")
    return f"<table class='data'>{head}{body}</table>"


_CSS = """
:root { font-family: -apple-system, system-ui, sans-serif; line-height: 1.5; }
body { max-width: 820px; margin: 2rem auto; padding: 0 1rem; color: #1f2328; }
.verdict { display: flex; align-items: center; gap: .8rem; padding: 1rem 1.2rem;
  border-radius: 10px; background: #f6f8fa; margin-bottom: 1.4rem; }
.light { color: #fff; font-weight: 700; font-size: .8rem; letter-spacing: .04em;
  padding: .3rem .6rem; border-radius: 6px; white-space: nowrap; }
.verdict p { margin: 0; font-size: 1.05rem; }
details { border: 1px solid #d0d7de; border-radius: 8px; margin: .6rem 0; }
summary { cursor: pointer; padding: .6rem .9rem; font-weight: 600; }
.body { padding: 0 .9rem .9rem; }
table { border-collapse: collapse; width: 100%; font-size: .92rem; }
table.kv th { text-align: left; width: 34%; color: #57606a; font-weight: 500; vertical-align: top; }
table.kv td, table.kv th { padding: .25rem .4rem; }
table.data th, table.data td { border: 1px solid #d0d7de; padding: .35rem .5rem; text-align: left; }
table.data th { background: #f6f8fa; }
.yes { color: #1a7f37; font-weight: 600; } .no { color: #57606a; }
.muted { color: #57606a; } .note { color: #57606a; font-size: .9rem; }
ul { margin: .3rem 0; padding-left: 1.2rem; } code { background: #eff1f3; padding: 0 .3rem; border-radius: 4px; }
h1 { font-size: 1.5rem; } footer { color: #8c959f; font-size: .82rem; margin-top: 2rem; }
"""


def _page(title: str, verdict_html: str, sections: str) -> str:
    return (f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<title>{_esc(title)}</title><style>{_CSS}</style></head><body>"
            f"<h1>{_esc(title)}</h1>{verdict_html}{sections}"
            f"<footer>Generated by Maestra — every number is measured, not asserted; "
            f"rejected interventions are shown on equal footing with accepted ones.</footer>"
            f"</body></html>")


def render_dossier(result, *, run_record: dict | None = None,
                   verdict_sentence: str | None = None,
                   metric_notes: dict | None = None) -> str:
    """Render a run's evidence dossier as one standalone HTML string.

    Args:
        result: A :class:`~maestra.pipeline.PipelineResult` (duck-typed to avoid an import cycle).
        run_record: Optional logged record (e.g. a ``runs.jsonl`` line or a multi-seed benchmark
            row); its ``mde`` is surfaced in the CV section when present.
        verdict_sentence: The LLM's stakeholder verdict (overrides the deterministic default
            sentence — never the colour). ``None`` uses the deterministic default.
        metric_notes: ``{metric_name: target-units sentence}`` from the LLM; missing metrics fall
            back to a deterministic direction note. Ensures no raw metric appears untranslated.

    Returns:
        A complete ``<!DOCTYPE html>`` document — inline CSS, no external assets.
    """
    colour, default_sentence = _verdict_light(result)
    bg, label = _LIGHT[colour]
    sentence = verdict_sentence or default_sentence
    verdict_html = (f"<div class='verdict'><span class='light' style='background:{bg}'>{label}"
                    f"</span><p>{_esc(sentence)}</p></div>")

    t = getattr(result, "training", None)
    cv = getattr(result, "cv", None)
    fs = getattr(result, "fold_strategy", None) or {}
    sections = []

    # (2) Setup
    setup_rows = [
        ("Problem type", _esc(getattr(t, "problem_type", None))),
        ("Eval metric", _esc(getattr(t, "eval_metric", None) or (cv.eval_metric if cv else None))),
        ("Columns", _esc(f"{getattr(result, 'n_cols_before', '?')} → "
                         f"{getattr(result, 'n_cols_after', '?')} (after cleaning + FE)")),
        ("Fold strategy", _esc(fs.get("strategy", "random (advisor off)"))
         + (f" on <code>{_esc(fs.get('group_column') or fs.get('time_column') or fs.get('period_column'))}</code>"
            if (fs.get("group_column") or fs.get("time_column") or fs.get("period_column")) else "")),
    ]
    if fs.get("rationale"):
        setup_rows.append(("Advisor reasoning", f"<span class='note'>{_esc(fs['rationale'])}</span>"))
    sections.append(_section("Setup", _kv_table(setup_rows), open_=True))

    # (3) Interventions — accepted and rejected, equally visible
    sections.append(_section("Interventions (proposed → measured → kept or dropped)",
                             _interventions_html(collect_interventions(result))))

    # (4) Cross-validation result, in target units
    if cv is not None:
        note = _metric_note(cv.eval_metric, cv.greater_is_better, metric_notes)
        folds = ", ".join(f"{s:.4g}" for s in cv.fold_scores)
        cv_rows = [
            (f"CV {cv.eval_metric}", f"<b>{cv.mean:.4g}</b> ± {cv.std:.4g} "
             f"<span class='note'>({note})</span>"),
            (f"{cv.n_folds}-fold scores", _esc(f"[{folds}]")),
        ]
        if run_record and run_record.get("mde") is not None:
            cv_rows.append(("Minimum detectable effect",
                            _esc(f"{run_record['mde']:.4g} (a smaller delta is 'undecided', not zero)")))
        sections.append(_section("Cross-validation estimate", _kv_table(cv_rows), open_=True))
    elif t is not None and getattr(t, "metrics", None):
        note_rows = [(k, f"{_esc(v)} <span class='note'>({_metric_note(k, True, metric_notes)})</span>")
                     for k, v in t.metrics.items()]
        sections.append(_section("Holdout metrics", _kv_table(note_rows)))

    # (5) CV budget
    budget = getattr(result, "cv_budget", None)
    if budget:
        cap = budget.get("limit")
        sections.append(_section("Intervention cost (CV budget)", _kv_table([
            ("Trial CVs spent", _esc(budget.get("trials_spent", 0))),
            ("Budget", _esc("unlimited" if cap is None else cap)),
        ])))

    # (6) Limitations — auto-derived
    lims = _limitations(result)
    if lims:
        items = "".join(f"<li>{_esc(x)}</li>" for x in lims)
        sections.append(_section("Limitations", f"<ul>{items}</ul>", open_=True))

    return _page("Maestra run dossier", verdict_html, "".join(sections))


def write_dossier(result, path: str, **kwargs) -> None:
    """Render (:func:`render_dossier`) and write the dossier HTML to ``path``."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(render_dossier(result, **kwargs))


_NARRATIVE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "verdict_sentence": {"type": "string",
                             "description": "One sentence, stakeholder language (no jargon), on "
                                            "whether this run's number can be trusted and why."},
        "metric_notes": {"type": "object", "additionalProperties": {"type": "string"},
                         "description": "Per metric name, one plain-language sentence translating "
                                        "the score into the target's own units / meaning."},
    },
    "required": ["verdict_sentence", "metric_notes"],
}

_NARRATIVE_PROMPT = (
    "You translate a finished AutoML run into plain language for a non-technical stakeholder. You "
    "are given FACTS as JSON and a deterministic traffic-light colour and verdict. Write ONE "
    "stakeholder-language sentence for the verdict (consistent with the given colour — you explain "
    "it, you do NOT change it) and, for each metric, ONE sentence translating the number into the "
    "target's own units or meaning. Use the supplied numbers VERBATIM; invent nothing."
)


def dossier_narrative(model: str, result) -> dict:
    """Ask the LLM for the dossier's prose only — the stakeholder verdict sentence and the
    per-metric target-units notes. Returns ``{"verdict_sentence", "metric_notes"}`` to feed
    straight into :func:`render_dossier`. The LLM explains; the colour and numbers are decided
    here. Mocked in tests; the pure renderer never needs it."""
    import json

    from maestra.llm import call_structured

    colour, default_sentence = _verdict_light(result)
    cv = getattr(result, "cv", None)
    t = getattr(result, "training", None)
    facts = {
        "traffic_light": colour,
        "deterministic_verdict": default_sentence,
        "cv": ({"metric": cv.eval_metric, "mean": cv.mean, "std": cv.std,
                "greater_is_better": cv.greater_is_better} if cv else None),
        "holdout_metrics": (getattr(t, "metrics", None) or None) if t else None,
        "interventions": collect_interventions(result),
    }
    return call_structured(
        model=model,
        system_prompt=_NARRATIVE_PROMPT,
        user_prompt=f"Run facts (JSON):\n{json.dumps(facts, ensure_ascii=False, indent=2, default=float)}",
        tool_name="write_dossier_prose",
        tool_description="Stakeholder verdict sentence + per-metric target-units notes.",
        parameters_schema=_NARRATIVE_SCHEMA,
    )
