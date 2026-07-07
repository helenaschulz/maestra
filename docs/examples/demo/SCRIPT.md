# P2b demo script

~3 minutes. Claude Desktop (or Code) with the `maestra` MCP server configured (see
`docs/MCP.md`), `docs/examples/demo/demand.csv` as the working file. Every number below is
copied verbatim from a real run against this exact file (`gpt-4o`, 2026-07-07) — not invented —
so if a re-run drifts (LLM non-determinism, a library update), fix the number here, not the
recording.

Read the lines verbatim; the "Claude says" blocks are illustrative paraphrase (the model's
actual wording will vary run to run) — the numbers and verdicts inside them must not.

---

**[0:00] Setup shot (no narration needed on camera):** Claude Desktop open, `demand.csv` visible
in Finder/terminal for orientation. One sentence to camera:

> "This is Kaggle's Bike Sharing Demand dataset — hourly rental counts. I want to know two
> things: can I trust a model built on this, and how good would it actually be."

**[0:15] Prompt 1 — type into Claude:**

> I have `docs/examples/demo/demand.csv`, target column `count`. Is this safe to model, and can
> I trust a validation score on it?

**Expected tool call:** `audit_csv("docs/examples/demo/demand.csv", "count")`

**Expected verdict (real, 2026-07-07):**
```json
{"verdict": "ok", "risk_level": "high", "fold_strategy": "time",
 "target_leaks": [["registered", 0.971]], "leakage_warnings": []}
```

**Claude says (paraphrase):** "High risk. Two problems: this is a time-ordered series
(`datetime`), so folds need to respect time, not be random — and `registered` correlates 0.97
with the target `count`, which is a near-certain leak (`count = casual + registered`). A model
trained with that column in would look great and be useless in production."

**[1:00] Prompt 2 — type into Claude:**

> How much would a naive random split have overstated things?

**Expected tool call:** `check_validation("docs/examples/demo/demand.csv", "count")`

**Expected verdict (real, 2026-07-07):**
```json
{"verdict": "ok",
 "recommended_strategy": {"strategy": "time", "column": "datetime"},
 "naive_cv": {"metric": "root_mean_squared_error", "mean": -5.29},
 "recommended_cv": {"metric": "root_mean_squared_error", "mean": -20.27},
 "optimism_gap": 14.98, "direction": "optimistic (dangerous)"}
```

**Claude says (paraphrase):** "Confirmed and measured, not just asserted: a naive random split
reports an RMSE around 5 — a time-ordered split, the honest one, reports around 20. That's a
15-point optimism gap. Anyone who validated this randomly would have shipped a model believing
it was four times better than it is."

*(Timing note: this step runs two real cross-validations, ~60–70s wall time — cut here in
editing, or narrate over the wait.)*

**[2:00] Prompt 3 — type into Claude:**

> Given all that, what quality could I actually get, and what drives it?

**Expected tool call:** `feasibility("docs/examples/demo/demand.csv", "count")`

**Expected verdict (real, 2026-07-07):**
```json
{"verdict": "ok",
 "achievable_quality": {"metric": "root_mean_squared_error", "mean": -77.6, "std": 31.1},
 "fold_strategy": "time",
 "strongest_drivers": [{"feature": "datetime_hour", "importance": 178.1},
                       {"feature": "datetime_year", "importance": 54.2},
                       {"feature": "workingday", "importance": 51.9}],
 "risk_level": "high"}
```

**Claude says (paraphrase):** "With honest, time-ordered folds and the leak removed: expect an
RMSE around 78 (± 31 — real spread, not a false-precision single number). Hour of day is by far
the strongest driver, followed by year (the service grew over time) and whether it's a working
day. The leakage risk from the audit still stands — that's already accounted for in this number,
the leak was dropped before training."

*(Timing note: this step runs a conservative full pipeline, ~3 min wall time — cut or narrate
over the wait, same as step 2.)*

**[2:45] Closing line to camera:**

> "Three questions, three measured answers — not three guesses. That's the difference: every
> number Maestra hands back was checked against the data itself, not asserted by the model."

---

## Generalprobe note (2026-07-07)

Run live against a real `gpt-4o` key and real AutoGluon (not mocked) before this script was
finalized. Found and fixed two real bugs in the MCP server itself along the way (see the
`p2-mcp-server` branch's fixup commit): `feasibility`'s feature-importance call crashed on the
raw input schema (fixed to use AutoGluon's own post-processed features), and both
`check_validation`'s and `feasibility`'s wall-clock backstops had almost no headroom over their
own nominal AutoGluon time on this exact file (loosened after measuring the real numbers above).
No further script corrections were needed — the verdicts above are what a live run actually
returns today.
