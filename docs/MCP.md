# Maestra as an MCP server

Maestra's MCP server exposes four tools to LLM frontends (Claude Desktop, Claude Code): the
non-DS channel. The frontend hands over a CSV path and a target column (plus a time column for the
forecasting audit); the tools consume Maestra's own verdicts and never build or return a model.
Every response is a structured record
(`{"verdict": "ok", ...}` or `{"verdict": "rejected", "reason": "..."}`) — never a traceback, never
a raw DataFrame.

| Tool | What it answers |
| --- | --- |
| `audit_csv(path, target, model="gpt-4o")` | Is this CSV safe to model at all? (validation design, leakage, structural traps) |
| `check_validation(path, target, model="gpt-4o")` | How must folds be built, and how optimistic is a naive random split — *measured*, not asserted |
| `feasibility(path, target, model="gpt-4o")` | What quality is achievable, what drives it, what's risky — without training a model you keep |
| `audit_backtest(path, target, time_column, series_column=None, model="gpt-4o")` | Does an existing *forecasting* setup leak? Future-known features, a *measured* naive-vs-embargoed backtest gap over rolling origins, and a null-controlled series-boundary shift check |

Each tool needs a real LLM call (the `model` argument, an OpenAI/Anthropic model name via
LiteLLM) and, for `check_validation`/`feasibility`/`audit_backtest`, real AutoGluon training under
a hard, fixed time budget — none of that is a knob the frontend can turn. Conservative defaults are baked in
by design (see `CLAUDE.md` — the LLM proposes, a deterministic gate decides).

## Install

```bash
pip install -e ".[mcp]"
```

This pulls in the optional `mcp` dependency group (the official Python MCP SDK, `FastMCP`) on
top of the core install. It is not part of `.[dev]`/`.[research]` and not installed by CI — the
core test suite never needs an MCP runtime.

You'll also need an LLM API key, same as every other Maestra entry point:

```bash
echo "OPENAI_API_KEY=sk-..." > .env
```

`maestra-mcp` loads `.env` itself on startup (same helper the `maestra`/`maestra-audit` CLIs
use) — but an MCP frontend launches the server as a bare subprocess, not through your shell, so
if you don't keep a `.env` next to where the frontend runs it, set the key directly in the
frontend's server config instead (below).

## Configure in Claude Desktop

Add an entry to Claude Desktop's MCP config (`~/Library/Application Support/Claude/claude_desktop_config.json`
on macOS):

```json
{
  "mcpServers": {
    "maestra": {
      "command": "/absolute/path/to/maestra/.venv/bin/maestra-mcp",
      "env": {
        "OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```

Use the absolute path to the `maestra-mcp` script inside your project's virtualenv (`which
maestra-mcp` after activating it) — Claude Desktop does not source your shell profile or `.env`,
so the `env` block above is the reliable way to hand it an API key. Restart Claude Desktop to
pick up the change.

## Configure in Claude Code

```bash
claude mcp add maestra /absolute/path/to/maestra/.venv/bin/maestra-mcp
```

Or add the same `mcpServers` block as above to a project's `.mcp.json`. Claude Code inherits your
shell environment when launching local MCP servers, so if `OPENAI_API_KEY` is already exported
(or sits in a `.env` in the directory `maestra-mcp` runs from), no extra `env` block is needed.

## Example dialog

> **You:** I have `data/churn.csv`, target column `churned`. Can you tell me if this is
> trustworthy to model, and roughly how well?
>
> **Claude** *(calls `audit_csv("data/churn.csv", "churned")`)*
> → `{"verdict": "ok", "risk_level": "elevated", "fold_strategy": "group", "target_leaks": [],
> "leakage_warnings": [], "html_report": "data/churn.csv.audit.html"}`
>
> Claude: "The audit flags an elevated risk: rows repeat per customer, so folds need to be
> grouped by customer, not random — otherwise your validation score will look better than it
> really is. No target leakage found. Full report: `data/churn.csv.audit.html`. Want me to
> quantify how much a naive split would have overstated things?"
>
> **You:** Yes.
>
> **Claude** *(calls `check_validation("data/churn.csv", "churned")`)*
> → `{"verdict": "ok", "recommended_strategy": {"strategy": "group", "column": "customer_id"},
> "optimism_gap": 0.06, "direction": "optimistic (dangerous)", ...}`
>
> Claude: "Confirmed and measured: a naive random split reports accuracy about 0.06 higher than
> the grouped, customer-aware split — that gap would have been invisible without checking. Use
> `group` folds on `customer_id`."
>
> **You:** And what quality could I actually get out of this?
>
> **Claude** *(calls `feasibility("data/churn.csv", "churned")`)*
> → `{"verdict": "ok", "achievable_quality": {"metric": "accuracy", "mean": 0.87, "std": 0.01},
> "strongest_drivers": [{"feature": "tenure_months", "importance": 0.31}, ...],
> "risk_level": "elevated", ...}`
>
> Claude: "With grouped, leakage-free folds, expect about 87% accuracy (±1%). The strongest
> driver is `tenure_months`, followed by [...]. The same elevated risk from the audit still
> applies — group your folds before trusting this number."

### Forecasting: `audit_backtest`

For a time-series setup, the frontend also passes the ordering column (and optionally a series-id
column):

> **You:** I have `data/sales.csv`, target `units_sold`, ordered by `date`, one row per
> `store_id` per day. Is my backtest trustworthy?
>
> **Claude** *(calls `audit_backtest("data/sales.csv", "units_sold", "date", series_column="store_id")`)*
> → `{"verdict": "ok", "risk_level": "high", "future_leaks": [{"column": "customers",
> "correlation_with_target": 0.892, "reason": "only known after the period has ended"}],
> "split_design": {"mean_gap": 12.4, "direction": "optimistic (dangerous)", "n_origins": 5},
> "series_leak_auc": 1.0, "html_report": "data/sales.csv.backtest_audit.html"}`
>
> Claude: "High risk. `customers` would not be known at real forecast time (|corr| 0.89 with the
> target, and it is absent from a true future window), so training on it leaks. And a naive
> no-gap backtest overstates quality by ~12.4 versus an embargoed one across 5 rolling origins.
> The series-boundary AUC is diagnostic only (it does not yet separate real series leakage from
> ordinary trend). Add an embargo and drop future-known columns before trusting your backtest."

The tool trains no deployable model: every fit is a budget-bounded diagnostic backtest
replication, and `series_leak_auc` is a caveated diagnostic that does **not** drive `risk_level`
(only the future-leak finding and the measured split-design gap do).

## Guardrails (what the tools will and won't do)

- **No tunable parameters beyond `path`/`target`/`model`.** CV settings, time budgets, and fold
  counts are fixed and conservative — the tools answer a question fast, they don't search for
  the best possible model.
- **A minimum row count** (50) below which any judgment — LLM or CV — is noise, not signal;
  under it, the tools return a structured rejection instead of guessing.
- **A wall-clock backstop per tool** (`audit_csv` 60s, `check_validation` 150s, `feasibility`
  360s, `audit_backtest` 150s) on top of AutoGluon's own `time_limit`, which is the real, primary
  cost bound. The
  backstop is best-effort (a background thread with a result timeout, not a hard process kill):
  if it fires, the tool returns a timeout rejection immediately, but the underlying AutoGluon
  call may keep running to completion in the background rather than being forcibly stopped.
- **Rejection is a normal, reasoned result**, not an error path: a missing target column, too
  few rows, or an exceeded budget all come back as `{"verdict": "rejected", "reason": "..."}`,
  never a raw traceback.
