"""Offline test for the example-report builder: the --dry-run path renders synthetic dossiers
without any LLM/AutoGluon call (the real generation needs API keys and is Helena's to run)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import build_example_reports as ber  # noqa: E402


def test_examples_are_well_formed():
    names = {e["name"] for e in ber.EXAMPLES}
    assert names == {"bike-sharing", "house-prices", "grunfeld"}
    for e in ber.EXAMPLES:
        assert e["kind"] in ("dossier", "audit") and e["csv"] and e["target"]


def test_dry_run_writes_one_html_per_example(tmp_path):
    rc = ber.main(["--dry-run", "--out-dir", str(tmp_path)])
    assert rc == 0
    files = sorted(p.name for p in tmp_path.glob("*.html"))
    assert files == ["bike-sharing.html", "grunfeld.html", "house-prices.html"]
    html = (tmp_path / "bike-sharing.html").read_text(encoding="utf-8")
    assert html.startswith("<!DOCTYPE html>") and "synthetic, offline dry-run" in html
