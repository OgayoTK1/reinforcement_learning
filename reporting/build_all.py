"""One command from experiment outputs to figures, tables, diagrams, API doc and the report.

    uv run main.py report --results experiments/results
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import api_doc, diagrams, figures, report, stats
from .data import Results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="experiments/results")
    ap.add_argument("--out", default="docs/report")
    ap.add_argument("--student", default="Ogayo Andrew")
    ap.add_argument("--video", default="[link to be added after recording]")
    ap.add_argument("--repo", default="[link to be added after push]")
    a = ap.parse_args()
    R = Results(a.results)
    out = Path(a.out)
    fig_dir = out / "figures"
    diagrams.all_diagrams(Path("docs/figures"))
    api_doc.build()
    figs = figures.all_figures(R, fig_dir)
    if R.test is not None:
        stats.summary_table(R.test).to_csv(out / "test_summary_ci.csv", index=False)
        stats.paired_tests(R.test).to_csv(out / "paired_tests.csv", index=False)
        docx, pdf = report.build(R, figs, out, Path("assets"), a.student, a.video, a.repo)
        print(f"report: {docx}  {pdf}  ({report.page_count(pdf)} pages)")
    else:
        print("no test_episodes.csv yet: run `uv run main.py evaluate` first to build the report")


if __name__ == "__main__":
    main()
