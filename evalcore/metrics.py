"""Aggregate an EvalRun into a scorecard.

Pass rates are computed over APPLICABLE judgements only - a judge that didn't
apply to a case (e.g. retrieval scoring on an unanswerable question) is excluded
from its denominator rather than counted as a failure, so the numbers mean what
they say.
"""
from __future__ import annotations


def _pct(n, d):
    return round(100.0 * n / d) if d else 0


def scorecard(run) -> dict:
    results = run.results
    n = len(results)

    names: list[str] = []
    for r in results:
        for j in r.judgements:
            if j.name not in names:
                names.append(j.name)

    per_judge = {}
    mean_score = {}
    for name in names:
        js = [j for r in results for j in r.judgements
              if j.name == name and j.applicable]
        if js:
            per_judge[name] = _pct(sum(1 for j in js if j.passed), len(js))
            mean_score[name] = round(sum(j.score for j in js) / len(js), 3)

    abstained = sum(1 for r in results if r.rag_result.answer.abstained)

    card = {
        "version": run.version,
        "n_cases": n,
        "per_judge_pass_rate": per_judge,
        "abstention_rate": _pct(abstained, n),
    }
    if "mrr" in mean_score:
        card["mrr"] = mean_score["mrr"]
    return card


def render_scorecard(card) -> str:
    lines = ["scorecard: %s" % card["version"], "-" * 40,
             "cases        : %d" % card["n_cases"]]
    for name, rate in card["per_judge_pass_rate"].items():
        lines.append("  %-16s: %d%%" % (name, rate))
    if "mrr" in card:
        lines.append("  %-16s: %.3f" % ("mrr (mean)", card["mrr"]))
    lines.append("abstention   : %d%%" % card["abstention_rate"])
    return "\n".join(lines)
