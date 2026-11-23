"""Aggregate an EvalRun into a scorecard.

Pass rates are computed over APPLICABLE judgements only - a judge that didn't
apply to a case (e.g. retrieval scoring on an unanswerable question) is excluded
from its denominator rather than counted as a failure, so the numbers mean what
they say.

The scorecard also reports abstention as a classification problem, which is the
measurement that actually matters for a grounded system. Treat "the system
abstained" as the positive prediction and "the corpus cannot answer this" as the
positive label:

  true positive  - unanswerable, abstained      (correct refusal)
  false negative - unanswerable, answered        (a confident wrong answer)
  false positive - answerable, abstained         (an unnecessary refusal)
  true negative  - answerable, answered          (correct answer)

From that: abstention precision and recall, and the two rates a reviewer cares
about - the false-answer rate (how often it answered when it should not have,
the dangerous failure) and the over-refusal rate (how often it refused a
question it could have answered, the annoying failure).
"""
from __future__ import annotations


def _pct(n, d):
    return round(100.0 * n / d) if d else 0


def abstention_stats(results) -> dict:
    """Confusion matrix and derived rates for the abstain/answer decision."""
    tp = fp = fn = tn = 0
    for r in results:
        abstained = r.rag_result.answer.abstained
        should_abstain = not r.case.answerable
        if should_abstain and abstained:
            tp += 1
        elif should_abstain and not abstained:
            fn += 1
        elif not should_abstain and abstained:
            fp += 1
        else:
            tn += 1
    n_unans = tp + fn
    n_ans = fp + tn
    precision = _pct(tp, tp + fp)
    recall = _pct(tp, tp + fn)
    f1 = round(2 * precision * recall / (precision + recall)) if (precision + recall) else 0
    return {
        "true_positive": tp, "false_negative": fn,
        "false_positive": fp, "true_negative": tn,
        "n_answerable": n_ans, "n_unanswerable": n_unans,
        "precision": precision, "recall": recall, "f1": f1,
        "false_answer_rate": _pct(fn, n_unans),
        "over_refusal_rate": _pct(fp, n_ans),
    }


def adversarial_stats(results) -> dict:
    """How the unanswerable adversarial traps went, split out from the plain
    unanswerable cases so the headline number is honest about difficulty."""
    cats: dict[str, dict] = {}
    for r in results:
        cat = getattr(r.case, "category", "standard")
        if r.case.answerable or cat == "standard":
            continue
        d = cats.setdefault(cat, {"n": 0, "caught": 0, "missed": 0})
        d["n"] += 1
        if r.rag_result.answer.abstained:
            d["caught"] += 1
        else:
            d["missed"] += 1
    return cats


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
        "abstention": abstention_stats(results),
    }
    adv = adversarial_stats(results)
    if adv:
        card["adversarial"] = adv
    if "mrr" in mean_score:
        card["mrr"] = mean_score["mrr"]
    return card


# Metrics the regression gate watches. abstention_recall and false_answer_rate
# live under card["abstention"]; gated_view flattens them next to the judges.
def gated_view(card: dict) -> dict:
    out = dict(card.get("per_judge_pass_rate", {}))
    ab = card.get("abstention", {})
    if "recall" in ab:
        out["abstention_recall"] = ab["recall"]
    if "precision" in ab:
        out["abstention_precision"] = ab["precision"]
    if "false_answer_rate" in ab:
        out["false_answer_rate"] = ab["false_answer_rate"]
    return out


def render_scorecard(card) -> str:
    lines = ["scorecard: %s" % card["version"], "-" * 44,
             "cases        : %d" % card["n_cases"]]
    for name, rate in card["per_judge_pass_rate"].items():
        lines.append("  %-18s: %d%%" % (name, rate))
    if "mrr" in card:
        lines.append("  %-18s: %.3f" % ("mrr (mean)", card["mrr"]))
    ab = card.get("abstention", {})
    if ab:
        lines.append("abstention decision (positive = abstain)")
        lines.append("  tp/fp/fn/tn       : %d/%d/%d/%d"
                     % (ab["true_positive"], ab["false_positive"],
                        ab["false_negative"], ab["true_negative"]))
        lines.append("  precision         : %d%%" % ab["precision"])
        lines.append("  recall            : %d%%" % ab["recall"])
        lines.append("  false-answer rate : %d%%  (answered when it should not)"
                     % ab["false_answer_rate"])
        lines.append("  over-refusal rate : %d%%  (refused when it could answer)"
                     % ab["over_refusal_rate"])
    adv = card.get("adversarial", {})
    for cat, d in adv.items():
        lines.append("  %-18s: caught %d/%d" % (cat, d["caught"], d["n"]))
    lines.append("abstention rate    : %d%%" % card["abstention_rate"])
    return "\n".join(lines)
