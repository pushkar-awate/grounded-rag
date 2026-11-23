"""The headline measurement: what does the grounding gate actually prevent?

Run the identical eval twice over the identical corpus, changing only one
thing - whether the system is allowed to abstain. The governed run uses the
normal brain, which refuses when the context does not support an answer. The
naive run uses NaiveBrain, which always answers from the best-matching passage.
The gap between their false-answer rates is the number this whole project
exists to produce: how many confident-but-wrong answers grounding removes.
"""
from __future__ import annotations

from .judges import default_judges
from .metrics import scorecard
from .runner import run_eval


def run_comparison(governed_pipe, naive_pipe, cases) -> dict:
    gov = scorecard(run_eval(governed_pipe, cases, default_judges(),
                             version="governed"))
    naive = scorecard(run_eval(naive_pipe, cases, default_judges(),
                               version="naive"))
    g, n = gov["abstention"], naive["abstention"]
    return {
        "governed": gov,
        "naive": naive,
        "headline": {
            "n_unanswerable": g["n_unanswerable"],
            "naive_false_answer_rate": n["false_answer_rate"],
            "governed_false_answer_rate": g["false_answer_rate"],
            "reduction_points": n["false_answer_rate"] - g["false_answer_rate"],
        },
    }


def render_comparison(cmp: dict) -> str:
    h = cmp["headline"]
    return "\n".join([
        "grounding gate: false-answer rate on %d unanswerable question(s)"
        % h["n_unanswerable"],
        "-" * 44,
        "  ungrounded (naive) : %d%%" % h["naive_false_answer_rate"],
        "  grounded           : %d%%" % h["governed_false_answer_rate"],
        "  reduction          : %d points" % h["reduction_points"],
    ])
