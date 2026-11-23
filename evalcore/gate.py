"""Regression gate: block a change that makes the system measurably worse.

Store a known-good scorecard as a baseline; a candidate fails the gate if any
GATED metric moves the wrong way by more than `tolerance` points. Retrieval,
grounding, correct-refusal, and the abstention precision/recall pair are gated
because regressions there are silent and harmful. Most metrics are
higher-is-better; the failure rates (false-answer, over-refusal) are
lower-is-better, so the gate checks each in the correct direction.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .metrics import gated_view

GATED = ("retrieval_hit", "groundedness", "refusal_correct", "faithfulness",
         "citation_present", "abstention_recall", "abstention_precision",
         "false_answer_rate")

# Metrics where a higher number is WORSE.
LOWER_IS_BETTER = ("false_answer_rate", "over_refusal_rate")


@dataclass
class Check:
    metric: str
    baseline: float | None
    candidate: float
    gated: bool
    regressed: bool
    note: str = ""


@dataclass
class GateResult:
    passed: bool
    checks: list[Check] = field(default_factory=list)
    regressions: list[Check] = field(default_factory=list)


def regression_gate(card: dict, baseline: dict, tolerance: float = 0.0) -> GateResult:
    cand = gated_view(card)
    base = gated_view(baseline) if baseline else {}
    checks: list[Check] = []
    regressions: list[Check] = []
    for metric, cval in cand.items():
        bval = base.get(metric)
        gated = metric in GATED and bval is not None
        if gated and metric in LOWER_IS_BETTER:
            regressed = cval > bval + tolerance
        elif gated:
            regressed = cval < bval - tolerance
        else:
            regressed = False
        note = "" if bval is not None else "not in baseline (informational)"
        if metric in LOWER_IS_BETTER:
            note = (note + " lower is better").strip()
        chk = Check(metric=metric, baseline=bval, candidate=cval, gated=gated,
                    regressed=regressed, note=note)
        checks.append(chk)
        if regressed:
            regressions.append(chk)
    return GateResult(passed=len(regressions) == 0, checks=checks,
                      regressions=regressions)
