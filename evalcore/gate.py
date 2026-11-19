"""Regression gate: block a change that makes the system measurably worse.

Store a known-good scorecard as a baseline; a candidate fails the gate if any
GATED metric drops more than `tolerance` points below it. Retrieval, grounding
and correct-refusal are gated because regressions there are silent and harmful;
MRR and abstention-rate are informational.
"""
from __future__ import annotations

from dataclasses import dataclass, field

GATED = ("retrieval_hit", "groundedness", "refusal_correct", "faithfulness",
         "citation_present")


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
    cand = card.get("per_judge_pass_rate", {})
    base = baseline.get("per_judge_pass_rate", {}) if baseline else {}
    checks: list[Check] = []
    regressions: list[Check] = []
    for metric, cval in cand.items():
        bval = base.get(metric)
        gated = metric in GATED and bval is not None
        regressed = gated and cval < bval - tolerance
        note = "" if bval is not None else "not in baseline (informational)"
        chk = Check(metric=metric, baseline=bval, candidate=cval, gated=gated,
                    regressed=regressed, note=note)
        checks.append(chk)
        if regressed:
            regressions.append(chk)
    return GateResult(passed=len(regressions) == 0, checks=checks,
                      regressions=regressions)
