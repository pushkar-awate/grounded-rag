"""Measure access-scoped retrieval across roles.

For every access case and every role, ask the same question and check two
things. First, compliance: a role cleared for the answer should get one, and a
role that is not cleared should be refused (the system abstains instead of
answering from data the caller cannot see). Second, and non-negotiable,
leakage: no passage above a caller's clearance may ever reach the context the
model reads. The leak count must be zero; a single leak means access-scoped
retrieval is broken, and that is the number an auditor checks first.
"""
from __future__ import annotations

from govcore.access import (Classification, allow_predicate, can_access,
                            chunk_classification)


def run_access_eval(make_pipeline, cases, principals) -> dict:
    """make_pipeline(principal) -> a RagPipeline over the governance corpus for
    that principal. cases come from load_access_cases. principals is the roster
    to test."""
    trials = []
    leaks = 0
    correct = 0
    for case in cases:
        req_class = Classification.parse(case["required_classification"])
        req_comp = case.get("required_compartment")
        for principal in principals:
            pipe = make_pipeline(principal)
            res = pipe.answer(case["question"])
            decision = can_access(principal, req_class, req_comp)
            expected = "answer" if decision.allowed else "abstain"
            got = "abstain" if res.answer.abstained else "answer"

            allow = allow_predicate(principal)
            leaked = [c.chunk_id for c in res.context if not allow(c)]
            if leaked:
                leaks += 1
            compliant = (got == expected) and not leaked
            if compliant:
                correct += 1
            trials.append({
                "case": case["id"],
                "role": principal.role,
                "clearance": principal.clearance.label,
                "required": req_class.label
                + (" + %s" % req_comp if req_comp else ""),
                "expected": expected,
                "got": got,
                "compliant": compliant,
                "leaked_chunks": leaked,
                "withheld": res.access_withheld,
            })
    n = len(trials)
    return {
        "trials": trials,
        "n": n,
        "correct": correct,
        "leaks": leaks,
        "scoping_rate": round(100.0 * correct / n) if n else 0,
    }


def render_access_eval(report: dict) -> str:
    lines = ["access-scoped retrieval: %d (question x role) trial(s)" % report["n"],
             "-" * 56,
             "  clearance violations (leaks) : %d   <- must be 0" % report["leaks"],
             "  correct access scoping       : %d/%d (%d%%)"
             % (report["correct"], report["n"], report["scoping_rate"]),
             ""]
    for t in report["trials"]:
        flag = "ok " if t["compliant"] else "XX "
        lines.append("  [%s] %-16s %-12s want=%-7s got=%-7s need=%s"
                     % (flag, t["case"], t["role"], t["expected"], t["got"],
                        t["required"]))
    return "\n".join(lines)
