"""Run the pipeline over every eval case and collect judgements.

Same pipeline object the app uses -> the scorecard describes production, not a
test double.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .judges import Judgement


@dataclass
class CaseResult:
    case: object
    rag_result: object
    judgements: list[Judgement] = field(default_factory=list)


@dataclass
class EvalRun:
    version: str
    results: list[CaseResult] = field(default_factory=list)


def run_eval(pipeline, cases, judges, version: str = "dev") -> EvalRun:
    run = EvalRun(version=version)
    for case in cases:
        rag_result = pipeline.answer(case.question)
        judgements = [j.judge(rag_result, case) for j in judges]
        run.results.append(CaseResult(case=case, rag_result=rag_result,
                                       judgements=judgements))
    return run
