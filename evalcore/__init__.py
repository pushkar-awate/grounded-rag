"""Evaluation + regression gate for the grounded-rag pipeline."""
from .dataset import EvalCase, load_cases
from .judges import (CitationJudge, GroundednessJudge, Judgement,
                     LLMFaithfulnessJudge, MRRJudge, RefusalJudge,
                     RetrievalHitJudge, default_judges)
from .runner import CaseResult, EvalRun, run_eval
from .metrics import render_scorecard, scorecard
from .gate import Check, GateResult, regression_gate

__all__ = [
    "EvalCase", "load_cases",
    "CitationJudge", "GroundednessJudge", "Judgement", "LLMFaithfulnessJudge",
    "MRRJudge", "RefusalJudge", "RetrievalHitJudge", "default_judges",
    "CaseResult", "EvalRun", "run_eval",
    "render_scorecard", "scorecard",
    "Check", "GateResult", "regression_gate",
]
