"""Evaluation + regression gate for the grounded-rag pipeline."""
from .dataset import EvalCase, load_cases
from .judges import (CitationJudge, GroundednessJudge, Judgement,
                     LLMFaithfulnessJudge, MRRJudge, RefusalJudge,
                     RetrievalHitJudge, default_judges)
from .runner import CaseResult, EvalRun, run_eval
from .metrics import abstention_stats, gated_view, render_scorecard, scorecard
from .gate import Check, GateResult, regression_gate
from .compare import render_comparison, run_comparison
from .access_eval import render_access_eval, run_access_eval

__all__ = [
    "EvalCase", "load_cases",
    "CitationJudge", "GroundednessJudge", "Judgement", "LLMFaithfulnessJudge",
    "MRRJudge", "RefusalJudge", "RetrievalHitJudge", "default_judges",
    "CaseResult", "EvalRun", "run_eval",
    "abstention_stats", "gated_view", "render_scorecard", "scorecard",
    "Check", "GateResult", "regression_gate",
    "render_comparison", "run_comparison",
    "render_access_eval", "run_access_eval",
]
