"""Judges score one RagResult against one EvalCase.

Most judges are deterministic and LLM-free, so the suite runs in CI for $0 and
never flakes on a model mood. One optional judge (LLMFaithfulness) uses the same
Groq brain the app uses, for a semantic check of whether the answer is actually
supported by its sources. Every judge returns the same Judgement shape so the
scorecard can aggregate them uniformly.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

_WORD = re.compile(r"[a-z0-9]{3,}")
_STOP = {"the", "and", "for", "that", "with", "this", "from", "are", "was",
         "its", "has", "have", "which", "into", "than", "then", "they", "can",
         "use", "used", "using", "all", "any", "not", "but", "how", "what",
         "does", "why", "when", "who"}


def _terms(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP}


@dataclass
class Judgement:
    name: str
    applicable: bool
    passed: bool
    score: float = 0.0
    detail: str = ""


class RetrievalHitJudge:
    """Did a relevant document make it into the final context?"""
    name = "retrieval_hit"

    def judge(self, result, case) -> Judgement:
        if not case.answerable or not case.relevant_doc_ids:
            return Judgement(self.name, applicable=False, passed=False,
                             detail="no gold doc")
        got = {c.doc_id for c in result.context}
        hit = bool(got & set(case.relevant_doc_ids))
        return Judgement(self.name, applicable=True, passed=hit, score=float(hit),
                         detail="context docs: %s" % sorted(got))


class MRRJudge:
    """Reciprocal rank of the first relevant chunk in the reranked context."""
    name = "mrr"

    def judge(self, result, case) -> Judgement:
        if not case.answerable or not case.relevant_doc_ids:
            return Judgement(self.name, applicable=False, passed=False)
        rank = 0.0
        for i, (chunk, _) in enumerate(result.reranked, start=1):
            if chunk.doc_id in case.relevant_doc_ids:
                rank = 1.0 / i
                break
        return Judgement(self.name, applicable=True, passed=rank > 0, score=rank,
                         detail="rr=%.3f" % rank)


class CitationJudge:
    """A substantive (non-abstained) answer must cite at least one source."""
    name = "citation_present"

    def judge(self, result, case) -> Judgement:
        ans = result.answer
        if ans.abstained:
            return Judgement(self.name, applicable=False, passed=False,
                             detail="abstained")
        ok = len(ans.cited_chunks) > 0
        return Judgement(self.name, applicable=True, passed=ok, score=float(ok),
                         detail="%d citations" % len(ans.cited_chunks))


class GroundednessJudge:
    """Structural grounding: every cited chunk is real context, and the answer
    shares real content terms with what it cited (a cheap, no-LLM proxy for
    'the answer actually came from the sources')."""
    name = "groundedness"

    def judge(self, result, case) -> Judgement:
        ans = result.answer
        if ans.abstained:
            return Judgement(self.name, applicable=False, passed=False,
                             detail="abstained")
        if not ans.cited_chunks:
            return Judgement(self.name, applicable=True, passed=False,
                             detail="no citations to ground")
        ctx_ids = {c.chunk_id for c in ans.context_chunks}
        all_real = all(c.chunk_id in ctx_ids for c in ans.cited_chunks)
        cited_terms = set()
        for c in ans.cited_chunks:
            cited_terms |= _terms(c.text)
        ans_terms = _terms(ans.text)
        overlap = ans_terms & cited_terms
        ok = all_real and len(overlap) > 0
        return Judgement(self.name, applicable=True, passed=ok,
                         score=float(ok),
                         detail="real_cites=%s overlap=%d" % (all_real, len(overlap)))


class RefusalJudge:
    """The decisive one: answer when you can, abstain when you can't."""
    name = "refusal_correct"

    def judge(self, result, case) -> Judgement:
        abstained = result.answer.abstained
        ok = (not abstained) if case.answerable else abstained
        want = "answer" if case.answerable else "abstain"
        got = "abstain" if abstained else "answer"
        return Judgement(self.name, applicable=True, passed=ok, score=float(ok),
                         detail="wanted %s, got %s" % (want, got))


class LLMFaithfulnessJudge:
    """Optional semantic check: ask the LLM whether every claim in the answer is
    supported by the provided sources. Uses the same brain as the app."""
    name = "faithfulness"

    def __init__(self, brain):
        self.brain = brain

    def judge(self, result, case) -> Judgement:
        ans = result.answer
        if ans.abstained or not ans.context_chunks:
            return Judgement(self.name, applicable=False, passed=False,
                             detail="abstained/no-context")
        sources = "\n\n".join("[%d] %s" % (i + 1, c.text)
                              for i, c in enumerate(ans.context_chunks))
        prompt = (
            "Sources:\n%s\n\nAnswer:\n%s\n\n"
            "Is EVERY factual claim in the Answer directly supported by the "
            "Sources above? Reply with only YES or NO." % (sources, ans.text))
        try:
            verdict = self.brain.complete(prompt, temperature=0.0).strip().upper()
        except Exception as e:  # a judge outage is neutral, never a failure
            return Judgement(self.name, applicable=False, passed=False,
                             detail="judge error: %s" % e)
        ok = verdict.startswith("YES")
        return Judgement(self.name, applicable=True, passed=ok, score=float(ok),
                         detail="llm verdict: %s" % verdict[:20])


def default_judges(brain=None, use_llm: bool = False) -> list:
    judges = [RetrievalHitJudge(), MRRJudge(), CitationJudge(),
              GroundednessJudge(), RefusalJudge()]
    if use_llm and brain is not None:
        judges.append(LLMFaithfulnessJudge(brain))
    return judges
