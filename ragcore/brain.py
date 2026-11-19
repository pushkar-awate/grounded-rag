"""The LLM behind answer generation.

Same shape as the Brain in Pushkar's agentcore: one method, complete(prompt),
so the pipeline never depends on a specific provider.

  * GroqBrain    - real generation via Groq's free API (what the live app uses).
  * ExtractiveBrain - deterministic, offline. It reads the grounded prompt,
    finds the source that best matches the question, and either returns an
    extractive cited answer or abstains when nothing matches. This makes the
    eval suite meaningful AND reproducible in CI for $0, with no API key.
  * MockBrain    - minimal stub for unit tests.
"""
from __future__ import annotations

import os
import re
from typing import Callable, Optional

_WORD = re.compile(r"[a-z0-9]{3,}")
_STOP = {"the", "and", "for", "that", "with", "this", "from", "are", "was",
         "its", "has", "have", "which", "into", "than", "then", "they", "can",
         "use", "used", "using", "all", "any", "not", "but", "how", "what",
         "does", "why", "when", "who", "according", "documents", "document"}
_ABSTAIN = ("I don't have enough information in the provided documents "
            "to answer that.")


def _terms(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP}


class MockBrain:
    """Offline stub. Returns a fixed marker unless given a responder."""

    def __init__(self, responder: Optional[Callable[[str], str]] = None):
        self.responder = responder
        self.calls = 0

    def complete(self, prompt: str, system: Optional[str] = None,
                 temperature: float = 0.0) -> str:
        self.calls += 1
        return self.responder(prompt) if self.responder else "MOCK_ANSWER"


class ExtractiveBrain:
    """Deterministic grounded 'generation' with no model and no key.

    Parses the numbered Sources and the Question out of the grounded prompt,
    scores each source by content-term overlap with the question, and returns an
    extractive answer citing the best source - or abstains when the best overlap
    is below `threshold`. Used for the committed baseline and CI.
    """

    def __init__(self, threshold: int = 2, sentences: int = 2):
        self.threshold = threshold
        self.sentences = sentences
        self.calls = 0

    def complete(self, prompt: str, system: Optional[str] = None,
                 temperature: float = 0.0) -> str:
        self.calls += 1
        if "Question:" in prompt:
            src_part, q_part = prompt.split("Question:", 1)
            question = q_part.split("Answer")[0].strip()
        else:
            src_part, question = prompt, ""
        blocks = re.findall(r"\[(\d+)\]\s*\([^\n]*\)\n(.*?)(?=\n\[\d+\]\s*\(|\Z)",
                            src_part, re.S)
        q_terms = _terms(question)
        best_num, best_text, best_ov = None, "", 0
        for num, text in blocks:
            ov = len(q_terms & _terms(text))
            if ov > best_ov:
                best_num, best_text, best_ov = num, text.strip(), ov
        if best_num is None or best_ov < self.threshold:
            return _ABSTAIN
        sents = re.split(r"(?<=[.!?])\s+", best_text)
        answer = " ".join(sents[:self.sentences]).strip()
        return "%s [%s]" % (answer, best_num)


class GroqBrain:
    """Groq chat-completions wrapper. Reads GROQ_API_KEY from the environment."""

    def __init__(self, model: str = "openai/gpt-oss-20b",
                 api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self._client = None

    def _lazy(self):
        if self._client is None:
            if not self.api_key:
                raise RuntimeError("GROQ_API_KEY is not set")
            from groq import Groq
            self._client = Groq(api_key=self.api_key)
        return self._client

    def complete(self, prompt: str, system: Optional[str] = None,
                 temperature: float = 0.0) -> str:
        client = self._lazy()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = client.chat.completions.create(
            model=self.model, messages=messages, temperature=temperature)
        return resp.choices[0].message.content or ""


def has_groq_key() -> bool:
    return bool(os.environ.get("GROQ_API_KEY"))


def default_brain(use_llm: bool = False):
    """GroqBrain when a key is present and LLM mode is on; else the deterministic
    ExtractiveBrain so everything still runs offline and free."""
    if use_llm and has_groq_key():
        return GroqBrain()
    return ExtractiveBrain()
