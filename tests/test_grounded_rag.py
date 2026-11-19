"""Tests run entirely on the deterministic, dependency-light path (hashing
embedder + identity reranker + ExtractiveBrain), so the suite is fast, offline,
and free in CI - no torch, no model download, no API key."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragcore import (ABSTAIN_PHRASE, Document, ExtractiveBrain, HashingEmbedder,
                     IdentityReranker, MockBrain, RagPipeline, build_store,
                     chunk_text, generate)
from ragcore.generate import parse_citations
from ragcore.ingest import chunk_documents
from evalcore import default_judges, regression_gate, run_eval
from evalcore.dataset import EvalCase
from evalcore.metrics import scorecard


DOCS = [
    Document("transformer", "Transformer",
             "The Transformer relies on self-attention. Queries and keys produce "
             "attention weights via a scaled dot product and softmax. Multi-head "
             "attention runs several attention operations in parallel."),
    Document("lora", "LoRA",
             "LoRA freezes the pretrained weights and trains small low-rank "
             "matrices. QLoRA quantises the frozen base model to 4-bit precision."),
]


def _pipeline(brain=None):
    emb = HashingEmbedder()
    store = build_store(DOCS, emb)
    return RagPipeline(emb, store, IdentityReranker(), brain or ExtractiveBrain(),
                       retrieve_k=10, context_k=3)


# --- chunking --------------------------------------------------------------

def test_chunk_text_respects_target_and_overlaps():
    text = "\n\n".join("word " * 100 for _ in range(4))  # 4 big paragraphs
    chunks = chunk_text(text, target_words=120, overlap_words=20)
    assert len(chunks) > 1
    assert all(len(c.split()) <= 160 for c in chunks)  # target + overlap headroom


def test_chunk_documents_are_citable():
    chunks = chunk_documents(DOCS, target_words=50, overlap_words=10)
    assert chunks and all(c.doc_id and c.chunk_id for c in chunks)
    assert "#" in chunks[0].citation()


# --- generation ------------------------------------------------------------

def test_citation_parsing_maps_markers_to_chunks():
    chunks = chunk_documents(DOCS)
    ans = "Self-attention uses queries and keys [1]."
    cited = parse_citations(ans, chunks)
    assert cited == [chunks[0]]


def test_generate_abstains_with_no_context():
    ans = generate("anything", [], MockBrain())
    assert ans.abstained and ans.text == ABSTAIN_PHRASE


def test_extractive_brain_answers_relevant_and_abstains_irrelevant():
    pipe = _pipeline()
    hit = pipe.answer("How do queries and keys produce attention weights?")
    assert not hit.answer.abstained
    assert hit.answer.cited_chunks  # it cited a source
    miss = pipe.answer("What is the capital of France?")
    assert miss.answer.abstained


# --- eval + gate -----------------------------------------------------------

def test_eval_scorecard_and_refusal():
    cases = [
        EvalCase("q1", "How does self-attention use queries and keys?",
                 answerable=True, relevant_doc_ids=["transformer"]),
        EvalCase("q2", "What is the capital of France?", answerable=False),
    ]
    pipe = _pipeline()
    run = run_eval(pipe, cases, default_judges())
    card = scorecard(run)
    assert card["n_cases"] == 2
    assert card["per_judge_pass_rate"]["refusal_correct"] == 100
    assert card["per_judge_pass_rate"]["retrieval_hit"] == 100


def test_regression_gate_flags_a_drop():
    baseline = {"per_judge_pass_rate": {"retrieval_hit": 100, "groundedness": 100}}
    worse = {"per_judge_pass_rate": {"retrieval_hit": 70, "groundedness": 100}}
    result = regression_gate(worse, baseline, tolerance=5.0)
    assert not result.passed
    assert any(c.metric == "retrieval_hit" for c in result.regressions)


def test_regression_gate_passes_within_tolerance():
    baseline = {"per_judge_pass_rate": {"retrieval_hit": 100}}
    ok = {"per_judge_pass_rate": {"retrieval_hit": 98}}
    assert regression_gate(ok, baseline, tolerance=5.0).passed
