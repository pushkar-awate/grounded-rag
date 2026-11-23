"""Tests for the governance layer: access-scoped retrieval, the tamper-evident
audit log, PII redaction, abstention precision/recall, and the grounded-vs-naive
comparison. All deterministic and offline (hashing embedder + ExtractiveBrain),
so they run in CI for $0."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragcore import (Document, ExtractiveBrain, HashingEmbedder,
                     IdentityReranker, NaiveBrain, RagPipeline, build_store)
from evalcore import run_comparison, run_access_eval, run_eval, default_judges
from evalcore.dataset import EvalCase
from evalcore.metrics import abstention_stats, scorecard
from evalcore.gate import regression_gate
from govcore import (AuditLog, Classification, ROLES, can_access, detect,
                     get_principal, redact)
from govcore.access import Principal


# --- classified corpus fixture --------------------------------------------

GOV_DOCS = [
    Document("pub", "Public saver", "The Everyday Saver pays 2.5% interest per annum.",
             classification=int(Classification.PUBLIC)),
    Document("conf", "Risk register",
             "The retail payments recovery time objective is two hours under CPS 230.",
             classification=int(Classification.CONFIDENTIAL), compartment="RISK"),
    Document("restr", "Incident report",
             "The March 2026 payments outage was a material operational risk incident.",
             classification=int(Classification.RESTRICTED), compartment="RISK"),
]


def _gov_pipeline(principal):
    emb = HashingEmbedder()
    store = build_store(GOV_DOCS, emb)
    return RagPipeline(emb, store, IdentityReranker(), ExtractiveBrain(),
                       retrieve_k=10, context_k=5, principal=principal)


# --- access control --------------------------------------------------------

def test_clearance_ordering_and_compartments():
    pub = ROLES["public_user"]
    exec_ = ROLES["executive"]
    assert not can_access(pub, Classification.CONFIDENTIAL, "RISK").allowed
    assert can_access(exec_, Classification.RESTRICTED, "RISK").allowed
    ro = ROLES["risk_officer"]
    assert can_access(ro, Classification.CONFIDENTIAL, "RISK").allowed
    assert not can_access(ro, Classification.CONFIDENTIAL, "FINANCE").allowed


def test_unknown_classification_fails_closed():
    assert Classification.parse("TOP_SECRET") == Classification.RESTRICTED


def test_access_scoped_retrieval_never_leaks_above_clearance():
    pipe = _gov_pipeline(get_principal("public_user"))
    res = pipe.answer("What is the recovery time objective for retail payments?")
    assert all(c.classification <= int(Classification.PUBLIC) for c in res.context)
    assert res.answer.abstained
    assert res.answer.abstain_reason == "access_scoped"
    assert res.access_withheld >= 1


def test_cleared_role_gets_the_answer():
    pipe = _gov_pipeline(get_principal("risk_officer"))
    res = pipe.answer("What is the recovery time objective for retail payments?")
    assert not res.answer.abstained
    assert res.answer.cited_chunks


def test_access_eval_zero_leaks():
    cases = [
        {"id": "rto", "question": "What is the recovery time objective for retail payments?",
         "relevant_doc_id": "conf", "required_classification": "CONFIDENTIAL",
         "required_compartment": "RISK"},
        {"id": "inc", "question": "How is the March 2026 payments outage classified?",
         "relevant_doc_id": "restr", "required_classification": "RESTRICTED",
         "required_compartment": "RISK"},
    ]
    emb = HashingEmbedder()
    store = build_store(GOV_DOCS, emb)

    def make_pipe(p):
        return RagPipeline(emb, store, IdentityReranker(), ExtractiveBrain(),
                           retrieve_k=10, context_k=5, principal=p)

    report = run_access_eval(make_pipe, cases, list(ROLES.values()))
    assert report["leaks"] == 0
    assert report["correct"] == report["n"]


# --- audit log -------------------------------------------------------------

def test_audit_records_and_verifies():
    log = AuditLog()
    pipe = _gov_pipeline(get_principal("executive"))
    pipe.audit = log
    pipe.answer("What is the recovery time objective for retail payments?")
    pipe.answer("What is the capital of France?")
    assert len(log) == 2
    ok, _ = log.verify()
    assert ok


def test_audit_detects_tampering():
    log = AuditLog()
    log.record(principal="exec", clearance="Restricted", question="q1",
               decision="answer")
    log.record(principal="exec", clearance="Restricted", question="q2",
               decision="abstain")
    assert log.verify()[0]
    log._entries[0]["question"] = "edited"
    ok, detail = log.verify()
    assert not ok and "seq 0" in detail


# --- PII --------------------------------------------------------------------

def test_pii_detects_email_card_phone():
    text = "Reach me at jane.doe@example.com or 0412 345 678. Card 4111 1111 1111 1111."
    kinds = {f.kind for f in detect(text)}
    assert "email" in kinds and "phone" in kinds and "card" in kinds


def test_pii_redaction_removes_values():
    red, finds = redact("email a@b.com now")
    assert "a@b.com" not in red and "[REDACTED:EMAIL]" in red and len(finds) == 1


def test_pii_card_requires_luhn():
    assert not any(f.kind == "card" for f in detect("ref 1234 5678 9012 3456"))


# --- abstention metrics -----------------------------------------------------

def _mini_run():
    emb = HashingEmbedder()
    docs = [Document("t", "T", "The Transformer uses self-attention with queries and keys.")]
    store = build_store(docs, emb)
    pipe = RagPipeline(emb, store, IdentityReranker(), ExtractiveBrain(),
                       retrieve_k=10, context_k=3)
    cases = [
        EvalCase("ans", "How does self-attention use queries and keys?", answerable=True,
                 relevant_doc_ids=["t"]),
        EvalCase("unans", "What is the capital of France?", answerable=False),
    ]
    return run_eval(pipe, cases, default_judges())


def test_abstention_precision_recall():
    run = _mini_run()
    ab = scorecard(run)["abstention"]
    assert ab["precision"] == 100 and ab["recall"] == 100
    assert ab["false_answer_rate"] == 0 and ab["over_refusal_rate"] == 0


def test_naive_brain_raises_false_answer_rate():
    emb = HashingEmbedder()
    docs = [Document("t", "T", "The Transformer uses self-attention with queries and keys.")]
    store = build_store(docs, emb)
    gov = RagPipeline(emb, store, IdentityReranker(), ExtractiveBrain(),
                      retrieve_k=10, context_k=3)
    naive = RagPipeline(emb, store, IdentityReranker(), NaiveBrain(),
                        retrieve_k=10, context_k=3)
    cases = [EvalCase("u1", "What is the capital of France?", answerable=False),
             EvalCase("u2", "Who won the 2020 election?", answerable=False)]
    cmp = run_comparison(gov, naive, cases)
    assert cmp["headline"]["naive_false_answer_rate"] == 100
    assert cmp["headline"]["governed_false_answer_rate"] == 0
    assert cmp["headline"]["reduction_points"] == 100


# --- gate direction ---------------------------------------------------------

def test_gate_flags_rising_false_answer_rate():
    baseline = {"abstention": {"recall": 90, "precision": 100, "false_answer_rate": 10}}
    worse = {"abstention": {"recall": 90, "precision": 100, "false_answer_rate": 40}}
    result = regression_gate(worse, baseline, tolerance=5.0)
    assert not result.passed
    assert any(c.metric == "false_answer_rate" for c in result.regressions)


def test_gate_allows_falling_false_answer_rate():
    baseline = {"abstention": {"false_answer_rate": 40}}
    better = {"abstention": {"false_answer_rate": 10}}
    assert regression_gate(better, baseline, tolerance=5.0).passed
