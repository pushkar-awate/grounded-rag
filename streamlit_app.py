"""grounded-rag - a RAG assistant that cites its sources, refuses to answer when
the documents don't support it, and governs the answer for regulated settings.

Ask questions over a bundled set of ML/LLM explainers, your own uploaded PDF, a
live arXiv search, or a synthetic classified corpus that shows access-scoped
retrieval. Every answer is grounded in retrieved passages with [n] citations;
when the corpus can't answer, or the caller isn't cleared to see what would
answer, the system abstains instead of guessing. Each decision is written to a
tamper-evident audit log. A one-click evaluation shows retrieval quality,
groundedness, abstention precision/recall, and the gate that runs in CI.
"""
import hashlib
import json
import os
import tempfile

import streamlit as st

st.set_page_config(page_title="grounded-rag", page_icon="\U0001F4DA", layout="wide")

# Streamlit Cloud serves secrets via st.secrets; GroqBrain reads os.environ.
try:
    if "GROQ_API_KEY" in st.secrets and not os.environ.get("GROQ_API_KEY"):
        os.environ["GROQ_API_KEY"] = str(st.secrets["GROQ_API_KEY"])
except Exception:
    pass

from ragcore import (RagPipeline, STEmbedder, HashingEmbedder,
                     CrossEncoderReranker, IdentityReranker, GroqBrain,
                     ExtractiveBrain, NaiveBrain, has_groq_key, build_store)
from ragcore.ingest import load_path
from evalcore import (default_judges, load_cases, regression_gate, run_eval,
                      run_comparison, run_access_eval)
from evalcore.metrics import scorecard
from govcore import AuditLog, ROLES, get_principal
from app.corpus import (load_bundled_corpus, fetch_arxiv,
                        load_governance_corpus, load_access_cases)

HERE = os.path.dirname(os.path.abspath(__file__))
GOV_LABEL = "Governance demo (classified)"

if "audit" not in st.session_state:
    st.session_state["audit"] = AuditLog()
audit = st.session_state["audit"]


# --- cached heavy resources (loaded once per session) ----------------------

@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedder():
    try:
        emb = STEmbedder()
        emb.embed(["warmup"])
        return emb, "MiniLM (sentence-transformers)"
    except Exception:
        return HashingEmbedder(), "hashing fallback (offline)"


@st.cache_resource(show_spinner="Loading reranker...")
def get_reranker():
    try:
        rr = CrossEncoderReranker()
        rr.rerank("warmup", [], top_k=0)
        return rr, "cross-encoder (ms-marco-MiniLM)"
    except Exception:
        return IdentityReranker(), "identity (offline)"


@st.cache_resource(show_spinner="Indexing documents...")
def get_store(signature: str, _docs):
    emb, _ = get_embedder()
    return build_store(_docs, emb)


def corpus_signature(docs) -> str:
    h = hashlib.sha1()
    for d in docs:
        h.update(d.doc_id.encode())
        h.update(str(len(d.text)).encode())
        h.update(str(getattr(d, "classification", 0)).encode())
    return h.hexdigest()


# --- sidebar: corpus + settings --------------------------------------------

st.title("grounded-rag")
st.caption("Retrieval-augmented QA that **cites its sources**, **abstains** when "
           "the documents can't answer, and **governs the output** for regulated "
           "settings: access-scoped retrieval, PII redaction, and a "
           "tamper-evident audit trail.")

with st.sidebar:
    st.header("Corpus")
    source = st.radio("Documents to search", [
        "Bundled ML/LLM papers", GOV_LABEL,
        "Upload a PDF / text file", "Live arXiv search"])

    docs = None
    if source == "Bundled ML/LLM papers":
        docs = load_bundled_corpus()
        st.caption("%d built-in explainers (Transformers, LoRA, RAG, BERT, "
                   "vector search, MLOps drift)." % len(docs))
    elif source == GOV_LABEL:
        docs = load_governance_corpus()
        st.caption("%d fictional bank documents at mixed classification levels. "
                   "Change the role below and watch the same question get "
                   "answered or refused based on clearance." % len(docs))
    elif source == "Upload a PDF / text file":
        ups = st.file_uploader("Drop a .pdf, .txt or .md (one or more)",
                               type=["pdf", "txt", "md"], accept_multiple_files=True)
        if ups:
            loaded = []
            for up in ups:
                suffix = os.path.splitext(up.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
                    tf.write(up.read())
                    tmp = tf.name
                try:
                    d = load_path(tmp)
                    d.title = up.name
                    loaded.append(d)
                finally:
                    os.unlink(tmp)
            docs = loaded
            st.caption("Loaded %d document(s)." % len(docs))
    else:
        q = st.text_input("arXiv query", value="retrieval augmented generation")
        n = st.slider("How many papers", 3, 20, 8)
        if st.button("Fetch from arXiv"):
            try:
                st.session_state["arxiv_docs"] = fetch_arxiv(q, n)
            except Exception as e:
                st.error("arXiv fetch failed: %s" % e)
        docs = st.session_state.get("arxiv_docs")
        if docs:
            st.caption("Fetched %d abstracts." % len(docs))

    st.header("Governance")
    role = st.selectbox(
        "Caller role (clearance)", list(ROLES.keys()),
        index=list(ROLES.keys()).index("public_user"),
        help="Retrieval only returns passages this role is cleared to read. On "
             "the classified corpus, lower clearance means the system abstains "
             "instead of answering from data the caller cannot see.")
    principal = get_principal(role)
    st.caption("Clearance: **%s**" % principal.label)
    pii_on = st.toggle("Redact PII from retrieved context", value=False,
                       help="Scrub emails, phone numbers, card and TFN-like "
                            "numbers out of context before the model reads it.")

    st.header("Retrieval")
    retrieve_k = st.slider("Candidates retrieved", 5, 40, 20)
    context_k = st.slider("Chunks sent to the model", 2, 10, 5)

    key_present = has_groq_key()
    use_llm = st.toggle("Use LLM (Groq) for generation", value=key_present,
                        disabled=not key_present,
                        help="On: a real LLM writes the grounded answer. Off: a "
                             "deterministic extractive answerer (no key needed).")
    if not key_present:
        st.caption("Set GROQ_API_KEY in app secrets to enable LLM generation.")


# --- build pipeline --------------------------------------------------------

if not docs:
    st.info("Pick or upload a corpus in the sidebar to begin.")
    st.stop()

emb, emb_name = get_embedder()
rr, rr_name = get_reranker()
sig = corpus_signature(docs)
store = get_store(sig, docs)
brain = GroqBrain() if (use_llm and key_present) else ExtractiveBrain()
brain_name = "Groq GPT-OSS-20B" if (use_llm and key_present) else "extractive (deterministic)"
pipe = RagPipeline(emb, store, rr, brain, retrieve_k=retrieve_k,
                   context_k=context_k, principal=principal, pii=pii_on,
                   audit=audit, corpus_sig=sig[:8])

st.caption("Embedder: **%s**  |  Reranker: **%s**  |  Generator: **%s**  |  "
           "Role: **%s**  |  %d chunks indexed"
           % (emb_name, rr_name, brain_name, principal.role, len(store.chunks)))


# --- ask -------------------------------------------------------------------

st.subheader("Ask a question")
examples = {
    "Bundled ML/LLM papers": "How does self-attention use queries and keys?",
    GOV_LABEL: "What is the recovery time objective for retail payments?",
    "Upload a PDF / text file": "What is this document about?",
    "Live arXiv search": "What problem do these papers address?",
}
question = st.text_input("Question", value=examples.get(source, ""))
go = st.button("Answer", type="primary")

if go and question.strip():
    with st.spinner("Retrieving and generating..."):
        res = pipe.answer(question)
    ans = res.answer

    if ans.abstained:
        st.warning("**Abstained.** " + res.abstain_explanation)
    else:
        st.markdown("### Answer")
        st.markdown(ans.text)
        if ans.cited_chunks:
            st.markdown("**Sources cited**")
            for c in ans.cited_chunks:
                with st.expander("%s" % c.citation()):
                    st.write(c.text)

    meta = []
    if res.access_withheld:
        meta.append("%d passage(s) withheld on access grounds" % res.access_withheld)
    if res.pii_redactions:
        meta.append("%d PII identifier(s) redacted from context" % res.pii_redactions)
    if res.decision_id:
        meta.append("audit id `%s`" % res.decision_id)
    if meta:
        st.caption("  |  ".join(meta))

    with st.expander("Retrieved context (reranked, after access filtering)"):
        if not res.reranked:
            st.caption("Nothing retrieved that this role is cleared to see.")
        for c, score in res.reranked:
            st.markdown("**%s**  (score %.3f)" % (c.citation(), score))
            st.caption(c.text[:400] + ("..." if len(c.text) > 400 else ""))


# --- access control demo ---------------------------------------------------

if source == GOV_LABEL:
    st.divider()
    st.subheader("Access-scoped retrieval")
    st.caption("The same five questions asked by every role. Cleared roles get "
               "an answer; others are refused, and no passage above a role's "
               "clearance ever reaches the model. Zero leaks is the whole point.")
    if st.button("Run the access-control check"):
        with st.spinner("Asking every question as every role..."):
            def make_pipe(p):
                return RagPipeline(emb, store, rr, brain, retrieve_k=retrieve_k,
                                   context_k=context_k, principal=p)
            report = run_access_eval(make_pipe, load_access_cases(),
                                     list(ROLES.values()))
        c1, c2 = st.columns(2)
        c1.metric("Clearance violations (leaks)", report["leaks"],
                  help="Must be zero.")
        c2.metric("Correct access scoping",
                  "%d/%d" % (report["correct"], report["n"]))
        st.dataframe([
            {"case": t["case"], "role": t["role"], "needs": t["required"],
             "expected": t["expected"], "got": t["got"],
             "ok": "yes" if t["compliant"] else "NO"}
            for t in report["trials"]], use_container_width=True, hide_index=True)


# --- grounding headline: grounded vs ungrounded ----------------------------

st.divider()
st.subheader("What the grounding gate prevents")
st.caption("Run the gold eval twice over the bundled corpus, changing only one "
           "thing: whether the system may abstain. The gap in false-answer rate "
           "is what grounding buys you.")
if st.button("Measure grounded vs ungrounded"):
    with st.spinner("Running both configurations..."):
        bundled = load_bundled_corpus()
        bsig = corpus_signature(bundled)
        bstore = get_store(bsig, bundled)
        gov_pipe = RagPipeline(emb, bstore, rr, brain, retrieve_k=retrieve_k,
                               context_k=context_k)
        naive_pipe = RagPipeline(emb, bstore, rr, NaiveBrain(),
                                 retrieve_k=retrieve_k, context_k=context_k)
        cases = load_cases(os.path.join(HERE, "data", "eval_cases.json"))
        cmp = run_comparison(gov_pipe, naive_pipe, cases)
    h = cmp["headline"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Ungrounded false-answer rate", "%d%%" % h["naive_false_answer_rate"])
    c2.metric("Grounded false-answer rate", "%d%%" % h["governed_false_answer_rate"])
    c3.metric("Reduction", "%d points" % h["reduction_points"])
    st.caption("On %d unanswerable question(s), including adversarial traps. "
               "Lower is better; grounding is the only thing that changed."
               % h["n_unanswerable"])


# --- evaluation ------------------------------------------------------------

st.divider()
st.subheader("Evaluation scorecard")
st.caption("Runs the gold eval set over the **bundled** corpus and gates the "
           "scorecard against the committed baseline. This is the same check CI runs.")
if st.button("Run evaluation on the bundled corpus"):
    with st.spinner("Scoring retrieval, grounding, and abstention..."):
        bundled = load_bundled_corpus()
        estore = get_store(corpus_signature(bundled), bundled)
        epipe = RagPipeline(emb, estore, rr, brain,
                            retrieve_k=retrieve_k, context_k=context_k)
        cases = load_cases(os.path.join(HERE, "data", "eval_cases.json"))
        judges = default_judges(brain=brain, use_llm=(use_llm and key_present))
        card = scorecard(run_eval(epipe, cases, judges, version="dashboard"))

    rates = card["per_judge_pass_rate"]
    pretty = {"retrieval_hit": "Retrieval hit", "mrr": "MRR pass",
              "citation_present": "Citations", "groundedness": "Groundedness",
              "refusal_correct": "Correct refusal", "faithfulness": "Faithfulness"}
    cols = st.columns(len(rates))
    for col, (name, rate) in zip(cols, rates.items()):
        col.metric(pretty.get(name, name), "%d%%" % rate)

    ab = card["abstention"]
    st.markdown("**Abstention decision** (positive = abstain)")
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Precision", "%d%%" % ab["precision"])
    a2.metric("Recall", "%d%%" % ab["recall"])
    a3.metric("False-answer rate", "%d%%" % ab["false_answer_rate"],
              help="Answered when it should have abstained. Lower is better.")
    a4.metric("Over-refusal rate", "%d%%" % ab["over_refusal_rate"],
              help="Refused when it could answer. Lower is better.")

    baseline_path = os.path.join(HERE, "baseline.json")
    if os.path.exists(baseline_path):
        with open(baseline_path) as fh:
            baseline = json.load(fh)
        result = regression_gate(card, baseline, tolerance=5.0)
        if result.passed:
            st.success("GATE PASS. No gated regression versus the committed baseline.")
        else:
            st.error("GATE FAIL. Regression in %s."
                     % ", ".join(c.metric for c in result.regressions))
        st.caption("Live numbers use the real model stack and may differ slightly "
                   "from the deterministic committed baseline; the gate tolerance "
                   "absorbs that, and the per-judge guardrails are the real signal.")


# --- audit trail -----------------------------------------------------------

st.divider()
st.subheader("Audit trail")
st.caption("Every answer-or-abstain decision this session, hash-chained so any "
           "edit to the record is detectable. This is the operational-risk "
           "evidence a reviewer asks for under a standard like CPS 230.")
entries = audit.entries()
if not entries:
    st.caption("No decisions logged yet. Ask a question above.")
else:
    ok, detail = audit.verify()
    (st.success if ok else st.error)("Integrity: %s" % detail)
    st.dataframe([
        {"id": e["decision_id"], "role": e["principal"],
         "clearance": e["clearance"], "decision": e["decision"],
         "reason": e["abstain_reason"], "withheld": e["access_withheld"],
         "pii": e["pii_redactions"], "question": e["question"][:60]}
        for e in reversed(entries)], use_container_width=True, hide_index=True)
    st.download_button("Download audit log (JSONL)", audit.to_jsonl(),
                       file_name="audit_log.jsonl", mime="application/x-ndjson")
