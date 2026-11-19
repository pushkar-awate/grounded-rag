"""grounded-rag - a RAG assistant that cites its sources and refuses to answer
when the documents don't support it.

Ask questions over a bundled set of ML/LLM explainers, your own uploaded PDF, or
a live arXiv search. Every answer is grounded in retrieved passages with [n]
citations; when the corpus can't answer, the system abstains instead of
guessing. A one-click evaluation shows retrieval quality, groundedness, and
correct-refusal rates, gated against a committed baseline - the same eval that
runs in CI.
"""
import hashlib
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
                     ExtractiveBrain, has_groq_key, build_store)
from ragcore.ingest import load_path
from evalcore import default_judges, load_cases, regression_gate, run_eval
from evalcore.metrics import scorecard
from app.corpus import load_bundled_corpus, fetch_arxiv

HERE = os.path.dirname(os.path.abspath(__file__))


# --- cached heavy resources (loaded once per session) ----------------------

@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedder():
    try:
        emb = STEmbedder()
        emb.embed(["warmup"])  # force model download/load now
        return emb, "MiniLM (sentence-transformers)"
    except Exception:
        return HashingEmbedder(), "hashing fallback (offline)"


@st.cache_resource(show_spinner="Loading reranker...")
def get_reranker():
    try:
        rr = CrossEncoderReranker()
        rr.rerank("warmup", [], top_k=0)  # no-op; model loads on first real call
        return rr, "cross-encoder (ms-marco-MiniLM)"
    except Exception:
        return IdentityReranker(), "identity (offline)"


@st.cache_resource(show_spinner="Indexing documents...")
def get_store(signature: str, _docs):
    # signature makes the cache key change when the corpus changes.
    emb, _ = get_embedder()
    return build_store(_docs, emb)


def corpus_signature(docs) -> str:
    h = hashlib.sha1()
    for d in docs:
        h.update(d.doc_id.encode())
        h.update(str(len(d.text)).encode())
    return h.hexdigest()


# --- sidebar: corpus + settings --------------------------------------------

st.title("grounded-rag")
st.caption("Retrieval-augmented QA that **cites its sources** and **abstains** "
           "when the documents can't answer. Retrieval, reranking, "
           "grounded generation, then an evaluation gate.")

with st.sidebar:
    st.header("Corpus")
    source = st.radio("Documents to search", [
        "Bundled ML/LLM papers", "Upload a PDF / text file", "Live arXiv search"])

    docs = None
    if source == "Bundled ML/LLM papers":
        docs = load_bundled_corpus()
        st.caption("%d built-in explainers (Transformers, LoRA, RAG, BERT, "
                   "vector search, MLOps drift)." % len(docs))
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
store = get_store(corpus_signature(docs), docs)
brain = GroqBrain() if (use_llm and key_present) else ExtractiveBrain()
brain_name = "Groq GPT-OSS-20B" if (use_llm and key_present) else "extractive (deterministic)"
pipe = RagPipeline(emb, store, rr, brain, retrieve_k=retrieve_k, context_k=context_k)

st.caption("Embedder: **%s**  |  Reranker: **%s**  |  Generator: **%s**  "
           "|  %d chunks indexed" % (emb_name, rr_name, brain_name, len(store.chunks)))


# --- ask -------------------------------------------------------------------

st.subheader("Ask a question")
examples = {
    "Bundled ML/LLM papers": "How does self-attention use queries and keys?",
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
        st.warning("**Abstained.** The retrieved context doesn't contain "
                   "the answer, so the system declined to guess.")
    else:
        st.markdown("### Answer")
        st.markdown(ans.text)
        if ans.cited_chunks:
            st.markdown("**Sources cited**")
            for i, c in enumerate(ans.cited_chunks, start=1):
                with st.expander("%s" % c.citation()):
                    st.write(c.text)

    with st.expander("Retrieved context (reranked)"):
        for c, score in res.reranked:
            st.markdown("**%s**  (score %.3f)" % (c.citation(), score))
            st.caption(c.text[:400] + ("..." if len(c.text) > 400 else ""))


# --- evaluation ------------------------------------------------------------

st.divider()
st.subheader("Evaluation")
st.caption("Runs the gold eval set over the **bundled** corpus and gates the "
           "scorecard against the committed baseline. This is the same check CI runs.")
if st.button("Run evaluation on the bundled corpus"):
    with st.spinner("Scoring retrieval, grounding, and refusal..."):
        bundled = load_bundled_corpus()
        estore = get_store(corpus_signature(bundled), bundled)
        epipe = RagPipeline(emb, estore, rr, brain,
                            retrieve_k=retrieve_k, context_k=context_k)
        cases = load_cases(os.path.join(HERE, "data", "eval_cases.json"))
        judges = default_judges(brain=brain, use_llm=(use_llm and key_present))
        run = run_eval(epipe, cases, judges, version="dashboard")
        card = scorecard(run)

    rates = card["per_judge_pass_rate"]
    cols = st.columns(len(rates) + 1)
    pretty = {"retrieval_hit": "Retrieval hit", "mrr": "MRR pass",
              "citation_present": "Citations", "groundedness": "Groundedness",
              "refusal_correct": "Correct refusal", "faithfulness": "Faithfulness"}
    for col, (name, rate) in zip(cols, rates.items()):
        col.metric(pretty.get(name, name), "%d%%" % rate)
    cols[-1].metric("Abstention", "%d%%" % card["abstention_rate"])

    baseline_path = os.path.join(HERE, "baseline.json")
    if os.path.exists(baseline_path):
        import json
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
