# Retrieval-Augmented Generation

Retrieval-Augmented Generation, or RAG, combines a language model with an
external document store so that answers are grounded in specific sources rather
than in the model's parametric memory. At query time the system retrieves the
passages most relevant to the question and places them in the prompt, and the
model answers using that supplied context.

A typical RAG pipeline has four stages. First, documents are split into chunks
and embedded into vectors that are stored in a vector index. Second, the user's
question is embedded and the nearest chunks are retrieved. Third, a reranker, 
often a cross-encoder, reorders the retrieved candidates by reading the question
and each chunk together for a more accurate relevance score. Fourth, the top
chunks are inserted into a grounded prompt and the model generates an answer
that cites them.

RAG reduces hallucination because the model is instructed to rely only on the
retrieved context, and it lets a system stay current by updating the document
store instead of retraining the model. A well-designed RAG system also abstains:
when the retrieved context does not contain the answer, it says so rather than
guessing. Evaluating RAG therefore means measuring both retrieval quality and
the faithfulness of answers to their sources.
