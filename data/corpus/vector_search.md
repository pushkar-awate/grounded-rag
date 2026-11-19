# Embeddings and Vector Search

An embedding is a dense vector that represents the meaning of a piece of text so
that semantically similar texts land close together in vector space. Embeddings
are produced by models trained so that related inputs have high similarity,
usually measured by cosine similarity or inner product.

Vector search finds the stored vectors nearest to a query vector. For small
collections an exact brute-force scan is fine, computing the similarity between
the query and every stored vector. For large collections, approximate
nearest-neighbour indexes such as HNSW or inverted-file indexes trade a small
amount of recall for a large speed-up. Libraries like FAISS provide these
indexes and run efficiently on CPU.

A common trick is to normalise all vectors to unit length. Once vectors are
normalised, their inner product equals their cosine similarity, so the simplest
possible index, a flat inner-product index, gives exact cosine search with no
training step. Embeddings and vector search are the retrieval backbone of modern
semantic search and of retrieval-augmented generation systems.
