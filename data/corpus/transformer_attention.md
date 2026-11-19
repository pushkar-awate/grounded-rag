# The Transformer and Self-Attention

The Transformer is a neural network architecture for sequence modelling that
replaced recurrence with attention. Instead of processing tokens one after
another, it processes a whole sequence in parallel and lets every token look at
every other token directly through a mechanism called self-attention.

Self-attention works by projecting each token into three vectors: a query, a
key, and a value. The relevance of one token to another is computed as the dot
product of a query with a key, scaled and passed through a softmax to produce
attention weights. Each token's new representation is the weighted sum of all
value vectors. Because any token can attend to any other in a single step, the
path length between distant tokens is constant, which makes long-range
dependencies far easier to learn than in recurrent networks.

Multi-head attention runs several attention operations in parallel, each with
its own learned projections, so the model can attend to different kinds of
relationships at once. The Transformer stacks these attention layers with
position-wise feed-forward networks, residual connections, and layer
normalisation. Since attention itself is order-agnostic, positional encodings
are added to the inputs so the model knows where each token sits in the
sequence.
