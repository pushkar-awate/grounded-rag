# BERT and Masked Language Model Pretraining

BERT is a Transformer encoder trained to produce contextual representations of
text. Unlike left-to-right language models, BERT is bidirectional: each token's
representation is informed by context on both sides. This is achieved through a
pretraining objective called masked language modelling.

In masked language modelling a fraction of the input tokens are replaced with a
special mask symbol, and the model is trained to predict the original tokens
from the surrounding context. Because the model must use both left and right
context to fill in the blanks, it learns deep bidirectional features. BERT was
additionally trained on a next-sentence prediction task, where it learns whether
one sentence plausibly follows another.

After pretraining on large unlabelled corpora, BERT is fine-tuned on downstream
tasks such as classification, named-entity recognition, and question answering
by adding a small task-specific head. This pretrain-then-fine-tune recipe became
the dominant paradigm in natural language processing, because a single
pretrained encoder could be adapted to many tasks with relatively little labelled
data.
