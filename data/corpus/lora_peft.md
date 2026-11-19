# LoRA and Parameter-Efficient Fine-Tuning

Fine-tuning a large language model by updating all of its weights is expensive:
it needs a full copy of the optimiser state and produces a whole new model per
task. Parameter-efficient fine-tuning (PEFT) methods avoid this by training only
a small number of extra parameters while keeping the original weights frozen.

LoRA, short for Low-Rank Adaptation, is the most widely used PEFT method. It
freezes the pretrained weight matrices and injects a pair of small, trainable
low-rank matrices alongside them. During training only those low-rank matrices
are updated; at inference their product is added to the frozen weights, so there
is no extra latency. Because the rank is small, the number of trainable
parameters can be a tiny fraction of the full model, which dramatically reduces
memory use and lets large models be adapted on a single consumer GPU.

QLoRA extends this idea by quantising the frozen base model to 4-bit precision
while still training LoRA adapters in higher precision. This cuts memory further
and makes it possible to fine-tune multi-billion-parameter models on modest
hardware. A key practical benefit of LoRA is that the adapters are small files
that can be stored and swapped per task without duplicating the base model.
