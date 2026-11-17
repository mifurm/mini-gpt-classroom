# Teaching plan: build a real, small GPT

## Goal and scope

Students will implement and train a decoder-only Transformer that predicts the
next UTF-8 byte. It follows the open-source GPT-2 architecture, not its size,
pretrained weights, or tokenizer. A small book-trained model should learn spelling,
local syntax, dialogue, and style; it will not be a useful chatbot or reliably
write coherent long stories. Do not call byte perplexity GPT-2 token perplexity.

## Implementation sequence (two 90-minute labs plus precomputed training)

1. **Data and task (20 min):** download the Internet Archive's public-domain
   *Through the Looking-Glass* scan. Inspect OCR, retain provenance and checksums,
   remove scan furniture, split contiguous passages 80/10/10 with boundary gaps.
   Encode UTF-8 bytes into IDs 0..255. Show `x = bytes[:-1]`, `y = bytes[1:]`.
2. **Architecture (40 min):** token + learned position embeddings; explicit
   multi-head scaled dot-product attention with a triangular causal mask;
   pre-LayerNorm residual attention; pre-LayerNorm residual 4x-width GELU MLP;
   final LayerNorm and a vocabulary projection tied to the token embedding.
3. **Learning (30 min):** cross-entropy, backpropagation, AdamW, gradient clipping.
   Show tensor shapes, a training step, and the future-token leakage test.
4. **Measurement (25 min):** identical held-out target positions across rounds;
   report next-byte cross-entropy, bits/byte, perplexity, and top-1 accuracy.
   Use only validation for choices; open test only after selecting the final model.
5. **Five improvement rounds (35 min):** compare a baseline with more training,
   longer context, more capacity, additional training, and warmup/cosine decay.
   Each experiment starts from scratch with the same seed. Keep the best
   validation checkpoint. These are a practical sequence, not an equal-compute
   or multi-seed causal study; regressions must be reported, not hidden.
6. **Generation and critique (30 min):** fixed prompts/seeds, temperature/top-k,
   inspect representative outputs from every round. Discuss misspellings,
   repetition, OCR artifacts, memorization, and lack of story-level reasoning.

## Exact architecture retained from GPT-2

OpenAI reference: https://github.com/openai/gpt-2/blob/master/src/model.py

- Learned token/position embeddings, causal self-attention, bias-bearing projections.
- Pre-normalization and residual connections in both sublayers.
- Tanh-approximate GELU, 4x expansion, final LayerNorm, tied vocabulary weights.

Intentional departures: byte tokenizer rather than 50,257-token BPE; fewer/smaller
blocks; short contexts; no pretrained weights, key/value cache, distributed
training, or mixed precision. Linear layers express the reference's `conv1d`
projections. These omissions simplify engineering, not the attention mechanism.

## Data and rights

https://archive.org/details/throughlookinggl00carruoft

Archive metadata dates the scan edition to approximately 1899 and marks it
`NOT_IN_COPYRIGHT` in the US. Carroll's original work is public domain.
Only the original story is used, not a modern annotated edition. This is an
Internet Archive digitized-book corpus, **not** a Wayback Machine web crawl.
OCR is imperfect and retained errors limit attainable quality. Students outside
the US should check local rights before redistributing the corpus.

## Deliverables and acceptance criteria

- Short readable `model.py`, data preparation, runnable training/generation CLIs.
- Correctness tests: causality, shifting, loss/backpropagation, checkpoint roundtrip,
  byte encoding, valid dimensions, and disjoint data boundaries.
- Baseline plus five actually executed improvement experiments; machine-readable
  metrics, learning curves, checkpoints, and fixed-seed samples.
- A measured final test score, a clearly limited quality assessment, and a simple
  browser slide deck with real results and speaker notes.
- Final model must beat a training-only unigram baseline on held-out loss and
  produce visibly learned English-like text. Do not promise monotonic gains.
