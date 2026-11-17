# Version 2: more context, depth, and regularization

## Keep two useful teaching versions

**V1 stays the simplest lesson:** two 128-wide decoder blocks, four heads,
64-byte context, no dropout, 437,760 parameters. Preserve the existing
`artifacts/final.pt`, six-round results, and original slide deck. Retrain the
same architecture separately to check that the implementation remains compatible.

**V2 is a small extension, not a different framework:** three 128-wide blocks,
four heads, 128-byte context, and dropout probability 0.1. Keep the same byte
tokenizer, learned positions, causal attention, pre-LayerNorm, 4x GELU MLP,
residual connections, tied embeddings, and initialization. One optional
configuration field lets the same readable implementation load both versions.

| Change | Why try it? | Cost / limitation |
|---|---|---|
| Context 64 → 128 bytes | Attend to more of a sentence instead of only a fragment | Attention scores grow quadratically; 128 bytes still is not a long paragraph |
| Blocks 2 → 3 | Add a stage for combining contextual features | More parameters and compute; can overfit a single book |
| Dropout 0 → 0.1 | Regularize embeddings, attention weights, residual attention output, and MLP output | Slower fitting; improvement must be measured; disabled during evaluation/generation |

## Training and measurement, decided before the new runs

1. Reuse the frozen, checksummed public-domain Internet Archive corpus and
   original contiguous splits. No extra corpus, tokenizer, or pretrained weights.
2. Train V1 from scratch for 1,800 updates; V2 from scratch for 3,600 updates.
   Both use seed 42, batch 32, AdamW LR 0.001, decay 0.01, and gradient clipping 1.
   Keep constant learning rate because the first study's schedule did not help.
3. Score the same 1,024 validation target offsets every 200 updates, with each
   version's configured context. Select each checkpoint by validation loss only.
   Report V2's 1,800-update checkpoint metric as an equal-update comparison too.
4. Compare train/validation gap, parameters, bytes processed, runtime, next-byte
   loss, bits/byte, perplexity, accuracy, and three unchanged fixed-prompt samples.
5. Only after training/selection, score all 16,300 original test targets for both.
   **This test was already reported in the first study: it is a reused holdout,
   not a new independent test.** Do not retune V2 using this comparison.
6. Deliver `artifacts/versions/v1.pt` and `artifacts/versions/v2.pt`, a JSON record,
   an unedited sample comparison, and CLI support for choosing either version.
   Preserve original artifact hashes; check old-checkpoint loading and generation.

Different architectures and different training budgets are bundled here. The
equal-update comparison still is not equal-compute: V2 processes twice as many
bytes per update and has more blocks. Do not attribute any gain to one change
without separate ablations. One seed and one OCR book limit conclusions.

## Further improvements to consider, not implemented in V2

- **Byte-level BPE:** retain open-vocabulary byte coverage while making the
  context more word-efficient. Adds a tokenizer subsystem and makes token-level
  perplexity incomparable; use bits per raw byte for cross-tokenizer evaluation.
- **More diverse public-domain Archive books:** likely more helpful for general
  writing than more layers on one book. Requires rights verification and
  document-level train/validation/test separation, ideally a fresh test corpus.
- **RoPE/relative positions or longer-context training:** useful in a later
  architecture lesson, but a deliberate departure from GPT-2's learned positions.
- **Fused attention / a KV cache:** improve training or generation speed, not
  intrinsic model quality; retain explicit attention for the first classroom lesson.
- **Multi-seed ablations:** isolate context, depth, and dropout under matched
  budgets before claiming reliable or causal architectural improvements.

The goal is better held-out prediction and visibly improved local language,
not instruction following, factual reliability, or coherent long stories.
