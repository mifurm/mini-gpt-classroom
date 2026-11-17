# Five more versions: broaden the data without hiding regressions

## Data expansion

Keep the original *Through the Looking-Glass* training/validation/test files
unchanged. Add five Internet Archive scans marked `NOT_IN_COPYRIGHT` in US metadata:

| Added in | Work | Archive item | Edition |
|---|---|---|---|
| V3 | Alice's Adventures in Wonderland | `alicesadventures00carruoft` | 1872 |
| V4 | The Wonderful Wizard of Oz | `wonderfulwizardo00baumiala` | 1900 |
| V5 | Peter and Wendy | `peterwendy00barr` | 1911 metadata; 1912 scan title page |
| V6 | The Wind in the Willows | `windinwillows00grah` | 1913 |
| V7 | The Secret Garden | `secretgarden00burn` | 1911 |

Use only each original story: exclude library labels, advertisements, and other
works bound in the same scan. In particular, Peter's scan also contains *Margaret
Ogilvy*, which must not be included; Alice's scan advertises *Looking-Glass*,
which must not become duplicate training data. Preserve source/cleaned hashes,
rights evidence, source URLs, and explicit extraction boundaries.

Split each new story into contiguous 80/10/10 passages, with 128-byte discarded
gaps. Store books separately. A training window must fit entirely within one
book's training split; concatenated book boundaries are never training examples.
Record a long exact-span overlap audit against both original and new holdouts.
These are within-book held-out passages, not entirely unseen-author test books.

All books are public domain in the US; check local rights before redistribution.
Peter Pan-related works have special royalty provisions in the UK.

## Version recipes, chosen before new test evaluation

| Version | Parent | Books trained on | Blocks / width / heads / context | New updates |
|---|---|---:|---|---:|
| V3 | V2 | 2 | 3 / 128 / 4 / 128 bytes | 1,800 |
| V4 | V3 | 3 | 3 / 128 / 4 / 128 bytes | 1,800 |
| V5 | V4 | 4 | 4 / 128 / 4 / 128 bytes | 1,800 |
| V6 | V5 | 5 | 4 / 128 / 4 / 128 bytes | 1,800 |
| V7 | V6 | 6 | 4 / 128 / 4 / 128 bytes | 2,600 |

These are sequential warm-started versions, **not five scratch-trained models**.
They inherit the previous validation-selected checkpoint. V5 adds one initially
identity residual block: zero its attention/MLP output projections so the old
model's inference function is preserved before training. All retain the same
GPT-2-style byte architecture and dropout 0.1.

Use CPU, seed 42, batch 32, AdamW learning rate 0.0003, weight decay 0.01, and
gradient clipping norm 1. Half of sampled windows come from the original
training document; half are distributed equally across the currently available
new books. This rehearsal mixture aims to preserve the original style while
learning broader language. Record actual window counts per document.

## Fair, explicit evaluation

- **Common validation:** fixed original 1,024 target bytes plus 256 target bytes
  per new book, all five books included for every version. Validation can include
  works not yet used for training; this makes scores directly comparable.
- **Checkpoint score:** 50% original validation loss plus 50% mean loss across
  the five new validation documents. Same weighting for all versions. Score
  every 200 updates and retain the lowest score, including the initial parent.
- **Legacy test:** after all five models are trained/selected, evaluate every
  original test byte after its first 128 bytes (16,300 targets). This is a reused
  holdout and must be labeled as such.
- **Fresh expanded test:** first evaluate after all five selections are final;
  fixed 2,048 distinct target bytes in each new book's test split. Report per-book
  accuracy and an equal-book macro average. This holdout was not scored by V1/V2.
- Evaluate V1/V2 on the same fresh targets as baselines. Select any recommended
  new checkpoint using common validation, not either test. Do not revise recipes
  after seeing test outcomes.
- Report accuracy, next-byte loss, bits/byte, byte perplexity, data volume,
  cumulative inheritance, runtime, model size, curves, and unedited samples.

More books do not guarantee higher original-book accuracy. Diversity, OCR
noise, capacity, extra updates, and continual learning are bundled; there is one
seed and unequal cumulative compute. Report every regression. None of these
models should be represented as a chatbot or reliable long-form writer.

## Deliverables

Five additional checkpoints `artifacts/versions/v3.pt` through `v7.pt`; expanded
data/provenance; the runnable expansion pipeline; a comparison including V1/V2;
and `generate.py --version v1` through `--version v7`. Preserve old checkpoints,
the first studies' reports, original data, and original slide deck.
