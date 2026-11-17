# A tiny GPT for the classroom

A real decoder-only Transformer, trained from scratch and then extended through
continued training on public-domain Internet Archive OCR.
**Not a pretrained GPT-2 or a chatbot.**

## Published results website

**Live site:** https://red-smoke-04a79c703.4.azurestaticapps.net/

The lab now includes a static results explorer with the six original experiments,
V1-V7 comparisons, fivefold-corpus results, unedited saved samples, and all three
teaching decks. It displays recorded outcomes, not a live inference API.

See [infrastructure and deployment instructions](infra/README.md) to provision
Azure hosting and publish it reproducibly with Terraform and Azure CLI:

```bash
export AZURE_SUBSCRIPTION_ID="<your-subscription-id>"
bash scripts/deploy.sh
```

Only allowlisted website assets and lesson/report outputs are uploaded.
Checkpoints, raw corpora, source files, credentials, and Terraform state are not.

### GPT under the microscope

The website's **How GPT works** section has three interactive, keyboard-accessible tabs:

1. **Predict one byte:** inspect byte IDs, actual embedding features, each block
   and attention head, causal masking, residual additions, logits and probabilities.
2. **Learn from one example:** follow shifted targets, per-position cross-entropy,
   backpropagated gradients and one actual AdamW update on a disposable model copy.
3. **Generate a sentence:** step through or play 12 recorded next-byte samples,
   showing the prefix, sampling distribution, chosen byte and appended output.

All values come from CPU execution of the original `artifacts/final.pt`, not
illustrative numbers or live browser inference. The training window is explicitly
synthetic (`"Alice s"`, from `"Alice sees a cat."`); the model copy starts with the
Archive-trained weights. It demonstrates an update, not a heldout quality gain.
The existing checkpoint, experiments and decks are never overwritten.
Only the first four hidden features are displayed; full attention weights and
next-byte logits/probabilities are included in the downloadable recording.

Regenerate the trace after changing its producer or the original model:

```bash
python3 build_microscope.py
python3 -m unittest -v test_model.MicroscopeTests
node --test scripts/site.test.mjs
node scripts/build-site.mjs
```

The builder rejects recordings whose checkpoint or Python source hashes no
longer match. The deployment remains static; PyTorch is needed to regenerate
traces, not to host the site.

Start with [PLAN.md](PLAN.md). Open [slides.html](slides.html) in a browser for the
17-slide teaching deck (arrow keys / space to advance, `N` for speaker notes).
[slides.pdf](slides.pdf) is the rendered, printable version. Read
[RESULTS.md](RESULTS.md) for measured comparisons and every fixed-seed sample.

## Full architecture and experiment lecture

The **85-slide deep dive** is available as [PDF](deep-dive.pdf),
[editable PowerPoint](deep-dive.pptx), and [browser slides with speaker notes](deep-dive.html).
Its 62 lecture slides explain tensor shapes, attention mathematics, actual
source-linked Python, Archive corpus preparation, optimization and evaluation,
the six initial experiments, and exactly how V1-V7 were constructed.
The 23 appendix slides contain all 59 recorded samples, unedited.
Results include learning curves, checkpoint selection, per-book comparisons,
loss, bits per byte, perplexity, accuracy, and limitations.

In the browser, use arrow keys or space to advance, the slide picker to jump,
and `N` to show notes. PowerPoint also contains speaker notes; diagrams, charts,
tables, and code text are native editable objects.
All 85 browser/print layouts and navigation were checked, and the PDF contains
85 landscape pages. PowerPoint contents, notes, geometry, and chart data are
checked structurally; the PDF is rendered from HTML, not from PowerPoint.
The original 17-slide deck and all trained models remain unchanged.

Regenerate the deep dive from the included source and experiment records,
without retraining:

```bash
python3 -m venv .slides-venv
.slides-venv/bin/python -m pip install -r requirements-slides.txt
.slides-venv/bin/python build_deep_deck.py
node render_deep_deck.mjs
```

PDF export requires Node.js 22+ and installed Edge, Chrome, or Chromium.
Set `SLIDE_BROWSER` to the browser executable if it is not found automatically.
The renderer uses an isolated temporary browser profile and checks every slide
for content/footer overlap before exporting. A preview and render-check JSON
are included alongside the deck.

## Seven selectable model versions

[PLAN_V2.md](PLAN_V2.md) and [VERSIONS.md](VERSIONS.md) cover the first two versions.
[PLAN_V3_V7.md](PLAN_V3_V7.md) and [EXPANDED_RESULTS.md](EXPANDED_RESULTS.md) cover
five more versions, expanded data, common accuracy comparisons, and unedited samples.
The available training corpus grows from 132,440 to 1,240,716 bytes (9.4x).

| Version | Blocks / byte context | Training books | Checkpoint |
|---|---|---:|---|
| V1 | 2 / 64 | 1 | `artifacts/versions/v1.pt` |
| V2 | 3 / 128 | 1 | `artifacts/versions/v2.pt` |
| V3 | 3 / 128 | 2 | `artifacts/versions/v3.pt` |
| V4 | 3 / 128 | 3 | `artifacts/versions/v4.pt` |
| V5 | 4 / 128 | 4 | `artifacts/versions/v5.pt` |
| V6 | 4 / 128 | 5 | `artifacts/versions/v6.pt` |
| V7 | 4 / 128 | 6 | `artifacts/versions/v7.pt` |

All models have width 128 and four heads; V1 has no dropout, V2-V7 use dropout 0.1.

```bash
python3 generate.py --version v1 --device cpu --prompt 'Alice said, "'
python3 generate.py --version v2 --device cpu --prompt 'Alice said, "'
python3 generate.py --version v7 --device cpu --prompt 'Dorothy said, "'

# First two versions: scratch training.
python3 train_versions.py --device cpu
python3 report_versions.py

# Five more: continued training, adding one Archive work per version.
python3 expanded_data.py
python3 train_expanded.py --device cpu
python3 report_expanded.py
```

V3-V7 are sequential warm starts, not independent scratch runs. The extra block
in V5 starts as an identity residual block to preserve its parent's predictions.
Expanded training leaves V1/V2, the original data, reports, and deck untouched.
Without `--version`, generation still defaults to `artifacts/final.pt`.
All seven are compared on the reused original holdout and a fresh five-book
holdout, first scored after all five new checkpoint selections.

## Fivefold corpus experiment

[FIVEFOLD_RESULTS.md](FIVEFOLD_RESULTS.md) gives the separate larger-data experiment:
paired small/large-corpus reruns of every V1-V7 recipe, coverage of the original
six experiments (R4 is identical to V1), and a longer V7 continuation.
The 14 added Archive narratives expand **each training corpus by at least 5x**,
without overwriting the earlier models, reports, or presentations.

| Version | Previous train bytes | Larger train bytes | Growth |
|---|---:|---:|---:|
| V1 | 132,440 | 685,236 | 5.17x |
| V2 | 132,440 | 685,236 | 5.17x |
| V3 | 250,152 | 1,377,265 | 5.51x |
| V4 | 418,197 | 2,189,179 | 5.23x |
| V5 | 625,761 | 3,252,191 | 5.20x |
| V6 | 893,030 | 4,514,489 | 5.06x |
| V7 | 1,240,716 | 6,295,276 | 5.07x |

Each pair uses identical starting weights, architecture, update budget, and
validation targets. V3-V7 use the same **original parent** in each arm, not a
newly retrained chain. Large-corpus sampling preserves the original rehearsal
distribution in 20% of windows and uses additional books in 80%.
All paired selections and the predeclared 15,600-update V7 continuation finish
before new-book performance testing. Legacy test targets are reused exactly.

All seven expanded-data versions improved added-book next-byte accuracy versus
their matched-budget controls, by 1.73-7.38 percentage points. Original-book
accuracy fell in six versions; the legacy five-book average improved in all seven.
V7's added-book accuracy increased from 55.13% (control) to 59.24% (larger data),
then 62.37% with the longer budget. These are byte predictions, not coherent
writing: the recorded samples still contain broken grammar and mixed characters.
All 24 unit tests and the original-artifact integrity checks passed.

```bash
python3 fivefold_data.py
python3 -m unittest -v
python3 fivefold_train.py --arm control --device cpu
python3 fivefold_train.py --arm large --device cpu
python3 fivefold_train.py --evaluate --device cpu
python3 report_fivefold.py
python3 generate.py --checkpoint artifacts/fivefold/large/v7_long.pt --device cpu
```

Rerunning these commands replaces only the new experiment outputs.
Models are in `artifacts/fivefold/{control,large}/`; full metrics, curves,
paired-book intervals, and fixed-seed samples are in
`artifacts/fivefold/comparison.json` and `artifacts/fivefold/metrics.csv`.
The corpus manifest, hashes, actual Archive rights statements, text-edition/reprint
notes, and exact-span audit are in `data/fivefold/manifest.json`.
The original deep-dive slides describe the earlier experiments, not these new reruns.

## Decision and reasoning model families

[CONCEPT_MODELS.md](CONCEPT_MODELS.md) explains two additional model types,
their architectures, exact supervision, training, and limitations. These use
explicitly synthetic labels; the existing language models retain Archive data.

| Model | Concept | Architecture | Heldout result |
|---|---|---|---|
| D1 | Supervised decision policy | 1,250-parameter ReLU MLP | 100% action accuracy; zero mean regret |
| R0 | Answer-only control | 118,528-parameter, two-block GPT | 100% final answers |
| R1 | Supervised scratchpad reasoning | Same GPT as R0, trained on intermediate states | 100% final answers and verified two-step traces |

Each result uses 100 heldout input triples, not next-byte accuracy on prose.
R0 already solves the small task: no accuracy advantage for reasoning is claimed.
R1 fails the required three-step format on unseen longer problems.
This is not reinforcement learning or general-purpose reasoning.
The unsuccessful modular-arithmetic attempt is preserved, not hidden.

```bash
python3 concept_models.py --model decision --values 3 8 5
python3 concept_models.py --model reasoning --values 8 2 5
```

[Results and samples](concepts/RESULTS.md) and [full JSON](concepts/results.json)
accompany a separate 16-slide lesson:
[HTML with notes](concepts/lesson.html), [PDF](concepts/lesson.pdf),
[editable PowerPoint](concepts/lesson.pptx). Original models and decks are unchanged.

## Run

Python 3.10+ and PyTorch 2.2+ are required. Dependencies were already available
on the machine used for the recorded experiments.

```bash
cd /Users/michalfu/mini-gpt-classroom
# If PyTorch is missing, use a virtual environment:
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# Data and final checkpoint are included; download again only if needed.
python data.py
python -m unittest -v
python generate.py --device cpu --prompt 'Alice said, "'

# One small classroom run (does not overwrite the six experiments).
python train.py --device cpu --steps 600
python generate.py --device cpu --checkpoint artifacts/classroom.pt

# Reproduce all six experiments; overwrites their checkpoints and metrics.
python train.py --experiments --device cpu
python build_deck.py
python verify_artifacts.py
```

After new experiments, regenerate the PDF by opening `slides.html` and printing
to PDF in landscape, with backgrounds enabled and browser headers/footers off.
The included PDF was browser-rendered and checked to contain exactly 17 pages;
slide navigation, notes, and horizontal overflow were also browser-tested.

Use `python3` instead of `python` outside the virtual environment.
For another computer, use `--device auto`, `--device mps`, or `--device cuda`.
The recorded CPU/GPU benchmark found CPU faster for this particular tiny model;
the report records the device and actual training times. Exact numbers can
differ by device and PyTorch version even with fixed seeds.

## Read the code in this order

| File | Purpose |
|---|---|
| `model.py` | Byte encoding, explicit causal attention, GPT blocks, generation |
| `data.py` | Archive download, story cleaning, frozen splits, checksums |
| `train.py` | Shifted batches, training, fair target scoring, six experiments |
| `generate.py` | Load the final model and sample new text |
| `test_model.py` | Ten focused correctness/learning/compatibility tests |
| `test_expanded.py` | Fourteen boundary, provenance, aggregation, depth-growth and fivefold-corpus tests |
| `train_versions.py` | Retrain V1 and train the regularized, longer-context V2 |
| `report_versions.py` | Verify both checkpoints and generate the measured comparison |
| `expanded_data.py` | Five rights-verified books, story extraction, disjoint splits and overlap audit |
| `train_expanded.py` | Train V3-V7 with rehearsal and common validation; then evaluate fresh tests |
| `report_expanded.py` | Verify all seven models and publish accuracy/data/sample comparisons |
| `build_deck.py` | Turn recorded metrics into the report and browser slides |
| `build_deep_deck.py` | Build the full lecture from actual source and all experiment records |
| `slide_formats.py` | Render native editable PowerPoint objects and standalone HTML |
| `render_deep_deck.mjs` | Browser-check all lecture slides and export the landscape PDF |
| `fivefold_data.py` | Prepare genuine 5x-or-larger Archive corpora, verify downloads and audit overlap |
| `fivefold_train.py` | Run matched-budget controls, larger-data pairs, and the longer V7 check |
| `report_fivefold.py` | Verify all new results and publish metrics plus every unedited sample |
| `concept_models.py` | Train and run supervised decision, answer-only and scratchpad models |
| `test_concepts.py` | Check labels, disjoint cases, response masks, gradients and trace verification |
| `build_concept_deck.py` | Build the separate decision/reasoning lesson from actual code and results |
| `verify_artifacts.py` | Check delivered files, results, and final checkpoint |

For the first lesson, concentrate on `model.py` and the `batch` and `run_round`
functions in `train.py`; experiment orchestration is separate from the model.
The model uses no Transformers library or pretrained weights.

## Important limits

- Byte IDs 0..255 replace GPT-2 BPE. One byte is not one word; even V7's
  128-byte context is short. All loss/perplexity comparisons here use byte tokens.
- OCR spelling mistakes remain. A contiguous book split avoids overlapping
  windows, but style, names, and repeated phrases recur across the same book.
- There is one seed and unequal compute. Changes are bundled; V3-V7 inherit
  prior training. These are not controlled causal or multi-seed ablations.
- Model/checkpoint selection uses validation repeatedly. In the first study,
  test was opened only after selection. V2 reuses that test for comparison;
  V3-V7 keep that comparison and add a new within-book holdout. New checkpoint
  selection still uses validation only; fresh test performance is scored after
  all five training runs and selections.
- Generation samples use the same prompts, temperature, top-k, and seeds in
  every round. Word-lexicon match is a rough spelling proxy, not coherence.
- Sampling is not an assistant: prompts are prose prefixes, not instructions.
  Cropped generation resets learned positions; no key/value cache is used.

## Sources and rights

- Original architecture: https://github.com/openai/gpt-2/blob/master/src/model.py
- Paper: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
- Corpus: https://archive.org/details/throughlookinggl00carruoft
- Download/rights/checksums: `data/provenance.json`
- Expanded books/rights/boundaries/checksums: `data/expanded/manifest.json`

Carroll's original text is public domain; Archive metadata marks this scan
`NOT_IN_COPYRIGHT` in the US. Check local rights before redistributing.
This is a digitized book from the Internet Archive, not a Wayback web crawl.
Expanded data adds *Alice's Adventures in Wonderland*, *The Wonderful Wizard
of Oz*, *Peter and Wendy*, *The Wind in the Willows*, and *The Secret Garden*.
All scan metadata marks them `NOT_IN_COPYRIGHT` (US); review older language for
classroom use and check local rights, including Peter Pan-related UK provisions.
