"""Build the complete V1-V7 lecture from actual source code and recorded results."""

import ast
import hashlib
import json
from pathlib import Path
import textwrap

from slide_formats import render_html, render_pptx


ROOT = Path(__file__).resolve().parent
TITLE = "A tiny GPT, deeply explained: architecture, code, training, V1-V7"
INPUTS = [
    "model.py", "data.py", "expanded_data.py", "train.py", "train_versions.py",
    "train_expanded.py", "test_model.py", "test_expanded.py", "generate.py",
    "data/provenance.json", "data/expanded/manifest.json", "artifacts/results.json",
    "artifacts/versions/comparison.json", "artifacts/versions/expanded_comparison.json",
]


def text(value):
    return dict(kind="text", text=value)


def points(*items):
    return dict(kind="points", items=list(items))


def table(headers, rows):
    return dict(kind="table", headers=headers, rows=rows)


def diagram(label, labels):
    return dict(kind="diagram", label=label, labels=labels)


def chart(label, series):
    return dict(kind="chart", label=label, series=series)


def source(filename, symbol, first=0, last=None):
    lines = (ROOT / filename).read_text().splitlines()
    node = ast.parse("\n".join(lines))
    for part in symbol.split("."):
        matches = [child for child in node.body if getattr(child, "name", None) == part]
        if len(matches) != 1:
            raise ValueError(f"Cannot resolve source symbol {filename}:{symbol}")
        node = matches[0]
    start = node.lineno - 1
    end = node.end_lineno
    section = lines[start:end]
    if last is None:
        last = len(section)
    if not 0 <= first < last <= len(section):
        raise ValueError(f"Invalid excerpt range {symbol}: {first}:{last}, length {len(section)}")
    code = textwrap.dedent("\n".join(section[first:last]))
    return dict(kind="code", text=code, source=f"{filename}:{start+first+1}-{start+last} · {symbol}",
                filename=filename, first_line=start+first+1, last_line=start+last)


def percent(value):
    return f"{value:.1%}"


def metric_matrix(results, key, digits=3):
    names = ["alice", "oz", "peter", "willows", "garden"]
    return [[r["name"].upper()] + [
        percent(r["tests"]["per_book"][name][key]) if key == "accuracy"
        else f"{r['tests']['per_book'][name][key]:.{digits}f}" for name in names
    ] for r in results]


def sampled(result, index, study):
    item = result["samples"][index]
    return dict(
        label=f"{result['name'].upper()} · seed {item['seed']} · T={item['temperature']} · k={item['top_k']}",
        text=item["prompt"] + item["continuation"],
        sample_id=f"{study}:{result['name']}:{index}",
    )


def make_slides(first, second, expanded, provenance, corpus):
    slides = []
    rounds = first["rounds"]
    v1, v2 = second["versions"]
    versions = expanded["versions"]
    all_results = expanded["baselines"] + versions

    def add(section, title, blocks, notes):
        slides.append(dict(section=section, title=title, blocks=blocks, notes=notes))

    add("Orientation", "A tiny GPT, deeply explained", [
        text("Architecture → actual Python → learning → seven version recipes → measured outcomes."),
        text("A decoder-only GPT-2-style model: 437,760–842,496 parameters, trained locally on public-domain Internet Archive OCR."),
        points("Main lecture: internals and experiments.", "Appendix: every recorded generated sample, unedited."),
    ], "Use the main lecture over two sessions, then browse the sample appendix during discussion. "
       "The deck is generated from source files and the three experiment JSON records. V1/V2 are "
       "scratch-trained; V3-V7 are warm-started continuations. No pretrained GPT-2 weights are used.")
    add("Orientation", "What students should be able to explain", [
        points("Trace every tensor from byte IDs to next-byte logits.",
               "Explain causality, pre-normalization, residual paths and weight tying.",
               "Build a training batch without future or document-boundary leakage.",
               "Distinguish scratch training, warm starts and identity depth growth.",
               "Read the results without confusing byte accuracy with good writing."),
    ], "Prerequisites: Python, basic PyTorch, matrix multiplication, softmax, and gradients. "
       "Allocate a first session to architecture/code and a second to data/training/version comparisons. "
       "The appendix is searchable evidence, not material that must all be presented aloud.")
    add("Architecture", "GPT-2-style does not mean pretrained GPT-2", [
        table(["Feature", "Retained", "Intentional simplification"],
              [["Decoder blocks", "Causal attention + pre-LN residual MLP", "1–4 blocks, not 12"],
               ["Embeddings", "Learned tokens and absolute positions", "256 bytes, not GPT-2 BPE"],
               ["MLP", "4× expansion, tanh GELU, biases", "Width 64 or 128"],
               ["Output", "Final LayerNorm + tied token matrix", "No separate vocabulary head"],
               ["Engineering", "AdamW, clipping, checkpoints", "No KV cache or distributed training"]]),
        text("Structurally faithful and readable; not compatible with the original GPT-2 checkpoint shapes."),
    ], "Reference implementation: https://github.com/openai/gpt-2/blob/master/src/model.py . "
       "The reference's conv1d projections are pointwise linear maps. Our initialization and training "
       "engineering are implementation choices, not a claim to reproduce every original GPT-2 detail.")
    add("Architecture", "The family at a glance", [
        table(["Version", "Blocks", "Width", "Heads", "Byte context", "Dropout", "Parameters"],
              [[r["name"].upper(), r["config"]["layers"], r["config"]["width"],
                r["config"]["heads"], r["config"]["context"], r["config"].get("dropout", 0),
                f"{r['parameters']:,}"] for r in all_results]),
        text("The core mechanism stays the same. V5 adds depth; V3–V7 mainly broaden data coverage."),
    ], "V1 uses no dropout. All later named versions use dropout 0.1. V2 doubles context and "
       "adds a block relative to V1. V5 adds the fourth block. The vocabulary remains 256 "
       "because bytes represent new words without needing a tokenizer vocabulary update.")
    add("Data", "Six real Archive works, not synthetic text", [
        table(["Work", "Scan edition", "Training bytes"],
              [["Through the Looking-Glass", provenance["publication_date"],
                f"{provenance['split_bytes']['train']:,}"]] +
              [[b["title"], b["publication_date"], f"{b['split_bytes']['train']:,}"]
               for b in corpus["books"]]),
        text(f"Available training data: {corpus['total_train_bytes']:,} bytes, "
             f"{corpus['total_train_bytes']/corpus['original_train_bytes']:.1f}× the original corpus."),
    ], "All scans are Internet Archive digitized books with NOT_IN_COPYRIGHT US metadata. "
       "This is not a Wayback crawl. Peter's metadata says c1911 but its title page says 1912. "
       "Rights are US-specific; Peter Pan-related material has special UK provisions. Review "
       "older language and OCR errors before classroom use. Source URLs are in the closing slide.")
    add("Data", "Extract the story, not the whole scanned volume", [
        source("expanded_data.py", "extract_story"),
        points("Alice: stop before advertisements for Looking-Glass.",
               "Peter: exclude the separately bound Margaret Ogilvy.",
               "Keep source, cleaned and split hashes; retain uncorrected OCR errors."),
    ], "This is actual code, not pseudocode. The verified start/end regular expressions are "
       "stored in BOUNDARIES and the corpus manifest. Missing reliable markers cause an "
       "explicit error. Decorative initial letters can be absent from OCR; the Oz opening "
       "is one example. Do not silently fill missing text with invented prose.")
    add("Data", "Split first; never move a holdout into training", [
        source("expanded_data.py", "spans_for"),
        text("Each new work: contiguous 80/10/10 passages, with 128 discarded bytes at both boundaries."),
        text("The original train/validation/test files remain frozen."),
    ], "A window must fit within one book's training split. The test is within-book, not "
       "an unseen-author benchmark. A 128-byte boundary gap guards the maximum context. "
       "Byte cuts need not align with words or Unicode characters; the learning task itself "
       "is defined on bytes. Checksums detect accidental changes.")
    add("Data", "Tokenization: a deliberately visible choice", [
        source("model.py", "encode"),
        source("model.py", "decode"),
        text('“Alice” → [65, 108, 105, 99, 101]. A Unicode character can occupy several bytes.'),
        text("256 IDs, no unknown-word token. Byte perplexity is not GPT-2 BPE perplexity."),
    ], "decode uses replacement characters for invalid or unfinished UTF-8. Generation can "
       "terminate mid-character or mid-word. Keeping bytes makes the Transformer visible "
       "without a tokenizer subsystem, but makes context much shorter in word terms. "
       "For cross-tokenizer comparisons, use normalized raw-byte metrics rather than token PPL.")
    add("Data", "A training example is a shifted window", [
        source("train.py", "batch"),
        text("x = A l i c     y = l i c e. Every training position predicts the next byte."),
        text("The sampled start leaves room for the final target: len(data) − context."),
    ], "starts are sampled from [0, len(data)-context), so y's largest possible index "
       "is len(data)-1. This is why the input and target share a window shifted by one. "
       "The helper receives only a training document, never a concatenation crossing "
       "validation or a book boundary.")
    add("Architecture", "The full computation graph", [
        diagram("Byte IDs to cross-entropy", [
            "Byte IDs\n[B,T]", "Token + position\n[B,T,C]", "Decoder blocks\n[B,T,C]",
            "Final norm\n[B,T,C]", "Tied logits\n[B,T,256]", "Next-byte loss\nscalar"]),
        text("V7: B=32, T=128, C=128, H=4, L=4. The loss trains all positions."),
        text("Generation uses only the final position’s logits, then appends one sampled byte."),
    ], "Residual connections preserve the [batch, time, channel] shape throughout the stack. "
       "Token and position embeddings are added, not concatenated. Logits are unnormalized "
       "scores; cross-entropy combines log-softmax with the correct-target lookup.")
    add("Architecture", "Tensor shapes inside one attention block", [
        table(["Tensor", "General shape", "V7 concrete shape"],
              [["Hidden x", "[B,T,C]", "[32,128,128]"],
               ["Packed QKV", "[B,T,3C]", "[32,128,384]"],
               ["Each of Q,K,V", "[B,H,T,C/H]", "[32,4,128,32]"],
               ["Attention scores", "[B,H,T,T]", "[32,4,128,128]"],
               ["Merged output", "[B,T,C]", "[32,128,128]"]]),
        text("Scores alone contain 2,097,152 floats ≈ 8 MiB per block, before gradients."),
    ], "Head width is C/H=32. Time appears twice in the score tensor: one axis for query "
       "positions and one for keys. Doubling context multiplies score memory by four, "
       "not two. Total training memory also includes activations, gradients and optimizer states.")
    add("Architecture / actual code", "Learned embeddings and positions", [
        source("model.py", "GPT.forward", 0, 6),
        text("Embeddings: token table [256,C], position table [context,C]. Positions start at zero."),
        text("Layer blocks preserve width; positional information is learned, not sinusoidal."),
    ], "Inspect GPT.__init__ for the two nn.Embedding tables. These are absolute learned "
       "positions like GPT-2. During cropped generation, positions reset for each window; "
       "this is a simplification, not a long-context or cached-decoding implementation.")
    add("Architecture / actual code", "Project Q, K and V; split the heads", [
        source("model.py", "CausalAttention.forward", 0, 7),
        text("One linear layer makes 3C features; chunk separates query, key and value roles."),
    ], "Q identifies what each position seeks, K describes positions to match against, and "
       "V provides their content. view splits C into heads and head-width; transpose puts "
       "heads before time. Each head gets an independent attention matrix, not independent tokens.")
    add("Architecture", "The triangular mask is the causality guarantee", [
        dict(kind="mask", size=5),
        text("1 = visible key; 0 = forbidden future key. A token may see itself and all earlier tokens."),
        text("Changing positions 8–15 must leave logits at positions 0–7 unchanged."),
    ], "Rows are queries and columns are keys. Mask the future before softmax with minus "
       "infinity, so forbidden weights become zero. Zeroing scores instead would still "
       "give future keys probability mass. Evaluation disables dropout for the exact-prefix test.")
    add("Architecture / actual code", "Scale, mask, normalize, then mix values", [
        source("model.py", "CausalAttention.forward", 7, 10),
        text("softmax(QKᵀ / √head_width) V. Scaling prevents large dot products from saturating softmax."),
    ], "The scale is 1/sqrt(32) for our 128-wide, four-head models. Before dropout, each "
       "attention row sums to one over visible keys. Training-time attention dropout scales "
       "surviving weights and does not renormalize the row. This is separate from the causal mask.")
    add("Architecture / actual code", "Merge heads and project back to the residual width", [
        source("model.py", "CausalAttention.forward", 10, 12),
        source("model.py", "CausalAttention.__init__", 2, 7),
        text("transpose → contiguous → view merges head outputs. The projection restores the C-wide residual interface."),
    ], "After transposing the attended values, contiguous makes the memory layout compatible "
       "with view. The output projection mixes heads. Its output dropout is a regularization "
       "site, not a change in tensor shape or vocabulary.")
    add("Architecture / actual code", "A block has two pre-normalized residual branches", [
        source("model.py", "Block.forward"),
        source("model.py", "Block.__init__", 5, 11),
        text("Attention communicates across time. The MLP transforms each position’s features."),
    ], "Pre-LN means normalize the branch input, then add its output to the untouched residual "
       "stream. LayerNorm normalizes features within one token, not across time. The MLP is "
       "position-wise: C→4C→C, with tanh-approximate GELU between its linear maps.")
    add("Architecture", "Why LayerNorm and GELU are not optional details", [
        points("LayerNorm(x) = gain·(x−μ)/√(variance+ε) + bias; statistics are across C.",
               "GELU approximately gates x by the probability it exceeds a Gaussian threshold.",
               "The residual path carries existing information and a direct gradient route.",
               "Final LayerNorm precedes the tied vocabulary projection."),
        text("Our nn.LayerNorm default ε is 10⁻⁵; GELU uses approximate='tanh'."),
    ], "Contrast LayerNorm with BatchNorm: its behavior does not depend on another example "
       "in the minibatch. GELU is not ReLU; it retains a smooth negative-side response. "
       "These operations are preserved from the GPT-2 structural pattern.")
    add("Architecture / actual code", "One embedding matrix serves two jobs", [
        source("model.py", "GPT.forward", 6, 12),
        text("F.linear(x, token.weight) uses the same [256,C] parameter matrix used for input lookup."),
    ], "Weight tying reduces parameters and connects token representations with token "
       "prediction. F.linear interprets the embedding matrix as output rows. The loss "
       "flattens [B,T,256] and [B,T] so every target contributes. No softmax is needed "
       "before F.cross_entropy; supplying probabilities there would be incorrect.")
    add("Architecture", "Count the parameters instead of guessing", [
        text("P = 256C + TC + L(12C² + 13C) + 2C"),
        table(["Component", "V7 parameters"],
              [["Token embedding: 256×128", "32,768"],
               ["Position embedding: 128×128", "16,384"],
               ["One decoder block: 12×128² + 13×128", "198,272"],
               ["Four blocks", "793,088"],
               ["Final LayerNorm", "256"],
               ["Total", "842,496"]]),
    ], "Attention contributes 4C²+4C parameters; the MLP contributes 8C²+5C; two "
       "LayerNorms contribute 4C. The final norm adds 2C. Dropout and the causal mask "
       "have no learned parameters. There is no second output vocabulary matrix.")
    add("Architecture / actual code", "Initialization keeps residual branches controlled", [
        source("model.py", "GPT._initialize"),
        source("model.py", "GPT.__init__", 10, 15),
        text("Linear/embedding weights start with σ=0.02; residual projections use σ=0.02/√(2L)."),
    ], "Biases are zero. Scaling residual output projections with depth controls accumulated "
       "residual variance. This is our training-oriented implementation choice, not a "
       "promise to match the original TensorFlow checkpoint initialization exactly.")
    add("Architecture / regularization", "Where V2–V7 apply dropout", [
        table(["Site", "Actual operation", "Training p", "Evaluation"],
              [["Embeddings", "token + position → dropout", "0.1", "Identity"],
               ["Attention weights", "softmax scores → dropout", "0.1", "Identity"],
               ["Attention output", "projection → dropout", "0.1", "Identity"],
               ["MLP output", "second linear → dropout", "0.1", "Identity"]]),
        text("model.eval() disables dropout, but does not disable gradients. no_grad() is a separate control."),
    ], "V1 has p=0, so its legacy function is unchanged. The same saved state dict loads "
       "with a default dropout field when old checkpoints omit that field. The dropout "
       "tests check train/eval behavior, generation determinism and training-mode restoration.")
    add("Training / actual code", "One optimization step", [
        source("train_expanded.py", "train_version", 22, 31),
        text("AdamW LR=0.0003 for new stages; weight decay=0.01; clip gradient norm to 1."),
    ], "The scratch studies used LR=0.001. Expanded stages use a lower fixed LR because "
       "they warm-start learned weights. zero_grad precedes backward; clipping follows "
       "backward and precedes optimizer.step. Nonfinite loss or gradients raise an explicit error.")
    add("Training / evaluation", "Score next bytes without giving away the target", [
        source("train.py", "evaluate", 5, 14),
        text("Inputs end immediately before the scored target. Only logits[:,−1] is compared with that target."),
    ], "Evaluation positions are sampled without replacement using fixed generators. All "
       "versions score identical targets within a comparison but may use different context "
       "lengths. Conditioning on prior held-out bytes is normal autoregressive evaluation, "
       "not training on test targets. eval and no_grad serve different purposes.")
    add("Study 1 / recipes", "Baseline plus five improvement attempts", [
        table(["Run", "Change", "Context", "Width / blocks", "Updates"],
              [[r["name"].split("_")[0].upper(), r["change"], r["context"],
                f"{r['width']} / {r['layers']}", r["steps"]] for r in rounds]),
    ], "Each scratch run restarts seed 42 and its own batch generator. R1 adds training; "
       "R2 adds context; R3 adds width and depth together; R4 adds training; R5 adds a "
       "schedule. These are practical attempts, not five guaranteed successes or an equal-compute ablation.")
    add("Study 1 / all results", "The baseline experiment table", [
        table(["Run", "Parameters", "Train loss", "Val loss", "Val BPB", "Val PPL", "Val accuracy"],
              [[r["name"].split("_")[0].upper(), f"{r['parameters']:,}",
                f"{r['train']['loss']:.3f}", f"{r['validation']['loss']:.3f}",
                f"{r['validation']['bits_per_byte']:.3f}", f"{r['validation']['perplexity']:.3f}",
                percent(r["validation"]["accuracy"])] for r in rounds]),
        text("R2 and R5 regressed. Keep R4, not merely the latest run."),
    ], "The scores use each run's validation-selected checkpoint. Loss is nats per byte; "
       "BPB=loss/ln(2), PPL=exp(loss). Uniform prediction has 8 bits/byte. Larger context "
       "processes more bytes per update, and width/depth changes are bundled.")
    add("Study 1 / all curves", "Learning curves expose diminishing returns", [
        chart("Updates within run →; validation loss in nats ↓",
              [dict(name=r["name"].split("_")[0].upper(),
                    points=[[e["step"], e["validation"]["loss"]] for e in r["curve"]])
               for r in rounds]),
        text("R0/R1 overlap early; R3/R4 overlap through update 600."),
    ], "This chart contains every recorded validation checkpoint from the first study. "
       "Initialization and sampled windows are identical within each matching prefix. "
       "Do not mix this validation loss with the expanded study's weighted score.")
    add("V1 / build recipe", "V1: retain the smallest useful teaching model", [
        table(["Setting", "V1"], [["Blocks / width / heads", "2 / 128 / 4"],
              ["Context / dropout", "64 bytes / 0.0"], ["Scratch updates / best", "1,800 / 1,800"],
              ["Parameters", f"{v1['parameters']:,}"],
              ["Test accuracy / BPB", f"{percent(v1['test']['accuracy'])} / {v1['test']['bits_per_byte']:.3f}"]]),
        text(f"Retraining reproduced the original selected weights exactly: {second['v1_exact_weight_reproduction']}."),
    ], "V1 repeats the selected R4 architecture and recipe. It was retrained separately, "
       "not merely copied under a new name. The exact tensor equality check protects "
       "the original teaching model while optional dropout/depth support was added.")
    add("V2 / build recipe", "V2: context, one more block, and dropout", [
        table(["Change from V1", "V2"], [["Context", "64 → 128 bytes"],
              ["Blocks", "2 → 3"], ["Dropout", "0 → 0.1"], ["Parameters", f"{v2['parameters']:,}"],
              ["Scratch updates / selected", f"{v2['steps']:,} / {v2['best_step']:,}"]]),
        text("Still width 128, four heads, byte tokens, learned positions and tied embeddings."),
    ], "V2 is scratch-trained, not warm-started from V1. Context, depth and regularization "
       "change together. It uses the original single-book training corpus. Adding dropout "
       "was intended to regularize, but this bundled comparison does not isolate its effect.")
    add("V1–V2 / measured comparison", "V2 improves prediction, with more compute", [
        table(["Version", "Val loss", "Full val loss", "Test BPB", "Test PPL", "Test accuracy"],
              [[r["name"].upper(), f"{r['validation']['loss']:.3f}",
                f"{r['full_validation']['loss']:.3f}", f"{r['test']['bits_per_byte']:.3f}",
                f"{r['test']['perplexity']:.3f}", percent(r["test"]["accuracy"])] for r in [v1, v2]]),
        text("V2 validation at 1,800 updates: " +
             f"{next(e for e in v2['curve'] if e['step']==1800)['validation']['loss']:.3f}; "
             "equal updates still are not equal compute."),
    ], "V2 presents twice as many bytes per update and was run for twice the updates. "
       "The initial test was already published, so this comparison reuses a holdout rather "
       "than introducing a new independent test. Checkpoints still use validation only.")
    add("V1–V2 / all curves", "Select a checkpoint, not the final optimizer step", [
        chart("Updates within scratch run →; original validation loss ↓", [
            dict(name=r["name"].upper(),
                 points=[[e["step"], e["validation"]["loss"]] for e in r["curve"]])
            for r in [v1, v2]]),
        text(f"V2: selected update {v2['best_step']:,}; final update {v2['steps']:,} was worse."),
    ], "The selection point is 2,600 of 3,600 executed updates. More training did not "
       "keep improving held-out loss. The original record preserves all checks, not just "
       "the best score. The expanded warm-start chain starts from this selected V2 state.")
    add("Evaluation design", "Three studies use two validation objectives", [
        table(["Study", "Selection target set", "Selection objective"],
              [["R0–R5", "1,024 original bytes", "Original validation loss"],
               ["V1–V2", "Same original bytes", "Original validation loss"],
               ["V3–V7", "Original 1,024 + 5×256 new bytes", "½ original + ½ new-book macro loss"]]),
        text("Compare all seven versions on the common expanded validation/test suites, not on incompatible selection scores."),
    ], "The full original validation scores for V1/V2 were measured after selection. "
       "Expanded common validation includes all five new books even in stages not yet "
       "training on those books. This keeps target offsets and weighting identical across stages.")
    add("Continual data expansion", "Add one work per version", [
        table(["Version", "New work", "Training books", "Available train bytes"],
              [[r["name"].upper(), b["title"], r["books"]+1, f"{r['available_train_bytes']:,}"]
               for r, b in zip(versions, corpus["books"])]),
        text("The original training file is rehearsed; its validation/test bytes are never added to training."),
    ], "V3 adds Alice; V4 Oz; V5 Peter; V6 Willows; V7 Secret Garden. Each stage trains on "
       "the cumulative allowed training documents. All per-document splits are frozen "
       "before training. These volumes are unique available training bytes, not repeated exposures.")
    add("Continual learning", "Checkpoint lineage is not optimizer lineage", [
        diagram("Warm-start chain", [f"{r['name'].upper()}\n{r['best_step']:,} selected"
                                   for r in [v2] + versions]),
        text("Load parent weights → create a fresh AdamW optimizer → train → select by common validation."),
        text("V1 is independent. V3–V7 inherit selected weights, not parent optimizer moments."),
    ], "All stages use seed 42, but they are not independent seeds or scratch runs. "
       "The parent SHA-256 is saved for each version. Executed updates can exceed the "
       "selected checkpoint's update count; inherited learning is based on the selected state.")
    add("Continual learning / actual code", "Warm starts plus controlled depth growth", [
        source("model.py", "grow_depth"),
    ], "This is the entire helper. It only permits adding blocks. Existing state tensors "
       "are copied, then new attention and MLP output projection weights are zeroed. "
       "Their biases already initialize to zero. With dropout disabled, the new residual "
       "branches add zero, preserving the parent's inference function before training.")
    add("Continual learning / correctness", "A new block starts as identity, but can learn", [
        source("test_expanded.py", "ExpandedTests.test_depth_growth_preserves_function_and_new_branch_can_learn", 0, 12),
        text("Zero output weights do not freeze the branch: output projections receive gradients on the first step."),
    ], "Upstream parameters in a zero-output branch initially get no gradient through that "
       "zero projection; after the projection begins learning, their gradients can flow. "
       "The test checks exact inference preservation and nonzero gradients on both output "
       "projection matrices. Parent parameters are not aliased or modified.")
    add("Document-aware training / code", "Choose a document before choosing its window", [
        source("train_expanded.py", "document_batch", 0, 9),
        text("P(original)=0.5; P(each new book)=0.5 / number_of_new_books."),
    ], "Rehearsal deliberately oversamples the small original document. It aims to reduce "
       "forgetting of the original style while sharing the rest of the windows across "
       "new works equally, not proportionally to their byte lengths.")
    add("Document-aware training / code", "No batch row crosses a book boundary", [
        source("train_expanded.py", "document_batch", 9, 13),
        text("Group selected rows by document; call the original shifted-window helper for that document only."),
    ], "x and y are assembled after document-local sampling. Concatenating documents "
       "and sampling arbitrary windows would create artificial transitions and could "
       "cross holdout boundaries. The tests use constant-valued synthetic books to make "
       "any crossing immediately visible.")
    add("Document-aware training / actual exposure", "How much each document is actually sampled", [
        table(["V7 document", "Windows", "Share of V7 windows"],
              [[name, f"{count:,}", f"{count/sum(versions[-1]['document_windows'].values()):.1%}"]
               for name, count in versions[-1]["document_windows"].items()]),
        text("Random mixture counts are recorded, rather than assumed to equal their expected percentages."),
    ], "Every row is a 128-byte context with 128 shifted targets. Windows are sampled with "
       "replacement and overlap within the training partition. Available training bytes "
       "and total presented bytes are different quantities.")
    add("Common validation / actual code", "The same weighted score selects every new version", [
        source("train_expanded.py", "suite"),
        text("loss = 0.5·original_loss + 0.1·(Alice + Oz + Peter + Willows + Garden losses)."),
    ], "combine averages losses and accuracies with explicit weights, then computes "
       "PPL=exp(averaged loss), not average(per-book PPL). The five-book macro prevents "
       "the longest book from dominating. The original gets half the selection weight "
       "by design; it is not an unbiased corpus-size-weighted benchmark.")
    add("Checkpoint selection / actual code", "Save only when held-out loss improves", [
        source("train_expanded.py", "train_version", 10, 15),
        source("train_expanded.py", "train_version", 43, 46),
        text("The initial parent is a valid candidate. Neither test score chooses a checkpoint."),
    ], "Every 200 updates, the same validation suite is measured. The initial parent is "
       "saved at step zero; a stage can honestly fail to improve. In the actual runs every "
       "new version selected a later checkpoint. Test performance is scored after all "
       "five choices are finalized.")
    add("Compute / all executed runs", "Budgets and selected checkpoints are different", [
        table(["Version", "Executed / selected", "Presented bytes", "Training seconds"],
              [[r["name"].upper(), f"{r['steps']:,} / {r['best_step']:,}",
                f"{r.get('presented_bytes',r.get('training_tokens')):,}",
                f"{r['training_seconds']:.1f}"] for r in [v1, v2] + versions]),
        text("Times include periodic validation/checkpoint overhead; exclude final test and sample generation."),
    ], "V2 executes 3,600 but selects 2,600. V3 selects 1,400 of 1,800. V7 selects "
       "1,000 of 2,600. Later versions also inherit earlier selected computation. "
       "Do not describe this as an equal-compute comparison.")
    for result, book in zip(versions, corpus["books"]):
        config = result["config"]
        inherited = result["initial_validation"]["combined"]["loss"]
        selected_loss = result["validation"]["combined"]["loss"]
        add(f"{result['name'].upper()} / exact build", f"{result['name'].upper()}: add {book['name']} and continue learning", [
            table(["Build detail", "Recorded value"],
                  [["Parent checkpoint", result["parent"].upper() + ".pt"],
                   ["Added book", book["title"]],
                   ["Blocks / width / heads / context", f"{config['layers']} / 128 / 4 / 128 bytes"],
                   ["Available training bytes", f"{result['available_train_bytes']:,}"],
                   ["Updates executed / selected", f"{result['steps']:,} / {result['best_step']:,}"],
                   ["Common loss: parent → selected", f"{inherited:.3f} → {selected_loss:.3f}"],
                   ["Original / fresh test accuracy",
                    f"{percent(result['tests']['per_book']['original']['accuracy'])} / "
                    f"{percent(result['tests']['new_books_macro']['accuracy'])}"]]),
        ], f"{result['name'].upper()} loads the recorded parent state and resets optimizer moments. "
           f"The additional work is {book['title']}. Only this and earlier new works are in training. "
           "All five new validation books remain in the common selection suite. "
           + ("V5 introduces the identity-initialized fourth block. " if result["name"] == "v5" else
              "No new block is introduced in this stage. ")
           + "Fixed examples are in the appendix; compare their errors, not just the aggregate score.")
    add("V3–V7 / all learning curves", "Common validation improves across the warm-start chain", [
        chart("Updates within this stage →; common weighted validation loss ↓",
              [dict(name=r["name"].upper(),
                    points=[[e["step"], e["validation"]["combined"]["loss"]] for e in r["curve"]])
               for r in versions]),
        text("All points use the same targets and weights. Stage zero is the inherited checkpoint."),
    ], "Each line includes every recorded point, including step zero. The curves have "
       "different initial states, data coverage and inherited compute. V7's selected update "
       "is 1,000 even though the run continued to 2,600. This is not a test-tuning curve.")
    add("All seven versions / comparable results", "The final accuracy and loss table", [
        table(["Version", "Common val loss", "Original test", "Fresh macro", "Fresh BPB", "Fresh PPL"],
              [[r["name"].upper(), f"{r['validation']['combined']['loss']:.3f}",
                percent(r["tests"]["per_book"]["original"]["accuracy"]),
                percent(r["tests"]["new_books_macro"]["accuracy"]),
                f"{r['tests']['new_books_macro']['bits_per_byte']:.3f}",
                f"{r['tests']['new_books_macro']['perplexity']:.3f}"] for r in all_results]),
        text("Original test: 16,300 targets, reused. New test: 5×2,048 targets, first scored after selection."),
    ], "All rows score identical target sets. The fresh set is new to this study but comes "
       "from held-out passages of the same five books, not unseen authors. The selected "
       "new recommendation is V7 by common validation, not by its test score.")
    headers = ["Version", "Alice", "Oz", "Peter", "Willows", "Garden"]
    for key, label, unit in [
        ("accuracy", "Fresh test accuracy by book", "Percent of next-byte targets predicted correctly."),
        ("loss", "Fresh test loss by book", "Nats per byte; lower is better."),
        ("bits_per_byte", "Fresh test bits per byte by book", "Loss divided by ln(2); lower is better."),
        ("perplexity", "Fresh test byte perplexity by book", "exp(loss), not BPE token perplexity."),
    ]:
        add("All per-book results", label, [
            table(headers, metric_matrix(all_results, key)), text(unit),
        ], "These are all seven versions on the same 2,048 distinct targets in each book. "
           "Equal-book macro metrics give each work equal weight. Report every domain, "
           "including regressions; do not cherry-pick only the added book or best sample.")
    add("Results / tradeoffs", "Aggregate gains do not mean every domain improved", [
        points("V6 improves Willows, but Alice, Oz and Peter accuracy each fall slightly from V5.",
               "V7 improves Garden, while Willows falls from V6.",
               "A rehearsal mixture protects the original book; it also biases the learning objective.",
               "Data, depth and cumulative compute are bundled: no causal attribution from this study."),
    ], "The actual per-book tables make these reversals visible. V6→V7 Willows goes "
       "from about 56.7% to 56.2%, despite improved macro accuracy. A model with broader "
       "training is not automatically superior for every target distribution.")
    add("Results / generalization", "Training loss is not a generalization guarantee", [
        table(["Version", "Original train loss", "Original validation loss", "Fresh test macro loss"],
              [[r["name"].upper(),
                f"{(r['train_per_book']['original'] if 'train_per_book' in r else r['train'])['loss']:.3f}",
                f"{(r['validation']['per_book']['original'] if 'per_book' in r['validation'] else r['validation'])['loss']:.3f}",
                f"{next(a for a in all_results if a['name']==r['name'])['tests']['new_books_macro']['loss']:.3f}"]
               for r in [v1, v2] + versions]),
    ], "Train diagnostics use fixed sampled training targets with dropout disabled, "
       "not the noisy last minibatch loss. The first and expanded studies use different "
       "diagnostic sample sizes, so treat train gaps as qualitative evidence, not a "
       "controlled gap benchmark. A small context and repeated book exposure still permit overfitting.")
    add("Generation / actual code", "Generate by repeatedly predicting the last position", [
        source("model.py", "GPT.generate", 9, 19),
        text("Crop to context; recompute learned positions; sample; append; repeat."),
    ], "The model is set to eval; generation uses no_grad and a local CPU sampling generator. "
       "After generation it restores the previous train/eval mode. No KV cache is used, "
       "so the full cropped window is recomputed for every emitted byte.")
    add("Generation / decoder choices", "Temperature and top-k do not change learned quality", [
        points("Temperature τ scales logits: probabilities = softmax(logits / τ).",
               "Lower τ sharpens probabilities; higher τ adds randomness.",
               "Top-k suppresses all but the k highest-scoring byte logits.",
               "Every recorded sample uses τ=0.8, k=40, 240 generated bytes, fixed seeds."),
        text("Reported loss/accuracy use raw model logits, not temperature or top-k filtering."),
    ], "Decoding can make samples look cleaner while leaving held-out predictive scores "
       "unchanged. Fixed seeds make a controlled qualitative comparison; they do not "
       "make one sample representative of all outputs. Invalid UTF-8 can be produced.")
    add("Generation / actual outcomes", "Improved byte prediction is not coherent writing", [
        dict(kind="sample_grid", items=[sampled(v1, 0, "v1v2"),
                                       sampled(versions[-1], 0, "expanded")]),
        text("Compare spelling, local syntax, repetition and meaning separately."),
    ], "These are unedited outputs for the same prompt, seed and decoding settings. "
       "Both can contain recognizable clauses while failing grammar or story continuity. "
       "A 100% training-lexicon hit rate can coexist with nonsense. The appendix contains "
       "every recorded output so the visible examples are not the only evidence.")
    add("Correctness / evidence", "What the 16 tests and artifact checks establish", [
        points("Future bytes do not change earlier logits; shifted targets stay in the split.",
               "Loss/backpropagation work; a toy sequence can be learned.",
               "Old checkpoints still load; dropout/generation modes are restored.",
               "Document windows cannot cross books; exact-span overlap and checksum changes are detected.",
               "Added depth preserves inference initially and has learnable output branches."),
    ], "Test sources: test_model.py and test_expanded.py. report_expanded.py reloaded all "
       "seven models, reproduced all 35 expanded samples and both metric suites, and "
       "checked parent lineage plus frozen artifacts. These checks establish implementation "
       "correctness and reproducibility, not general intelligence.")
    add("Limits / next experiments", "What the evidence does and does not support", [
        points("Supports: learned local language and better held-out next-byte prediction.",
               "Does not support: a reliable chatbot, factual answers or coherent long stories.",
               "Limits: one seed, short contexts, within-book holdouts, OCR noise, repeated validation.",
               "Next: clean/diverse books, byte-level BPE with comparable BPB, multi-seed ablations, fresh document-level tests."),
    ], "The strided 128-byte audit found no matches in 2,406 checked holdout spans but is "
       "not exhaustive semantic deduplication. The original test was published repeatedly. "
       "Do not tune further on the fresh expanded test and continue calling it fresh. "
       "Future RoPE/cache/fused-attention lessons should distinguish quality from efficiency.")
    add("Live lab / reproduction", "Run the models and verify the evidence", [
        dict(kind="code", text=(
            "python3 -m unittest -v\n"
            "python3 generate.py --version v1 --device cpu\n"
            "python3 generate.py --version v7 --device cpu\n"
            "python3 verify_artifacts.py\n"
            "python3 report_expanded.py\n\n"
            "# Reproduce five warm-started versions (overwrites V3-V7):\n"
            "python3 train_expanded.py --device cpu"
        ), source="README.md · run from /Users/michalfu/mini-gpt-classroom"),
        text("All seven checkpoints are included; no new training is required to use this lecture."),
    ], "The dependency is PyTorch. Presentation export additionally uses python-pptx in "
       "an isolated environment. Defaults preserve the original final.pt; select v7 explicitly "
       "for the expanded model. Existing reports retain code/data provenance and all curves.")
    add("Sources / closing", "Source-linked results, not invented examples", [
        points("OpenAI GPT-2 model: github.com/openai/gpt-2/blob/master/src/model.py",
               "Architecture/code: model.py; training: train.py, train_versions.py, train_expanded.py.",
               "Data provenance: data/provenance.json and data/expanded/manifest.json.",
               "Results: RESULTS.md, VERSIONS.md, EXPANDED_RESULTS.md and their JSON sources.",
               "Appendix follows: all recorded generation samples, unchanged."),
    ], "Original corpus item: throughlookinggl00carruoft. Added items: "
       "alicesadventures00carruoft; wonderfulwizardo00baumiala; peterwendy00barr; "
       "windinwillows00grah; secretgarden00burn. Each is accessible at "
       "https://archive.org/details/ITEM . Model and data hashes are in the lecture manifest. "
       "Use the original short deck for an introduction, this deck for the complete walkthrough.")
    main_count = len(slides)
    for study, results in [("baseline", rounds), ("v1v2", [v1, v2]), ("expanded", versions)]:
        for result in results:
            count = len(result["samples"])
            groups = [list(range(count))] if count == 3 else [list(range(3)), [3, 4], [5, 6]]
            for group_index, indices in enumerate(groups):
                add("Appendix / all recorded samples",
                    f"{result['name'].upper()}: unedited samples {group_index+1}/{len(groups)}", [
                        dict(kind="sample_grid", items=[sampled(result, i, study) for i in indices]),
                        text("240 generated bytes each. Same prompts, seeds and decoding settings within each study."),
                    ], "These outputs are copied exactly from the experiment JSON, including "
                       "misspellings, OCR-like fragments, whitespace and unfinished endings. "
                       "Do not describe training-lexicon hits as grammar or coherence scores. "
                       "Compare the named prompts against the same prompts in other versions.")
    return slides, main_count


def validate_content(slides, first, second, expanded):
    expected = {
        f"{study}:{r['name']}:{index}": item["prompt"] + item["continuation"]
        for study, results in [("baseline", first["rounds"]), ("v1v2", second["versions"]),
                               ("expanded", expanded["versions"])]
        for r in results for index, item in enumerate(r["samples"])
    }
    found = {}
    for slide in slides:
        if not slide["notes"] or not slide["blocks"]:
            raise AssertionError("Every slide requires content and speaker notes.")
        for block in slide["blocks"]:
            if block["kind"] == "code" and "filename" in block:
                original = (ROOT / block["filename"]).read_text().splitlines()
                actual = textwrap.dedent("\n".join(original[block["first_line"]-1:block["last_line"]]))
                if actual != block["text"]:
                    raise AssertionError("Displayed code differs from source.")
            if block["kind"] == "sample_grid":
                for item in block["items"]:
                    if expected[item["sample_id"]] != item["text"]:
                        raise AssertionError("A generated sample was edited.")
                    found[item["sample_id"]] = item["text"]
    if found != expected:
        raise AssertionError("Not every recorded sample is present.")
    return len(expected)


def main():
    first = json.loads((ROOT / "artifacts/results.json").read_text())
    second = json.loads((ROOT / "artifacts/versions/comparison.json").read_text())
    expanded = json.loads((ROOT / "artifacts/versions/expanded_comparison.json").read_text())
    provenance = json.loads((ROOT / "data/provenance.json").read_text())
    corpus = json.loads((ROOT / "data/expanded/manifest.json").read_text())
    for filename, digest in expanded["preserved_hashes"].items():
        if hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() != digest:
            raise AssertionError(f"Preserved original artifact changed: {filename}")
    slides, main_count = make_slides(first, second, expanded, provenance, corpus)
    sample_count = validate_content(slides, first, second, expanded)
    render_pptx(slides, ROOT / "deep-dive.pptx")
    (ROOT / "deep-dive.html").write_text(render_html(slides, TITLE), encoding="utf-8")
    hashes = {filename: hashlib.sha256((ROOT / filename).read_bytes()).hexdigest()
              for filename in INPUTS}
    deck = dict(title=TITLE, slide_count=len(slides), main_slides=main_count,
                appendix_slides=len(slides)-main_count, recorded_samples=sample_count,
                source_hashes=hashes, slides=slides)
    (ROOT / "deep_dive_slides.json").write_text(json.dumps(deck, indent=2) + "\n")
    print(f"Created {len(slides)} slides: {main_count} lecture + {len(slides)-main_count} appendix.")
    print(f"All {sample_count} recorded samples and actual source excerpts verified.")
    print("Created editable deep-dive.pptx and standalone deep-dive.html with speaker notes.")


if __name__ == "__main__":
    main()
