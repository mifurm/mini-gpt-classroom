"""Build a standalone HTML teaching deck and a report from actual experiment JSON."""

from html import escape
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def f(value, digits=3):
    return f"{value:.{digits}f}"


def percent(value):
    return f"{100 * value:.1f}%"


def sample_text(round_result, index=0):
    sample = round_result["samples"][index]
    return sample["prompt"] + sample["continuation"]


def plot_curves(rounds):
    width, height = 900, 280
    left, top, right, bottom = 60, 20, 20, 45
    max_steps = max(r["steps"] for r in rounds)
    low, high = 1.4, 2.7
    colors = ["#697586", "#1984a2", "#e5a000", "#9852cc", "#02785b", "#d34f47"]

    def xy(step, loss):
        return (left + step / max_steps * (width - left - right),
                top + (high - loss) / (high - low) * (height - top - bottom))

    elements = [f'<svg viewBox="0 0 {width} {height}" role="img" '
                'aria-label="Validation cross-entropy learning curves">'
                '<rect width="100%" height="100%" fill="white"/>']
    for value in [1.5, 2.0, 2.5]:
        _, y = xy(0, value)
        elements.append(f'<path d="M{left} {y} H{width-right}" stroke="#ddd"/>'
                        f'<text x="5" y="{y+5}" font-size="16">{value:.1f}</text>')
    for step in [0, 600, 1200, 1800]:
        x, _ = xy(step, low)
        elements.append(f'<text x="{x-12}" y="{height-20}" font-size="16">{step}</text>')
    for result, color in zip(rounds, colors):
        points = " ".join(f"{x:.1f},{y:.1f}"
                          for x, y in (xy(v["step"], v["validation"]["loss"])
                                       for v in result["curve"]))
        elements.append(f'<polyline points="{points}" fill="none" stroke="{color}" '
                        'stroke-width="3"/>')
        for event in result["curve"]:
            x, y = xy(event["step"], event["validation"]["loss"])
            elements.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
    elements.append("</svg>")
    legend = " ".join(f'<span style="color:{color}">{escape(r["name"].split("_")[0])}</span>'
                      for r, color in zip(rounds, colors))
    return "".join(elements) + f'<p class="small">Updates → &nbsp; | &nbsp; loss in nats ↓ &nbsp; {legend}</p>'


def make_report(report, provenance):
    rounds = report["rounds"]
    winner = next(r for r in rounds if r["name"] == report["selected"])
    baseline = rounds[0]
    reduction = 1 - winner["validation"]["loss"] / baseline["validation"]["loss"]
    lines = [
        "# Actual training results",
        "",
        f"**Selected: {winner['name']}**, {winner['parameters']:,} parameters, "
        f"{winner['config']['layers']} blocks, {winner['config']['width']}-wide embeddings, "
        f"{winner['config']['context']}-byte context. Baseline-to-best validation "
        f"cross-entropy decreased {percent(reduction)}.",
        "",
        "All six experiments were actually trained from scratch. The following table "
        "uses each run's best validation checkpoint; `best step` can differ from total updates.",
        "",
        "| Round | Change | Updates / best step | Parameters | Train loss | Val loss | Val bits/byte | Val byte PPL | Val top-1 | Training seconds |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in rounds:
        val = result["validation"]
        lines.append(
            f"| {result['name']} | {result['change']} | {result['steps']} / "
            f"{result['best_step']} | {result['parameters']:,} | "
            f"{f(result['train']['loss'])} | {f(val['loss'])} | "
            f"{f(val['bits_per_byte'])} | {f(val['perplexity'])} | "
            f"{percent(val['accuracy'])} | {f(result['training_seconds'], 1)} |"
        )
    lines += [
        "",
        "## What improved, and what did not",
        "",
        "- More training helped the small model (R1); increasing context alone did not (R2).",
        "- More capacity helped (R3), and training that model longer helped again (R4).",
        "- Warmup/cosine decay (R5) was slightly worse than R4 at this budget. "
        "Retain the validation-selected R4, not simply the last run.",
        "- This satisfies baseline plus five measured improvement **attempts**, "
        "not five guaranteed or monotonically successful gains.",
        "",
        "## Final held-out test",
        "",
        "The test file was first loaded after all rounds and validation-based selection. "
        "Every test target after the first 128 bytes was scored once. Test scores did "
        "not determine the selected checkpoint or any training settings.",
        "",
        "| Model | Targets | Loss (nats/byte) | Bits/byte | Byte perplexity | Top-1 next-byte accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, metrics in [("Selected GPT", report["final_test"]),
                           ("Train-only smoothed unigram", report["unigram_test"])]:
        lines.append(
            f"| {label} | {metrics['targets']:,} | {f(metrics['loss'])} | "
            f"{f(metrics['bits_per_byte'])} | {f(metrics['perplexity'])} | "
            f"{percent(metrics['accuracy'])} |"
        )
    lines += [
        "",
        "Unigram probabilities use training byte counts with add-one smoothing. "
        "A uniformly random predictor has loss ln(256) = 5.545 and 8 bits/byte. "
        "Perplexity is exp(loss), bits/byte is loss/ln(2). These are **not** GPT-2 BPE metrics.",
        "",
        "## Data, compute, and reproducibility",
        "",
        f"- Corpus: [{provenance['title']}]({provenance['text_url']}), "
        f"Archive item `{provenance['item']}`, edition `{provenance['publication_date']}`, "
        f"status `{provenance['copyright_status']}` (US).",
        f"- Cleaned corpus: {provenance['clean_bytes']:,} bytes. "
        f"Splits: {json.dumps(provenance['split_bytes'])}; 128-byte boundary gaps.",
        f"- Cleaned SHA-256: `{provenance['clean_sha256']}`.",
        "- Every run: seed 42, batch 32, AdamW, LR 0.001, weight decay 0.01, "
        "gradient clipping at norm 1. R5 warms up over 100 updates and decays to 0.0001.",
        "- Validation: 1,024 fixed distinct next-byte target offsets, seed 2026; "
        "each model conditions on its configured context. The scored target set "
        "is identical, not the context length or compute budget.",
        "- Checkpoint selection every 200 updates; training diagnostics use 512 fixed "
        "training targets. Curves and per-round JSON are in `artifacts/`.",
        f"- Environment: `{json.dumps(report['environment'])}`.",
        "- Measured benchmark (40 steps, 2-block/128-width/64-context, batch 32): "
        "CPU 0.0141 seconds/step; Apple MPS 0.0645 seconds/step. CPU was chosen "
        "for the recorded runs. Training times exclude sample generation and final test.",
        "",
        "| Round | Training bytes presented (including repeated windows) | Approx. corpus passes |",
        "|---|---:|---:|",
    ]
    for result in rounds:
        lines.append(
            f"| {result['name']} | {result['training_tokens']:,} | "
            f"{result['training_tokens']/provenance['split_bytes']['train']:.1f} |"
        )
    lines += [
        "",
        "## Uncurated fixed-seed samples from every round",
        "",
        "All prompts and 240 generated-byte budgets are identical across rounds. "
        "Temperature 0.8, top-k 40, seeds 123/124/125. These are recorded model outputs, "
        "not edited prose. Samples can terminate mid-word.",
        "",
        "The lexicon hit rate counts generated ASCII word runs that appear in training. "
        "It is a rough spelling proxy, rewards short/common words, and can increase "
        "with memorization. Repeated word-trigram fraction is a crude repetition flag. "
        "Neither measures grammar, factuality, or narrative coherence.",
    ]
    for result in rounds:
        lines += ["", f"### {result['name']}"]
        for index, sample in enumerate(result["samples"]):
            lines += [
                "",
                f"Prompt {index+1}; seed {sample['seed']}; lexicon hit "
                f"{percent(sample['lexicon_hit_rate'])}; repeated trigram fraction "
                f"{percent(sample['repeated_word_trigram_fraction'])}; "
                f"UTF-8 replacement characters {sample['utf8_replacements']}.",
                "",
                "```text", sample_text(result, index), "```",
            ]
    lines += [
        "",
        "## Is this good enough for students?",
        "",
        "**Yes, as a transparent language-model demonstration; no, as a general-purpose "
        "text generator.** The selected two-block model beats a train-only unigram "
        "predictor on completely held-out target bytes and visibly learns words, "
        "dialogue punctuation, names, and short English-like clauses.",
        "",
        "The generated text still misuses words, misspells them, repeats fragments, and "
        "loses narrative meaning. For example, the uncurated R4 sample starts "
        "\"The little for a little path rough the wood, of course\" but soon becomes "
        "ungrammatical. Teach this failure rather than presenting the output as coherent.",
        "",
        "The train/validation gap shows overfitting risk. Only one OCR book, one seed, "
        "and short context were used. Validation is reused for checkpoint/model selection; "
        "the reported validation scores are not independent confirmations. Contiguous "
        "splits prevent overlapping windows, not repeated phrases within the same book. "
        "No pretrained GPT-2 weights, web crawl, instruction tuning, or assistant alignment "
        "are included.",
        "",
        "## Sources",
        "",
        "- https://github.com/openai/gpt-2/blob/master/src/model.py",
        "- https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf",
        f"- https://archive.org/details/{provenance['item']}",
        f"- {provenance['metadata_url']}",
    ]
    return "\n".join(lines) + "\n"


def make_slides(report, provenance):
    rounds = report["rounds"]
    winner = next(r for r in rounds if r["name"] == report["selected"])
    test = report["final_test"]
    slides = []

    def add(title, body, notes):
        slides.append(dict(title=title, body=body, notes=notes))

    def code(text):
        return f"<pre><code>{escape(text)}</code></pre>"

    def sample(result, index=0):
        return f'<blockquote class="sample">{escape(sample_text(result, index))}</blockquote>'

    add("Build a real, tiny GPT",
        '<p class="lead">From Internet Archive text to next-byte prediction.</p>'
        f'<p>{winner["parameters"]:,} parameters · two decoder blocks · trained from scratch</p>'
        '<p>Baseline + five measured improvement rounds. No pretrained weights.</p>',
        "Two 90-minute labs plus precomputed experiments. Students should know Python, "
        "matrices, gradients, and softmax. This is a faithful small GPT-2-style architecture, "
        "not the pretrained GPT-2 model.")
    add("The whole plan",
        "<ol><li>Freeze public-domain data and held-out passages.</li>"
        "<li>Implement embeddings, causal attention, residual blocks.</li>"
        "<li>Train with next-byte cross-entropy.</li>"
        "<li>Compare five changes using the same held-out targets.</li>"
        "<li>Generate text, inspect failures, keep the best checkpoint.</li></ol>",
        "Start with PLAN.md. Ask students what evidence would establish that the model "
        "learns instead of just memorizing. We will show both held-out metrics and samples.")
    add("Real Internet Archive data",
        f'<p><strong>{escape(provenance["title"])}</strong> — Lewis Carroll</p>'
        f'<p>Scan edition: {escape(provenance["publication_date"])}; Archive status: '
        '<code>NOT_IN_COPYRIGHT</code> (US).</p>'
        f'<p>{provenance["clean_bytes"]:,} cleaned UTF-8 bytes. Extract story; remove '
        'page numbers/headers; preserve paragraphs and remaining OCR mistakes.</p>'
        f'<p class="small"><a href="{escape(provenance["text_url"])}">Direct OCR text</a>'
        ' · checksums and source details in data/provenance.json</p>',
        "This is the Internet Archive digitized-book collection, not a Wayback crawl. "
        "Use only the original public-domain story, not modern annotations. Check local "
        "rights outside the US. OCR artifacts are a real data-quality issue.")
    add("Bytes and honest splits",
        code('ids = list("Alice".encode("utf-8"))\n# [65, 108, 105, 99, 101]\nx = ids[:-1]  # A l i c\ny = ids[1:]   # l i c e') +
        f'<p>Train {provenance["split_bytes"]["train"]:,} · validation '
        f'{provenance["split_bytes"]["validation"]:,} · test '
        f'{provenance["split_bytes"]["test"]:,} bytes.</p>'
        '<p>Contiguous 80/10/10 passages; discard 128 bytes at both boundaries. '
        'Never let a training window cross a split.</p>',
        "Bytes avoid a tokenizer training subsystem and unknown tokens. They are not "
        "GPT-2 BPE: byte context and byte perplexity are not word/BPE context or perplexity. "
        "Unicode characters can span multiple bytes.")
    add("From tokens to logits",
        code("[B,T] byte IDs\n    ↓ token embedding + learned positions\n[B,T,C] hidden states\n    ↓ decoder blocks × L\n[B,T,C] final LayerNorm\n    ↓ tied token-embedding projection\n[B,T,256] next-byte logits") +
        '<p>Final model: B = 32, T = 64, C = 128, L = 2, heads = 4.</p>',
        "The output at position t predicts the byte after t. F.linear uses the same "
        "token embedding matrix; no independent output matrix is learned. "
        "Inspect model.py together and track tensor dimensions.")
    add("Causal multi-head attention",
        code("Q, K, V = linear(x).chunk(3, dim=-1)\nscores = Q @ K.transpose(-2, -1) / sqrt(head_width)\nscores[future_positions] = -infinity\nweights = softmax(scores)\noutput = project(weights @ V)") +
        '<p>Each head sees only its own position and the past. '
        'Four heads each use 32 features in the final model.</p>',
        "Show the lower-triangular mask. Explain scaling by square root of head width. "
        "Changing future input bytes must leave all earlier logits exactly unchanged; "
        "our test checks this. Without this property training can cheat.")
    add("A GPT-2-style block",
        code("x = x + attention(layer_norm_1(x))\nx = x + mlp(layer_norm_2(x))\n\nmlp = Linear(C, 4*C) → GELU → Linear(4*C, C)") +
        '<p>Pre-normalization, residual paths, bias-bearing projections, '
        'tanh-approximate GELU. Final LayerNorm follows the blocks.</p>',
        "These are the structural choices in OpenAI's open-source src/model.py. "
        "nn.Linear expresses the reference's conv1d projection. Residual projection "
        "initialization is scaled by depth. We omit KV caching and large-scale "
        "training engineering, not the Transformer mechanism.")
    add("Learning: one visible training step",
        code("x, y = sample_shifted_windows(train_bytes)\noptimizer.zero_grad(set_to_none=True)\nlogits, loss = model(x, y)\nloss.backward()\nclip_grad_norm_(model.parameters(), 1.0)\noptimizer.step()") +
        '<p>Cross-entropy = −log probability of the correct next byte. '
        'AdamW: LR 0.001, decay 0.01.</p>',
        "All positions in each training window contribute to loss. Explain gradient "
        "clipping and batch 32. Contrast training with held-out evaluation: we score "
        "only each selected target's final logit, using its preceding context.")
    add("Five improvement attempts",
        '<table><thead><tr><th>Run</th><th>Change from previous run</th><th>Updates</th>'
        '</tr></thead><tbody>' +
        "".join(f'<tr><td>{r["name"].split("_")[0]}</td><td>{escape(r["change"])}</td>'
                f'<td>{r["steps"]:,}</td></tr>' for r in rounds) +
        '</tbody></table><p class="small">Fresh initialization, same seed 42. '
        'Unequal compute; practical experiments, not controlled causal proof.</p>',
        "R0 is the baseline. R1/R4 increase training; R2 increases context; R3 changes "
        "width and depth together; R5 changes the learning-rate schedule. "
        "We do not claim every attempted improvement will succeed.")
    rows = "".join(
        f'<tr{" class=selected" if r["name"] == winner["name"] else ""}>'
        f'<td>{r["name"].split("_")[0]}</td><td>{r["parameters"]:,}</td>'
        f'<td>{f(r["validation"]["loss"])}</td>'
        f'<td>{f(r["validation"]["bits_per_byte"])}</td>'
        f'<td>{percent(r["validation"]["accuracy"])}</td></tr>'
        for r in rounds)
    add("Measured validation quality",
        '<table><thead><tr><th>Run</th><th>Parameters</th><th>Loss ↓</th>'
        '<th>Bits/byte ↓</th><th>Accuracy ↑</th></tr></thead>'
        f'<tbody>{rows}</tbody></table>'
        '<p>R2 and R5 regressed. Keep R4: best validation loss, not the newest model.</p>',
        "All runs score the same 1,024 distinct validation targets, with their respective "
        "context lengths. Checkpoint selection occurs every 200 updates. Validation "
        "reuse introduces selection bias; the final test is reserved until all choices "
        "are made. See RESULTS.md for train losses, compute and all samples.")
    add("Learning curves, not just a final number", plot_curves(rounds),
        "Curves show validation loss at checkpoints. R0/R1 overlap early because their "
        "initialization and batches are identical; R3/R4 do too. More training has "
        "diminishing returns. Lower training loss alone does not prove generalization.")
    add("Actual samples: baseline → more capacity",
        '<div class="columns"><div><h3>R0 · baseline</h3>' + sample(rounds[0]) +
        '</div><div><h3>R3 · two larger blocks</h3>' + sample(rounds[3]) +
        '</div></div><p class="small">Same prompt, seed 123, temperature 0.8, top-k 40; '
        '240 generated bytes. No editing.</p>',
        "Have students identify what improved: word boundaries, common words, quoted "
        "speech and Alice's name. R3 is still substantially ungrammatical. "
        "The samples are uncurated and always taken from the same prompt/seed.")
    add("Actual sample: the selected model",
        sample(winner, 2) +
        '<p>Learned words and short clauses; still misspelled, repetitive, '
        'and not a coherent story.</p><p class="small">Every sample from every run '
        'is included in RESULTS.md, not just these examples.</p>',
        "Discuss the odd opening 'The little for a little path' and the later errors. "
        "This is sufficient to demonstrate learning, not sufficient for a writing "
        "assistant. Word-lexicon hit rates are rough spelling proxies, not a coherence score.")
    add("A final test, opened only after selection",
        f'<p class="lead">{f(test["bits_per_byte"])} bits/byte · '
        f'{percent(test["accuracy"])} next-byte accuracy</p>'
        f'<p>{test["targets"]:,} distinct target bytes; byte perplexity '
        f'{f(test["perplexity"])}.</p>'
        f'<p>Training-only unigram baseline: {f(report["unigram_test"]["bits_per_byte"])} '
        f'bits/byte · {percent(report["unigram_test"]["accuracy"])} accuracy.</p>'
        '<p class="small">Not comparable to GPT-2 BPE perplexity. '
        'This is next-byte prediction, not answer correctness.</p>',
        "The test contains only held-out passages. Score every byte after a 128-byte "
        "prefix, once, with the selected model's 64-byte context. Unigram baseline "
        "is fit using training counts only. Do not use test to choose later models.")
    add("Limits students should understand",
        '<ul><li>One OCR book; one seed; short, 64-byte context.</li>'
        '<li>Training/validation gap and memorization risk.</li>'
        '<li>Repeated validation selection; unequal training budgets.</li>'
        '<li>No instruction following, reliable facts, or long-story reasoning.</li>'
        '<li>Byte tokens replace BPE; no pretrained weights or KV cache.</li></ul>',
        "This is honest experimental evidence, not a claim of production readiness. "
        "Contiguous splits avoid window leakage but cannot remove recurring names "
        "and phrases within one book. More parameters and more training are not "
        "automatically better; use validation and preserve an untouched test.")
    add("Live classroom demonstration",
        code("python3 -m unittest -v\npython3 generate.py --device cpu --prompt 'Alice said, \"'\npython3 train.py --device cpu --steps 600\npython3 generate.py --device cpu --checkpoint artifacts/classroom.pt") +
        '<p>Exercises: print attention shapes; verify the causal mask; '
        'change temperature; compare train and validation loss.</p>'
        '<p class="small">README.md: setup · PLAN.md: lesson sequence · '
        'RESULTS.md: evidence · artifacts/final.pt: selected checkpoint</p>',
        "Use the included final checkpoint for an immediate demo. The standalone run "
        "writes classroom.pt, not final.pt. Explain temperature/top-k without claiming "
        "they change learned probabilities. For larger machines choose another device. "
        "CPU was faster than MPS for this particular tiny benchmark.")
    add("Sources and reproducibility",
        '<p><a href="https://github.com/openai/gpt-2/blob/master/src/model.py">'
        'OpenAI GPT-2: original model implementation</a></p>'
        '<p><a href="https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf">'
        'GPT-2 paper: Language Models are Unsupervised Multitask Learners</a></p>'
        f'<p><a href="https://archive.org/details/{provenance["item"]}">'
        'Internet Archive: corpus scan and rights metadata</a></p>'
        '<p class="small">Frozen data hashes, every checkpoint, learning curves, '
        'configurations, seeds, and metrics are included locally.</p>',
        "Reproduce with python3 train.py --experiments --device cpu, then "
        "python3 build_deck.py. Exact numbers can vary with PyTorch/device. "
        "The slides and report are generated from the recorded results, not invented.")
    sections = []
    for index, slide in enumerate(slides):
        sections.append(
            f'<section class="slide{" active" if index == 0 else ""}" id="slide-{index+1}" '
            f'aria-label="Slide {index+1}: {escape(slide["title"], quote=True)}">'
            f'<div class="eyebrow">Tiny GPT · teaching lab</div><h1>{escape(slide["title"])}</h1>'
            f'{slide["body"]}<aside class="notes"><strong>Speaker notes:</strong> '
            f'{escape(slide["notes"])}</aside><footer>{index+1} / {len(slides)}</footer></section>'
        )
    css = """
*{box-sizing:border-box}body{margin:0;background:#142238;color:#142238;
font-family:system-ui,-apple-system,sans-serif}.slide{display:none;position:relative;
width:min(1180px,100%);min-height:680px;margin:30px auto;background:#fffdf8;
padding:45px 65px 65px;border-top:9px solid #02785b}.slide.active{display:block}
h1{font-size:39px;line-height:1.15;margin:16px 0 30px}h3{font-size:21px}
p,li{font-size:23px;line-height:1.5}li{margin-bottom:13px}.lead{font-size:32px;
font-weight:650;color:#02785b}.eyebrow{text-transform:uppercase;letter-spacing:2px;
font-size:13px;font-weight:700;color:#02785b}.small{font-size:16px}
pre{background:#edf1f4;padding:22px;overflow-x:auto;font-size:20px;line-height:1.55;
border-left:4px solid #02785b;white-space:pre-wrap}code{font-family:ui-monospace,monospace}
table{width:100%;border-collapse:collapse;font-size:19px}th,td{padding:12px 9px;
border-bottom:1px solid #d7dfdd;text-align:left}th{background:#edf1f4}
tr.selected{background:#d9f2e6;font-weight:700}.sample{white-space:pre-wrap;
font-size:21px;line-height:1.55;margin:12px 0;padding:20px;background:#edf1f4;
border-left:4px solid #02785b}.columns{display:grid;grid-template-columns:1fr 1fr;
gap:25px}.columns .sample{font-size:18px}footer{position:absolute;right:35px;
bottom:22px;color:#697586;font-size:15px}.notes{display:none;font-size:16px;
line-height:1.5;margin-top:22px;border-top:1px dashed #697586;padding-top:14px}
body.show-notes .notes{display:block}nav{display:flex;justify-content:center;
align-items:center;gap:12px;padding:0 15px 25px;color:white;font-size:14px}
button{padding:10px 15px;border:0;border-radius:4px;cursor:pointer;background:#e1ede7}
a{color:#02785b}svg{width:100%;height:auto}
@media(max-width:750px){.slide{padding:28px 23px 65px;margin:0;min-height:100vh}
h1{font-size:29px}p,li{font-size:19px}pre,table{font-size:15px}.columns{grid-template-columns:1fr}}
@media print{@page{size:landscape;margin:0}body{background:white}
.slide{display:block!important;margin:0;width:100%;height:100vh;min-height:0;
break-after:page;padding:28px 45px 45px}h1{font-size:29px;margin-bottom:18px}
p,li{font-size:18px}pre{font-size:15px;padding:15px}th,td{padding:8px}
table{font-size:15px}.sample{font-size:17px}.columns .sample{font-size:15px}
.notes,nav{display:none!important}}
"""
    javascript = """
const slides=[...document.querySelectorAll('.slide')];let index=0;
function show(i){index=Math.max(0,Math.min(slides.length-1,i));
slides.forEach((s,n)=>s.classList.toggle('active',n===index));
document.getElementById('counter').textContent=`${index+1} / ${slides.length}`;
history.replaceState(null,'',`#slide-${index+1}`);}
document.getElementById('prev').onclick=()=>show(index-1);
document.getElementById('next').onclick=()=>show(index+1);
document.getElementById('notes').onclick=()=>document.body.classList.toggle('show-notes');
document.addEventListener('keydown',e=>{
if(['ArrowRight','PageDown',' '].includes(e.key)){e.preventDefault();show(index+1);}
if(['ArrowLeft','PageUp'].includes(e.key)){e.preventDefault();show(index-1);}
if(e.key==='Home')show(0);if(e.key==='End')show(slides.length-1);
if(e.key.toLowerCase()==='n')document.body.classList.toggle('show-notes');});
const match=location.hash.match(/^#slide-(\\d+)$/);show(match?Number(match[1])-1:0);
"""
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>Build a real, tiny GPT — teaching deck</title><style>' + css +
        '</style></head><body>' + "".join(sections) +
        '<nav aria-label="Slide navigation"><button id="prev">← Previous</button>'
        '<span id="counter"></span><button id="next">Next →</button>'
        '<button id="notes">Notes (N)</button><span>Arrow keys · print to PDF</span></nav>'
        '<script>' + javascript + '</script></body></html>\n'
    )


def main():
    report = json.loads((ROOT / "artifacts" / "results.json").read_text())
    provenance = json.loads((ROOT / "data" / "provenance.json").read_text())
    (ROOT / "RESULTS.md").write_text(make_report(report, provenance), encoding="utf-8")
    (ROOT / "slides.html").write_text(make_slides(report, provenance), encoding="utf-8")
    print("Created RESULTS.md and slides.html from six recorded experiments.")


if __name__ == "__main__":
    main()
