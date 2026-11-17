"""Verify V3-V7, preserve V1/V2, and publish the expanded-data comparison."""

import json
import math

import torch

from data import ROOT, load_split
from expanded_data import BOOKS, DIRECTORY as DATA_DIRECTORY, checksum, load_book, overlap_audit
from model import decode, encode, load_model
from train import eval_positions
from train_expanded import DIRECTORY, RECIPES, preserved_hashes, suite, tensor


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def verify(report, manifest):
    torch.set_num_threads(4)
    require(report["preserved_hashes"] == preserved_hashes(), "Original artifacts changed.")
    require(checksum((DATA_DIRECTORY / "manifest.json").read_bytes()) ==
            report["expanded_manifest_sha256"], "Expanded corpus manifest changed.")
    require([r["name"] for r in report["versions"]] == [s["name"] for s in RECIPES],
            "All five new versions must be complete.")
    training = {"original": load_split("train")}
    heldout = {f"original/{split}": load_split(split) for split in ["validation", "test"]}
    tests = {"original": tensor(load_split("test"))}
    validation = {"original": tensor(load_split("validation"))}
    for book in manifest["books"]:
        require(book["copyright_status"] == "NOT_IN_COPYRIGHT", "Unverified source rights.")
        name = book["name"]
        require(checksum((DATA_DIRECTORY / f"{name}_source.txt").read_bytes()) ==
                book["raw_sha256"], f"{name} raw source changed.")
        clean = (DATA_DIRECTORY / f"{name}_clean.txt").read_bytes()
        require(checksum(clean) == book["clean_sha256"], f"{name} clean story changed.")
        for split, (start, end) in book["split_spans"].items():
            content = load_book(name, split)
            require(content == clean[start:end], f"{name} split does not match its recorded span.")
            if split == "train":
                training[name] = content
            else:
                heldout[f"{name}/{split}"] = content
        tests[name] = tensor(load_book(name, "test"))
        validation[name] = tensor(load_book(name, "validation"))
    require(not overlap_audit(training, heldout)["exact_matches"], "Holdout overlap found.")
    positions = {
        name: eval_positions(content, count=len(content) - 128 if name == "original" else 2048,
                             seed=2027 if name == "original" else 2028)
        for name, content in tests.items()
    }
    val_positions = {name: eval_positions(content, count=1024 if name == "original" else 256)
                     for name, content in validation.items()}
    selected = min(report["versions"], key=lambda r: r["validation"]["combined"]["loss"])
    require(selected["name"] == report["selected_by_common_validation"],
            "Recommended model was not selected by common validation.")
    all_results = report["baselines"] + report["versions"]
    for result in all_results:
        model = load_model(DIRECTORY / f"{result['name']}.pt")
        require(sum(p.numel() for p in model.parameters()) == result["parameters"],
                "Parameter count mismatch.")
        if result["name"] not in {"v1", "v2"}:
            spec = next(s for s in RECIPES if s["name"] == result["name"])
            require(result["curve"][-1]["step"] == spec["steps"], "Incomplete training run.")
            best = min(result["curve"], key=lambda event: event["validation"]["combined"]["loss"])
            require(best["step"] == result["best_step"], "Wrong best checkpoint.")
            parent_path = DIRECTORY / f"{result['parent']}.pt"
            require(checksum(parent_path.read_bytes()) == result["parent_sha256"],
                    "Warm-start parent differs from recorded lineage.")
            require(sum(result["document_windows"].values()) == result["steps"] * 32,
                    "Document exposure counts do not cover the training run.")
            require(result["available_train_bytes"] ==
                    sum(len(training[name]) for name in result["training_documents"]),
                    "Training dataset size mismatch.")
            require(len(result["samples"]) == 7, "Missing fixed-prompt samples.")
            for sample in result["samples"]:
                prompt = torch.tensor([encode(sample["prompt"])], dtype=torch.long)
                generated = model.generate(prompt, count=240, seed=sample["seed"],
                                           temperature=sample["temperature"], top_k=sample["top_k"])
                continuation = decode(generated[0].tolist()[prompt.shape[1]:])
                require(continuation == sample["continuation"], "Sample does not match checkpoint.")
        actual = suite(model, tests, positions, "cpu")
        actual_validation = suite(model, validation, val_positions, "cpu")
        for section, expected in [(actual, result["tests"]),
                                  (actual_validation, result["validation"])]:
            for name, metrics in section["per_book"].items():
                for key, value in metrics.items():
                    require(math.isclose(value, expected["per_book"][name][key],
                                         rel_tol=1e-6, abs_tol=1e-6),
                            f"{result['name']} metric mismatch: {name}/{key}")
        print(f"Verified {result['name']}: checkpoint, validation, and both test suites.", flush=True)


def report_markdown(report, manifest):
    versions = report["versions"]
    selected = next(r for r in versions if r["name"] == report["selected_by_common_validation"])
    baselines = report["baselines"]
    old_size = manifest["original_train_bytes"]
    total = manifest["total_train_bytes"]
    lines = [
        "# V3-V7: five more models, a larger Archive corpus",
        "",
        f"**Five additional trained checkpoints are delivered.** Training data expands "
        f"from {old_size:,} to {total:,} available bytes ({total/old_size:.1f}x) across "
        "six original public-domain works. V1/V2 and the previous data, reports and deck "
        "remain unchanged. All versions remain selectable with `generate.py --version`.",
        "",
        f"**Recommended new version: {selected['name']}**, chosen by the same common "
        "validation score for every new version, not by either test score.",
        "",
        "[PLAN_V3_V7.md](PLAN_V3_V7.md) records the data/training/evaluation plan. "
        "`data/expanded/manifest.json` contains extraction boundaries, rights evidence, "
        "raw/clean/split hashes, byte spans, and the exact-span overlap audit.",
        "",
        "## Directly comparable measured accuracy",
        "",
        "| Version | Training books | Parameters | Common validation loss | Reused original test accuracy | Fresh five-book test macro accuracy | Fresh macro bits/byte |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in baselines + versions:
        tests = result["tests"]
        lines.append(
            f"| {result['name']} | {1 if result['name'] in ['v1','v2'] else result['books']+1} | "
            f"{result['parameters']:,} | {result['validation']['combined']['loss']:.3f} | "
            f"{tests['per_book']['original']['accuracy']:.1%} | "
            f"{tests['new_books_macro']['accuracy']:.1%} | "
            f"{tests['new_books_macro']['bits_per_byte']:.3f} |"
        )
    v2 = next(r for r in baselines if r["name"] == "v2")
    old_gain = selected["tests"]["per_book"]["original"]["accuracy"] - v2["tests"]["per_book"]["original"]["accuracy"]
    new_gain = selected["tests"]["new_books_macro"]["accuracy"] - v2["tests"]["new_books_macro"]["accuracy"]
    lines += [
        "",
        f"Against V2, validation-selected {selected['name']} changes original-test accuracy "
        f"by {100*old_gain:+.2f} percentage points and fresh expanded-test macro accuracy "
        f"by {100*new_gain:+.2f} points. **Accuracy means next UTF-8 byte prediction**, "
        "not factual correctness, grammar, or instruction following.",
        "",
        "These scores use identical test targets for all seven models. The original "
        "16,300-byte target set is a **reused holdout**. The new test contains 2,048 "
        "distinct targets in each of the five new books (10,240 total); its performance "
        "was first measured only after all five checkpoint selections were final.",
        "",
        "## Fresh test accuracy by book",
        "",
        "| Version | Alice | Oz | Peter and Wendy | Wind in the Willows | Secret Garden |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for result in baselines + versions:
        accuracies = [f"{result['tests']['per_book'][book['name']]['accuracy']:.1%}" for book in BOOKS]
        lines.append(f"| {result['name']} | " + " | ".join(accuracies) + " |")
    lines += [
        "",
        f"Train-only unigram using the final expanded training corpus: fresh-book macro "
        f"accuracy {report['unigram_new_macro']['accuracy']:.1%}, "
        f"{report['unigram_new_macro']['bits_per_byte']:.3f} bits/byte.",
        "",
        "## What each version changes",
        "",
        "| Version | Parent | Newly added work | Blocks | New updates / best step | Available training bytes | New bytes presented | Training seconds |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for result, book in zip(versions, BOOKS):
        lines.append(
            f"| {result['name']} | {result['parent']} | {book['name']} | {result['layers']} | "
            f"{result['steps']:,} / {result['best_step']:,} | "
            f"{result['available_train_bytes']:,} | {result['presented_bytes']:,} | "
            f"{result['training_seconds']:.1f} |"
        )
    lines += [
        "",
        "**Lineage:** V2 → V3 → V4 → V5 → V6 → V7. These are warm-started models, "
        "not independent scratch runs. Each inherits its parent's best checkpoint, "
        "not its parent's final optimizer state. Training exposure above counts each "
        "new run only; inherited computation is additional.",
        "",
        "Every new model: width 128, four heads, context 128 bytes, dropout 0.1. "
        "V5 adds a fourth residual block initialized to preserve the parent inference "
        "function. The other upgrades focus on data coverage and continued learning. "
        "They retain learned positions, pre-LayerNorm, explicit causal attention, "
        "4x GELU MLPs, final LayerNorm, and tied byte embeddings.",
        "",
        "Training: seed 42, batch 32, AdamW LR 0.0003, decay 0.01, clipping norm 1. "
        "Half of windows rehearse the original training document; the remaining half "
        "are shared equally among the new books available to that version. Actual "
        "document-window counts and full validation curves are in per-version JSON.",
        "",
        "Selection every 200 updates (including the initial parent) uses 50% original "
        "validation loss plus 50% equal-book mean loss across all five new validation "
        "books. Every version scores the same 1,024 original targets and 256 targets "
        "per new book. Validation on future training books is intentional: the target "
        "set and weighting stay fixed throughout the comparison.",
        "",
        "## Dataset provenance",
        "",
        "| Work | Metadata edition | Clean bytes | Training bytes | Source |",
        "|---|---|---:|---:|---|",
    ]
    for book in manifest["books"]:
        lines.append(
            f"| {book['title']} | {book['publication_date']} | {book['clean_bytes']:,} | "
            f"{book['split_bytes']['train']:,} | "
            f"[{book['item']}]({book['text_url']}) |"
        )
    lines += [
        "",
        "Archive metadata marks all five scans `NOT_IN_COPYRIGHT` in the US. Use only "
        "the verified original story, not advertisements, library labels, or other "
        "works bound in the same volume. Peter's scan includes *Margaret Ogilvy*, "
        "which was excluded. The Peter title page says 1912 while metadata says c1911.",
        "",
        "Each new work has contiguous 80/10/10 splits and discarded 128-byte boundary "
        "gaps. Batches are sampled **within one document's training split**; never across "
        "book boundaries. The original partitions were not expanded to include their "
        "validation or test passages.",
        "",
        f"The audit checked {manifest['overlap_audit']['checked_spans']:,} strided "
        "128-byte held-out spans against every training document and found no exact "
        "matches. This is not exhaustive semantic deduplication. Test passages remain "
        "within the same books, so recurring names, motifs and shorter phrases can "
        "appear in both training and holdouts.",
        "",
        "Spelling/OCR errors remain, including missing decorative initial letters in "
        "the Oz scan. Older books can contain dated or objectionable language; review "
        "the material before classroom use. Public-domain status here is US-specific; "
        "check local rights, including special Peter Pan-related UK royalty provisions.",
        "",
        "## Fixed-prompt, unedited samples",
        "",
        "Every version uses the same seven prompts, seeds 123–129, temperature 0.8, "
        "top-k 40, and 240 generated bytes per prompt. Samples may end mid-word. "
        "Lexicon hits are a rough spelling proxy, not a coherence or grammar measure.",
    ]
    for result in versions:
        lines += ["", f"### {result['name']}"]
        for sample in result["samples"]:
            lines += [
                "",
                f"Prompt `{sample['prompt']}`; seed {sample['seed']}; "
                f"training-lexicon hits {sample['lexicon_hit_rate']:.1%}; "
                f"repeated word trigrams {sample['repeated_word_trigram_fraction']:.1%}.",
                "",
                "```text", sample["prompt"] + sample["continuation"], "```",
            ]
    lines += [
        "",
        "## Limits and honest conclusions",
        "",
        "More data and more updates need not improve every metric in every round. "
        "The tables include all versions and all regressions; the recommended model "
        "uses validation only. Data coverage, continual training, depth and unequal "
        "cumulative compute are bundled, with one seed, so these are not causal ablations.",
        "",
        "The fresh holdout is new to this study but comprises within-book passages, "
        "not a new author or a general-purpose writing benchmark. Byte perplexity "
        "cannot be compared directly with GPT-2 BPE token perplexity. All models remain "
        "small demonstrations, not chatbots or reliable narrative writers.",
        "",
        "## Reproduce and choose any version",
        "",
        "```bash",
        "cd /Users/michalfu/mini-gpt-classroom",
        "python3 generate.py --version v3 --device cpu",
        f"python3 generate.py --version {selected['name']} --device cpu --prompt 'Dorothy said, \"'",
        "python3 generate.py --version v7 --device cpu",
        "",
        "# Included expanded corpus is frozen; prepare again only to reproduce data.",
        "python3 expanded_data.py",
        "python3 -m unittest -v",
        "python3 train_expanded.py --device cpu",
        "python3 report_expanded.py",
        "```",
        "",
        "All seven checkpoints: `artifacts/versions/v1.pt` through `v7.pt`.",
        "New metrics/lineage: `artifacts/versions/expanded_comparison.json`, "
        "`v3.json` through `v7.json`, and the corresponding `*_curve.jsonl` files.",
        "",
        f"Environment: `{json.dumps(report['environment'])}`.",
    ]
    return "\n".join(lines) + "\n"


def main():
    report = json.loads((DIRECTORY / "expanded_comparison.json").read_text())
    manifest = json.loads((DATA_DIRECTORY / "manifest.json").read_text())
    verify(report, manifest)
    (ROOT / "EXPANDED_RESULTS.md").write_text(report_markdown(report, manifest))
    print("All seven models, 35 new samples, both test suites, corpus and old artifacts verified.")
    print("Created EXPANDED_RESULTS.md.")


if __name__ == "__main__":
    main()
