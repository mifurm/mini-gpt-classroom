"""Verify the fivefold experiment and publish complete quality comparisons."""

import csv
import hashlib
import json
import math
import unittest

from data import ROOT
from fivefold_data import DIRECTORY as DATA_DIRECTORY, load_document
from fivefold_train import DIRECTORY, SPECS, check_preserved
from model import load_model


def percentage(value):
    return f"{value:.2%}"


def change(value):
    return f"{100 * value:+.2f} pp"


def metrics(record, group):
    return record["tests"][group]


def main():
    tests = unittest.TextTestRunner().run(unittest.defaultTestLoader.discover(str(ROOT)))
    if not tests.wasSuccessful():
        raise RuntimeError("Unit tests failed; results were not published.")
    report = json.loads((DIRECTORY / "comparison.json").read_text())
    manifest = json.loads((DATA_DIRECTORY / "manifest.json").read_text())
    assert report["corpus_manifest_sha256"] == hashlib.sha256((DATA_DIRECTORY / "manifest.json").read_bytes()).hexdigest()
    assert not manifest["overlap_audit"]["exact_matches"]
    for book in manifest["books"]:
        for suffix, expected in [("source", book["raw_sha256"]), ("clean", book["clean_sha256"])]:
            assert hashlib.sha256((DATA_DIRECTORY / f"{book['name']}_{suffix}.txt").read_bytes()).hexdigest() == expected
        for split in ["train", "validation", "test"]:
            content = load_document(book["name"], split)
            assert len(content) == book["split_bytes"][split]
    for plan in manifest["versions"].values():
        actual = sum(len(load_document(name, "train"))
                     for name in plan["legacy_documents"] + plan["additional_documents"])
        assert actual == plan["train_bytes"] and actual >= 5 * plan["old_train_bytes"]
    for record in report["records"] + report["previous"]:
        path = ROOT / record["checkpoint"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["checkpoint_sha256"]
        if record["arm"] != "previous":
            assert record["corpus_manifest_sha256"] == report["corpus_manifest_sha256"]
            if record["parent_checkpoint"]:
                assert hashlib.sha256((ROOT / record["parent_checkpoint"]).read_bytes()).hexdigest() == record["parent_sha256"]
            model = load_model(path)
            assert sum(p.numel() for p in model.parameters()) == record["parameters"]
            assert sum(record["document_windows"].values()) == record["steps"] * 32
            assert record["available_train_bytes"] == sum(len(load_document(name, "train"))
                                                         for name in record["train_documents"])
            best = min(event["validation"]["selection"]["loss"] for event in record["curve"])
            assert math.isclose(best, record["best_validation_loss"], abs_tol=1e-8)
            assert math.isclose(record["validation"]["selection"]["loss"], best, abs_tol=1e-6)
            assert record["steps"] * 32 * model.config.context == record["presented_bytes"]
    old = json.loads((ROOT / "artifacts/versions/expanded_comparison.json").read_text())
    for recorded in old["baselines"] + old["versions"]:
        current = next(r for r in report["previous"] if r["name"] == recorded["name"])
        assert current["tests"]["legacy_new_macro"]["accuracy"] == recorded["tests"]["new_books_macro"]["accuracy"]
    check_preserved()

    def find(name, arm):
        return next(record for record in report["records"] + report["previous"]
                    if record["name"] == name and record["arm"] == arm)

    for spec in SPECS:
        control, large = find(spec["name"], "control"), find(spec["name"], "large")
        for field in ["initial_model_sha256", "parent_sha256", "recipe", "config",
                      "seed", "batch_size", "steps"]:
            assert control[field] == large[field], (spec["name"], field)
    gains = [
        metrics(find(name, "large"), "additional_macro")["accuracy"]
        - metrics(find(name, "control"), "additional_macro")["accuracy"]
        for name in manifest["versions"]
    ]
    regressions = {
        group: [name.upper() for name in manifest["versions"]
                if metrics(find(name, "large"), group)["accuracy"]
                < metrics(find(name, "control"), group)["accuracy"]]
        for group in ["original", "legacy_new_macro"]
    }
    protected_count = len(json.loads((DIRECTORY / "preserved.json").read_text()))
    lines = [
        "# Fivefold Archive corpus experiment",
        "",
        "**Completed:** every V1-V7 corpus is at least five times its previous unique training-byte count. "
        "The five distinct preliminary recipes were also rerun on the larger V1 corpus; R4 is the exact V1 recipe.",
        "",
        "Each matched-budget pair has identical architecture, initialization/parent weights, seed, optimizer, "
        "batch size, completed update budget, and common validation targets. The intervention changes training text "
        "and its distribution. The control retains the original rehearsal mix; the large arm gives 20% "
        "of windows to that mix and 80% to new books (byte-weighted). Validation can select different "
        "checkpoint updates within the same completed budget; selected updates are reported below.",
        "",
        "V3-V7 each start from the **same original parent in both arms**, isolating a stage's data change. "
        "They are not a newly chained fivefold-trained lineage. Previously delivered checkpoints and decks are intact.",
        "",
        "## Main findings and verification",
        "",
        f"Expanded-data arms improved added-book next-byte accuracy in **{sum(gain > 0 for gain in gains)}/7** "
        f"versions at matched completed update budgets; differences range from {change(min(gains))} "
        f"to {change(max(gains))}.",
        "",
        f"Original-test accuracy regressions: {', '.join(regressions['original']) or 'none'}. "
        f"Legacy five-book accuracy regressions: {', '.join(regressions['legacy_new_macro']) or 'none'}. "
        "More varied data is not a guarantee of improvement on every existing domain.",
        "",
        f"**Verified:** {tests.testsRun} unit tests passed; {len(report['records'])} new runs and "
        f"{len(report['previous'])} previous checkpoints checked; paired initialization/configuration/budgets "
        f"match; all corpus/checkpoint hashes match; {protected_count} protected original files are unchanged.",
        "",
        f"Each model was scored on {sum(value['targets'] for value in report['records'][0]['tests']['per_book'].values()):,} "
        "fixed next-byte targets, including 7,168 targets from the 14 additional books.",
        "",
        "Qualitative inspection of fixed seeds 123 and 130 for V1-large, V7-control, V7-large, "
        "and V7-long found English-like fragments but persistent broken grammar, invented words, "
        "and mixed fictional characters. Even the longer model is a teaching demonstration, "
        "not a coherent writing assistant. All samples below remain unedited.",
        "",
        "## Unique training corpus sizes",
        "",
        "| Version | Previous bytes | Expanded bytes | Factor | Added books |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, plan in manifest["versions"].items():
        lines.append(f"| {name.upper()} | {plan['old_train_bytes']:,} | {plan['train_bytes']:,} | "
                     f"{plan['expansion_factor']:.2f}x | {len(plan['additional_documents'])} |")
    lines += [
        "",
        "Counts exclude validation/test bytes and boundary gaps; no repetition or duplication counts as new data.",
        "",
        "## Matched-budget quality: larger data versus rerun controls",
        "",
        "| Version | Updates | Original test control -> large | Legacy five-book control -> large | "
        "Additional-book control -> large | Added-book gain |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in manifest["versions"]:
        small, large = find(name, "control"), find(name, "large")
        cells = [name.upper(), str(large["steps"])]
        for group in ["original", "legacy_new_macro", "additional_macro"]:
            cells.append(f"{percentage(metrics(small, group)['accuracy'])} -> "
                         f"{percentage(metrics(large, group)['accuracy'])}")
        cells.append(change(metrics(large, "additional_macro")["accuracy"] -
                            metrics(small, "additional_macro")["accuracy"]))
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Full loss and quality metrics",
        "",
        "| Version | Arm | Dataset | Loss (nats/byte) | Bits/byte | Byte perplexity | Next-byte accuracy |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for name in list(manifest["versions"]) + ["v7_long"]:
        arms = ["previous", "control", "large"] if name != "v7_long" else ["large"]
        for arm in arms:
            record = find(name, arm)
            for group in ["original", "legacy_new_macro", "additional_macro"]:
                result = metrics(record, group)
                lines.append(f"| {name.upper()} | {arm} | {group} | {result['loss']:.4f} | "
                             f"{result['bits_per_byte']:.4f} | {result['perplexity']:.4f} | "
                             f"{percentage(result['accuracy'])} |")
    long, matched = find("v7_long", "large"), find("v7", "large")
    lines += [
        "",
        "## Predeclared longer-training check",
        "",
        f"V7-large continued its live model/optimizer trajectory from {matched['steps']:,} to "
        f"{long['steps']:,} updates ({long['step_multiplier']}x the original budget). "
        "The multiplier is the ceiling of corpus growth, fixed before training to approximately "
        "restore presented bytes per available byte. No test score chose the extension.",
        "",
        f"Additional-book accuracy: {percentage(metrics(matched, 'additional_macro')['accuracy'])} -> "
        f"**{percentage(metrics(long, 'additional_macro')['accuracy'])}**. "
        f"Legacy five-book accuracy: {percentage(metrics(matched, 'legacy_new_macro')['accuracy'])} -> "
        f"**{percentage(metrics(long, 'legacy_new_macro')['accuracy'])}**.",
        "",
        "This extended result includes additional optimization as well as the expanded corpus; "
        "it is not a matched-compute estimate of data improvement alone. With a fixed update budget, "
        "increasing available bytes reduces exposure per corpus byte.",
        "",
        "## Preliminary six-experiment replication",
        "",
        "| Recipe | Corpus growth | Updates | Added-book control accuracy | Large accuracy | Difference |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for spec in [s for s in SPECS if s["name"].startswith("r")] + [
        dict(name="r4_train_longer", steps=1800)
    ]:
        alias = "v1" if spec["name"] == "r4_train_longer" else spec["name"]
        small, large = find(alias, "control"), find(alias, "large")
        lines.append(f"| {spec['name']} | {large['corpus_expansion_factor']:.2f}x | {spec['steps']} | "
                     f"{percentage(metrics(small, 'additional_macro')['accuracy'])} | "
                     f"{percentage(metrics(large, 'additional_macro')['accuracy'])} | "
                     f"{change(metrics(large, 'additional_macro')['accuracy'] - metrics(small, 'additional_macro')['accuracy'])} |")
    lines += [
        "",
        "## Checkpoint selection, compute, and training exposure",
        "",
        "| Version/recipe | Arm | Updates | Selected update | Seconds | Presented MB | Bytes presented / available byte |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for record in report["records"]:
        lines.append(f"| {record['name']} | {record['arm']} | {record['steps']:,} | "
                     f"{record['best_step']:,} | {record['training_seconds']:.1f} | "
                     f"{record['presented_bytes']/1e6:.2f} | "
                     f"{record['presented_bytes_per_available_byte']:.2f} |")
    lines += [
        "",
        "Timing includes periodic validation/checkpoint overhead and excludes final test scoring/sampling. "
        "The two arms ran concurrently, so wall-clock timings are not isolated hardware benchmarks. "
        "Window sampling with replacement means presented/available bytes is exposure, not exact epochs.",
        "",
        "## Paired uncertainty over added books",
        "",
        "| Version | Added-book accuracy difference | 95% paired-book bootstrap interval |",
        "|---|---:|---:|",
    ]
    for comparison in report["paired_comparisons"]:
        interval = comparison["additional_accuracy"]
        lines.append(f"| {comparison['name']} | {change(interval['mean'])} | "
                     f"{change(interval['lower'])} to {change(interval['upper'])} |")
    lines += [
        "",
        "These intervals describe variation across this book collection, not training-seed uncertainty "
        "or a representative population of all writing. Books share authors; this descriptive "
        "book bootstrap does not account for author clustering.",
        "",
        "## Generated-text quality proxies",
        "",
        "All models use the same nine prompts, seeds 123-131, temperature 0.8, top-k 40, and 240 generated bytes. "
        "Lexicon matching uses one fixed dictionary from all training documents, never test text. "
        "Neither spelling match nor low repetition proves grammaticality or coherence.",
        "",
        "| Version | Arm | Fixed-lexicon word hit rate | Repeated word-trigram fraction | Mean UTF-8 replacements |",
        "|---|---|---:|---:|---:|",
    ]
    for record in report["records"] + report["previous"]:
        result = record["sample_metrics"]
        lines.append(f"| {record['name']} | {record['arm']} | {percentage(result['lexicon_hit_rate'])} | "
                     f"{result['repeated_word_trigram_fraction']:.3f} | {result['utf8_replacements']:.2f} |")
    lines += [
        "",
        "## Data, rights, and contamination checks",
        "",
        f"Added scans prepared: {len(manifest['books'])}. Their pre-1926 narrative texts/text editions "
        "are assessed as public domain in the US under the [Cornell copyright-term reference]"
        "(https://guides.library.cornell.edu/copyright/publicdomain). Some Archive metadata lacks a binary "
        "rights flag; its actual statement and the separate date-based rights basis are preserved, not fabricated. "
        "Some scans are later reprints of those public-domain texts, not physical pre-1926 editions.",
        "",
        "Acquisition rejected an undated Fauntleroy candidate in favor of the dated 1886 scan, "
        "and a damaged Treasure Island candidate lacking its final narrative paragraph in favor "
        "of a complete copy. Neither rejected candidate counts toward training bytes.",
        "",
        "| Work | Edition | Clean bytes | Train bytes | Source |",
        "|---|---|---:|---:|---|",
    ]
    for book in manifest["books"]:
        lines.append(f"| {book['title']} | {book['publication_date']} | {book['clean_bytes']:,} | "
                     f"{book['split_bytes']['train']:,} | [Archive](https://archive.org/details/{book['item']}) |")
    lines += ["", "Metadata publication dates can refer to the original text edition, not the scanned reprint."]
    for book in manifest["books"]:
        if book.get("scan_note"):
            lines.append(f"- {book['name']}: {book['scan_note']}")
    audit = manifest["overlap_audit"]
    lines += [
        "",
        f"Audit: **{audit['checked_spans']:,} heldout 128-byte spans**, checked against every one of "
        f"**{audit['training_windows']:,} document-local training offsets**; **zero exact matches**. "
        "Rolling-hash candidates are verified byte-for-byte. This remains a strided, exact-byte audit, "
        "not exhaustive semantic deduplication. Every book uses contiguous 80/10/10 partitions, "
        "128-byte gaps, checksums, and document-local training windows.",
        "",
        "OCR mistakes, dated language, and historical stereotypes remain. Check local rights and "
        "classroom suitability before redistributing. The common added-book test includes books not "
        "trained by early versions; all versions nevertheless use the same target positions.",
        "",
        "## Protocol and limits",
        "",
        report["validation_protocol"], "", report["test_protocol"], "", report["limitations"],
        "",
        "Tests and checkpoints are never chosen by test results. Legacy holdouts were reported earlier "
        "and are reused, not fresh. New-book performance was first opened only after every paired and "
        "extended selection finished. One random seed cannot establish a robust causal improvement; "
        "larger corpora also change genre and token distributions.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "python3 fivefold_data.py",
        "python3 -m unittest -v",
        "python3 fivefold_train.py --arm control --device cpu",
        "python3 fivefold_train.py --arm large --device cpu",
        "python3 fivefold_train.py --evaluate --device cpu",
        "python3 report_fivefold.py",
        "python3 generate.py --checkpoint artifacts/fivefold/large/v7_long.pt --device cpu",
        "```",
        "",
        "These commands overwrite only the new fivefold experiment outputs. Detailed per-book scores, "
        "all curves, paired intervals, hashes, and every sample are in "
        "`artifacts/fivefold/comparison.json`; flat metrics are in `artifacts/fivefold/metrics.csv`.",
        "",
        "## All unedited fixed-seed samples",
        "",
    ]
    for record in report["records"] + report["previous"]:
        lines += [f"### {record['arm']} / {record['name']}", ""]
        for sample in record["samples"]:
            lines += [f"Seed {sample['seed']}, prefix {sample['prompt']!r}", "",
                      "```text", sample["prompt"] + sample["continuation"], "```", ""]
    (ROOT / "FIVEFOLD_RESULTS.md").write_text("\n".join(lines) + "\n")
    fields = ["name", "arm", "steps", "best_step", "train_bytes", "dataset",
              "loss", "bits_per_byte", "perplexity", "accuracy", "targets"]
    with (DIRECTORY / "metrics.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for record in report["records"] + report["previous"]:
            groups = {group: result for group, result in record["tests"].items()
                      if group != "per_book"}
            groups.update({f"book/{name}": result
                           for name, result in record["tests"]["per_book"].items()})
            for group, result in groups.items():
                writer.writerow(dict(name=record["name"], arm=record["arm"],
                                     steps=record.get("steps"), best_step=record.get("best_step"),
                                     train_bytes=record.get("available_train_bytes"),
                                     dataset=group, **result_to_csv(result)))
    print(f"Verified {len(report['records'])} new runs and {len(report['previous'])} previous models.")
    print(f"Wrote FIVEFOLD_RESULTS.md, metrics.csv, and {sum(len(r['samples']) for r in report['records'] + report['previous'])} unedited samples.")


def result_to_csv(result):
    return {key: result[key] for key in ["loss", "bits_per_byte", "perplexity", "accuracy", "targets"]}


if __name__ == "__main__":
    main()
