"""Paired small/large-corpus runs; all heldout performance is opened after selection."""

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import platform
import random
import shutil
import statistics
import time

import torch

from data import ROOT
from expanded_data import BOOKS as LEGACY_BOOKS
from fivefold_data import DIRECTORY as DATA_DIRECTORY, load_document
from model import Config, GPT, grow_depth, load_model, save_model
from train import ROUNDS, PROMPTS, batch, choose_device, eval_positions, evaluate, samples
from train_expanded import RECIPES, EXTRA_PROMPTS, combine, tensor
from train_versions import VERSIONS


DIRECTORY = ROOT / "artifacts" / "fivefold"
PROMPT_SET = PROMPTS + EXTRA_PROMPTS + ['Anne said, "', "The children "]
SPECS = [
    dict(**spec, corpus="v1", parent=None, dropout=0.0, learning_rate=0.001)
    for spec in ROUNDS if spec["name"] != "r4_train_longer"
] + [
    dict(**spec, corpus=spec["name"], parent=None, learning_rate=0.001)
    for spec in VERSIONS
] + [
    dict(**spec, corpus=spec["name"], context=128, width=128, dropout=0.1,
         schedule=False, learning_rate=0.0003)
    for spec in RECIPES
]


def preserved_hashes():
    files = []
    for directory in [ROOT / "data", ROOT / "artifacts"]:
        files.extend(path for path in directory.rglob("*")
                     if path.is_file() and "fivefold" not in path.relative_to(ROOT).parts)
    files.extend(ROOT / name for name in [
        "model.py", "train.py", "train_versions.py", "train_expanded.py", "data.py",
        "expanded_data.py", "RESULTS.md", "VERSIONS.md", "EXPANDED_RESULTS.md",
        "slides.html", "slides.pdf", "deep-dive.html", "deep-dive.pdf", "deep-dive.pptx",
    ])
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(set(files))}


def check_preserved():
    path = DIRECTORY / "preserved.json"
    current = preserved_hashes()
    if path.exists():
        if json.loads(path.read_text()) != current:
            raise RuntimeError("A previously delivered model, dataset, report, or slide changed.")
    else:
        path.write_text(json.dumps(current, indent=2) + "\n")


def model_hash(model):
    digest = hashlib.sha256(json.dumps(asdict(model.config), sort_keys=True).encode())
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def sampling_weights(plan, training, arm):
    legacy = plan["legacy_documents"]
    old_weights = [1.0] if len(legacy) == 1 else [0.5] + [0.5 / (len(legacy) - 1)] * (len(legacy) - 1)
    if arm == "control":
        return legacy, old_weights
    additional = plan["additional_documents"]
    total = sum(len(training[name]) for name in additional)
    return (legacy + additional,
            [0.2 * weight for weight in old_weights]
            + [0.8 * len(training[name]) / total for name in additional])


def weighted_batch(documents, weights, context, rng, batch_size=32):
    if not documents or any(len(document) <= context for document in documents):
        raise ValueError("Every document needs a complete shifted training window.")
    if len(weights) != len(documents) or any(not math.isfinite(w) or w <= 0 for w in weights):
        raise ValueError("Positive finite sampling weights must align with documents.")
    if not math.isclose(sum(weights), 1.0):
        raise ValueError("Sampling weights must sum to one.")
    if len(documents) == 1:
        x, y = batch(documents[0], context, batch_size, rng, "cpu")
        return x, y, torch.zeros(batch_size, dtype=torch.long)
    chosen = torch.multinomial(torch.tensor(weights), batch_size, replacement=True, generator=rng)
    x = torch.empty((batch_size, context), dtype=torch.long)
    y = torch.empty_like(x)
    for index in chosen.unique().tolist():
        rows = (chosen == index).nonzero().flatten()
        x[rows], y[rows] = batch(documents[index], context, len(rows), rng, "cpu")
    return x, y, chosen


def suite(model, documents, positions, additional_names, device):
    per_book = {name: evaluate(model, content, positions[name], device)
                for name, content in documents.items()}
    old_macro = combine([per_book[book["name"]] for book in LEGACY_BOOKS], [0.2] * 5)
    new_macro = combine([per_book[name] for name in additional_names],
                        [1 / len(additional_names)] * len(additional_names))
    legacy = combine([per_book["original"], old_macro], [0.5, 0.5])
    return dict(per_book=per_book, original=per_book["original"],
                legacy_new_macro=old_macro, additional_macro=new_macro,
                selection=combine([legacy, new_macro], [0.5, 0.5]))


def train_one(spec, arm, manifest, training, validation, positions, additional, device):
    manifest_hash = hashlib.sha256((DATA_DIRECTORY / "manifest.json").read_bytes()).hexdigest()
    torch.manual_seed(42)
    rng = torch.Generator().manual_seed(42)
    parent = ROOT / "artifacts" / "versions" / f"{spec['parent']}.pt" if spec["parent"] else None
    model = (grow_depth(load_model(parent, device), spec["layers"]) if parent else
             GPT(Config(context=spec["context"], width=spec["width"], layers=spec["layers"],
                        dropout=spec["dropout"])).to(device))
    initial_hash = model_hash(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec["learning_rate"], weight_decay=0.01)
    plan = manifest["versions"][spec["corpus"]]
    names, weights = sampling_weights(plan, training, arm)
    documents = [training[name] for name in names]
    directory = DIRECTORY / arm
    directory.mkdir(exist_ok=True)
    path = directory / f"{spec['name']}.pt"
    initial = suite(model, validation, positions, additional, device)
    best_loss, best_step = initial["selection"]["loss"], 0
    save_model(model, path)
    curve = [dict(step=0, validation=initial)]
    counts = torch.zeros(len(names), dtype=torch.long)
    matched_steps = spec["steps"]
    multiplier = math.ceil(plan["expansion_factor"]) if arm == "large" and spec["name"] == "v7" else 1
    total_steps = matched_steps * multiplier
    long_path = directory / "v7_long.pt"
    started = time.perf_counter()
    matched = None

    def result(step, seconds, selected_step, selected_loss, checkpoint, windows):
        return dict(
            name=spec["name"], arm=arm, recipe=spec, seed=42, batch_size=32,
            config=asdict(model.config), parameters=sum(p.numel() for p in model.parameters()),
            initial_model_sha256=initial_hash,
            corpus_manifest_sha256=manifest_hash,
            parent_checkpoint=str(parent.relative_to(ROOT)) if parent else None,
            parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest() if parent else None,
            steps=step, best_step=selected_step, best_validation_loss=selected_loss,
            training_seconds=seconds, checkpoint=str(checkpoint.relative_to(ROOT)),
            train_documents=names, sampling_weights=weights,
            document_windows={name: int(count) for name, count in zip(names, windows)},
            available_train_bytes=sum(len(document) for document in documents),
            old_train_bytes=plan["old_train_bytes"],
            corpus_expansion_factor=plan["expansion_factor"] if arm == "large" else 1.0,
            presented_bytes=step * 32 * model.config.context,
            presented_bytes_per_available_byte=step * 32 * model.config.context /
                                               sum(len(document) for document in documents),
            initial_validation=initial, curve=list(curve),
        )

    with (directory / f"{spec['name']}_curve.jsonl").open("w") as progress:
        progress.write(json.dumps(curve[0]) + "\n")
        for step in range(1, total_steps + 1):
            if spec["schedule"]:
                phase = (step - 100) / (matched_steps - 100)
                rate = (0.001 * step / 100 if step <= 100 else
                        0.0001 + 0.00045 * (1 + math.cos(math.pi * phase)))
                for group in optimizer.param_groups:
                    group["lr"] = rate
            x, y, chosen = weighted_batch(documents, weights, model.config.context, rng)
            counts += torch.bincount(chosen, minlength=len(names))
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(x.to(device), y.to(device))
            if not torch.isfinite(loss).item():
                raise RuntimeError(f"Nonfinite loss: {arm}/{spec['name']}/{step}")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(norm).item():
                raise RuntimeError(f"Nonfinite gradients: {arm}/{spec['name']}/{step}")
            optimizer.step()
            if step % 200 == 0 or step == total_steps:
                metrics = suite(model, validation, positions, additional, device)
                elapsed = time.perf_counter() - started
                event = dict(step=step, batch_loss=loss.item(), validation=metrics,
                             learning_rate=optimizer.param_groups[0]["lr"], elapsed_seconds=elapsed)
                curve.append(event)
                progress.write(json.dumps(event) + "\n")
                progress.flush()
                score = metrics["selection"]["loss"]
                if score < best_loss:
                    best_loss, best_step = score, step
                    save_model(model, path if step <= matched_steps else long_path)
                print(f"{arm}/{spec['name']} {step}/{total_steps} "
                      f"val={score:.4f} legacy={metrics['legacy_new_macro']['accuracy']:.3%} "
                      f"new={metrics['additional_macro']['accuracy']:.3%} "
                      f"elapsed={elapsed:.1f}s", flush=True)
            if step == matched_steps:
                matched = result(step, time.perf_counter() - started, best_step, best_loss, path,
                                 counts.clone())
                (directory / f"{spec['name']}.json").write_text(json.dumps(matched, indent=2) + "\n")
                if total_steps > matched_steps:
                    shutil.copyfile(path, long_path)
        if total_steps > matched_steps:
            extended = result(total_steps, time.perf_counter() - started, best_step, best_loss,
                              long_path, counts)
            extended["name"] = "v7_long"
            extended["step_multiplier"] = multiplier
            (directory / "v7_long.json").write_text(json.dumps(extended, indent=2) + "\n")
    if matched is None:
        raise RuntimeError("Matched-budget checkpoint was not recorded.")


def load_validation(manifest):
    additional = manifest["versions"]["v7"]["additional_documents"]
    names = ["original"] + [book["name"] for book in LEGACY_BOOKS] + additional
    training = {name: tensor(load_document(name, "train")) for name in names}
    validation = {name: tensor(load_document(name, "validation")) for name in names}
    positions = {name: eval_positions(content, count=512 if name == "original" else 128)
                 for name, content in validation.items()}
    return training, validation, positions, additional


def paired_interval(first, second, names, key, trials=2000):
    differences = [first["per_book"][name][key] - second["per_book"][name][key] for name in names]
    rng = random.Random(2033)
    estimates = sorted(statistics.mean(rng.choices(differences, k=len(differences)))
                       for _ in range(trials))
    return dict(mean=statistics.mean(differences), lower=estimates[int(trials * 0.025)],
                upper=estimates[int(trials * 0.975)],
                protocol="Paired book bootstrap; does not measure training-seed uncertainty.")


def final_evaluation(manifest, device):
    training, validation, positions, additional = load_validation(manifest)
    records = []
    for arm in ["control", "large"]:
        for spec in SPECS:
            path = DIRECTORY / arm / f"{spec['name']}.json"
            if not path.exists():
                raise RuntimeError(f"Finish all training before opening tests: {path}")
            records.append(json.loads(path.read_text()))
    long_file = DIRECTORY / "large" / "v7_long.json"
    if not long_file.exists():
        raise RuntimeError("Finish the predeclared longer V7 training before opening tests.")
    records.append(json.loads(long_file.read_text()))
    digest = hashlib.sha256((DATA_DIRECTORY / "manifest.json").read_bytes()).hexdigest()
    if any(record["corpus_manifest_sha256"] != digest for record in records):
        raise RuntimeError("Corpus manifest changed between training runs; do not open tests.")
    test_documents = {name: tensor(load_document(name, "test")) for name in validation}
    test_positions = {
        name: eval_positions(content, count=len(content) - 128 if name == "original" else
                             2048 if name in {book["name"] for book in LEGACY_BOOKS} else 512,
                             seed=2027 if name == "original" else 2028 if name in
                             {book["name"] for book in LEGACY_BOOKS} else 2032)
        for name, content in test_documents.items()
    }
    fixed_lexicon_bytes = b"\n".join(load_document(name, "train") for name in training)
    original = []
    for spec in SPECS:
        path = (ROOT / "artifacts" / "versions" / f"{spec['name']}.pt" if spec["name"].startswith("v")
                else ROOT / "artifacts" / f"{spec['name']}.pt")
        original.append(dict(name=spec["name"], arm="previous", checkpoint=str(path.relative_to(ROOT))))
    for record in records + original:
        model = load_model(ROOT / record["checkpoint"], device)
        record["validation"] = suite(model, validation, positions, additional, device)
        record["tests"] = suite(model, test_documents, test_positions, additional, device)
        record["samples"] = samples(model, fixed_lexicon_bytes, device, prompts=PROMPT_SET)
        record["sample_metrics"] = {
            key: statistics.mean(sample[key] for sample in record["samples"])
            for key in ["lexicon_hit_rate", "repeated_word_trigram_fraction", "utf8_replacements"]
        }
        record["checkpoint_sha256"] = hashlib.sha256((ROOT / record["checkpoint"]).read_bytes()).hexdigest()
        print(f"{record['arm']}/{record['name']}: original={record['tests']['original']['accuracy']:.2%} "
              f"legacy={record['tests']['legacy_new_macro']['accuracy']:.2%} "
              f"new={record['tests']['additional_macro']['accuracy']:.2%}", flush=True)
    comparisons = []
    for spec in SPECS:
        small = next(r for r in records if r["name"] == spec["name"] and r["arm"] == "control")
        large = next(r for r in records if r["name"] == spec["name"] and r["arm"] == "large")
        assert small["initial_model_sha256"] == large["initial_model_sha256"]
        comparisons.append(dict(
            name=spec["name"],
            additional_accuracy=paired_interval(large["tests"], small["tests"], additional, "accuracy"),
            additional_loss=paired_interval(large["tests"], small["tests"], additional, "loss"),
        ))
    check_preserved()
    report = dict(
        records=records, previous=original, paired_comparisons=comparisons,
        r4_alias="r4_train_longer is exactly the V1 architecture/recipe; uses the V1 paired reruns.",
        corpus_manifest_sha256=hashlib.sha256((DATA_DIRECTORY / "manifest.json").read_bytes()).hexdigest(),
        environment=dict(python=platform.python_version(), torch=torch.__version__, device=str(device),
                         threads=torch.get_num_threads()),
        validation_protocol="Same targets, all versions/arms: original 512, other books 128 each. "
                            "Selection loss = .25 original + .25 legacy five-book macro + "
                            ".50 additional-book macro. Check every 200 updates; seed 2026.",
        test_protocol="All paired and long selections finish before performance testing. Original "
                      "all 16300 targets and five legacy books 2048 each reuse exact prior targets. "
                      "New books: 512 distinct targets each, seed 2032, equal-book macro.",
        sampling_protocol="Control preserves original rehearsal weights. Large gives .20 to that "
                          "legacy distribution and .80 to additional books weighted by training bytes.",
        limitations="One seed, changed genre/domain distribution, within-book holds, imperfect OCR; "
                    "V3-V7 use the same previous parent in each arm, not a newly chained lineage. "
                    "Byte accuracy is not coherence; fixed-lexicon sample scores are spelling proxies.",
    )
    (DIRECTORY / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--arm", choices=["control", "large"])
    action.add_argument("--evaluate", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    check_preserved()
    manifest = json.loads((DATA_DIRECTORY / "manifest.json").read_text())
    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    if args.evaluate:
        final_evaluation(manifest, device)
    else:
        training, validation, positions, additional = load_validation(manifest)
        for spec in SPECS:
            train_one(spec, args.arm, manifest, training, validation, positions, additional, device)
        check_preserved()


if __name__ == "__main__":
    main()
