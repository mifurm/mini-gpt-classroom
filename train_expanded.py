"""Train V3-V7 on growing Archive data; score fresh tests only after selection."""

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import platform
import time

import torch

from data import ROOT, load_split
from expanded_data import BOOKS, DIRECTORY as DATA_DIRECTORY, load_book
from model import grow_depth, load_model, save_model
from train import PROMPTS, batch, choose_device, eval_positions, evaluate, samples, unigram
from train_versions import PRESERVED


DIRECTORY = ROOT / "artifacts" / "versions"
RECIPES = [
    dict(name="v3", parent="v2", books=1, layers=3, steps=1800),
    dict(name="v4", parent="v3", books=2, layers=3, steps=1800),
    dict(name="v5", parent="v4", books=3, layers=4, steps=1800),
    dict(name="v6", parent="v5", books=4, layers=4, steps=1800),
    dict(name="v7", parent="v6", books=5, layers=4, steps=2600),
]
EXTRA_PROMPTS = ["Dorothy said, \"", "Wendy ", "The Mole ", "Mary Lennox "]


def preserved_hashes():
    names = PRESERVED + [
        "artifacts/versions/v1.pt", "artifacts/versions/v2.pt",
        "artifacts/versions/comparison.json", "VERSIONS.md",
        "data/train.bin", "data/validation.bin", "data/test.bin", "data/provenance.json",
    ]
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}


def tensor(content):
    return torch.tensor(list(content), dtype=torch.long)


def document_batch(documents, context, rng, batch_size=32):
    if not documents or any(len(document) <= context for document in documents):
        raise ValueError("Each training document must contain a complete shifted window.")
    if len(documents) < 2:
        raise ValueError("Expanded training needs the original plus at least one new book.")
    weights = torch.tensor([0.5] + [0.5 / (len(documents) - 1)] * (len(documents) - 1))
    chosen = torch.multinomial(weights, batch_size, replacement=True, generator=rng)
    x = torch.empty((batch_size, context), dtype=torch.long)
    y = torch.empty_like(x)
    for index in chosen.unique().tolist():
        rows = (chosen == index).nonzero().flatten()
        x[rows], y[rows] = batch(documents[index], context, len(rows), rng, "cpu")
    return x, y, chosen


def combine(metrics, weights):
    if len(metrics) != len(weights) or not metrics:
        raise ValueError("Metrics and weights must be nonempty and aligned.")
    if any(weight < 0 for weight in weights) or not math.isclose(sum(weights), 1.0):
        raise ValueError("Metric weights must be nonnegative and sum to one.")
    loss = sum(weight * metric["loss"] for weight, metric in zip(weights, metrics))
    return dict(loss=loss, bits_per_byte=loss / math.log(2), perplexity=math.exp(loss),
                accuracy=sum(weight * metric["accuracy"]
                             for weight, metric in zip(weights, metrics)),
                targets=sum(metric["targets"] for metric in metrics))


def suite(model, documents, positions, device):
    per_book = {name: evaluate(model, content, positions[name], device)
                for name, content in documents.items()}
    new_metrics = [per_book[book["name"]] for book in BOOKS]
    macro = combine(new_metrics, [1 / len(new_metrics)] * len(new_metrics))
    combined = combine([per_book["original"], macro], [0.5, 0.5])
    return dict(per_book=per_book, new_books_macro=macro, combined=combined)


def train_version(spec, training, validation, positions, device):
    torch.manual_seed(42)
    rng = torch.Generator().manual_seed(42)
    parent_path = DIRECTORY / f"{spec['parent']}.pt"
    parent = load_model(parent_path, device)
    model = grow_depth(parent, spec["layers"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.01)
    names = ["original"] + [book["name"] for book in BOOKS[:spec["books"]]]
    documents = [training[name] for name in names]
    path = DIRECTORY / f"{spec['name']}.pt"
    initial = suite(model, validation, positions, device)
    best_loss = initial["combined"]["loss"]
    best_step = 0
    save_model(model, path)
    curve = [dict(step=0, validation=initial, elapsed_seconds=0.0)]
    counts = torch.zeros(len(documents), dtype=torch.long)
    started = time.perf_counter()
    with (DIRECTORY / f"{spec['name']}_curve.jsonl").open("w") as progress:
        progress.write(json.dumps(curve[0]) + "\n")
        for step in range(1, spec["steps"] + 1):
            x, y, chosen = document_batch(documents, model.config.context, rng)
            counts += torch.bincount(chosen, minlength=len(documents))
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(x.to(device), y.to(device))
            if not torch.isfinite(loss).item():
                raise RuntimeError(f"Nonfinite {spec['name']} loss at step {step}")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(norm).item():
                raise RuntimeError(f"Nonfinite {spec['name']} gradient at step {step}")
            optimizer.step()
            if step % 200 == 0 or step == spec["steps"]:
                metrics = suite(model, validation, positions, device)
                event = dict(step=step, batch_loss=loss.item(), validation=metrics,
                             elapsed_seconds=time.perf_counter() - started)
                curve.append(event)
                progress.write(json.dumps(event) + "\n")
                progress.flush()
                print(f"{spec['name']} {step}/{spec['steps']} "
                      f"joint_val={metrics['combined']['loss']:.4f} "
                      f"old_acc={metrics['per_book']['original']['accuracy']:.3f} "
                      f"new_acc={metrics['new_books_macro']['accuracy']:.3f} "
                      f"seconds={event['elapsed_seconds']:.1f}", flush=True)
                if metrics["combined"]["loss"] < best_loss:
                    best_loss, best_step = metrics["combined"]["loss"], step
                    save_model(model, path)
    elapsed = time.perf_counter() - started
    model = load_model(path, device)
    train_metrics = {
        name: evaluate(model, training[name], eval_positions(training[name], count=256), device)
        for name in names
    }
    result = dict(
        **spec, checkpoint=path.name, parent_sha256=hashlib.sha256(parent_path.read_bytes()).hexdigest(),
        config=asdict(model.config), parameters=sum(p.numel() for p in model.parameters()),
        seed=42, batch_size=32, learning_rate=0.0003, weight_decay=0.01,
        best_step=best_step, training_seconds=elapsed, training_documents=names,
        available_train_bytes=sum(len(document) for document in documents),
        presented_bytes=spec["steps"] * 32 * model.config.context,
        document_windows={name: int(count) for name, count in zip(names, counts)},
        initial_validation=initial, validation=suite(model, validation, positions, device),
        train_per_book=train_metrics, curve=curve,
        samples=samples(model, b"\n".join(bytes(document.tolist()) for document in documents),
                        device, prompts=PROMPTS + EXTRA_PROMPTS),
    )
    (DIRECTORY / f"{spec['name']}.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    DIRECTORY.mkdir(exist_ok=True)
    before = preserved_hashes()
    manifest = json.loads((DATA_DIRECTORY / "manifest.json").read_text())
    manifest_hash = hashlib.sha256((DATA_DIRECTORY / "manifest.json").read_bytes()).hexdigest()
    device = choose_device(args.device)
    torch.set_num_threads(4)
    training = {"original": tensor(load_split("train"))}
    validation = {"original": tensor(load_split("validation"))}
    for book in BOOKS:
        training[book["name"]] = tensor(load_book(book["name"], "train"))
        validation[book["name"]] = tensor(load_book(book["name"], "validation"))
    positions = {name: eval_positions(content, count=1024 if name == "original" else 256)
                 for name, content in validation.items()}
    results = [train_version(spec, training, validation, positions, device) for spec in RECIPES]
    selected = min(results, key=lambda result: result["validation"]["combined"]["loss"])["name"]
    # Fresh expanded-test performance is first measured after all five selections.
    tests = {"original": tensor(load_split("test"))}
    for book in BOOKS:
        tests[book["name"]] = tensor(load_book(book["name"], "test"))
    test_positions = {
        name: eval_positions(content, count=len(content) - 128 if name == "original" else 2048,
                             seed=2027 if name == "original" else 2028)
        for name, content in tests.items()
    }
    baselines = []
    for name in ["v1", "v2"]:
        model = load_model(DIRECTORY / f"{name}.pt", device)
        baselines.append(dict(
            name=name, config=asdict(model.config),
            parameters=sum(p.numel() for p in model.parameters()),
            validation=suite(model, validation, positions, device),
            tests=suite(model, tests, test_positions, device),
        ))
    for result in results:
        result["tests"] = suite(load_model(DIRECTORY / result["checkpoint"], device),
                                tests, test_positions, device)
        print(f"{result['name']}: legacy_test={result['tests']['per_book']['original']['accuracy']:.3%}; "
              f"fresh_new_test_macro={result['tests']['new_books_macro']['accuracy']:.3%}", flush=True)
    all_training = torch.cat(list(training.values()))
    unigram_per_book = {name: unigram(all_training, content, test_positions[name])
                        for name, content in tests.items()}
    if before != preserved_hashes():
        raise RuntimeError("A preserved V1/V2 model, data file, report, or deck changed.")
    report = dict(
        versions=results, baselines=baselines, selected_by_common_validation=selected,
        preserved_hashes=before, expanded_manifest_sha256=manifest_hash,
        dataset_train_bytes=manifest["total_train_bytes"],
        environment=dict(python=platform.python_version(), torch=torch.__version__,
                         device=str(device), threads=torch.get_num_threads()),
        validation_protocol="Fixed original 1024 targets and 256 targets per new book. "
                            "Selection score = 0.5*original loss + 0.5*new-book macro loss.",
        test_protocol="Original reused holdout: all 16300 targets. Fresh expanded holdout: "
                      "2048 distinct targets/book; equal-book macro. First scored after all "
                      "five checkpoint selections; neither test selects recipes or checkpoints.",
        unigram_per_book=unigram_per_book,
        unigram_new_macro=combine([unigram_per_book[book["name"]] for book in BOOKS], [0.2] * 5),
        limitation="Within-book held-out passages, imperfect OCR, one seed, unequal cumulative "
                   "training, bundled data/depth changes. Exact-span audit is not semantic deduplication.",
    )
    (DIRECTORY / "expanded_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Validation-selected new model: {selected}; preserved old artifacts verified.", flush=True)


if __name__ == "__main__":
    main()
