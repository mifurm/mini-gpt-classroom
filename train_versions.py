"""Retrain and deliver two compatible classroom GPT versions without replacing V1."""

import argparse
import hashlib
import json
import platform

import torch

from data import MAX_CONTEXT, ROOT, load_split
from model import load_model
from train import choose_device, eval_positions, evaluate, run_round, unigram


VERSIONS = [
    dict(name="v1", steps=1800, context=64, width=128, layers=2, dropout=0.0,
         schedule=False, change="Original two-block model, retrained from scratch"),
    dict(name="v2", steps=3600, context=128, width=128, layers=3, dropout=0.1,
         schedule=False, change="Longer context, another block, and dropout"),
]
PRESERVED = ["artifacts/final.pt", "artifacts/results.json", "RESULTS.md",
             "slides.html", "slides.pdf"]


def original_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in PRESERVED}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    directory = ROOT / "artifacts" / "versions"
    directory.mkdir(exist_ok=True)
    before = original_hashes()
    device = choose_device(args.device)
    torch.set_num_threads(4)
    train = torch.tensor(list(load_split("train")), dtype=torch.long)
    validation = torch.tensor(list(load_split("validation")), dtype=torch.long)
    positions = eval_positions(validation)
    results = [run_round(spec, directory, device, train, validation, positions)
               for spec in VERSIONS]
    full_positions = eval_positions(validation, count=len(validation) - MAX_CONTEXT)
    for result in results:
        model = load_model(directory / result["checkpoint"], device)
        result["full_validation"] = evaluate(model, validation, full_positions, device)
    original = load_model(ROOT / "artifacts" / "final.pt")
    retrained = load_model(directory / "v1.pt")
    same_weights = all(torch.equal(value, retrained.state_dict()[name])
                       for name, value in original.state_dict().items())
    # The original test is reused for comparison, never for checkpoint selection.
    test = torch.tensor(list(load_split("test")), dtype=torch.long)
    test_positions = eval_positions(test, count=len(test) - MAX_CONTEXT, seed=2027)
    for result in results:
        model = load_model(directory / result["checkpoint"], device)
        result["test"] = evaluate(model, test, test_positions, device)
    if before != original_hashes():
        raise RuntimeError("An original teaching artifact changed during version training.")
    report = dict(
        environment=dict(python=platform.python_version(), torch=torch.__version__,
                         device=str(device), threads=torch.get_num_threads()),
        versions=results, original_hashes=before, v1_exact_weight_reproduction=same_weights,
        unigram_test=unigram(train, test, test_positions),
        protocol="Both train from scratch, same seed and optimizer, same frozen Archive "
                 "splits and target offsets. Checkpoint selection uses sampled validation "
                 "only. Full validation/test metrics are computed after training. "
                 "Original test was previously reported: reused holdout, not a fresh test. "
                 "V2 uses more parameters, twice the bytes per update, and twice the updates.",
    )
    (directory / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(
        v1_exact_weight_reproduction=same_weights,
        versions=[dict(name=r["name"], parameters=r["parameters"],
                       validation=r["validation"], test=r["test"]) for r in results],
    ), indent=2), flush=True)


if __name__ == "__main__":
    main()
