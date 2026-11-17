"""Verify both delivered models and write a measured, unedited comparison report."""

import json
import math

import torch

from data import MAX_CONTEXT, ROOT, load_split
from model import decode, encode, load_model
from train import eval_positions, evaluate
from train_versions import VERSIONS, original_hashes


def verify(report):
    torch.set_num_threads(4)
    if report["original_hashes"] != original_hashes():
        raise AssertionError("Original model/deck/results were modified.")
    if [r["name"] for r in report["versions"]] != [s["name"] for s in VERSIONS]:
        raise AssertionError("Both model versions must be delivered.")
    directory = ROOT / "artifacts" / "versions"
    test = torch.tensor(list(load_split("test")), dtype=torch.long)
    positions = eval_positions(test, count=len(test) - MAX_CONTEXT, seed=2027)
    for result, spec in zip(report["versions"], VERSIONS):
        if result["curve"][-1]["step"] != spec["steps"]:
            raise AssertionError(f"{spec['name']} training is incomplete.")
        best = min(result["curve"], key=lambda event: event["validation"]["loss"])
        if result["best_step"] != best["step"]:
            raise AssertionError("Delivered checkpoint was not selected by validation.")
        model = load_model(directory / result["checkpoint"])
        if model.config.dropout != spec["dropout"]:
            raise AssertionError("Wrong version dropout configuration.")
        if sum(p.numel() for p in model.parameters()) != result["parameters"]:
            raise AssertionError("Recorded parameter count is wrong.")
        metrics = evaluate(model, test, positions, "cpu")
        for key, value in metrics.items():
            if not math.isclose(value, result["test"][key], rel_tol=1e-6, abs_tol=1e-6):
                raise AssertionError(f"{result['name']} test metric mismatch: {key}")
        if len(result["samples"]) != 3:
            raise AssertionError("Missing comparison samples.")
        for sample in result["samples"]:
            prompt = torch.tensor([encode(sample["prompt"])], dtype=torch.long)
            generated = model.generate(prompt, count=240, seed=sample["seed"],
                                       temperature=sample["temperature"], top_k=sample["top_k"])
            continuation = decode(generated[0].tolist()[prompt.shape[1]:])
            if continuation != sample["continuation"]:
                raise AssertionError("Recorded sample does not match checkpoint.")
    original = load_model(ROOT / "artifacts" / "final.pt")
    v1 = load_model(directory / "v1.pt")
    exact = all(torch.equal(value, v1.state_dict()[key])
                for key, value in original.state_dict().items())
    if exact != report["v1_exact_weight_reproduction"]:
        raise AssertionError("V1 reproduction claim is incorrect.")


def write_report(report):
    v1, v2 = report["versions"]
    lines = [
        "# Two classroom GPT versions",
        "",
        "Both models were trained from scratch again on the same frozen public-domain "
        "Internet Archive OCR. The original final checkpoint, six-run report, and HTML/PDF "
        "deck remain unchanged. [PLAN_V2.md](PLAN_V2.md) explains the architectural changes "
        "and further improvements worth investigating.",
        "",
        "| Version | Role | Blocks | Width / heads | Byte context | Dropout | Parameters | Updates / best checkpoint |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result, role in [(v1, "Simplest teaching model"), (v2, "Extended teaching model")]:
        config = result["config"]
        lines.append(
            f"| {result['name']} | {role} | {config['layers']} | "
            f"{config['width']} / {config['heads']} | {config['context']} | "
            f"{config['dropout']} | {result['parameters']:,} | "
            f"{result['steps']:,} / {result['best_step']:,} |"
        )
    lines += [
        "",
        "V2 adds context, one decoder block, and dropout at embeddings, attention weights, "
        "attention residual output, and MLP output. Dropout is active only during training. "
        "Both versions retain GPT-2-style learned positions, pre-LayerNorm, explicit causal "
        "attention, 4x-width GELU MLPs, residual paths, and tied byte embeddings.",
        "",
        "## Measured quality",
        "",
        "| Version | Train loss | Sampled validation loss | Full validation loss | Reused test loss | Test bits/byte | Test byte PPL | Test accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in [v1, v2]:
        test = result["test"]
        lines.append(
            f"| {result['name']} | {result['train']['loss']:.3f} | "
            f"{result['validation']['loss']:.3f} | {result['full_validation']['loss']:.3f} | "
            f"{test['loss']:.3f} | {test['bits_per_byte']:.3f} | "
            f"{test['perplexity']:.3f} | {test['accuracy']:.1%} |"
        )
    val_change = 1 - v2["validation"]["loss"] / v1["validation"]["loss"]
    test_change = 1 - v2["test"]["loss"] / v1["test"]["loss"]
    last_v2_loss = v2["curve"][-1]["validation"]["loss"]
    lines += [
        "",
        f"V2's selected checkpoint changes sampled validation loss by "
        f"{-val_change:+.1%} and reused test loss by {-test_change:+.1%} relative to V1 "
        "(negative means lower/better). Accuracy is next-byte prediction, not answer correctness.",
        "",
        f"V2's best checkpoint is update {v2['best_step']:,}, not the last update. "
        f"Its final training checkpoint's validation loss was {last_v2_loss:.3f}, "
        f"versus {v2['validation']['loss']:.3f} at the selected checkpoint. "
        "Additional training did not keep improving generalization.",
        "",
        "Qualitatively, the first V2 sample contains longer recognizable phrases such "
        "as 'cried Tweedledee' and 'the King looked', but still starts 'She were said' "
        "and invents misspellings. Both models remain ungrammatical and lack story-level "
        "coherence. Dropout was intended to regularize; these bundled runs do not prove "
        "it reduced overfitting, and a substantial train/validation gap remains.",
        "",
        "**Selection protocol:** same 1,024 distinct validation targets, checkpoint checks "
        "every 200 updates, same seed 42, optimizer, learning rate and clipping. Full "
        "validation scores cover 16,299 target bytes; test covers 16,300. Only sampled "
        "validation chooses the checkpoint. Each model uses its own context length.",
        "",
        "**Test limitation:** the original test was previously reported during V1. It is "
        "a reused holdout, not a new independent final test. No V2 settings or checkpoint "
        "were chosen from the new test scores. Architecture, dropout, and compute changes "
        "are bundled; this is not a causal ablation or multi-seed benchmark.",
        "",
        "## Training budget and equal-update comparison",
        "",
        "| Version | Training bytes presented | Training seconds | Validation loss at 1,800 updates |",
        "|---|---:|---:|---:|",
    ]
    for result in [v1, v2]:
        event = next(e for e in result["curve"] if e["step"] == 1800)
        lines.append(
            f"| {result['name']} | {result['training_tokens']:,} | "
            f"{result['training_seconds']:.1f} | {event['validation']['loss']:.3f} |"
        )
    lines += [
        "",
        "The 1,800-update comparison removes the difference in update count, **not** "
        "the difference in compute: V2 has more blocks and processes twice as many "
        "bytes per update. Its full training run processes four times V1's bytes. "
        "Training seconds exclude sample generation and full validation/test scoring.",
        "",
        f"Exact original V1 weight reproduction after retraining: "
        f"**{report['v1_exact_weight_reproduction']}**. Original teaching artifact "
        "checksums were checked before/after training and again during report verification.",
        "",
        f"Environment: `{json.dumps(report['environment'])}`.",
        "",
        "## Unedited, fixed-prompt samples",
        "",
        "Same three prompts, seeds 123/124/125, temperature 0.8, top-k 40, "
        "and 240 generated bytes. Outputs may end mid-word. Lexicon hits are a crude "
        "spelling proxy, not a grammar/coherence score.",
    ]
    for index in range(3):
        lines += ["", f"### Prompt {index+1}: `{v1['samples'][index]['prompt']}`"]
        for result in [v1, v2]:
            sample = result["samples"][index]
            lines += [
                "",
                f"**{result['name']}** — training-lexicon hit rate "
                f"{sample['lexicon_hit_rate']:.1%}; repeated word-trigram fraction "
                f"{sample['repeated_word_trigram_fraction']:.1%}.",
                "",
                "```text", sample["prompt"] + sample["continuation"], "```",
            ]
    lines += [
        "",
        "## Choose either version",
        "",
        "Use **V1** to introduce the smallest architecture and demonstrate the original "
        "results. Use **V2** for a second lesson on longer context, extra depth, "
        "regularization, and compute/quality tradeoffs. Neither is a chatbot or a "
        "reliable long-form writer.",
        "",
        "```bash",
        "cd /Users/michalfu/mini-gpt-classroom",
        "python3 generate.py --version v1 --device cpu --prompt 'Alice said, \"'",
        "python3 generate.py --version v2 --device cpu --prompt 'Alice said, \"'",
        "",
        "# Reproduce both; preserves original artifacts/final.pt and first-study results.",
        "python3 train_versions.py --device cpu",
        "python3 report_versions.py",
        "```",
        "",
        "Checkpoints: `artifacts/versions/v1.pt`, `artifacts/versions/v2.pt`.",
        "Raw results and learning curves: `artifacts/versions/comparison.json`, "
        "`v1.json`, `v2.json`, and the two `*_curve.jsonl` files.",
    ]
    (ROOT / "VERSIONS.md").write_text("\n".join(lines) + "\n")


def main():
    report = json.loads((ROOT / "artifacts" / "versions" / "comparison.json").read_text())
    verify(report)
    write_report(report)
    print("Verified both checkpoints, all six samples, test metrics, and original hashes.")
    print("Created VERSIONS.md with measured quality and compute comparisons.")


if __name__ == "__main__":
    main()
