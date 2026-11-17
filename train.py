"""Train one tiny GPT or run a baseline plus five improvement experiments."""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import platform
import re
import time

import torch
from torch.nn import functional as F

from data import MAX_CONTEXT, ROOT, load_split
from model import Config, GPT, decode, encode, load_model, save_model


ROUNDS = [
    dict(name="r0_baseline", steps=200, context=32, width=64, layers=1,
         schedule=False, change="Baseline: one small block, 200 updates"),
    dict(name="r1_more_training", steps=600, context=32, width=64, layers=1,
         schedule=False, change="Increase updates from 200 to 600"),
    dict(name="r2_longer_context", steps=600, context=64, width=64, layers=1,
         schedule=False, change="Increase context from 32 to 64 bytes"),
    dict(name="r3_more_capacity", steps=600, context=64, width=128, layers=2,
         schedule=False, change="Increase width to 128 and depth to two blocks"),
    dict(name="r4_train_longer", steps=1800, context=64, width=128, layers=2,
         schedule=False, change="Increase updates from 600 to 1800"),
    dict(name="r5_scheduled_lr", steps=1800, context=64, width=128, layers=2,
         schedule=True, change="Add 100-step warmup and cosine learning-rate decay"),
]
PROMPTS = ["Alice said, \"", "The Queen ", "The little "]


def choose_device(requested="auto"):
    if requested != "auto":
        device = torch.device(requested)
        torch.zeros(1, device=device)  # Fail explicitly if the requested device is unavailable.
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def batch(data, context, batch_size, rng, device):
    starts = torch.randint(len(data) - context, (batch_size,), generator=rng)
    offsets = torch.arange(context)
    x = data[starts[:, None] + offsets]
    y = data[starts[:, None] + offsets + 1]
    return x.to(device), y.to(device)


def eval_positions(data, count=1024, seed=2026):
    if len(data) <= MAX_CONTEXT:
        raise ValueError("Evaluation needs more than MAX_CONTEXT bytes.")
    rng = torch.Generator().manual_seed(seed)
    # Sample without replacement. Every round scores exactly these next-byte targets.
    return torch.randperm(len(data) - MAX_CONTEXT, generator=rng)[:count] + MAX_CONTEXT


@torch.no_grad()
def evaluate(model, data, positions, device):
    previous_mode = model.training
    model.eval()
    losses, correct = [], 0
    try:
        for indices in positions.split(64):
            offsets = torch.arange(-model.config.context, 0)
            x = data[indices[:, None] + offsets].to(device)
            y = data[indices].to(device)
            logits = model(x)[0][:, -1]
            losses.extend(F.cross_entropy(logits, y, reduction="none").cpu().tolist())
            correct += int((logits.argmax(-1) == y).sum().item())
    finally:
        model.train(previous_mode)
    loss = sum(losses) / len(losses)
    return dict(loss=loss, bits_per_byte=loss / math.log(2),
                perplexity=math.exp(loss), accuracy=correct / len(losses),
                targets=len(losses))


def unigram(train, data, positions):
    counts = torch.bincount(train, minlength=256).float() + 1
    probabilities = counts / counts.sum()
    targets = data[positions]
    loss = -probabilities[targets].log().mean().item()
    return dict(loss=loss, bits_per_byte=loss / math.log(2),
                perplexity=math.exp(loss),
                accuracy=(targets == probabilities.argmax()).float().mean().item(),
                targets=len(positions))


def samples(model, training_bytes, device, prompts=None):
    lexicon = set(re.findall(r"[a-z]+", training_bytes.decode("utf-8", "replace").lower()))
    output = []
    for index, prompt in enumerate(PROMPTS if prompts is None else prompts):
        tokens = torch.tensor([encode(prompt)], device=device)
        generated = model.generate(tokens, count=240, seed=123 + index)[0].cpu().tolist()
        continuation = decode(generated[len(encode(prompt)):])
        words = re.findall(r"[a-z]+", continuation.lower())
        trigrams = list(zip(words, words[1:], words[2:]))
        output.append(dict(prompt=prompt, continuation=continuation,
                           seed=123 + index, temperature=0.8, top_k=40,
                           lexicon_hit_rate=sum(w in lexicon for w in words) / max(1, len(words)),
                           repeated_word_trigram_fraction=(
                               1 - len(set(trigrams)) / len(trigrams) if trigrams else 0),
                           utf8_replacements=continuation.count("\ufffd")))
    return output


def run_round(spec, directory, device, train, validation, positions, seed=42):
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed)
    config = Config(context=spec["context"], width=spec["width"], layers=spec["layers"],
                    dropout=spec.get("dropout", 0.0))
    model = GPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    train_positions = eval_positions(train, count=512)
    initial = evaluate(model, validation, positions, device)
    best_loss, best_step, curve = initial["loss"], 0, []
    path = directory / f"{spec['name']}.pt"
    save_model(model, path)
    started = time.perf_counter()
    progress_path = directory / f"{spec['name']}_curve.jsonl"
    with progress_path.open("w") as progress:
        for step in range(1, spec["steps"] + 1):
            if spec["schedule"]:
                if step <= 100:
                    rate = 1e-3 * step / 100
                else:
                    phase = (step - 100) / (spec["steps"] - 100)
                    rate = 1e-4 + 0.5 * 9e-4 * (1 + math.cos(math.pi * phase))
                for group in optimizer.param_groups:
                    group["lr"] = rate
            x, y = batch(train, config.context, 32, rng, device)
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(x, y)
            if not torch.isfinite(loss).item():
                raise RuntimeError(f"Nonfinite loss in {spec['name']} step {step}")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gradient_norm).item():
                raise RuntimeError(f"Nonfinite gradient in {spec['name']} step {step}")
            optimizer.step()
            if step % 200 == 0 or step == spec["steps"]:
                metrics = evaluate(model, validation, positions, device)
                event = dict(step=step, batch_loss=loss.item(),
                             validation=metrics, lr=optimizer.param_groups[0]["lr"],
                             elapsed_seconds=time.perf_counter() - started)
                curve.append(event)
                progress.write(json.dumps(event) + "\n")
                progress.flush()
                print(f"{spec['name']} {step:4d}/{spec['steps']} "
                      f"val={metrics['loss']:.4f} bpb={metrics['bits_per_byte']:.3f} "
                      f"acc={metrics['accuracy']:.3f} elapsed={event['elapsed_seconds']:.1f}s",
                      flush=True)
                if metrics["loss"] < best_loss:
                    best_loss, best_step = metrics["loss"], step
                    save_model(model, path)
    training_seconds = time.perf_counter() - started
    model = load_model(path, device)
    result = dict(
        **spec, config=asdict(config), seed=seed, batch_size=32,
        learning_rate=1e-3, weight_decay=0.01,
        parameters=sum(p.numel() for p in model.parameters()),
        training_tokens=spec["steps"] * 32 * config.context,
        training_seconds=training_seconds, best_step=best_step,
        initial_validation=initial, validation=evaluate(model, validation, positions, device),
        train=evaluate(model, train, train_positions, device), curve=curve,
        samples=samples(model, bytes(train.tolist()), device), checkpoint=path.name,
    )
    (directory / f"{spec['name']}.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def experiments(args):
    directory = ROOT / "artifacts"
    directory.mkdir(exist_ok=True)
    device = choose_device(args.device)
    torch.set_num_threads(4)
    train = torch.tensor(list(load_split("train")), dtype=torch.long)
    validation = torch.tensor(list(load_split("validation")), dtype=torch.long)
    positions = eval_positions(validation)
    results = []
    for spec in ROUNDS:
        results.append(run_round(spec, directory, device, train, validation, positions))
    winner = min(results, key=lambda r: r["validation"]["loss"])
    model = load_model(directory / winner["checkpoint"], device)
    save_model(model, directory / "final.pt")
    # Test is first read only after all training and validation-based model selection.
    test = torch.tensor(list(load_split("test")), dtype=torch.long)
    test_positions = eval_positions(test, count=len(test) - MAX_CONTEXT, seed=2027)
    report = dict(
        environment=dict(python=platform.python_version(), torch=torch.__version__,
                         platform=platform.platform(), device=str(device),
                         threads=torch.get_num_threads()),
        evaluation=dict(validation_seed=2026, test_seed=2027,
                        max_context=MAX_CONTEXT, target_count=len(positions),
                        protocol="Same target positions; each model uses its configured context. "
                                 "Each target is scored once; no replacement. Final test "
                                 "scores every target after the first 128 test bytes."),
        unigram_validation=unigram(train, validation, positions),
        rounds=results, selected=winner["name"],
        final_test=evaluate(model, test, test_positions, device),
        unigram_test=unigram(train, test, test_positions),
        quality_caveat="Single seed, one OCR book, unequal compute, repeated validation "
                       "selection, sparse evaluation; byte perplexity is not BPE perplexity.",
    )
    (directory / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ["selected", "final_test", "unigram_test"]},
                     indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--experiments", action="store_true")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--context", type=int, default=64)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if not 1 <= args.context <= MAX_CONTEXT:
        parser.error(f"--context must be between 1 and {MAX_CONTEXT}")
    if args.experiments:
        experiments(args)
    else:
        directory = ROOT / "artifacts"
        directory.mkdir(exist_ok=True)
        torch.set_num_threads(4)
        train = torch.tensor(list(load_split("train")), dtype=torch.long)
        validation = torch.tensor(list(load_split("validation")), dtype=torch.long)
        spec = dict(name="classroom", steps=args.steps, context=args.context,
                    width=args.width, layers=args.layers, schedule=False,
                    change="Standalone classroom training run")
        result = run_round(spec, directory, choose_device(args.device),
                           train, validation, eval_positions(validation))
        print(json.dumps(result["validation"], indent=2))


if __name__ == "__main__":
    main()
