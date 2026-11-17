"""Small supervised decision and scratchpad-reasoning classroom experiments."""

import argparse
import hashlib
import itertools
import json
import random
import re
import time

import torch
from torch import nn
from torch.nn import functional as F

from data import ROOT
from fivefold_train import check_preserved, model_hash
from model import Config, GPT, decode, encode, load_model, save_model


DIRECTORY = ROOT / "concepts"
ACTIONS = ["wait", "accept"]


class DecisionPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(3, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 2),
        )

    def forward(self, features):
        return self.network(features)


def decision_reward(case, action):
    cost, benefit, budget = case
    return 0 if action == 0 else benefit - cost if cost <= budget else -10


def decision_label(case):
    return int(decision_reward(case, 1) > decision_reward(case, 0))


def validate_values(values):
    if len(values) < 2 or any(not isinstance(value, int) or not 0 <= value <= 9
                              for value in values):
        raise ValueError("Supply at least two integer digits in [0, 9].")


def reasoning_example(values, scratchpad):
    validate_values(values)
    prompt = "+".join(map(str, values)) + "="
    running, steps = values[0], []
    for value in values[1:]:
        running = (running + value) % 10
        steps.append(running)
    response = ",".join(map(str, steps if scratchpad else steps[-1:])) + "\n"
    return prompt, response


def split_cases(seed=2026):
    cases = list(itertools.product(range(10), repeat=3))
    random.Random(seed).shuffle(cases)
    return dict(train=cases[:800], validation=cases[800:900], test=cases[900:])


def response_batch(cases, scratchpad):
    if not cases:
        raise ValueError("A supervised batch cannot be empty.")
    examples = [reasoning_example(case, scratchpad) for case in cases]
    length = max(len(encode(prompt + response)) - 1 for prompt, response in examples)
    x = torch.zeros((len(cases), length), dtype=torch.long)
    y = torch.full_like(x, -100)
    for row, (prompt, response) in enumerate(examples):
        tokens = encode(prompt + response)
        x[row, :len(tokens) - 1] = torch.tensor(tokens[:-1])
        # The final prompt token predicts the first response token.
        start = len(encode(prompt)) - 1
        y[row, start:start + len(encode(response))] = torch.tensor(encode(response))
    return x, y


@torch.no_grad()
def greedy_responses(model, cases, scratchpad):
    prompts = [reasoning_example(case, scratchpad)[0] for case in cases]
    if not prompts or len({len(prompt) for prompt in prompts}) != 1:
        raise ValueError("Greedy evaluation needs nonempty, equal-length prompts.")
    count = 2 * (len(cases[0]) - 1) + 2 if scratchpad else 4
    if len(prompts[0]) + count > model.config.context:
        raise ValueError("Prompt and response exceed this demonstration's context.")
    generated = model.generate(torch.tensor([encode(prompt) for prompt in prompts]),
                               count=count, top_k=1)
    responses = []
    for tokens in generated[:, len(prompts[0]):].tolist():
        text = decode(tokens)
        end = text.find("\n")
        responses.append(text[:end + 1] if end >= 0 else text)
    return responses


def score_response(values, response, scratchpad):
    _, expected = reasoning_example(values, scratchpad)
    count = len(values) - 1 if scratchpad else 1
    valid = re.fullmatch(r"[0-9](?:,[0-9])*\n", response) is not None
    predicted = [int(value) for value in response.strip().split(",")] if valid else []
    valid = valid and len(predicted) == count
    answer = valid and predicted[-1] == sum(values) % 10
    verified = []
    if scratchpad:
        previous = values[0]
        for index, value in enumerate(values[1:]):
            verified.append(valid and predicted[index] == (previous + value) % 10)
            if valid:
                previous = predicted[index]
    return dict(format_correct=valid, answer_correct=answer,
                trace_correct=all(verified) if scratchpad else None,
                step_accuracy=sum(verified) / count if scratchpad else None,
                expected=expected, generated=response)


def reasoning_metrics(model, cases, scratchpad):
    responses = greedy_responses(model, cases, scratchpad)
    samples = [dict(values=list(case), prompt=reasoning_example(case, scratchpad)[0],
                    **score_response(case, response, scratchpad))
               for case, response in zip(cases, responses)]
    result = dict(cases=len(cases), samples=samples)
    for key in ["format_correct", "answer_correct", "trace_correct", "step_accuracy"]:
        result[key] = (sum(sample[key] for sample in samples) / len(samples)
                       if scratchpad or key not in ["trace_correct", "step_accuracy"] else None)
    return result


def decision_tensors(cases):
    return (torch.tensor(cases, dtype=torch.float32) / 9,
            torch.tensor([decision_label(case) for case in cases]))


@torch.no_grad()
def decision_metrics(model, cases):
    x, labels = decision_tensors(cases)
    choices = model(x).argmax(-1).tolist()
    samples = [
        dict(values=list(case), action=ACTIONS[action], expected=ACTIONS[label],
             reward=decision_reward(case, action),
             regret=decision_reward(case, label) - decision_reward(case, action))
        for case, action, label in zip(cases, choices, labels.tolist())
    ]
    return dict(cases=len(cases), accuracy=sum(a == b for a, b in zip(choices, labels.tolist())) / len(cases),
                mean_reward=sum(sample["reward"] for sample in samples) / len(cases),
                mean_regret=sum(sample["regret"] for sample in samples) / len(cases),
                samples=samples)


def train_decision(splits, steps):
    torch.manual_seed(42)
    model = DecisionPolicy()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
    train_x, train_y = decision_tensors(splits["train"])
    val_x, val_y = decision_tensors(splits["validation"])
    rng = torch.Generator().manual_seed(42)
    curve, best, best_step = [], float("inf"), 0
    path = DIRECTORY / "decision.pt"
    started = time.perf_counter()
    for step in range(1, steps + 1):
        rows = torch.randint(len(train_x), (64,), generator=rng)
        loss = F.cross_entropy(model(train_x[rows]), train_y[rows])
        update(model, optimizer, loss)
        if step % 200 == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                val_loss = F.cross_entropy(model(val_x), val_y).item()
            curve.append(dict(step=step, validation_loss=val_loss))
            if val_loss < best:
                best, best_step = val_loss, step
                torch.save(dict(model=model.state_dict(), actions=ACTIONS,
                                features=["cost", "benefit", "budget"]), path)
            model.train()
    model.load_state_dict(torch.load(path, weights_only=True)["model"])
    return model.eval(), dict(name="D1", parameters=sum(p.numel() for p in model.parameters()),
                              steps=steps, best_step=best_step, best_validation_loss=best,
                              seconds=time.perf_counter() - started, curve=curve,
                              checkpoint=str(path.relative_to(ROOT)))


def update(model, optimizer, loss):
    if not torch.isfinite(loss):
        raise RuntimeError("Nonfinite supervised training loss.")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not torch.isfinite(norm):
        raise RuntimeError("Nonfinite supervised gradients.")
    optimizer.step()


def train_reasoning(splits, scratchpad, steps):
    torch.manual_seed(42)
    model = GPT(Config(context=32, width=64, heads=4, layers=2))
    initial_hash = model_hash(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    train_x, train_y = response_batch(splits["train"], scratchpad)
    val_x, val_y = response_batch(splits["validation"], scratchpad)
    rng = torch.Generator().manual_seed(42)
    name = "reasoning" if scratchpad else "answer_only"
    path = DIRECTORY / f"{name}.pt"
    curve, best, best_step = [], float("inf"), 0
    started = time.perf_counter()
    for step in range(1, steps + 1):
        rows = torch.randint(len(train_x), (64,), generator=rng)
        _, loss = model(train_x[rows], train_y[rows])
        update(model, optimizer, loss)
        if step % 200 == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                _, val_loss = model(val_x, val_y)
            accuracy = reasoning_metrics(model, splits["validation"], scratchpad)["answer_correct"]
            curve.append(dict(step=step, validation_loss=val_loss.item(), answer_accuracy=accuracy))
            if val_loss.item() < best:
                best, best_step = val_loss.item(), step
                save_model(model, path)
            print(f"{name} {step}/{steps}: val loss={val_loss.item():.4f}, "
                  f"answer accuracy={accuracy:.1%}", flush=True)
            model.train()
    return load_model(path), dict(
        name="R1" if scratchpad else "R0", scratchpad=scratchpad,
        parameters=sum(p.numel() for p in model.parameters()), steps=steps, best_step=best_step,
        best_validation_loss=best, initial_model_sha256=initial_hash, curve=curve,
        seconds=time.perf_counter() - started, checkpoint=str(path.relative_to(ROOT)),
        supervised_response_tokens_per_batch=64 * (4 if scratchpad else 2),
    )


def publish(report):
    (DIRECTORY / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    rows = [
        "# Decision and reasoning teaching models", "",
        "These are separate task demonstrations, not replacements for the Archive-trained V1-V7. "
        "Their data are explicitly synthetic: exact labels let students test decisions and steps.", "",
        "| Model | Parameters | Selected / completed updates | Heldout metric | Result |",
        "|---|---:|---:|---|---:|",
    ]
    for record in report["models"]:
        key = "accuracy" if record["name"] == "D1" else "answer_correct"
        rows.append(f"| {record['name']} | {record['parameters']:,} | "
                    f"{record['best_step']} / {record['steps']} | "
                    f"{'Decision accuracy' if key == 'accuracy' else 'Exact final answer'} | "
                    f"{record['test'][key]:.1%} |")
    decision, answer, reasoning = report["models"]
    rows += [
        "", f"D1 mean heldout regret: {decision['test']['mean_regret']:.3f}. "
        f"Always-wait baseline: {report['decision_baseline']['accuracy']:.1%} accuracy.", "",
        f"R1 fully verified two-step traces: **{reasoning['test']['trace_correct']:.1%}**. "
        f"Valid output format: {reasoning['test']['format_correct']:.1%}.", "",
        "## Longer, unseen computation", "",
        f"On {reasoning['longer_test']['cases']} four-operand problems (three steps, never trained): "
        f"R0 final accuracy {answer['longer_test']['answer_correct']:.1%}; "
        f"R1 final accuracy {reasoning['longer_test']['answer_correct']:.1%}; "
        f"R1 verified full traces {reasoning['longer_test']['trace_correct']:.1%}.", "",
        "## Protocol and limits", "",
        "800/100/100 disjoint input triples, frozen before training. All test generation occurs after "
        "all three checkpoint selections. R0 and R1 share initial GPT weights, training cases, "
        "batch size, optimizer, seed and completed update budget, but R1 has longer responses and "
        "more supervised tokens/compute. Their response losses are not comparable measures of task quality. "
        "Selection uses validation response loss; test uses greedy free generation, never true prior steps. "
        "D1 is supervised imitation of a supplied reward rule, not reinforcement learning, a Decision "
        "Transformer, or a safety-critical policy. R1 is supervised scratchpad computation, not proof "
        "of general reasoning or faithfulness of natural-language explanations. One seed and a small "
        "finite synthetic task do not establish broad reasoning superiority.", "",
        "All individual predictions (including malformed outputs), validation curves, and checkpoint "
        "hashes are in `results.json`. Data partitions are in `dataset.json`.", "",
        "## Fixed heldout examples", "",
    ]
    for record in report["models"]:
        rows += [f"### {record['name']}", "", "```json",
                 json.dumps(record["test"]["samples"][:5], indent=2), "```", ""]
    (DIRECTORY / "RESULTS.md").write_text("\n".join(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--model", choices=["decision", "answer_only", "reasoning"])
    parser.add_argument("--values", nargs="+", type=int)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.steps < 1 or args.threads < 1:
        parser.error("Steps and threads must be positive.")
    if args.train and (args.model or args.values):
        parser.error("--train cannot be combined with inference arguments.")
    torch.set_num_threads(args.threads)
    check_preserved()
    if not args.train:
        if not args.model or args.values is None:
            parser.error("Use --train, or --model NAME --values DIGITS.")
        validate_values(args.values)
        if args.model == "decision":
            if len(args.values) != 3:
                parser.error("Decision features are cost benefit budget, in that order.")
            model = DecisionPolicy()
            model.load_state_dict(torch.load(DIRECTORY / "decision.pt", weights_only=True)["model"])
            with torch.no_grad():
                action = model(torch.tensor([args.values], dtype=torch.float32) / 9).argmax(-1).item()
            print(ACTIONS[action])
        else:
            model = load_model(DIRECTORY / f"{args.model}.pt")
            print(greedy_responses(model, [args.values], args.model == "reasoning")[0], end="")
        return
    DIRECTORY.mkdir(exist_ok=True)
    splits = split_cases()
    longer = list(itertools.product(range(10), repeat=4))
    random.Random(2030).shuffle(longer)
    longer = longer[:200]
    (DIRECTORY / "dataset.json").write_text(json.dumps(dict(
        source="Synthetic, generated by exact public arithmetic/reward rules; not Archive text.",
        seed=2026, splits=splits, longer_test=longer), indent=2) + "\n")
    decision, d_record = train_decision(splits, 2000)
    answer, a_record = train_reasoning(splits, False, args.steps)
    reasoning, r_record = train_reasoning(splits, True, args.steps)
    assert a_record["initial_model_sha256"] == r_record["initial_model_sha256"]
    d_record["test"] = decision_metrics(decision, splits["test"])
    for model, record in [(answer, a_record), (reasoning, r_record)]:
        record["test"] = reasoning_metrics(model, splits["test"], record["scratchpad"])
        record["longer_test"] = reasoning_metrics(model, longer, record["scratchpad"])
    for record in [d_record, a_record, r_record]:
        record["checkpoint_sha256"] = hashlib.sha256((ROOT / record["checkpoint"]).read_bytes()).hexdigest()
    baseline = sum(decision_label(case) == 0 for case in splits["test"]) / len(splits["test"])
    publish(dict(seed=42, split_seed=2026, models=[d_record, a_record, r_record],
                 decision_baseline=dict(name="always_wait", accuracy=baseline),
                 dataset_sha256=hashlib.sha256((DIRECTORY / "dataset.json").read_bytes()).hexdigest()))
    check_preserved()
    print((DIRECTORY / "RESULTS.md").read_text().split("## Fixed heldout examples")[0])


if __name__ == "__main__":
    main()
