"""Record real GPT forward, learning, and generation steps for the static lesson."""

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json

import torch
from torch import nn
from torch.nn import functional as F

from data import ROOT
from model import GPT, decode, encode, load_model


PROMPT = "Alice "
TEACHING_TEXT = "Alice sees a cat."
GENERATION = dict(count=12, temperature=0.8, top_k=40, seed=123)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preview(tensor: torch.Tensor) -> dict:
    return dict(shape=list(tensor.shape), first_four=tensor.detach()[..., :4].tolist())


def candidates(probabilities: torch.Tensor, count=5) -> list[dict]:
    values, ids = probabilities.topk(count)
    return [dict(id=token, probability=value) for token, value in zip(ids.tolist(), values.tolist())]


@torch.no_grad()
def record_forward(model: GPT, tokens: torch.Tensor) -> dict:
    captured: dict[str, torch.Tensor] = {}
    handles = []

    def capture(name):
        def hook(_module: nn.Module, _inputs, output: torch.Tensor):
            captured[name] = output.detach().clone()
        return hook

    modules = dict(token=model.token, position=model.position, embedded=model.embedding_dropout,
                   normalized=model.final_norm)
    for index, block in enumerate(model.blocks):
        modules.update({
            f"attention_{index}": block.attention.attention_dropout,
            f"attention_output_{index}": block.attention,
            f"mlp_{index}": block.mlp,
            f"block_{index}": block,
        })
    try:
        for name, module in modules.items():
            handles.append(module.register_forward_hook(capture(name)))
        logits, _ = model(tokens)
    finally:
        for handle in handles:
            handle.remove()
    probabilities = logits.softmax(-1)[0]
    blocks = []
    for index in range(model.config.layers):
        incoming = captured["embedded"] if index == 0 else captured[f"block_{index - 1}"]
        after_attention = incoming + captured[f"attention_output_{index}"]
        blocks.append(dict(
            attention=captured[f"attention_{index}"][0].tolist(),
            attention_output=preview(captured[f"attention_output_{index}"]),
            after_attention=preview(after_attention),
            mlp_output=preview(captured[f"mlp_{index}"]),
            output=preview(captured[f"block_{index}"]),
        ))
    changed = tokens.clone()
    changed[0, -1] = (changed[0, -1] + 1) % 256
    altered_logits, _ = model(changed)
    return dict(
        text=decode(tokens[0].tolist()), ids=tokens[0].tolist(),
        token_embeddings=preview(captured["token"]),
        position_embeddings=preview(captured["position"]),
        combined_embeddings=preview(captured["embedded"]),
        blocks=blocks, normalized=preview(captured["normalized"]),
        logits=logits[0].tolist(), probabilities=probabilities.tolist(),
        logits_shape=list(logits.shape),
        causality=dict(changed_position=tokens.shape[1] - 1,
                       original_id=int(tokens[0, -1]), changed_id=int(changed[0, -1]),
                       earlier_max_logit_difference=float(
                           (logits[:, :-1] - altered_logits[:, :-1]).abs().max())),
    )


def record_training(model: GPT, tokens: torch.Tensor) -> dict:
    learner = deepcopy(model)
    learner.train()
    targets = torch.tensor([encode(TEACHING_TEXT)[1:tokens.shape[1] + 1]], dtype=torch.long)
    optimizer = torch.optim.AdamW(learner.parameters(), lr=0.001, weight_decay=0.01)
    before_logits, loss = learner(tokens, targets)
    if loss is None:
        raise RuntimeError("The supervised forward pass did not return a loss.")
    before_losses = F.cross_entropy(before_logits[0], targets[0], reduction="none")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient = learner.token.weight.grad
    if gradient is None or not torch.isfinite(gradient).all():
        raise RuntimeError("The token embedding gradient is missing or nonfinite.")
    flat_index = int(gradient.abs().argmax())
    token_id, feature = divmod(flat_index, learner.config.width)
    parameter = learner.token.weight
    before_weight = float(parameter.detach()[token_id, feature])
    selected_gradient = float(gradient[token_id, feature])
    gradient_norm = sum(float(p.grad.square().sum()) for p in learner.parameters()
                        if p.grad is not None) ** 0.5
    optimizer.step()
    learner.eval()
    with torch.no_grad():
        after_logits, after_loss = learner(tokens, targets)
        after_losses = F.cross_entropy(after_logits[0], targets[0], reduction="none")
    if after_loss is None:
        raise RuntimeError("The updated supervised forward pass did not return a loss.")
    state = optimizer.state[parameter]
    settings = optimizer.param_groups[0]
    return dict(
        synthetic_example=TEACHING_TEXT,
        window=TEACHING_TEXT[:tokens.shape[1] + 1],
        input_ids=tokens[0].tolist(), target_ids=targets[0].tolist(),
        batch_size=1, updates=1, loss_before=float(loss.detach()), loss_after=float(after_loss),
        gradient_norm=gradient_norm,
        positions=[dict(
            input_id=int(tokens[0, position]), target_id=int(targets[0, position]),
            probability_before=float(before_logits.detach().softmax(-1)[0, position, target]),
            probability_after=float(after_logits.softmax(-1)[0, position, target]),
            loss_before=float(before_losses.detach()[position]),
            loss_after=float(after_losses[position]),
        ) for position, target in enumerate(targets[0].tolist())],
        optimizer=dict(name="AdamW", learning_rate=settings["lr"], weight_decay=settings["weight_decay"],
                       betas=list(settings["betas"]), epsilon=settings["eps"]),
        weight=dict(name="token.weight", token_id=token_id, feature=feature,
                    before=before_weight, gradient=selected_gradient,
                    after=float(parameter.detach()[token_id, feature]),
                    first_moment=float(state["exp_avg"][token_id, feature]),
                    second_moment=float(state["exp_avg_sq"][token_id, feature])),
    )


@torch.no_grad()
def record_generation(model: GPT, tokens: torch.Tensor) -> dict:
    logits_by_step = []

    def capture(_module: nn.Module, _inputs, output):
        logits_by_step.append(output[0][0, -1].detach().clone())

    handle = model.register_forward_hook(capture)
    try:
        generated = model.generate(tokens, **GENERATION)[0].tolist()
    finally:
        handle.remove()
    steps = []
    for index, logits in enumerate(logits_by_step):
        scaled = logits.float() / GENERATION["temperature"]
        scaled = scaled.masked_fill(scaled < scaled.topk(GENERATION["top_k"]).values[-1], float("-inf"))
        probabilities = scaled.softmax(-1)
        prefix = generated[:tokens.shape[1] + index]
        chosen = generated[tokens.shape[1] + index]
        steps.append(dict(
            prefix=decode(prefix), context_ids=prefix[-model.config.context:],
            sampled_id=chosen, sampled_probability=float(probabilities[chosen]),
            candidates=candidates(probabilities), text_after=decode(prefix + [chosen]),
        ))
    return dict(**GENERATION, prompt=PROMPT, ids=generated, text=decode(generated), steps=steps)


def build_trace() -> dict:
    checkpoint = ROOT / "artifacts/final.pt"
    initial_hash = sha256(checkpoint)
    model = load_model(checkpoint, "cpu")
    if model.config.dropout != 0:
        raise ValueError("This single-step lesson requires the original zero-dropout checkpoint.")
    original_weights = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    tokens = torch.tensor([encode(PROMPT)], dtype=torch.long)
    result = dict(
        schema_version=1,
        provenance=dict(
            checkpoint="artifacts/final.pt", checkpoint_sha256=initial_hash,
            source_hashes={name: sha256(ROOT / name) for name in ["model.py", "build_microscope.py"]},
            torch_version=torch.__version__, device="cpu", config=asdict(model.config),
            parameters=sum(p.numel() for p in model.parameters()),
        ),
        tokenizer=dict(text="caf\u00e9", ids=encode("caf\u00e9"), vocabulary_size=256),
        forward=record_forward(model, tokens),
        training=record_training(model, tokens),
        generation=record_generation(model, tokens),
    )
    if sha256(checkpoint) != initial_hash or any(
        not torch.equal(tensor, original_weights[name]) for name, tensor in model.state_dict().items()
    ):
        raise RuntimeError("The microscope must not change the original checkpoint or inference model.")
    return result


if __name__ == "__main__":
    torch.set_num_threads(2)
    trace = build_trace()
    destination = ROOT / "artifacts/microscope.json"
    destination.write_text(json.dumps(trace, indent=2, allow_nan=False) + "\n")
    print(f"Recorded real forward pass, one disposable update and "
          f"{len(trace['generation']['steps'])} generation steps in {destination.relative_to(ROOT)}.")
