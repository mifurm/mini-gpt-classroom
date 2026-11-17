"""A small GPT-2-style decoder. Tokens are UTF-8 bytes, not GPT-2 BPE."""

from dataclasses import asdict, dataclass, replace
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
    context: int = 64
    width: int = 128
    heads: int = 4
    layers: int = 2
    dropout: float = 0.0

    def __post_init__(self):
        if min(self.context, self.width, self.heads, self.layers) < 1:
            raise ValueError("All dimensions must be positive.")
        if self.width % self.heads:
            raise ValueError("Embedding width must be divisible by head count.")
        if not 0 <= self.dropout < 1:
            raise ValueError("Dropout probability must be in [0, 1).")


def encode(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def decode(tokens: list[int]) -> str:
    # An unfinished or invalid UTF-8 sequence is visible as a replacement character.
    return bytes(tokens).decode("utf-8", errors="replace")


class CausalAttention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.heads = config.heads
        self.qkv = nn.Linear(config.width, 3 * config.width)
        self.project = nn.Linear(config.width, config.width)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.output_dropout = nn.Dropout(config.dropout)
        mask = torch.tril(torch.ones(config.context, config.context, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [
            t.view(batch, length, self.heads, width // self.heads).transpose(1, 2)
            for t in (q, k, v)
        ]
        scores = q @ k.transpose(-2, -1) / math.sqrt(width // self.heads)
        scores = scores.masked_fill(~self.mask[:length, :length], float("-inf"))
        weights = self.attention_dropout(scores.softmax(dim=-1))
        attended = (weights @ v).transpose(1, 2).contiguous().view(batch, length, width)
        return self.output_dropout(self.project(attended))


class Block(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.width)
        self.attention = CausalAttention(config)
        self.ln2 = nn.LayerNorm(config.width)
        self.mlp = nn.Sequential(
            nn.Linear(config.width, 4 * config.width),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * config.width, config.width),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.token = nn.Embedding(256, config.width)
        self.position = nn.Embedding(config.context, config.width)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*(Block(config) for _ in range(config.layers)))
        self.final_norm = nn.LayerNorm(config.width)
        self.apply(self._initialize)
        # Scale residual projections with depth to keep residual variance controlled.
        for block in self.blocks:
            nn.init.normal_(block.attention.project.weight,
                            std=0.02 / math.sqrt(2 * config.layers))
            nn.init.normal_(block.mlp[2].weight,
                            std=0.02 / math.sqrt(2 * config.layers))

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.bias)

    def forward(self, tokens, targets=None):
        if tokens.ndim != 2 or not 1 <= tokens.shape[1] <= self.config.context:
            raise ValueError("Input must have shape [batch, 1..context].")
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.embedding_dropout(self.token(tokens) + self.position(positions))
        x = self.final_norm(self.blocks(x))
        # F.linear uses the very same matrix as the input token embedding.
        logits = F.linear(x, self.token.weight)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, 256), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, tokens, count=200, temperature=0.8, top_k=40, seed=123):
        if temperature <= 0 or not 1 <= top_k <= 256 or count < 0:
            raise ValueError("Require temperature > 0, 1 <= top_k <= 256, count >= 0.")
        if tokens.ndim != 2 or tokens.shape[1] == 0:
            raise ValueError("Generation requires a nonempty batched prompt.")
        training = self.training
        self.eval()
        rng = torch.Generator(device="cpu").manual_seed(seed)
        try:
            for _ in range(count):
                logits, _ = self(tokens[:, -self.config.context:])
                logits = logits[:, -1].float() / temperature
                cutoff = logits.topk(top_k).values[:, -1:]
                logits = logits.masked_fill(logits < cutoff, float("-inf"))
                # CPU sampling gives a reproducible RNG on CPU and Apple MPS.
                next_token = torch.multinomial(logits.softmax(-1).cpu(), 1, generator=rng)
                tokens = torch.cat((tokens, next_token.to(tokens.device)), dim=1)
        finally:
            self.train(training)
        return tokens


def save_model(model, path):
    torch.save({"config": asdict(model.config), "model": model.state_dict()}, path)


def load_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = GPT(Config(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()


def grow_depth(parent, layers):
    if layers < parent.config.layers:
        raise ValueError("Depth growth cannot remove existing blocks.")
    model = GPT(replace(parent.config, layers=layers)).to(parent.token.weight.device)
    state = model.state_dict()
    state.update(parent.state_dict())
    model.load_state_dict(state)
    # New residual branches start as identity, preserving the parent's inference.
    for block in model.blocks[parent.config.layers:]:
        nn.init.zeros_(block.attention.project.weight)
        nn.init.zeros_(block.mlp[2].weight)
    return model
