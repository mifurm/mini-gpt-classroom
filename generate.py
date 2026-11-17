"""Generate a continuation from a locally trained checkpoint."""

import argparse

import torch

from data import ROOT
from model import decode, encode, load_model
from train import choose_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--checkpoint")
    source.add_argument("--version", choices=[f"v{i}" for i in range(1, 8)])
    parser.add_argument("--prompt", default='Alice said, "')
    parser.add_argument("--tokens", type=int, default=240)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = choose_device(args.device)
    torch.set_num_threads(4)
    checkpoint = args.checkpoint or (
        ROOT / "artifacts" / "versions" / f"{args.version}.pt"
        if args.version else ROOT / "artifacts" / "final.pt"
    )
    model = load_model(checkpoint, device)
    tokens = torch.tensor([encode(args.prompt)], dtype=torch.long, device=device)
    generated = model.generate(tokens, args.tokens, args.temperature, args.top_k, args.seed)
    print(decode(generated[0].cpu().tolist()))


if __name__ == "__main__":
    main()
