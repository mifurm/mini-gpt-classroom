"""Verify frozen data, experiment completeness, delivered deck, and selected model."""

from html.parser import HTMLParser
import json
import math
import re

import torch

from data import ROOT, load_split
from model import encode, load_model
from train import ROUNDS, eval_positions, evaluate


class DeckParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.slides = 0
        self.notes = 0
        self.ids = set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "section" and "slide" in attrs.get("class", "").split():
            self.slides += 1
        if tag == "aside" and attrs.get("class") == "notes":
            self.notes += 1
        if "id" in attrs:
            if attrs["id"] in self.ids:
                raise ValueError(f"Duplicate HTML id: {attrs['id']}")
            self.ids.add(attrs["id"])


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    torch.set_num_threads(4)
    for name in ["train", "validation", "test"]:
        load_split(name)
    report = json.loads((ROOT / "artifacts" / "results.json").read_text())
    rounds = report["rounds"]
    require([r["name"] for r in rounds] == [r["name"] for r in ROUNDS],
            "Baseline plus five recorded rounds are required.")
    for result in rounds:
        for key in ["loss", "bits_per_byte", "perplexity", "accuracy"]:
            require(math.isfinite(result["validation"][key]), f"Nonfinite {key}")
        require(result["curve"][-1]["step"] == result["steps"], "Run did not finish.")
        require(len(result["samples"]) == 3, "Missing uncurated samples.")
        path = ROOT / "artifacts" / result["checkpoint"]
        require(path.exists(), f"Missing checkpoint {path}")
        model = load_model(path)
        require(sum(p.numel() for p in model.parameters()) == result["parameters"],
                "Parameter count mismatch.")
    selected = min(rounds, key=lambda r: r["validation"]["loss"])
    require(report["selected"] == selected["name"], "Selection did not use validation.")
    final = load_model(ROOT / "artifacts" / "final.pt")
    winner = load_model(ROOT / "artifacts" / selected["checkpoint"])
    for key, tensor in final.state_dict().items():
        require(torch.equal(tensor, winner.state_dict()[key]), f"Final differs at {key}")
    test = torch.tensor(list(load_split("test")), dtype=torch.long)
    positions = eval_positions(test, count=len(test) - 128, seed=2027)
    metrics = evaluate(final, test, positions, "cpu")
    for key, value in metrics.items():
        require(abs(value - report["final_test"][key]) < 1e-6, f"Test {key} mismatch")
    require(metrics["loss"] < report["unigram_test"]["loss"], "GPT fails unigram baseline.")
    require(selected["validation"]["loss"] < rounds[0]["validation"]["loss"],
            "Final validation fails initial baseline.")
    prompt = torch.tensor([encode("The little ")], dtype=torch.long)
    tokens = final.generate(prompt, count=240, seed=125)[0].tolist()
    require(tokens[:len(encode("The little "))] == encode("The little "),
            "Generation did not preserve prompt.")
    from model import decode
    expected = selected["samples"][2]["prompt"] + selected["samples"][2]["continuation"]
    require(decode(tokens) == expected, "Recorded sample is not reproducible.")
    parser = DeckParser()
    parser.feed((ROOT / "slides.html").read_text())
    require(parser.slides == 17 and parser.notes == 17, "Incomplete teaching deck.")
    pdf = (ROOT / "slides.pdf").read_bytes()
    require(pdf.startswith(b"%PDF"), "Invalid PDF deck.")
    require(len(re.findall(rb"/Type\s*/Page\b", pdf)) == 17, "PDF page count mismatch.")
    for name in ["PLAN.md", "README.md", "RESULTS.md"]:
        require((ROOT / name).stat().st_size > 500, f"Missing/empty {name}")
    print(f"Verified: frozen splits; six complete runs; {parser.slides}-slide HTML/PDF deck; "
          "all speaker notes; final checkpoint; reproducible sample; full test metrics.")
    print(f"Final test: {metrics['bits_per_byte']:.3f} bits/byte, "
          f"{metrics['accuracy']:.1%} accuracy on {metrics['targets']:,} targets.")


if __name__ == "__main__":
    main()
