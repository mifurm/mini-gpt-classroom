from dataclasses import asdict
import tempfile
from pathlib import Path
import unittest

import torch
from torch.nn import functional as F

from data import clean_story, split_bytes
from model import Config, GPT, decode, encode, load_model, save_model
from train import batch, eval_positions, evaluate
from build_microscope import build_trace


class GPTTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)
        self.model = GPT(Config(context=16, width=32, heads=4, layers=2))

    def test_causal_prefix_is_unchanged_by_future_tokens(self):
        first = torch.randint(256, (2, 16))
        second = first.clone()
        second[:, 8:] = (second[:, 8:] + 17) % 256
        a, _ = self.model(first)
        b, _ = self.model(second)
        torch.testing.assert_close(a[:, :8], b[:, :8], rtol=0, atol=0)
        self.assertFalse(torch.allclose(a[:, 8:], b[:, 8:]))

    def test_shift_and_no_out_of_split_targets(self):
        data = torch.arange(200)
        x, y = batch(data, 16, 32, torch.Generator().manual_seed(0), "cpu")
        self.assertTrue(torch.equal(y, x + 1))
        self.assertLess(int(y.max()), len(data))

    def test_loss_backward_and_tied_projection(self):
        x = torch.randint(256, (2, 12))
        y = torch.randint(256, (2, 12))
        logits, loss = self.model(x, y)
        self.assertEqual(logits.shape, (2, 12, 256))
        torch.testing.assert_close(loss, F.cross_entropy(logits.reshape(-1, 256),
                                                       y.reshape(-1)))
        loss.backward()
        self.assertTrue(torch.isfinite(self.model.token.weight.grad).all())
        self.assertGreater(self.model.token.weight.grad.abs().sum().item(), 0)
        embedding_parameters = [
            name for name, value in self.model.named_parameters()
            if value.shape == (256, 32)
        ]
        self.assertEqual(embedding_parameters, ["token.weight"])

    def test_checkpoint_roundtrip_and_generation(self):
        x = torch.tensor([encode("Alice")])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.pt"
            save_model(self.model, path)
            restored = load_model(path)
            torch.testing.assert_close(self.model(x)[0], restored(x)[0])
            a = restored.generate(x, count=20, seed=5)
            b = restored.generate(x, count=20, seed=5)
            self.assertTrue(torch.equal(a, b))
            self.assertEqual(a.shape, (1, 25))
            self.assertTrue(torch.equal(a[:, :5], x))

    def test_bytes_and_validation_errors(self):
        self.assertEqual(decode(encode("Alice, café — ♟")), "Alice, café — ♟")
        self.assertEqual(decode([255]), "\ufffd")
        with self.assertRaises(ValueError):
            Config(width=33, heads=4)
        for value in [-0.1, 1.0, float("nan")]:
            with self.assertRaises(ValueError):
                Config(dropout=value)
        with self.assertRaises(ValueError):
            self.model(torch.zeros((1, 17), dtype=torch.long))
        with self.assertRaises(ValueError):
            self.model.generate(torch.ones((1, 1), dtype=torch.long), temperature=0)

    def test_disjoint_splits_and_story_cleaning(self):
        content = bytes(range(256)) * 100
        splits, spans = split_bytes(content)
        self.assertGreaterEqual(spans["validation"][0] - spans["train"][1], 128)
        self.assertGreaterEqual(spans["test"][0] - spans["validation"][1], 128)
        for name, (start, end) in spans.items():
            self.assertEqual(splits[name], content[start:end])
        raw = ("Library labels\nCHAPTER  I.\nLOOKING-GLASS HOUSE.\n\n"
               "12\nLOOKING-GLASS HOUSE. 13\nThe kit-\nten sat.\nTHE END.\nLabels")
        cleaned = clean_story(raw)
        self.assertNotIn("Library", cleaned)
        self.assertNotIn("13", cleaned)
        self.assertIn("The kitten sat.", cleaned)
        self.assertTrue(cleaned.endswith("THE END.\n"))
        with self.assertRaises(ValueError):
            clean_story("No reliable boundaries")

    def test_evaluation_targets_fixed_and_modes_restored(self):
        data = torch.tensor(list(bytes(range(256)) * 3), dtype=torch.long)
        positions = eval_positions(data, count=128)
        self.assertEqual(len(positions.unique()), 128)
        self.assertTrue(torch.equal(positions, eval_positions(data, count=128)))
        metrics = evaluate(self.model, data, positions, "cpu")
        self.assertEqual(metrics["targets"], 128)
        self.assertTrue(self.model.training)
        self.assertGreater(metrics["perplexity"], 1)

    def test_can_overfit_a_tiny_sequence(self):
        model = GPT(Config(context=8, width=32, heads=4, layers=1))
        x = torch.tensor([[65, 66, 65, 66, 65, 66, 65, 66]])
        y = torch.tensor([[66, 65, 66, 65, 66, 65, 66, 65]])
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
        initial = model(x, y)[1].item()
        for _ in range(100):
            optimizer.zero_grad()
            loss = model(x, y)[1]
            loss.backward()
            optimizer.step()
        self.assertLess(model(x, y)[1].item(), initial * 0.1)
        self.assertTrue(torch.equal(model(x)[0].argmax(-1), y))

    def test_legacy_checkpoint_without_dropout_is_compatible(self):
        config = asdict(self.model.config)
        del config["dropout"]
        x = torch.tensor([encode("Alice")])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            torch.save({"config": config, "model": self.model.state_dict()}, path)
            restored = load_model(path)
            self.assertEqual(restored.config.dropout, 0.0)
            torch.testing.assert_close(self.model(x)[0], restored(x)[0], rtol=0, atol=0)

    def test_dropout_train_eval_causality_and_generation_modes(self):
        model = GPT(Config(context=16, width=32, layers=3, dropout=0.1))
        x = torch.randint(256, (2, 16))
        a, _ = model(x)
        b, _ = model(x)
        self.assertFalse(torch.equal(a, b))
        model.eval()
        a, _ = model(x)
        b, _ = model(x)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        changed = x.clone()
        changed[:, 8:] = (changed[:, 8:] + 1) % 256
        torch.testing.assert_close(a[:, :8], model(changed)[0][:, :8], rtol=0, atol=0)
        model.train()
        first = model.generate(x[:1], count=8, seed=3)
        self.assertTrue(model.training)
        second = model.generate(x[:1], count=8, seed=3)
        self.assertTrue(torch.equal(first, second))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dropout.pt"
            save_model(model, path)
            restored = load_model(path)
            self.assertEqual(restored.config.dropout, 0.1)
            torch.testing.assert_close(restored(x)[0], model.eval()(x)[0])


class MicroscopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.trace = build_trace()

    def test_forward_trace_matches_real_model_and_causal_attention(self):
        from data import ROOT
        model = load_model(ROOT / "artifacts/final.pt")
        trace = self.trace["forward"]
        tokens = torch.tensor([trace["ids"]])
        with torch.no_grad():
            logits, _ = model(tokens)
        torch.testing.assert_close(torch.tensor(trace["logits"]), logits[0], rtol=0, atol=0)
        probabilities = torch.tensor(trace["probabilities"])
        torch.testing.assert_close(probabilities, logits[0].softmax(-1), rtol=0, atol=0)
        self.assertEqual(self.trace["tokenizer"]["ids"], [99, 97, 102, 195, 169])
        self.assertEqual(trace["causality"]["earlier_max_logit_difference"], 0)
        for block in trace["blocks"]:
            attention = torch.tensor(block["attention"])
            self.assertEqual(int(torch.count_nonzero(torch.triu(attention, diagonal=1))), 0)
            torch.testing.assert_close(attention.sum(-1), torch.ones_like(attention.sum(-1)))
        token = torch.tensor(trace["token_embeddings"]["first_four"])[0]
        position = torch.tensor(trace["position_embeddings"]["first_four"])
        combined = torch.tensor(trace["combined_embeddings"]["first_four"])[0]
        torch.testing.assert_close(token + position, combined)

    def test_training_records_shifted_loss_and_real_adamw_update(self):
        import math
        trace = self.trace["training"]
        self.assertEqual(trace["input_ids"][1:], trace["target_ids"][:-1])
        for phase in ["before", "after"]:
            losses = [position[f"loss_{phase}"] for position in trace["positions"]]
            self.assertAlmostEqual(sum(losses) / len(losses), trace[f"loss_{phase}"], places=6)
            for position in trace["positions"]:
                self.assertAlmostEqual(-math.log(position[f"probability_{phase}"]),
                                       position[f"loss_{phase}"], places=5)
        weight, optimizer = trace["weight"], trace["optimizer"]
        beta1, beta2 = optimizer["betas"]
        first = weight["first_moment"] / (1 - beta1)
        second = weight["second_moment"] / (1 - beta2)
        expected = weight["before"] * (1 - optimizer["learning_rate"] * optimizer["weight_decay"])
        expected -= optimizer["learning_rate"] * first / (math.sqrt(second) + optimizer["epsilon"])
        self.assertAlmostEqual(weight["after"], expected, places=6)
        self.assertNotEqual(weight["before"], weight["after"])
        self.assertGreater(trace["gradient_norm"], 0)

    def test_generation_is_actual_sampling_from_unchanged_checkpoint(self):
        from data import ROOT
        from build_microscope import GENERATION
        trace = self.trace["generation"]
        model = load_model(ROOT / "artifacts/final.pt")
        original = {name: value.clone() for name, value in model.state_dict().items()}
        tokens = torch.tensor([encode(trace["prompt"])])
        self.assertEqual(model.generate(tokens, **GENERATION)[0].tolist(), trace["ids"])
        self.assertEqual(trace["steps"][-1]["text_after"], trace["text"])
        for step in trace["steps"]:
            with torch.no_grad():
                logits = model(torch.tensor([step["context_ids"]]))[0][0, -1] / trace["temperature"]
            logits = logits.masked_fill(logits < logits.topk(trace["top_k"]).values[-1], float("-inf"))
            self.assertEqual(float(logits.softmax(-1)[step["sampled_id"]]), step["sampled_probability"])
        self.assertTrue(all(torch.equal(value, original[name]) for name, value in model.state_dict().items()))


if __name__ == "__main__":
    unittest.main()
