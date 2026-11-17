import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from concept_models import (DecisionPolicy, decision_label, decision_reward, greedy_responses, reasoning_example,
                            response_batch, score_response, split_cases)
from model import Config, GPT


class ConceptTests(unittest.TestCase):
    def test_decision_reward_feasibility_and_ties(self):
        self.assertEqual(decision_label((3, 8, 5)), 1)
        self.assertEqual(decision_label((6, 8, 5)), 0)
        self.assertEqual(decision_label((3, 3, 5)), 0)
        self.assertEqual(decision_reward((6, 8, 5), 1), -10)
        self.assertEqual(decision_reward((3, 8, 5), 1), 5)
        with self.assertRaises(ValueError):
            decision_reward((3, 8, 5), 2)

    def test_splits_hold_out_input_identities(self):
        splits = split_cases()
        self.assertEqual(splits, split_cases())
        sets = [set(cases) for cases in splits.values()]
        self.assertEqual([len(cases) for cases in sets], [800, 100, 100])
        self.assertEqual(len(set.union(*sets)), 1000)
        for first in range(3):
            for second in range(first + 1, 3):
                self.assertFalse(sets[first] & sets[second])

    def test_response_only_shift_and_padding(self):
        x, y = response_batch([(2, 3, 4), (9, 9)], True)
        self.assertEqual(reasoning_example((2, 3, 4), True), ("cap(2,3,4)=", "3,3\n"))
        self.assertTrue((y[0, :10] == -100).all())
        self.assertEqual(y[0, 10:].tolist(), list(b"3,3\n"))
        self.assertEqual(x[0, 10].item(), ord("="))
        self.assertTrue((y[1, 10:] == -100).all())
        logits = torch.randn(2, x.shape[1], 256, requires_grad=True)
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        loss.backward()
        self.assertTrue((logits.grad[y == -100] == 0).all())
        self.assertGreater(logits.grad[y != -100].abs().sum().item(), 0)

    def test_verifier_distinguishes_answer_from_steps(self):
        correct = score_response((2, 3, 4), "3,3\n", True)
        self.assertTrue(correct["trace_correct"])
        wrong_trace = score_response((8, 2, 5), "9,5\n", True)
        self.assertTrue(wrong_trace["answer_correct"])
        self.assertFalse(wrong_trace["trace_correct"])
        consistent_wrong = score_response((8, 2, 9), "7,7\n", True)
        self.assertFalse(consistent_wrong["answer_correct"])
        self.assertEqual(consistent_wrong["step_accuracy"], 0.5)
        for text in ["3,3", "3,3\njunk", "x,3\n", "3\n", "3,3,0\n"]:
            with self.subTest(text=text):
                result = score_response((2, 3, 4), text, True)
                self.assertFalse(result["format_correct"])
                self.assertFalse(result["answer_correct"])
        self.assertIsNone(score_response((2, 3, 4), "3\n", False)["trace_correct"])

    def test_supervised_models_receive_gradients(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)
        policy = DecisionPolicy()
        F.cross_entropy(policy(torch.tensor([[3, 8, 5]], dtype=torch.float32) / 9),
                        torch.tensor([1])).backward()
        self.assertGreater(policy.network[0].weight.grad.abs().sum().item(), 0)
        model = GPT(Config(context=32, width=32, layers=2))
        x, y = response_batch([(2, 3, 4)], True)
        _, loss = model(x, y)
        loss.backward()
        self.assertTrue(torch.isfinite(model.token.weight.grad).all())
        self.assertGreater(model.token.weight.grad.abs().sum().item(), 0)

    def test_generation_receives_no_reference_steps(self):
        model = GPT(Config(context=32, width=32, layers=1))

        def generate(tokens, count, top_k):
            self.assertEqual(tokens.tolist(), [list(b"cap(8,2,5)=")])
            self.assertEqual(top_k, 1)
            self.assertEqual(count, 6)
            return torch.cat((tokens, torch.tensor([list(b"9,5\n??")])), dim=1)

        with patch.object(model, "generate", side_effect=generate):
            response = greedy_responses(model, [(8, 2, 5)], True)[0]
        self.assertEqual(response, "9,5\n")
        self.assertTrue(score_response((8, 2, 5), response, True)["answer_correct"])
        self.assertFalse(score_response((8, 2, 5), response, True)["trace_correct"])

    def test_invalid_examples_fail_explicitly(self):
        for values in [(), (1,), (1, 10), (-1, 3), (True, 3)]:
            with self.assertRaises(ValueError):
                reasoning_example(values, True)
        with self.assertRaises(ValueError):
            response_batch([], True)


if __name__ == "__main__":
    unittest.main()
