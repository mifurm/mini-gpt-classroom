from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import expanded_data
import fivefold_data
from expanded_data import checksum, extract_story, overlap_audit, spans_for
from fivefold_data import exact_audit, verify_rights, version_corpora
from fivefold_train import weighted_batch
from model import Config, GPT, grow_depth
from train_expanded import combine, document_batch


class ExpandedTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)

    def test_training_windows_never_cross_documents(self):
        documents = [torch.full((1000,), value, dtype=torch.long) for value in [11, 101, 201]]
        x, y, chosen = document_batch(documents, 128, torch.Generator().manual_seed(7), 128)
        self.assertEqual(chosen.shape, (128,))
        self.assertEqual(set(chosen.tolist()), {0, 1, 2})
        for row, label in enumerate(chosen.tolist()):
            self.assertTrue(torch.equal(x[row], documents[label][:128]))
            self.assertTrue(torch.equal(y[row], documents[label][:128]))
        with self.assertRaises(ValueError):
            document_batch([torch.ones(50), torch.ones(50)], 128, torch.Generator())

    def test_depth_growth_preserves_function_and_new_branch_can_learn(self):
        parent = GPT(Config(context=16, width=32, layers=2, dropout=0.1)).eval()
        x = torch.randint(256, (2, 16))
        expected = parent(x)[0].detach()
        grown = grow_depth(parent, 3).eval()
        torch.testing.assert_close(grown(x)[0], expected, rtol=0, atol=0)
        self.assertEqual(grown.config, replace(parent.config, layers=3))
        grown(x, torch.randint(256, (2, 16)))[1].backward()
        self.assertGreater(grown.blocks[2].attention.project.weight.grad.abs().sum().item(), 0)
        self.assertGreater(grown.blocks[2].mlp[2].weight.grad.abs().sum().item(), 0)
        torch.testing.assert_close(parent(x)[0], expected, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            grow_depth(parent, 1)

    def test_extraction_excludes_other_bound_works_and_advertisements(self):
        peter = ("Labels\nCHAPTER I\nPETER BREAKS THROUGH\nStory.\n"
                 "so long as children are gay and innocent and heartless.\n"
                 "THE END\nMARGARET OGILVY\nNot the target work.")
        story = extract_story("peter", peter)
        self.assertIn("Story.", story)
        self.assertNotIn("MARGARET", story)
        alice = ("Labels\nCHAPTER I.\nDOWN THE RABBIT-HOLE.\nStory.\n"
                 "happy summer days.\nTHE END\nTHROUGH THE LOOKING-GLASS ads")
        self.assertNotIn("LOOKING-GLASS", extract_story("alice", alice))
        with self.assertRaises(ValueError):
            extract_story("alice", "Missing reliable ending")

    def test_splits_and_exact_overlap_audit(self):
        content = bytes(range(256)) * 100
        spans = spans_for(content)
        self.assertEqual(spans["validation"][0] - spans["train"][1], 128)
        self.assertEqual(spans["test"][0] - spans["validation"][1], 128)
        audit = overlap_audit({"book": b"x" * 128 + b"y" * 128},
                              {"heldout": b"y" * 128 + b"z" * 128})
        self.assertEqual(audit["exact_matches"], [dict(holdout="heldout", offset=0, train="book")])
        self.assertEqual(audit["checked_spans"], 2)

    def test_split_checksum_tampering_is_rejected(self):
        content = bytes(range(256))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            manifest = {"books": [dict(name="alice", split_sha256={"train": checksum(content)})]}
            (path / "manifest.json").write_text(json.dumps(manifest))
            split_path = path / "alice_train.bin"
            split_path.write_bytes(content)
            with patch.object(expanded_data, "DIRECTORY", path):
                self.assertEqual(expanded_data.load_book("alice", "train"), content)
                split_path.write_bytes(content[:-1] + b"x")
                with self.assertRaises(ValueError):
                    expanded_data.load_book("alice", "train")

    def test_macro_perplexity_is_exponential_of_averaged_loss(self):
        first = dict(loss=1.0, accuracy=0.25, targets=100)
        second = dict(loss=3.0, accuracy=0.75, targets=200)
        result = combine([first, second], [0.5, 0.5])
        self.assertAlmostEqual(result["perplexity"], torch.exp(torch.tensor(2.)).item(), places=6)
        self.assertAlmostEqual(result["accuracy"], 0.5)
        self.assertEqual(result["targets"], 300)
        with self.assertRaises(ValueError):
            combine([first, second], [1.0, 1.0])

    def test_fivefold_weights_preserve_shifted_document_local_windows(self):
        documents = [torch.arange(400) % 100 + offset for offset in [0, 100, 200]]
        weights = [0.2, 0.3, 0.5]
        x, y, chosen = weighted_batch(documents, weights, 16, torch.Generator().manual_seed(7), 2048)
        self.assertTrue(torch.equal(x // 100, chosen[:, None].expand_as(x)))
        self.assertTrue(torch.equal(y, (x % 100 + 1) % 100 + chosen[:, None] * 100))
        for index, weight in enumerate(weights):
            self.assertAlmostEqual((chosen == index).float().mean().item(), weight, delta=0.04)
        for invalid in [[0.5, 0.5], [0.2, float("nan"), 0.8], [0.2, 0.3, 0.4]]:
            with self.assertRaises(ValueError):
                weighted_batch(documents, invalid, 16, torch.Generator())
        with self.assertRaises(ValueError):
            weighted_batch([torch.ones(16)], [1.0], 16, torch.Generator())

    def test_single_document_control_preserves_original_sampling_stream(self):
        from train import batch
        document = torch.arange(400)
        expected = batch(document, 16, 32, torch.Generator().manual_seed(42), "cpu")
        x, y, chosen = weighted_batch([document], [1.0], 16, torch.Generator().manual_seed(42))
        torch.testing.assert_close(x, expected[0])
        torch.testing.assert_close(y, expected[1])
        self.assertEqual(chosen.count_nonzero().item(), 0)

    def test_every_version_has_fivefold_unique_training_data(self):
        old_sizes = {"original": 100, **{book["name"]: 50 for book in expanded_data.BOOKS}}
        entries = [dict(name=f"new{i}", split_bytes={"train": 120}) for i in range(20)]
        plans = version_corpora(old_sizes, entries)
        previous = set()
        for plan in plans.values():
            self.assertGreaterEqual(plan["train_bytes"], 5 * plan["old_train_bytes"])
            names = plan["additional_documents"]
            self.assertEqual(len(names), len(set(names)))
            self.assertTrue(previous.issubset(names))
            previous = set(names)
        with self.assertRaises(ValueError):
            version_corpora(old_sizes, entries[:1])
        with self.assertRaises(ValueError):
            version_corpora(old_sizes, entries + [entries[0]])

    def test_rolling_audit_finds_copy_at_unaligned_training_offset(self):
        span = bytes(range(128))
        result = exact_audit({"train": b"x" * 23 + span + b"y" * 21},
                             {"test": span + b"z" * 128})
        self.assertEqual(result["checked_spans"], 2)
        self.assertEqual(result["exact_matches"],
                         [dict(train="train", train_offset=23, holdout="test", offset=0)])

    def test_fivefold_extraction_removes_front_matter_and_other_works(self):
        raw = ("Labels\nCHAPTER I\nBEGIN\nStory.\nThe last line.\n"
               "THE END\nADVERTISEMENTS\nAnother novel")
        with patch.dict(fivefold_data.BOUNDARIES,
                        {"demo": (r"CHAPTER\s+I\s+BEGIN", r"The last line\.")}):
            text = fivefold_data.extract("demo", raw)
            self.assertIn("Story.", text)
            self.assertNotIn("Labels", text)
            self.assertNotIn("ADVERTISEMENTS", text)
            with self.assertRaises(ValueError):
                fivefold_data.extract("demo", "CHAPTER I BEGIN\nNo verified ending.")

    def test_pre1926_rights_basis_is_explicit_and_rejects_conflicting_dates(self):
        self.assertEqual(verify_rights({"date": "1905"}), "PUBLIC_DOMAIN_US_PRE1926")
        with self.assertRaises(ValueError):
            verify_rights({"date": "1905", "possible-copyright-status": "IN_COPYRIGHT"})
        for date in ["1905; 1990", "2026", "", "unknown"]:
            with self.assertRaises(ValueError):
                verify_rights({"date": date})

    def test_fivefold_split_checksum_tampering_is_rejected(self):
        content = b"verified data" * 20
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            manifest = dict(books=[dict(name="extra", split_sha256={"train": checksum(content)})])
            (path / "manifest.json").write_text(json.dumps(manifest))
            split = path / "extra_train.bin"
            split.write_bytes(content)
            with patch.object(fivefold_data, "DIRECTORY", path):
                self.assertEqual(fivefold_data.load_document("extra", "train"), content)
                split.write_bytes(content[:-1] + b"!")
                with self.assertRaises(ValueError):
                    fivefold_data.load_document("extra", "train")

    def test_cached_archive_download_checks_file_integrity_and_item_identity(self):
        content = b"valid OCR " * 20
        book = dict(name="fixture", item="fixture")
        record = dict(metadata={"identifier": "fixture", "date": "1905", "title": "Fixture"},
                      files=[dict(name="fixture_djvu.txt", size=str(len(content)),
                                  md5=hashlib.md5(content, usedforsecurity=False).hexdigest())])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            metadata = path / "fixture_metadata.json"
            metadata.write_text(json.dumps(record))
            raw = path / "fixture_source.txt"
            raw.write_bytes(content)
            with patch.object(fivefold_data, "DIRECTORY", path):
                self.assertEqual(fivefold_data.download(book)[0], content)
                raw.write_bytes(content[:-1] + b"!")
                with self.assertRaises(ValueError):
                    fivefold_data.download(book)
                record["metadata"]["identifier"] = "wrong-item"
                metadata.write_text(json.dumps(record))
                with self.assertRaises(ValueError):
                    fivefold_data.download(book)


if __name__ == "__main__":
    unittest.main()
