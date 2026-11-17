"""Freeze a genuinely larger, rights-verified Archive corpus for every V1-V7 recipe."""

import argparse
from collections import defaultdict
import hashlib
import json
import re
import subprocess

from data import ROOT, load_split
from expanded_data import BOOKS as LEGACY_BOOKS, checksum, load_book, spans_for, unwrap_story


DIRECTORY = ROOT / "data" / "fivefold"
BOOKS = [
    dict(name="fauntleroy_1886", item="littlelordfauntl00burn_1"),
    dict(name="princess", item="littleprincessbe00burn",
         scan_note="1917 reprint of the 1905 narrative; title page inspected."),
    dict(name="railway", item="railwaychildren00nesb_1",
         scan_note="Metadata says 1906; title/copyright page says 1905."),
    dict(name="treasure_complete", item="treasureisl00stev"),
    dict(name="beauty", item="blackbeauty00sewe_10"),
    dict(name="heidi", item="heidispyr00spyruoft",
         scan_note="English text edition first published 1910, reprinted 1944. "
                   "Only that public-domain narrative is extracted; front editorial matter excluded."),
    dict(name="anne_green", item="anneofgreengable00mont_11"),
    dict(name="anne_avonlea", item="anneofavonlea00mont_5"),
    dict(name="tom", item="adventuresoftoms00twaiuoft"),
    dict(name="huck", item="adventuresofhuck00twaiuoft"),
    dict(name="jungle", item="junglebook00kipluoft",
         scan_note="1899 uniform text edition; reprint list ends 1915."),
    dict(name="second_jungle", item="secondjunglebook00kipluoft"),
    dict(name="little_women", item="littlewomenormeg00alcouoft"),
    dict(name="kidnapped", item="kidnappedbeingme00stevuoft"),
]
BOUNDARIES = {
    "fauntleroy_1886": (r"(?m)^\s*edric\s+himself\s+knew", r"(?m)^\s*THE\s+END\.\s*$"),
    "princess": (r"CHAPTER\s+I\s+SARA\s+ONCE",
                 r"Indian\s+gentleman,\s+and\s+they\s+got\s+into\s+the\s+carriage\s+and\s+drove\s+away\."),
    "railway": (r"CHAPTER\s+I\s+THE\s+BEGINNING\s+OF\s+THINGS\s+They\s+were",
                r"neither\s+we\s+nor\s+any\s+one\s+else\s+is\s+wanted\s+now\."),
    "treasure_complete": (r"CHAPTER\s+I\s+The\s+Old\s+Sea\s+Dog\s+at\s+the\s+Admiral\s+Benbow\s+SQUIRE",
                          r"(?m)^\s*THE\s+END\s*$"),
    "beauty": (r"CHAPTER\s+I\.\s+MY\s+EARLY\s+HOME\.", r"(?m)^\s*THE\s+END\.\s*$"),
    "heidi": (r"CHAPTER\s+I\s+UP\s+THE\s+MOUNTAIN\s+TO\s+ALM.UNCLB\s+From\s+the\s+old",
              r"the\s+mercies\s+He\s+has\s+shown\s+us\s+I\s+['*]+"),
    "anne_green": (r"CHAPTER\s+I\s+MRS\.\s+RACHEL\s+LYNDE\s+IS\s+SURPRISED\s+Mrs\.",
                   r"(?m)^\s*THE\s+END\.\s*$"),
    "anne_avonlea": (r"CHAPTER\s+I\s+AN\s+IRATE\s+NEIGHBOUR\s+A\s+TALL",
                     r"(?m)^\s*THE\s+END\.\s*$"),
    "tom": (r"CHAPTER\s+I\s+TOM\s+PLAYS,\s+FIGHTS,\s+AND\s+HIDES\s+'OM!",
            r"(?m)^\s*THE\s+END\s*$"),
    "huck": (r"CHAPTER\s+I\s+YOU\s+don.t\s+know\s+about\s+me", r"(?m)^\s*THE\s+END\s*$"),
    "jungle": (r"Now\s+Rann\s+the\s+Kite\s+brings\s+home\s+the\s+night",
               r"(?m)^\s*THE\s+END\s*$"),
    "second_jungle": (r"The\s+stream\s+is\s+shrunk",
                      r"Wisdom,\s+Strength,\s+and\s+Courtesy,\s+Jungle-\s*Favor\s+go\s+with\s+thee\s+J"),
    "little_women": (r"Christmas\s+won.t\s+be\s+Christmas\s+without\s+any\s+presents",
                     r"greater\s+happiness\s+than\s+this\s*!\s*\""),
    "kidnapped": (r"CHAPTER\s+I\s+I\s+SET\s+OFF\s+UPON\s+MY\s+JOURNEY\s+TO\s+THE\s+HOUSE\s+OF\s+SHAWS\s+I\s+WILL\s+begin",
                  r"very\s+doors\s+of\s+the\s+British\s+Linen\s+Com-\s*pany.s\s+bank\."),
}
RIGHTS_SOURCE = "https://guides.library.cornell.edu/copyright/publicdomain"


def verify_rights(metadata):
    years = [int(year) for year in re.findall(r"(?<!\d)\d{4}(?!\d)", str(metadata.get("date", "")))]
    statement = metadata.get("possible-copyright-status")
    if not years or any(not 1800 <= year <= 1925 for year in years):
        raise ValueError(f"Review printed edition date: {metadata.get('date')}")
    if statement not in {None, "NOT_IN_COPYRIGHT",
                         "The Library of Congress is unaware of any copyright restrictions on this item."}:
        raise ValueError(f"Contradictory or unreviewed Archive rights statement: {statement}")
    return "PUBLIC_DOMAIN_US_PRE1926"


def fetch(url, path, resume=False):
    partial = path.with_suffix(path.suffix + ".part")
    subprocess.run(
        ["curl", "--fail", "--location", "--silent", "--show-error", "--http1.1",
         "--retry", "5", "--retry-all-errors", "--connect-timeout", "20",
         "--max-time", "300", *(["--continue-at", "-"] if resume else []),
         "--output", str(partial), url],
        check=True,
    )
    if partial.stat().st_size < 100:
        raise ValueError(f"Unexpectedly short Archive response: {url}")
    partial.replace(path)


def download(book):
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    metadata_path = DIRECTORY / f"{book['name']}_metadata.json"
    metadata_url = f"https://archive.org/metadata/{book['item']}"
    if not metadata_path.exists():
        fetch(metadata_url, metadata_path)
    record = json.loads(metadata_path.read_text())
    metadata = record["metadata"]
    if metadata.get("identifier") != book["item"]:
        raise ValueError(f"Cached metadata belongs to a different Archive item: {book['name']}")
    verify_rights(metadata)
    files = [entry["name"] for entry in record["files"] if entry["name"].endswith("_djvu.txt")]
    if len(files) != 1:
        raise ValueError(f"Inspect OCR file selection for {book['item']}: {files}")
    text_url = f"https://archive.org/download/{book['item']}/{files[0]}?download=1"
    path = DIRECTORY / f"{book['name']}_source.txt"
    if not path.exists():
        fetch(text_url, path, resume=True)
    raw = path.read_bytes()
    entry = next(entry for entry in record["files"] if entry["name"] == files[0])
    if len(raw) != int(entry["size"]) or hashlib.md5(raw, usedforsecurity=False).hexdigest() != entry["md5"]:
        raise ValueError(f"Incomplete or corrupted Archive OCR: {book['name']}")
    raw.decode("utf-8")
    print(f"{book['name']}: {metadata['title']}, {metadata['date']}, "
          f"pre-1926 public-domain text (US), {len(raw):,} OCR bytes", flush=True)
    return raw, metadata, metadata_url, text_url


def extract(name, raw):
    if name not in BOUNDARIES:
        raise ValueError(f"Verify story boundaries before preparing {name}.")
    start_pattern, end_pattern = BOUNDARIES[name]
    start = re.search(start_pattern, raw, re.IGNORECASE)
    if start is None:
        raise ValueError(f"Story start missing: {name}")
    end = re.search(end_pattern, raw[start.end():], re.IGNORECASE)
    if end is None:
        raise ValueError(f"Story end missing: {name}")
    return unwrap_story(raw[start.start():start.end() + end.end()])


def exact_audit(training, heldout, length=128):
    """Match strided heldout spans against every document-local training offset."""
    base, mask = 257, (1 << 64) - 1
    power = pow(base, length - 1, 1 << 64)
    candidates = defaultdict(list)
    checked = 0
    for name, content in heldout.items():
        for offset in range(0, len(content) - length + 1, length):
            span = content[offset:offset + length]
            value = 0
            for byte in span:
                value = (value * base + byte) & mask
            candidates[value].append((name, offset, span))
            checked += 1
    matches, windows = [], 0
    for name, content in training.items():
        if len(content) < length:
            continue
        value = 0
        for byte in content[:length]:
            value = (value * base + byte) & mask
        for offset in range(len(content) - length + 1):
            windows += 1
            for heldout_name, heldout_offset, span in candidates.get(value, ()):
                if content[offset:offset + length] == span:
                    matches.append(dict(train=name, train_offset=offset,
                                        holdout=heldout_name, offset=heldout_offset))
            if offset + length < len(content):
                value = ((value - content[offset] * power) * base
                         + content[offset + length]) & mask
    return dict(span_bytes=length, heldout_stride=length, checked_spans=checked,
                training_windows=windows, exact_matches=matches,
                limitation="Exact-byte, strided heldout audit; not exhaustive semantic deduplication.")


def version_corpora(old_sizes, entries):
    if len({entry["name"] for entry in entries}) != len(entries):
        raise ValueError("Repeated documents cannot count toward corpus growth.")
    plans = {}
    for version in range(1, 8):
        old_names = ["original"] + [book["name"] for book in LEGACY_BOOKS[:max(0, version - 2)]]
        original_bytes = sum(old_sizes[name] for name in old_names)
        added, names = 0, []
        for entry in entries:
            names.append(entry["name"])
            added += entry["split_bytes"]["train"]
            if original_bytes + added >= 5 * original_bytes:
                break
        if original_bytes + added < 5 * original_bytes:
            raise ValueError(f"Need more unique Archive data for v{version}: "
                             f"{original_bytes + added:,} < {5 * original_bytes:,}.")
        plans[f"v{version}"] = dict(
            legacy_documents=old_names, additional_documents=names,
            old_train_bytes=original_bytes, train_bytes=original_bytes + added,
            expansion_factor=(original_bytes + added) / original_bytes,
        )
    return plans


def prepare():
    training = {"original": load_split("train")}
    heldout = {f"original/{split}": load_split(split) for split in ["validation", "test"]}
    for book in LEGACY_BOOKS:
        training[book["name"]] = load_book(book["name"], "train")
        for split in ["validation", "test"]:
            heldout[f"{book['name']}/{split}"] = load_book(book["name"], split)
    old_sizes = {name: len(content) for name, content in training.items()}
    entries = []
    for book in BOOKS:
        raw, metadata, metadata_url, text_url = download(book)
        content = extract(book["name"], raw.decode("utf-8")).encode("utf-8")
        spans = spans_for(content)
        splits = {name: content[start:end] for name, (start, end) in spans.items()}
        (DIRECTORY / f"{book['name']}_clean.txt").write_bytes(content)
        for split, data in splits.items():
            (DIRECTORY / f"{book['name']}_{split}.bin").write_bytes(data)
        training[book["name"]] = splits["train"]
        for split in ["validation", "test"]:
            heldout[f"{book['name']}/{split}"] = splits[split]
        entries.append(dict(
            **book, title=metadata["title"], publication_date=metadata["date"],
            copyright_status=verify_rights(metadata), copyright_region="US",
            archive_copyright_statement=metadata.get("possible-copyright-status"),
            rights_source=RIGHTS_SOURCE,
            rights_basis="Pre-1926 published narrative/text edition; later reprinting does not "
                         "renew the underlying copyright. Non-narrative editorial matter excluded.",
            metadata_url=metadata_url, text_url=text_url, raw_sha256=checksum(raw),
            clean_sha256=checksum(content), clean_bytes=len(content), boundaries=BOUNDARIES[book["name"]],
            split_spans=spans, split_bytes={name: len(data) for name, data in splits.items()},
            split_sha256={name: checksum(data) for name, data in splits.items()},
        ))
    if len({entry["clean_sha256"] for entry in entries}) != len(entries):
        raise ValueError("Duplicated cleaned works do not enlarge a corpus.")
    plans = version_corpora(old_sizes, entries)
    audit = exact_audit(training, heldout)
    if audit["exact_matches"]:
        raise ValueError(f"Training/holdout contamination: {audit['exact_matches']}")
    manifest = dict(books=entries, versions=plans, boundary_gap_bytes=128, overlap_audit=audit,
                    total_train_bytes=sum(map(len, training.values())),
                    cleaning="Verified story only; existing NFC/page/line cleaning; OCR errors retained.")
    (DIRECTORY / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(dict(versions=plans, overlap_audit=audit), indent=2), flush=True)


def load_document(name, split):
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Unknown split: {split}")
    if name == "original":
        return load_split(split)
    if name in {book["name"] for book in LEGACY_BOOKS}:
        return load_book(name, split)
    manifest = json.loads((DIRECTORY / "manifest.json").read_text())
    entry = next((book for book in manifest["books"] if book["name"] == name), None)
    if entry is None:
        raise ValueError(f"Unknown fivefold document: {name}")
    content = (DIRECTORY / f"{name}_{split}.bin").read_bytes()
    if checksum(content) != entry["split_sha256"][split]:
        raise ValueError(f"Checksum mismatch: {name}/{split}")
    return content


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--book", choices=[book["name"] for book in BOOKS])
    args = parser.parse_args()
    if args.download_only:
        for book in BOOKS:
            if args.book is None or args.book == book["name"]:
                download(book)
    else:
        if args.book:
            parser.error("--book requires --download-only")
        prepare()


if __name__ == "__main__":
    main()
