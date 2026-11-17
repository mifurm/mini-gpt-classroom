"""Prepare five rights-verified Archive books without changing the original corpus."""

import argparse
import hashlib
import json
import re
import unicodedata
from urllib.request import urlopen

from data import MAX_CONTEXT, ROOT, load_split


DIRECTORY = ROOT / "data" / "expanded"
BOOKS = [
    dict(name="alice", item="alicesadventures00carruoft"),
    dict(name="oz", item="wonderfulwizardo00baumiala"),
    dict(name="peter", item="peterwendy00barr"),
    dict(name="willows", item="windinwillows00grah"),
    dict(name="garden", item="secretgarden00burn"),
]
BOUNDARIES = {
    "alice": (r"CHAPTER\s+I\.\s+DOWN\s+THE\s+RABBIT-HOLE\.",
              r"happy\s+summer\s+days\."),
    "oz": (r"LIVED\s+IN\s+the\s+midst\s+of\s+the",
           r"home\s+again!\s*[\"”]"),
    "peter": (r"CHAPTER\s+I\s+PETER\s+BREAKS\s+THROUGH",
              r"so\s+long\s+as\s+children\s+are\s+gay\s+and\s+innocent\s+and\s+heartless\."),
    "willows": (r"THE\s+Mole\s+had\s+been\s+working",
                r"but\s+it\s+never\s+failed\s+to\s+have\s+its\s+full\s+effect\."),
    "garden": (r"CHAPTER\s+I\s+THERE\s+IS\s+NO\s+ONE\s+LEFT",
               r"Master\s+Colin!"),
}


def checksum(content):
    return hashlib.sha256(content).hexdigest()


def download(book):
    DIRECTORY.mkdir(exist_ok=True)
    raw_path = DIRECTORY / f"{book['name']}_source.txt"
    meta_path = DIRECTORY / f"{book['name']}_metadata.json"
    meta_url = f"https://archive.org/metadata/{book['item']}"
    text_url = f"https://archive.org/download/{book['item']}/{book['item']}_djvu.txt"
    if not meta_path.exists():
        with urlopen(meta_url, timeout=90) as response:
            metadata = json.load(response)["metadata"]
        meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    metadata = json.loads(meta_path.read_text())
    if metadata.get("possible-copyright-status") != "NOT_IN_COPYRIGHT":
        raise ValueError(f"{book['item']} does not have verified public-domain status.")
    if not raw_path.exists():
        with urlopen(text_url, timeout=120) as response:
            raw_path.write_bytes(response.read())
    print(f"{book['name']}: {metadata['title']}, {metadata['date']}, "
          f"NOT_IN_COPYRIGHT (US), {raw_path.stat().st_size:,} OCR bytes", flush=True)
    return raw_path.read_bytes(), metadata, text_url, meta_url


def unwrap_story(story):
    lines = []
    for line in unicodedata.normalize("NFC", story).splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()
        if re.fullmatch(r"\d+", line):
            continue
        if re.fullmatch(r"[A-Z ,.'?!&-]+\s+\d+", line):
            continue
        if re.fullmatch(r"\d+\s+[A-Z ,.'?!&-]+", line):
            continue
        lines.append(line)
    story = "\n".join(lines)
    story = re.sub(r"(?<=[a-z])-\n(?=[a-z])", "", story)
    story = re.sub(r"(?<=\S)\n(?=\S)", " ", story)
    return re.sub(r"\n{3,}", "\n\n", story).strip() + "\n"


def spans_for(content):
    first, second = int(len(content) * 0.8), int(len(content) * 0.9)
    spans = dict(train=(0, first), validation=(first + MAX_CONTEXT, second),
                 test=(second + MAX_CONTEXT, len(content)))
    if any(end - start <= 4 * MAX_CONTEXT for start, end in spans.values()):
        raise ValueError("Expanded book is too short for useful disjoint splits.")
    return spans


def extract_story(name, raw):
    if name not in BOUNDARIES:
        raise ValueError(f"Story boundaries for {name} have not been verified.")
    start_pattern, end_pattern = BOUNDARIES[name]
    start = re.search(start_pattern, raw, re.IGNORECASE)
    endings = list(re.finditer(end_pattern, raw, re.IGNORECASE))
    if start is None or not endings or endings[-1].end() <= start.start():
        raise ValueError(f"Reliable story boundaries not found for {name}.")
    return unwrap_story(raw[start.start():endings[-1].end()])


def overlap_audit(training, heldout, length=128):
    matches = []
    checked = 0
    for name, content in heldout.items():
        for offset in range(0, len(content) - length + 1, length):
            span = content[offset:offset + length]
            checked += 1
            for training_name, document in training.items():
                if span in document:
                    matches.append(dict(holdout=name, offset=offset, train=training_name))
    return dict(span_bytes=length, stride_bytes=length, checked_spans=checked,
                exact_matches=matches,
                limitation="Strided exact-byte audit, not exhaustive semantic deduplication.")


def prepare():
    entries = []
    training = {"original": load_split("train")}
    heldout = {f"original/{split}": load_split(split) for split in ["validation", "test"]}
    for book in BOOKS:
        raw, metadata, text_url, meta_url = download(book)
        content = extract_story(book["name"], raw.decode("utf-8")).encode("utf-8")
        (DIRECTORY / f"{book['name']}_clean.txt").write_bytes(content)
        spans = spans_for(content)
        splits = {name: content[start:end] for name, (start, end) in spans.items()}
        for split, data in splits.items():
            (DIRECTORY / f"{book['name']}_{split}.bin").write_bytes(data)
        training[book["name"]] = splits["train"]
        for split in ["validation", "test"]:
            heldout[f"{book['name']}/{split}"] = splits[split]
        entries.append(dict(
            **book, title=metadata["title"], publication_date=metadata["date"],
            copyright_status=metadata["possible-copyright-status"], copyright_region="US",
            text_url=text_url, metadata_url=meta_url, raw_sha256=checksum(raw),
            clean_sha256=checksum(content), clean_bytes=len(content),
            boundaries=BOUNDARIES[book["name"]], split_spans=spans,
            split_bytes={name: len(data) for name, data in splits.items()},
            split_sha256={name: checksum(data) for name, data in splits.items()},
        ))
    audit = overlap_audit(training, heldout)
    if audit["exact_matches"]:
        raise ValueError(f"Exact holdout spans appear in training: {audit['exact_matches']}")
    manifest = dict(
        books=entries, original_train_bytes=len(training["original"]),
        total_train_bytes=sum(len(content) for content in training.values()),
        total_new_clean_bytes=sum(book["clean_bytes"] for book in entries),
        boundary_gap_bytes=MAX_CONTEXT, overlap_audit=audit,
        cleaning="Extract verified story only; normalize spaces/NFC; remove numeric page "
                 "lines and uppercase running headings with numbers; unwrap lines and "
                 "join lowercase line-break hyphenation. Remaining OCR errors retained.",
    )
    (DIRECTORY / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(dict(
        manifest=str(DIRECTORY / "manifest.json"),
        total_train_bytes=manifest["total_train_bytes"],
        books=[dict(name=book["name"], clean_bytes=book["clean_bytes"],
                    train_bytes=book["split_bytes"]["train"]) for book in entries],
        overlap_audit=audit,
    ), indent=2))


def load_book(name, split):
    if name not in {book["name"] for book in BOOKS}:
        raise ValueError(f"Unknown book: {name}")
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Unknown split: {split}")
    manifest = json.loads((DIRECTORY / "manifest.json").read_text())
    entry = next(book for book in manifest["books"] if book["name"] == name)
    content = (DIRECTORY / f"{name}_{split}.bin").read_bytes()
    if checksum(content) != entry["split_sha256"][split]:
        raise ValueError(f"Checksum mismatch for {name}/{split}.")
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
        if args.book is not None:
            parser.error("--book is only supported with --download-only")
        prepare()


if __name__ == "__main__":
    main()
