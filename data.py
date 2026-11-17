"""Download a public-domain Internet Archive scan and freeze disjoint byte splits."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parent
ITEM = "throughlookinggl00carruoft"
META_URL = f"https://archive.org/metadata/{ITEM}"
TEXT_URL = f"https://archive.org/download/{ITEM}/{ITEM}_djvu.txt"
MAX_CONTEXT = 128


def clean_story(raw: str) -> str:
    start = re.search(r"CHAPTER\s+I\.\s+LOOKING-GLASS\s+HOUSE\.", raw)
    endings = list(re.finditer(r"THE\s+END\.", raw))
    if start is None or not endings or endings[-1].start() <= start.end():
        raise ValueError("Story boundaries not found; inspect the source, do not guess.")
    story = unicodedata.normalize("NFC", raw[start.start():endings[-1].end()])
    lines = []
    for line in story.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()
        if re.fullmatch(r"\d+", line):
            continue
        # Page headers, not chapter headings: uppercase words plus a page number.
        if re.fullmatch(r"[A-Z ,.'?!-]+\s+\d+", line):
            continue
        lines.append(line)
    story = "\n".join(lines)
    story = re.sub(r"(?<=[a-z])-\n(?=[a-z])", "", story)
    story = re.sub(r"(?<=\S)\n(?=\S)", " ", story)
    return re.sub(r"\n{3,}", "\n\n", story).strip() + "\n"


def split_bytes(content: bytes):
    cut1, cut2 = int(len(content) * 0.8), int(len(content) * 0.9)
    spans = {
        "train": (0, cut1),
        "validation": (cut1 + MAX_CONTEXT, cut2),
        "test": (cut2 + MAX_CONTEXT, len(content)),
    }
    if any(end - start <= 2 * MAX_CONTEXT for start, end in spans.values()):
        raise ValueError("Corpus is too small for three useful splits.")
    return {name: content[start:end] for name, (start, end) in spans.items()}, spans


def prepare():
    directory = ROOT / "data"
    directory.mkdir(exist_ok=True)
    with urlopen(META_URL, timeout=60) as response:
        metadata = json.load(response)
    with urlopen(TEXT_URL, timeout=90) as response:
        raw = response.read()
    status = metadata["metadata"].get("possible-copyright-status", "")
    if status != "NOT_IN_COPYRIGHT":
        raise ValueError(f"Unexpected copyright status: {status!r}; review before use.")
    (directory / "source.txt").write_bytes(raw)
    content = clean_story(raw.decode("utf-8")).encode("utf-8")
    (directory / "clean.txt").write_bytes(content)
    splits, spans = split_bytes(content)
    for name, split in splits.items():
        (directory / f"{name}.bin").write_bytes(split)
    provenance = {
        "item": ITEM, "metadata_url": META_URL, "text_url": TEXT_URL,
        "title": metadata["metadata"]["title"],
        "publication_date": metadata["metadata"]["date"],
        "copyright_status": status, "copyright_region": "US",
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "clean_sha256": hashlib.sha256(content).hexdigest(),
        "clean_bytes": len(content), "split_spans": spans,
        "split_bytes": {name: len(split) for name, split in splits.items()},
        "split_sha256": {name: hashlib.sha256(split).hexdigest()
                         for name, split in splits.items()},
        "boundary_gap_bytes": MAX_CONTEXT,
        "cleaning": [
            "Extract CHAPTER I through THE END; exclude library labels/front matter.",
            "Normalize NFC and horizontal spaces; remove standalone page numbers.",
            "Remove uppercase running headings ending in page numbers.",
            "Join lowercase words hyphenated at a line break; unwrap adjacent lines.",
            "Retain paragraph boundaries and uncorrected OCR spelling errors.",
        ],
    }
    (directory / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


def load_split(name: str) -> bytes:
    if name not in {"train", "validation", "test"}:
        raise ValueError(f"Unknown split: {name}")
    provenance = json.loads((ROOT / "data" / "provenance.json").read_text())
    content = (ROOT / "data" / f"{name}.bin").read_bytes()
    if hashlib.sha256(content).hexdigest() != provenance["split_sha256"][name]:
        raise ValueError(f"{name} split checksum mismatch; restore the frozen corpus.")
    return content


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    prepare()
