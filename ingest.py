"""
ingest.py — Digital Interview Twin (BM25 edition)
Reads markdown/PDF/text prep documents, chunks them, and saves to brain.json.

No embedding model required — zero ML dependencies, zero OOM risk.
BM25 keyword retrieval handles dense interview prep content extremely well
(technical keywords like "outbox", "multi-tenant", "KEDA" map cleanly to docs).

Usage:
    python3 ingest.py                          # ingest everything in docs/
    python3 ingest.py --file docs/resume.md   # ingest a single file
    python3 ingest.py --reset                 # wipe brain.json and re-ingest
"""

import sys
import json
import uuid
import argparse
from pathlib import Path
from typing import List

from config import (
    CHUNK_SIZE, CHUNK_OVERLAP,
    TAG_TECHNICAL, TAG_BEHAVIORAL, TAG_COMPANY, TAG_GENERAL,
)

BRAIN_FILE = "brain.json"


# ─── Text chunker ─────────────────────────────────────────────────────────────
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Splits text into overlapping character-level chunks.
    Tries to break on sentence boundaries ('. ', '? ', '! ') when possible.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end]

        if end < len(text):
            for sep in [". ", "? ", "! ", "\n\n", "\n"]:
                idx = chunk.rfind(sep, size - overlap)
                if idx != -1:
                    end = start + idx + len(sep)
                    chunk = text[start:end]
                    break

        chunks.append(chunk.strip())
        start = end - overlap

    return [c for c in chunks if len(c) > 40]


# ─── Tag detector ─────────────────────────────────────────────────────────────
def detect_tag(text: str, filename: str) -> str:
    """
    Infers a metadata tag from filename hints or content keywords.
    Tag stored in brain.json for BM25 pre-filtering by question type.
    """
    fname = filename.lower()

    if any(k in fname for k in ["behavioral", "star", "story", "stories", "leadership"]):
        return TAG_BEHAVIORAL
    if any(k in fname for k in ["technical", "design", "system", "architecture", "code"]):
        return TAG_TECHNICAL
    if any(k in fname for k in ["company", "research", "why_", "target"]):
        return TAG_COMPANY

    t = text.lower()
    behavioral_score = sum(t.count(k) for k in [
        "i led", "i owned", "i drove", "we had", "the team", "conflict",
        "stakeholder", "disagreement", "influence", "failed", "learned",
    ])
    technical_score = sum(t.count(k) for k in [
        "architecture", "database", "api", "latency", "queue", "kubernetes",
        "azure", "aws", "cache", "throughput", "microservice", "deployment",
    ])

    if behavioral_score > technical_score:
        return TAG_BEHAVIORAL
    if technical_score > 0:
        return TAG_TECHNICAL
    return TAG_GENERAL


# ─── File reader ──────────────────────────────────────────────────────────────
def read_file(path: Path) -> str:
    """Reads .md, .txt, or .pdf files and returns plain text."""
    suffix = path.suffix.lower()

    if suffix in (".md", ".txt"):
        return path.read_text(encoding="utf-8")

    if suffix == ".pdf":
        try:
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                return "\n\n".join(
                    page.extract_text() or "" for page in pdf.pages
                )
        except ImportError:
            print(f"  ⚠️  pdfplumber not installed — skipping {path.name}")
            print("     pip3 install pdfplumber --break-system-packages")
            return ""

    print(f"  ⚠️  Unsupported file type: {path.suffix} — skipping {path.name}")
    return ""


# ─── Main ingestion pipeline ──────────────────────────────────────────────────
def ingest_file(path: Path, brain: list) -> int:
    """
    Ingests a single file: read → chunk → append to brain list.
    Returns the number of chunks stored.
    """
    raw = read_file(path)
    if not raw.strip():
        return 0

    tag = detect_tag(raw, path.name)
    chunks = chunk_text(raw)

    if not chunks:
        return 0

    print(f"  📄  {path.name}  ({len(chunks)} chunks, tag={tag})")

    for chunk in chunks:
        brain.append({
            "id":     str(uuid.uuid4()),
            "text":   chunk,
            "source": path.name,
            "tag":    tag,
        })

    return len(chunks)


def ingest_directory(docs_dir: Path, brain: list) -> int:
    """Ingests all supported files in a directory (non-recursive)."""
    supported = {".md", ".txt", ".pdf"}
    files = [f for f in docs_dir.iterdir() if f.is_file() and f.suffix.lower() in supported]

    if not files:
        print(f"  ⚠️  No supported files found in {docs_dir}")
        return 0

    total = 0
    for f in sorted(files):
        total += ingest_file(f, brain)
    return total


# ─── Entry point ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Ingest prep docs into brain.json (BM25 mode)")
    parser.add_argument("--file",  type=str, help="Ingest a single file")
    parser.add_argument("--dir",   type=str, default="docs", help="Directory to ingest (default: docs/)")
    parser.add_argument("--reset", action="store_true", help="Wipe brain.json before ingesting")
    args = parser.parse_args()

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  Digital Interview Twin — BM25 Ingestion Pipeline")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
    print("  No embedding model — pure keyword chunking. RAM: ~0 MB extra.\n")

    brain = []

    if args.reset and Path(BRAIN_FILE).exists():
        Path(BRAIN_FILE).unlink()
        print(f"  🗑  Cleared existing {BRAIN_FILE}\n")

    if args.file:
        path = Path(args.file)
        if not path.exists():
            print(f"  ❌ File not found: {path}")
            sys.exit(1)
        total = ingest_file(path, brain)
    else:
        docs_dir = Path(args.dir)
        if not docs_dir.exists():
            print(f"  ❌ Directory not found: {docs_dir}")
            print(f"     Create it and add your resume/prep docs.")
            sys.exit(1)
        total = ingest_directory(docs_dir, brain)

    with open(BRAIN_FILE, "w", encoding="utf-8") as f:
        json.dump(brain, f, ensure_ascii=False, indent=2)

    print(f"\n  ✅ Ingestion complete — {total} chunks written to {BRAIN_FILE}")
    print(f"  📦 Total chunks: {len(brain)}")
    print(f"  💡 No embedding model needed — BM25 retrieval handles the rest\n")


if __name__ == "__main__":
    main()
