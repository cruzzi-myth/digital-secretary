"""
build_brain.py — standalone brain builder, no config imports needed.
Run: python3 build_brain.py
"""
import json, uuid, sys
from pathlib import Path

HERE = Path(__file__).parent
DOCS = HERE / "docs"
OUT  = HERE / "brain.json"

TAG_MAP = {
    "behavioral_stories.md": "behavioral",
    "projects.md":           "technical",
    "resume.md":             "general",
    "technical_prep.md":     "technical",
}

CHUNK_SIZE    = 400
CHUNK_OVERLAP = 80

brain = []

for name, tag in sorted(TAG_MAP.items()):
    path = DOCS / name
    if not path.exists():
        print(f"  SKIP (not found): {name}")
        continue
    raw = path.read_text(encoding="utf-8")
    start, n = 0, 0
    while start < len(raw):
        end = min(start + CHUNK_SIZE, len(raw))
        chunk = raw[start:end].strip()
        if len(chunk) > 40:
            brain.append({
                "id":     str(uuid.uuid4()),
                "text":   chunk,
                "source": name,
                "tag":    tag,
            })
            n += 1
        start = end - CHUNK_OVERLAP
    print(f"  {name}: {n} chunks", flush=True)

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(brain, f, ensure_ascii=False, indent=2)

print(f"\nDone: {len(brain)} chunks written to brain.json")
