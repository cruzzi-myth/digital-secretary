"""
rag.py — Digital Interview Twin (BM25 edition)
Core retrieval + inference engine.

Flow:
  question → classify tag → BM25 rank → format context → Groq Llama 3 → answer

No embedding model, no PyTorch, no ONNX, no ChromaDB.
Pure Python BM25 keyword search on brain.json.
Target end-to-end latency < 2s on any machine.
"""

import json
import time
import random
from pathlib import Path
from typing import Optional

from rank_bm25 import BM25Okapi
from groq import Groq

from config import (
    GROQ_API_KEY, GROQ_MODEL, GROQ_TIMEOUT,
    TOP_K, FILLER_PHRASES, validate,
)
from persona import build_system_prompt, classify_question

BRAIN_FILE = "brain.json"

# ─── Singleton state (load once, reuse across calls) ──────────────────────────
_brain: Optional[list] = None
_groq_client: Optional[Groq] = None


def _load_brain() -> list:
    global _brain
    if _brain is None:
        if not Path(BRAIN_FILE).exists():
            raise RuntimeError(
                f"{BRAIN_FILE} not found.\n"
                "Run: python3 ingest.py --reset\n"
                "Then add your resume/prep docs to the docs/ folder."
            )
        with open(BRAIN_FILE, "r", encoding="utf-8") as f:
            _brain = json.load(f)
        if not _brain:
            raise RuntimeError(
                f"{BRAIN_FILE} is empty.\n"
                "Run: python3 ingest.py --reset"
            )
    return _brain


def _get_groq() -> Groq:
    global _groq_client
    if _groq_client is None:
        validate()
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


# ─── BM25 Retrieval ───────────────────────────────────────────────────────────
def _tokenize(text: str) -> list:
    """Simple lowercase whitespace tokenizer."""
    return text.lower().split()


def retrieve(question: str, tag: Optional[str] = None, top_k: int = TOP_K) -> list:
    """
    Retrieves top-k relevant chunks using BM25 keyword ranking.

    If a tag is provided (from classify_question), first searches the tagged
    subset (top_k - 2 results), then fills remaining slots from the full corpus.

    Returns list of dicts: {text, source, tag, score}
    """
    brain = _load_brain()
    results = []
    seen_ids = set()
    q_tokens = _tokenize(question)

    # Step 1: Tagged retrieval (biased toward question type)
    if tag and top_k > 2:
        tagged = [c for c in brain if c.get("tag") == tag]
        if tagged:
            corpus = [_tokenize(c["text"]) for c in tagged]
            bm25 = BM25Okapi(corpus)
            scores = bm25.get_scores(q_tokens)
            ranked = sorted(zip(scores, tagged), key=lambda x: x[0], reverse=True)
            for score, chunk in ranked[:top_k - 2]:
                if score > 0:
                    results.append({
                        "text":   chunk["text"],
                        "source": chunk["source"],
                        "tag":    chunk["tag"],
                        "score":  round(score, 4),
                    })
                    seen_ids.add(chunk["id"])

    # Step 2: Unfiltered fill for remaining slots (coverage + general facts)
    needed = top_k - len(results)
    if needed > 0:
        corpus = [_tokenize(c["text"]) for c in brain]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(q_tokens)
        ranked = sorted(zip(scores, brain), key=lambda x: x[0], reverse=True)
        for score, chunk in ranked:
            if chunk["id"] not in seen_ids and len(results) < top_k:
                results.append({
                    "text":   chunk["text"],
                    "source": chunk["source"],
                    "tag":    chunk["tag"],
                    "score":  round(score, 4),
                })
                seen_ids.add(chunk["id"])

    return results[:top_k]


def format_context(chunks: list) -> str:
    """Formats retrieved chunks into a readable context block for the prompt."""
    lines = []
    for i, chunk in enumerate(chunks, 1):
        lines.append(f"[{i}] (source: {chunk['source']}, type: {chunk['tag']})")
        lines.append(chunk["text"])
        lines.append("")
    return "\n".join(lines)


# ─── Inference ────────────────────────────────────────────────────────────────
def answer(question: str, stream: bool = False):
    """
    End-to-end: question → BM25 retrieval → Groq inference → spoken answer.

    Args:
        question: The interviewer's question as a string.
        stream:   If True, prints tokens as they arrive (for CLI real-time feel).

    Returns:
        Tuple of (answer_text: str, metadata: dict)
    """
    t0 = time.perf_counter()

    # Step 1: Classify question type
    tag = classify_question(question)

    # Step 2: BM25 retrieval
    chunks  = retrieve(question, tag=tag)
    context = format_context(chunks)

    # Step 3: Build prompt
    system = build_system_prompt(context)

    # Step 4: Groq inference
    client = _get_groq()

    if stream:
        stream_resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": question},
            ],
            max_tokens=350,
            temperature=0.4,
            stream=True,
        )
        full_text = []
        for chunk in stream_resp:
            delta = chunk.choices[0].delta.content or ""
            print(delta, end="", flush=True)
            full_text.append(delta)
        print()
        response_text = "".join(full_text)
    else:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": question},
            ],
            max_tokens=350,
            temperature=0.4,
        )
        response_text = resp.choices[0].message.content

    t1 = time.perf_counter()

    top_score = chunks[0]["score"] if chunks else 0

    return response_text, {
        "latency_ms":     round((t1 - t0) * 1000),
        "tag":            tag,
        "chunks_used":    len(chunks),
        "sources":        list({c["source"] for c in chunks}),
        "top_similarity": round(top_score, 3),   # kept as top_similarity for demo.py compat
    }


def filler_phrase() -> str:
    """Returns a random spoken filler phrase to buy time while BM25 + Groq processes."""
    return random.choice(FILLER_PHRASES)
