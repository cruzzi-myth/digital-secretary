"""
demo.py — Digital Interview Twin
Interactive CLI demo. Type an interview question, get a spoken-style answer.

Usage:
    python demo.py                 # interactive loop
    python demo.py --once          # single question, then exit (for piping/testing)
    python demo.py --bench         # run the built-in benchmark suite

Prerequisites:
    1. export GROQ_API_KEY=gsk_...
    2. Add your resume/prep docs to docs/
    3. python ingest.py
    4. python demo.py
"""

import sys
import time
import argparse

# ─── Color helpers ────────────────────────────────────────────────────────────
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
GRAY   = "\033[90m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

BANNER = f"""
{CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}
{BOLD}  🎙  Digital Interview Twin — RAG Demo{RESET}
{GRAY}  Model: Llama 3 70B via Groq  |  DB: ChromaDB{RESET}
{CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}
  Type an interview question and press Enter.
  Type {YELLOW}quit{RESET} or press Ctrl+C to exit.
"""

BENCH_QUESTIONS = [
    # Behavioral
    "Tell me about a time you had to influence a team without direct authority.",
    "Describe a situation where you had to make a technical decision under uncertainty.",
    "Give me an example of a time you failed and what you learned.",
    # Technical
    "How would you design a system to handle 500k daily transactions with at-least-once delivery?",
    "Walk me through how you'd approach a multi-tenant architecture for a SaaS product.",
    "What's the difference between the outbox pattern and a saga, and when would you use each?",
    # General
    "Why are you looking for a new role?",
    "What's your ideal team size and engineering culture?",
]


def print_meta(meta: dict):
    """Prints retrieval metadata after each answer (debug info)."""
    print(
        f"\n{GRAY}  ↳ {meta['latency_ms']}ms · "
        f"tag={meta['tag']} · "
        f"chunks={meta['chunks_used']} · "
        f"similarity={meta['top_similarity']} · "
        f"sources: {', '.join(meta['sources']) or 'none'}{RESET}\n"
    )


def run_once():
    """Handles a single question from stdin (for testing/piping)."""
    from rag import answer, filler_phrase
    question = input(f"{CYAN}Question:{RESET} ").strip()
    if not question:
        sys.exit(0)
    print(f"\n{YELLOW}  [{filler_phrase()}]{RESET}\n")
    text, meta = answer(question, stream=True)
    print_meta(meta)


def run_interactive():
    """Main interactive loop."""
    from rag import answer, filler_phrase
    print(BANNER)

    while True:
        try:
            question = input(f"{CYAN}❓  Question:{RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n\n{GREEN}  Goodbye.{RESET}\n")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print(f"\n{GREEN}  Goodbye.{RESET}\n")
            break

        # Print filler phrase immediately while RAG + LLM runs
        print(f"\n{YELLOW}  \"{filler_phrase()}\"{RESET}\n")
        print(f"{GREEN}  Answer:{RESET}")

        try:
            text, meta = answer(question, stream=True)
            print_meta(meta)
        except RuntimeError as e:
            print(f"\n{YELLOW}  ⚠️  {e}{RESET}\n")
        except Exception as e:
            print(f"\n{YELLOW}  ⚠️  Error: {e}{RESET}\n")


def run_benchmark():
    """Runs the built-in question suite and prints a latency table."""
    from rag import answer

    print(f"\n{BOLD}  Running benchmark suite ({len(BENCH_QUESTIONS)} questions)...{RESET}\n")
    results = []

    for q in BENCH_QUESTIONS:
        print(f"  {GRAY}Q: {q[:70]}...{RESET}")
        t0 = time.perf_counter()
        try:
            _, meta = answer(q, stream=False)
            results.append({
                "q":       q[:55] + "...",
                "ms":      meta["latency_ms"],
                "tag":     meta["tag"],
                "sim":     meta["top_similarity"],
            })
            print(f"     {GREEN}✅ {meta['latency_ms']}ms  tag={meta['tag']}  sim={meta['top_similarity']}{RESET}")
        except Exception as e:
            print(f"     {YELLOW}⚠️  {e}{RESET}")
            results.append({"q": q[:55], "ms": -1, "tag": "error", "sim": 0})

    # Summary
    success = [r for r in results if r["ms"] > 0]
    if success:
        avg_ms = sum(r["ms"] for r in success) // len(success)
        max_ms = max(r["ms"] for r in success)
        print(f"\n{BOLD}  Benchmark complete{RESET}")
        print(f"  Avg latency: {avg_ms}ms  |  Max: {max_ms}ms  |  Questions: {len(success)}/{len(results)}\n")


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Digital Interview Twin CLI demo")
    parser.add_argument("--once",  action="store_true", help="Single question mode")
    parser.add_argument("--bench", action="store_true", help="Run benchmark suite")
    args = parser.parse_args()

    if args.bench:
        run_benchmark()
    elif args.once:
        run_once()
    else:
        run_interactive()
