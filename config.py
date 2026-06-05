"""
config.py — Digital Interview Twin
Central config for Groq, ChromaDB, and persona settings.
All secrets come from environment variables — never hardcode keys.
"""

import os

# ─── Groq LLM ─────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")           # export GROQ_API_KEY=gsk_...
GROQ_MODEL   = "llama-3.3-70b-versatile"                # 500+ tok/sec, free dev tier
GROQ_TIMEOUT = 8                                        # seconds — must beat filler phrase

# ─── ChromaDB ─────────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), ".chroma_db")
CHROMA_COLLECTION  = "interview_brain"

# ─── Embedding model (local, no API key needed) ───────────────────────────────
EMBED_MODEL = "all-MiniLM-L6-v2"                       # 384-dim, ~80ms/chunk on CPU

# ─── RAG retrieval ────────────────────────────────────────────────────────────
TOP_K          = 6       # chunks returned per query
CHUNK_SIZE     = 400     # characters
CHUNK_OVERLAP  = 80      # characters — preserves context across boundaries

# ─── Metadata tags for RAG filtering ─────────────────────────────────────────
# Chunks are tagged at ingest time so retrieval can filter by question type.
TAG_TECHNICAL  = "technical"    # architecture, code, system design
TAG_BEHAVIORAL = "behavioral"   # STAR stories, leadership, conflict resolution
TAG_COMPANY    = "company"      # target-company research, role context
TAG_GENERAL    = "general"      # resume facts, bio, education, contact

# ─── Persona ──────────────────────────────────────────────────────────────────
PERSONA_LEVEL = "staff"         # "senior" | "staff" | "principal"
RESPONSE_STYLE = "concise"      # "concise" (spoken) | "detailed" (written)

# ─── Filler audio phrases (played while RAG processes) ────────────────────────
FILLER_PHRASES = [
    "That's a great question — let me think through that.",
    "Good question. So, the way I've approached this...",
    "Yeah, I've dealt with this exact problem before.",
    "Interesting — let me walk you through my thinking on this.",
]

# ─── Validation ───────────────────────────────────────────────────────────────
def validate():
    if not GROQ_API_KEY:
        raise EnvironmentError(
            "GROQ_API_KEY not set.\n"
            "  Get a free key at https://console.groq.com\n"
            "  Then: export GROQ_API_KEY=gsk_..."
        )
