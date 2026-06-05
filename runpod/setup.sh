#!/bin/bash
# setup.sh — Digital Interview Twin / RunPod GPU setup
# Run once after spinning up a RunPod pod (NVIDIA A100 or RTX 4090)
#
# Usage:
#   bash runpod/setup.sh
#
# What this installs:
#   Phase 1: RAG brain (chromadb, groq, sentence-transformers)
#   Phase 2: faster-whisper STT (GPU-accelerated)
#   Phase 3: FasterLivePortrait (video synthesis)

set -e

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info() { echo -e "${BLUE}  ➜  $1${NC}"; }
pass() { echo -e "${GREEN}  ✅ $1${NC}"; }
warn() { echo -e "${YELLOW}  ⚠️  $1${NC}"; }

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  Digital Interview Twin — RunPod Setup${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# ─── Verify GPU ───────────────────────────────────────────────────────────────
info "Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || warn "No GPU detected — Phase 2/3 features will not work"
pass "GPU check done"

# ─── Python env ───────────────────────────────────────────────────────────────
info "Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip --quiet
pass "Virtual environment ready"

# ─── Phase 1: RAG brain ───────────────────────────────────────────────────────
info "Installing Phase 1 (RAG brain)..."
pip install groq chromadb sentence-transformers pdfplumber numpy --quiet
pass "Phase 1 installed"

# ─── Phase 2: STT (faster-whisper on GPU) ────────────────────────────────────
info "Installing Phase 2 (faster-whisper STT)..."
pip install faster-whisper pyaudio --quiet
pass "Phase 2 installed"

# ─── Phase 3: FasterLivePortrait (optional — large install) ──────────────────
INSTALL_VIDEO=${INSTALL_VIDEO:-"no"}
if [ "$INSTALL_VIDEO" = "yes" ]; then
  info "Installing Phase 3 (FasterLivePortrait)..."
  git clone https://github.com/warmshao/FasterLivePortrait --depth=1
  pip install -r FasterLivePortrait/requirements.txt --quiet
  pass "Phase 3 installed"
else
  warn "Skipping FasterLivePortrait (set INSTALL_VIDEO=yes to include)"
fi

# ─── Verify GROQ_API_KEY ──────────────────────────────────────────────────────
if [ -z "$GROQ_API_KEY" ]; then
  warn "GROQ_API_KEY not set!"
  echo ""
  echo "  Get your free key at: https://console.groq.com"
  echo "  Then run: export GROQ_API_KEY=gsk_..."
  echo ""
else
  pass "GROQ_API_KEY is set"
fi

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  Setup complete — next steps:${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  1. Fill in your resume:     docs/resume.md"
echo "  2. Add behavioral stories:  docs/behavioral_stories.md"
echo "  3. Add technical prep:      docs/technical_prep.md"
echo "  4. Ingest your docs:        python ingest.py"
echo "  5. Run the demo:            python demo.py"
echo ""
