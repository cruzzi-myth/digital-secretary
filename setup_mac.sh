#!/bin/bash
# setup_mac.sh — Run this once on your Mac to install deps and populate the brain
# Usage: bash setup_mac.sh
# Then: export GROQ_API_KEY=gsk_... && python3 demo.py

set -e
cd "$(dirname "$0")"

GREEN='\033[0;32m'; BLUE='\033[0;34m'; NC='\033[0m'
info() { echo -e "${BLUE}  ➜  $1${NC}"; }
pass() { echo -e "${GREEN}  ✅ $1${NC}"; }

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  Digital Interview Twin — Mac Setup (BM25 edition)${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

info "Installing Python dependencies (groq, rank-bm25, pdfplumber)..."
pip3 install -r requirements.txt --break-system-packages
pass "Dependencies installed"

echo ""
info "Ingesting docs into brain.json (no ML model — finishes in seconds)..."
python3 ingest.py --reset
pass "Brain populated!"

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  Setup complete! Next steps:${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  1. Get a free Groq key at: https://console.groq.com"
echo "  2. export GROQ_API_KEY=gsk_..."
echo "  3. python3 demo.py          ← interactive interview mode"
echo "  4. python3 demo.py --bench  ← run latency benchmark"
echo ""
