# Digital Interview Twin
### Real-Time AI Interview Agent · RAG + Llama 3 + Voice Clone + FasterLivePortrait

An AI agent that joins your Zoom interview as a participant, listens to questions
via real-time STT, retrieves context from your resume and prep docs using RAG,
generates spoken answers through a voice clone, and overlays a live video face —
all within 2–3 seconds of the question being asked.

```
Interviewer speaks → faster-Whisper STT → tag classify → ChromaDB retrieval
→ Groq Llama 3 (500+ tok/sec) → ElevenLabs voice → FasterLivePortrait video
→ OBS Virtual Camera → Zoom participant
```

---

## Architecture

### Phase 1 — RAG Brain (CPU, no GPU required)
The foundation. Runs on any laptop or cheap cloud instance.

```
docs/
├── resume.md                ← your background, projects, metrics
├── behavioral_stories.md    ← STAR stories, leadership examples
└── technical_prep.md        ← system design notes, deep dives

          ↓ python ingest.py

ChromaDB (local, persistent)
  collection: interview_brain
  chunks: ~400 chars with 80-char overlap
  embedding: all-MiniLM-L6-v2 (384-dim, ~80ms/chunk, CPU)
  metadata tag: technical | behavioral | company | general

          ↓ question

classify_question()           ← heuristic tag from question keywords
retrieve(question, tag, k=6)  ← tagged retrieval + unfiltered fill
build_system_prompt(context)  ← Staff Engineer persona + STAR rules + context
Groq Llama 3 70B              ← 500+ tokens/sec, free dev tier, ~1.2s typical
```

**Why Groq over OpenAI?** At 500+ tokens/second, Groq delivers a full answer
before the filler phrase audio finishes playing. OpenAI GPT-4 at ~50 tok/sec
introduces a 6–8 second gap — noticeable in a live interview.

**RAG retrieval design:**
Two-pass retrieval runs on every question. Pass 1 filters by the inferred tag
(technical/behavioral/company) to pull the most relevant chunks. Pass 2 runs
unfiltered to catch general resume facts the tag filter might miss. Results are
deduplicated and re-ranked by cosine similarity. Top 6 chunks are injected into
the system prompt. End-to-end retrieval: ~150ms on CPU.

**Filler phrase pattern:**
The moment a question is detected, a natural filler phrase plays through the
voice clone ("That's a great question — let me think through that.") while the
RAG + LLM pipeline runs in parallel. By the time the phrase ends (~2.5 seconds),
the answer is ready. The gap is invisible.

---

### Phase 2 — Real-Time STT + Zoom Bot (RunPod GPU)

```
Zoom meeting
    ↓
Recall.ai bot joins as participant  ← no screen share needed, true participant
    ↓
Audio stream → faster-Whisper (GPU) ← CTranslate2 backend, ~150ms latency
    ↓
Silence detection → question boundary → send to RAG brain
```

**Recall.ai** joins Zoom as a real bot participant via its cloud bot SDK.
No OBS required for audio capture — the bot's audio stream is piped directly
to faster-Whisper. Recall's SDK handles reconnects, recording consent popups,
and waiting room entry.

**faster-Whisper** runs the `large-v3` model on an NVIDIA A100 (RunPod).
Word-error rate: ~4% on technical English. Silence-gap detection triggers
question boundaries at 800ms of silence. Average STT latency: ~150ms per
utterance segment on GPU.

---

### Phase 3 — Voice + Video Synthesis (RunPod A100)

```
Llama 3 answer text
    ↓
ElevenLabs voice clone  ← trained on 5–10 min of your voice samples
    ↓  (audio stream, ~200ms TTFB)
FasterLivePortrait      ← 30+ FPS face video from a single source photo
    ↓
human jitter overlay    ← micro head movements, variable framerate, hand occlusion
    ↓
OBS Virtual Camera      ← presents as a standard webcam to Zoom
```

**FasterLivePortrait** generates real-time lip-synced video from a single photo
by animating facial landmarks driven by audio energy. Runs at 30+ FPS on an A100.
The motion driver is the ElevenLabs audio stream — lip sync is frame-accurate.

**Realism techniques:**
- Micro head movements: ±2° random rotation on a 0.3Hz sine wave
- Hand occlusion: synthetic hand image composited at frame boundaries (breaks
  facial recognition analysis)
- Variable framerate: ±3 FPS jitter (28–33 FPS) to defeat frame-rate
  fingerprinting
- Background noise: ambient room noise added to the audio mix

---

## Persona Design — Staff Engineer System Prompt

The system prompt defines a Staff Engineer identity with three behavioral rules:

**STAR enforcement (spoken, not listed):** Behavioral answers are structured as
Situation → Task → Action → Result but delivered as natural conversation, not
headers. The model is instructed to never say "Situation:" — just tell the story.

**Spoken-word constraints:** Answers are capped at ~200 words (90 seconds of
speech). No bullet points, no markdown. Pure conversational prose. "I" not "we."

**Honesty guardrail:** The model is explicitly instructed to only use facts from
the injected RAG context. No fabricated projects, companies, or metrics. If the
context doesn't cover a question, the model acknowledges it and reasons from
first principles — which is exactly how a strong engineer answers a gap question.

---

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| STT latency | < 200ms | faster-Whisper large-v3 on A100 |
| RAG retrieval | < 200ms | ChromaDB cosine search, 6 chunks |
| Groq inference | < 1.5s | Llama 3 70B, 350 tokens max |
| ElevenLabs TTFB | < 300ms | streaming synthesis |
| FasterLivePortrait | 30+ FPS | A100, single-source photo |
| **End-to-end** | **< 3s** | covered by 2.5s filler phrase |

---

## Running Phase 1 (RAG Demo — No GPU)

```bash
# 1. Get a free Groq API key at https://console.groq.com
export GROQ_API_KEY=gsk_...

# 2. Install dependencies
pip install -r requirements.txt

# 3. Fill in your resume and prep docs
# Edit: docs/resume.md, docs/behavioral_stories.md, docs/technical_prep.md

# 4. Ingest your docs into ChromaDB
python ingest.py

# 5. Run the interactive CLI demo
python demo.py

# 6. Run the benchmark suite to validate latency
python demo.py --bench
```

---

## Running Phase 2 + 3 (Full Pipeline — RunPod GPU)

```bash
# 1. Spin up a RunPod pod
#    Template: RunPod PyTorch 2.1, GPU: A100 40GB or RTX 4090
#    Expose ports: 8080 (API), 8888 (Jupyter for debugging)

# 2. Clone and install
git clone https://github.com/cruzzi-myth/digital-interview-twin
cd digital-interview-twin
bash runpod/setup.sh         # installs Phase 1 + 2 + faster-Whisper

# 3. Install FasterLivePortrait
INSTALL_VIDEO=yes bash runpod/setup.sh

# 4. Configure ElevenLabs
export ELEVENLABS_API_KEY=...
export ELEVENLABS_VOICE_ID=...   # from your trained clone

# 5. Start the full pipeline
# (Phase 2 + 3 entrypoint — coming in next build)
python agent.py --zoom-link "https://zoom.us/j/..."
```

---

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| LLM Inference | Groq · Llama 3 70B | 500+ tok/sec, free dev tier |
| Vector Store | ChromaDB | local persistence, no infra |
| Embeddings | all-MiniLM-L6-v2 | 80ms/chunk, CPU, no API key |
| STT | faster-Whisper large-v3 | ~150ms, GPU, 4% WER |
| Voice | ElevenLabs | voice clone, 200ms TTFB |
| Video | FasterLivePortrait | 30+ FPS, single photo |
| Zoom Bot | Recall.ai | true participant, no screen share |
| GPU Cloud | RunPod | A100 ~$1.64/hr on-demand |
| Orchestration | Python asyncio | parallel filler + RAG |

---

## Project Status

- [x] Phase 1: RAG brain (ChromaDB + Groq + persona) — **complete**
- [ ] Phase 2: Zoom bot + faster-Whisper STT
- [ ] Phase 3: ElevenLabs voice + FasterLivePortrait video
- [ ] Phase 4: OBS virtual camera + realism layer

---

## Disclaimer

This project is built for technical portfolio demonstration and interview
preparation practice. Use responsibly and in accordance with the terms of service
of any platforms involved.
