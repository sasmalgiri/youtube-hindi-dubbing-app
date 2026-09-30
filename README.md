# YouTube Hindi Dubbing App

An end-to-end platform that takes an English YouTube video (URL or file) and produces a naturally-timed Hindi dub with synchronised subtitles. Built with a Next.js web UI and a FastAPI processing backend.

---

## What it does

Given a YouTube link or uploaded video, the system:

1. Downloads the video and extracts clean audio
2. Transcribes English speech with Whisper (word-level timestamps)
3. Segments transcript into natural sentence units
4. Translates to Hindi with duration-aware LLM prompting (keeps target length close to the source slot)
5. Synthesises Hindi speech with a chosen TTS voice
6. Fits each segment to its original time slot (per-segment stretch + verification)
7. Assembles the final dubbed audio, muxes it back onto the video, and emits SRT / WebVTT subtitles

The result is a dubbed MP4, a Hindi audio track, and subtitle files — all downloadable from the web UI.

---

## Hindi Dialogue profile (multi-speaker)

For videos with conversation, choose the **Hindi Dialogue** mode in the UI, or run
`cd backend && python -m dubbing.dialogue dub <link-or-file>`. It works as follows:

- **Speakers:** each speaker is detected from the audio and keeps one voice throughout.
- **Translation:** dialogue is translated with context, turn by turn.
- **Timing:** the video keeps its original speed.
- **Honest result:** every job ends with a `report.md` and one of these statuses: `completed`,
  `completed_with_warnings` or `draft_incomplete`.

Setup, the doctor/preflight check, options and troubleshooting are in
[docs/dialogue/README.md](docs/dialogue/README.md). The requirement matrix, including which checks
were not run, is in [docs/dialogue/requirements_matrix.md](docs/dialogue/requirements_matrix.md).

---

## Highlights

- **Four pipeline modes** — switchable from the UI (fast / balanced / high-fidelity / studio), each trading speed against accuracy of lip-timing and prosody.
- **Multi-provider resilience** — translation rotates across OpenAI, Gemini, Groq, and Cerebras keys; TTS supports ElevenLabs, Google Cloud TTS, Sarvam, Fish-Speech, and CosyVoice.
- **Slot-accurate timing** — every translated sentence is verified against the original clip's duration, with recompute + global stretch passes so the dub never drifts out of sync.
- **Async job model** — long jobs run in the background; the UI streams live `[N%]` progress for every stage (download, ASR, translate, TTS, fit, assembly).
- **Local-first job store** — SQLite WAL with optional Supabase dual-write for real-time cross-device sync.
- **Persistent state** — saved links, completed URLs, and a translation glossary so re-runs reuse prior decisions.

---

## Architecture

```
┌──────────────────────┐          ┌────────────────────────────────┐
│  Web UI (Next.js)    │  HTTPS   │  Backend API (FastAPI)         │
│  • Job submit/track  │ ───────▶ │  • /jobs, /voices, /presets    │
│  • Live progress     │ ◀─────── │  • Job queue + SQLite store    │
│  • Result download   │          │  • Dubbing pipeline modules    │
└──────────────────────┘          └────────────────────────────────┘
                                             │
                            ┌────────────────┼────────────────┐
                            ▼                ▼                ▼
                       Whisper ASR     LLM Translate      TTS Voices
                    (word timestamps)  (OpenAI/Gemini/    (ElevenLabs/
                                        Groq/Cerebras)     Google/Sarvam/
                                                           Fish-Speech)
```

### Repository layout

```
youtube-dubbing-app
├── web/                          # Next.js frontend (deployed to Vercel)
│   └── src/
│       ├── app/                  # Pages: home, jobs, batch, API routes
│       └── components/           # UI: ProgressPipeline, PresetTabs, VoiceSelector, ...
├── backend/                      # FastAPI server + dubbing engine
│   ├── app.py                    # HTTP API
│   ├── pipeline.py               # Top-level job orchestration
│   ├── jobstore.py               # SQLite + optional Supabase sync
│   ├── cache.py, metrics.py      # Cache + telemetry
│   └── dubbing/
│       ├── oneflow.py            # End-to-end single-flow runner
│       ├── runner.py             # Mode dispatcher
│       ├── asr_runner.py         # Whisper wrapper
│       ├── translation.py        # Multi-provider translate w/ key rotation
│       ├── tts_bridge.py         # Unified TTS interface
│       ├── srtdub.py             # SRT-driven dubbing flow
│       ├── wordchunk.py          # Word-level chunking
│       ├── sentence_segmenter.py # Natural sentence boundaries
│       ├── slot_recompute.py     # Timing slot recomputation
│       ├── slot_verify.py        # Duration verification
│       ├── global_stretch.py     # Global tempo alignment
│       ├── presets.py            # Pipeline-mode presets
│       └── ...
├── src/                          # Standalone CLI (legacy + scripting entry)
├── scripts/                      # Utility scripts
└── tests/                        # Unit tests
```

---

## Quick start (local)

### Prerequisites

- Python 3.10+ with `ffmpeg` on PATH
- Node.js 18+
- At least one LLM API key (Groq free tier works out of the box)

### 1. Backend

```bash
cd backend
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env          # then fill in API keys
uvicorn app:app --host 0.0.0.0 --port 8000
```

### 2. Frontend

```bash
cd web
npm install
# point the UI at your backend
echo "NEXT_PUBLIC_API_BASE_URL=http://localhost:8000" > .env.local
npm run dev
```

Open http://localhost:3000 and paste a YouTube URL.

---

## Configuration

Copy `backend/.env.example` to `backend/.env` and fill the keys you want to use:

| Service       | Purpose                         | Free tier                       |
| ------------- | ------------------------------- | ------------------------------- |
| Groq          | Translation (Llama 3.3 70B)     | Yes                             |
| Gemini        | Translation (Gemma / Gemini)    | Yes (30 RPM per key)            |
| OpenAI        | Translation (GPT-4o)            | Paid                            |
| Cerebras      | Translation (fastest LLM)       | 1M tokens/day                   |
| ElevenLabs    | TTS (premium voices)            | Limited                         |
| Google TTS    | TTS                             | 1M chars/month                  |
| Supabase      | Real-time job sync (optional)   | Yes                             |
| HuggingFace   | Speaker diarization (optional)  | Yes                             |

Only add the keys for services you actually plan to use — any missing provider is skipped gracefully.

---

## Deployment

- **Frontend**: deploy the `web/` folder to Vercel; set `NEXT_PUBLIC_API_BASE_URL` to your backend's public URL.
- **Backend**: any host that runs Python + FFmpeg with GPU access for Whisper (Render, Railway, Fly.io, or a dedicated GPU VM). The repo includes `start.bat` / `start-backend-stable.bat` for local Windows runs.

---

## Optional: Wav2Lip lip-sync

Re-syncs the on-screen speaker's lips to the dubbed audio as a final post-process. Opt-in per job via the **Wav2Lip Lip Sync** toggle in Advanced Settings, or by setting `use_wav2lip: true` on the job request.

**Requirements** (one-time setup):

```bash
# 1. Clone the Wav2Lip repo into backend/
cd backend
git clone https://github.com/Rudrabha/Wav2Lip.git wav2lip

# 2. Download the checkpoint (wav2lip_gan.pth is higher quality)
mkdir -p wav2lip/checkpoints
# Fetch wav2lip_gan.pth from the Wav2Lip repo's README and place it here:
#   backend/wav2lip/checkpoints/wav2lip_gan.pth

# 3. Install Wav2Lip's Python dependencies
pip install -r wav2lip/requirements.txt
```

**Hardware**: GPU strongly recommended — CPU inference is ~10-20× realtime and impractical for real-length videos.

**Graceful failure**: If the repo is missing, the checkpoint isn't present, or inference fails for any reason (no face detected, subprocess error, timeout), the step is skipped and the original assembled video is kept. Never a hard job failure.

Both `backend/wav2lip/` and the checkpoints are in `.gitignore` — never committed.

---

## API surface

| Method | Endpoint              | Purpose                         |
| ------ | --------------------- | ------------------------------- |
| GET    | `/voices`             | List available TTS voices       |
| GET    | `/presets`            | List pipeline-mode presets      |
| POST   | `/jobs`               | Submit a new dubbing job        |
| GET    | `/jobs/{id}`          | Job status + progress stream    |
| GET    | `/jobs/{id}/result`   | Download final assets           |

---

## License

MIT — see the LICENSE file for details.
