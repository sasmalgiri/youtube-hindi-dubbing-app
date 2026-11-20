# Source inventory (Stage A, §1.1)

Base SHA: `dcaa0a07e2c341cd635e7b095fd8cd17a6405603` (master, same as the commit the release
instructions inspected) plus the uncommitted `backend/dubbing/storypkg/` work in progress.
Recorded 2026-11-20 on Windows 11 Pro, Python 3.10, Node per `web/package.json` (Next.js 14.2.25),
torch 2.4.1+cu121, RTX 3060 12 GB.

**Hosted column:** `allow` = may be exposed on the public hosted profile once its tracker rows are
verified; `deny` = local/desktop profile only; `replace` = must be rebuilt behind the shared intake,
ownership and billing services first. Nothing is exposed publicly until the `features` registry
(`backend/features.py`, tracker X01) and the hosted allowlist (S03) say so.

Trust boundary key: **U** = untrusted customer input, **P** = third-party provider call,
**FS** = local filesystem, **N** = outbound network to arbitrary hosts.

## Backend API routes (`backend/app.py`, 50 routes)

| Route | Purpose | Data touched | Boundary | Hosted | Owner | Test |
| --- | --- | --- | --- | --- | --- | --- |
| `GET /api/health` | liveness, provider status | env, GPU | – | allow | Eng | – |
| `GET/POST/DELETE /api/glossary…` | global glossary | `glossary.json` (process-wide) | U, FS | replace (per account) | Eng | – |
| `GET/DELETE /api/cache`, `POST /api/cache/purge` | ASR/source cache admin | cache dir | FS | deny (admin only) | Eng | – |
| `GET /api/voices` | Edge voice list | Edge | P | allow | Eng | – |
| `GET /api/cookies/status` | local YouTube cookie status | browser cookies | FS | **deny** (S02) | Eng | – |
| `GET /api/jobs` | list **all** jobs | every job | – | **replace** (S01 ownership) | Eng | – |
| `POST /api/jobs` | start job from link | URL → yt-dlp | U, N, P | **replace** (P01 intake, rights) | Eng | partial |
| `POST /api/jobs/upload` | start job from upload | upload | U, FS | replace (bounded stream, ownership) | Eng | partial |
| `POST /api/jobs/story`, `GET …/story` | story text → video | story, GPU | U, P | deny (local GPU feature) | Eng | `test_story.py` |
| `POST /api/jobs/with-srt` | SRT + video job | SRT, video | U, N | replace (P01 SRT roles) | Eng | partial |
| `GET /api/jobs/{id}` `…/events` `…/transcript` `…/report` `…/qa` `…/description` `…/manual-review` | job status and artifacts by ID | job | – | **replace** (ownership on every read) | Eng | partial |
| `GET /api/jobs/{id}/result` `…/srt` `…/source-srt` `…/original` | downloads by ID | media | – | **replace** (ownership, signed URLs) | Eng | – |
| `POST /api/jobs/{id}/compare-translations` | side-by-side MT | transcript | P | replace | Eng | – |
| `POST /api/jobs/{id}/continue` `…/resume-with-srt` | step-by-step/resume | job | U | replace (ownership, billing) | Eng | – |
| `GET/POST /api/jobs/{id}/dialogue/review`, `POST …/revoice`, `GET …/clip/{name}` | review & revoice | job, clips | U, P | replace (ownership, billing for revoice) | Eng | dialogue tests |
| `GET /api/dialogue/modules` `…/presets`, `POST …/resolve` | module matrix | probe | – | allow (read-only) | Eng | `test_modules*` |
| `GET /api/dialogue/voices`, `POST …/voice-preview` | voice list and preview | provider | P | allow (rate-limited) | Eng | – |
| `DELETE /api/jobs/{id}` | delete job | job, files | – | replace (ownership, P09 consequences) | Eng | – |
| `GET /api/outputs` | list saved output folder | `D:\…` folders | FS | **deny** | Eng | – |
| `GET/POST/PATCH/DELETE /api/links…` | saved links | `links` table (global) | U | replace (per account) | Eng | – |
| `GET/POST/DELETE /api/presets…` | presets | presets (global) | U | replace (per account; builtins allow) | Eng | – |

## Web (Next.js)

| Path | Purpose | Boundary | Hosted |
| --- | --- | --- | --- |
| `web/src/app/page.tsx` | creator form (URL, upload, batch, SRT, story) | U | replace (P01/P02) |
| `web/src/app/jobs/[id]/page.tsx` | progress and results | – | replace (ownership, P07/P08) |
| `web/src/app/batch/page.tsx` + `hooks/useBatchManager.ts` | batch queue in **sessionStorage** | – | replace (S04 server batches) |
| `web/src/app/api/local-dub/route.ts` | spawns downloader, buffers whole video in Node, forwards | U, N, FS | **deny** (S02) |

## Entry points, workers, launchers

| Entry | Purpose | Hosted |
| --- | --- | --- |
| `VoiceDub.bat`, `start*.bat`, `setup*.bat`, `desktop.py` | Windows desktop launch/install | deny (desktop profile) |
| `backend/app.py` threads + `_pipeline_semaphore = Semaphore(1)` | in-process job execution | **replace** (S03 durable queue/workers) |
| `backend/dubbing/story/wan_worker.py`, `dialogue/workers/*`, diarization child processes | model child processes | deny until worker isolation (R-rows) |
| `python -m dubbing.dialogue` | CLI | deny |
| `colab_backend.ipynb` | Colab + ngrok tunnel (reason for `allow_origins=["*"]`) | **deny** |
| `HINGLISH_TRAINER_URL` (`_send_training_data`, 3 call sites) | forwards customer source/target SRT to a trainer | **deny** (S03: must be impossible in hosted) |
| scheduled tasks, migrations, CI | **none exist** | A02 adds CI |

## Not first-party / separate provenance (§1.1)

| Item | Note |
| --- | --- |
| `CosyVoice` gitlink | submodule pointer without `.gitmodules`; licence and weights unverified (L07) |
| Model weights (faster-whisper, pyannote community-1/3.1, wav2vec2 gender, Demucs/audio-separator, IndicF5, Indic Parler, IndicTrans2, Wan 2.1, chatterbox) | downloaded at runtime into HF cache; each needs a licence row (L07) |
| FFmpeg build | system binary; codec licence (GPL/LGPL) unverified (L07) |
| Providers | Edge TTS (unofficial consumer endpoint), Sarvam, ElevenLabs, Google TTS, Groq, Gemini, Cerebras, SambaNova, Google free translate, Ollama, YouTube/yt-dlp |

## Hygiene findings

- `backend/pipeline-DESKTOP-UJHC7A1*.py` (3 files) are tracked OneDrive conflict copies of
  `pipeline.py`: dead code, not imported. Keep (no deletion per owner), mark `deny` and exclude
  from packaging.
- `backend/test_google_tts.py`, `backend/test_sarvam_tts.py`, `backend/test_srtdub_runtime.py` are
  manual scripts that **call paid/live providers**. They are not pytest tests and must stay out of CI.
- Four dependency manifests (`requirements.txt`, `requirements-gpu.txt`, `backend/requirements*.txt`,
  `pyproject.toml`) plus `backend/constraints.txt`; no lock file for Python (A02).
