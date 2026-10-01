# Hindi Dialogue: module matrix, presets and automatic on/off

Every option of the Hindi Dialogue profile is a **choice** inside a **stage**. Each choice declares what
it needs and what it costs.

You can pick a **preset** and optionally override stages. The resolver
(`backend/dubbing/dialogue/modules.py`) then checks what is installed on **this PC** and:

- **switches off** a choice that cannot run, showing the reason and the exact fix;
- **switches on** the first workable fallback when a stage would otherwise be empty;
- makes dependent stages follow each other. For example, a Hindi SRT turns off speech-to-text and
  translation, and line shortening needs an LLM translator;
- drops **paid** services unless "Allow paid services" is on;
- drops **cloud AI** services when "Local AI only" is on. Edge voices are kept only as the last fallback.

Nothing changes silently. Every change appears in the UI preview, in the job's events and in
`report.json`, under `config.modules`.

## Where to see it

- **UI:** choose **Hindi Dialogue** to get the preset cards, a "Will run on this PC" preview and
  *Customise modules*.
- **CLI:** run `cd backend && python -m dubbing.dialogue modules --preset free-local`. You can override
  stages with `--set voices=indic_parler,edge`.
- **API:**
  - `GET /api/dialogue/modules` returns the matrix with availability on this PC.
  - `GET /api/dialogue/presets` returns the presets.
  - `POST /api/dialogue/resolve` previews a preset plus overrides.
  - A job takes `dialogue_preset` and `dialogue_modules_json`.

## The matrix

| Stage | Choice | Cost | Runs | Needs | If it cannot run |
| --- | --- | --- | --- | --- | --- |
| Text source | Speech-to-text | free | – | – | – |
| | My English SRT | free | – | an uploaded English SRT | falls back to speech-to-text |
| | My Hindi SRT | free | – | an uploaded Hindi SRT | falls back to speech-to-text; when used, **speech-to-text and translation are skipped** |
| Speech-to-text | Automatic | free / free tier | local or cloud | `GROQ_API_KEY` **or** faster-whisper | → Whisper on this PC → Groq |
| | Whisper on this PC | free | local | faster-whisper (GPU recommended) | → Groq |
| | Groq Whisper | free tier | cloud | `GROQ_API_KEY` | → Whisper on this PC |
| Speakers | Detect speakers | free | local | pyannote.audio ≥ 4, torch, `HF_TOKEN` (GPU recommended) | → **Single voice** (reported) |
| | Single voice | free | – | – | – |
| Translation (chain) | Gemini | free tier | cloud | `GEMINI_API_KEY` | next in chain |
| | Groq LLM | free tier | cloud | `GROQ_API_KEY` | next in chain |
| | Cerebras | free tier | cloud | `CEREBRAS_API_KEY` | next in chain |
| | Ollama | free | local | Ollama running + a pulled model | next in chain |
| | IndicTrans2 | free | local | IndicTransToolkit + transformers 4.x (separate venv), `HF_TOKEN` | next in chain |
| | Google basic | free | cloud | deep-translator | next in chain |
| | OpenAI | **paid** | cloud | `OPENAI_API_KEY` + Allow paid | next in chain |
| Voices (chain) | Microsoft Edge | free | online | edge-tts | next provider |
| | Indic Parler-TTS | free | local | parler-tts (separate venv), `HF_TOKEN`, GPU strongly recommended | next provider (whole speaker) |
| | IndicF5 | free | local | curated authorised references in `backend/voices/indicf5/` | next provider |
| | Sarvam / ElevenLabs / Google | **paid** | cloud | key(s) + Allow paid | next provider |
| Background | Keep background | free | local | audio-separator **or** demucs, torch | → **Hindi voice only** (reported) |
| | Hindi voice only | free | – | – | – |
| Speech check | Check speech | free | local | faster-whisper | → **Skip** (reported) |
| Shorten long lines | On | free | – | an **LLM** in the final translation chain | → **Off** |
| Subtitles | Embed / files only | free | – | – | – |

The fallback order for an empty stage is in each stage's `fallback_order` in `modules.py`.

## Presets

| Preset | Speakers | Translation | Voices | Background | Check | Shorten | Paid |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Free — Online (recommended)** | detect | Gemini → Groq → Cerebras → Google basic | Edge | keep | on | on | no |
| **Free — Local AI (unlimited)** | detect | Ollama → IndicTrans2 | Indic Parler → Edge (last resort) | keep | on | on | no, local only |
| **Fast Draft (free)** | detect | Gemini → Groq → Cerebras → Google basic | Edge | none | off | off | no |
| **Single Narrator (free)** | single voice | Gemini → Groq → Cerebras → Google basic | Edge | keep | on | on | no |
| **Premium Voices (paid)** | detect | Gemini → Groq → Cerebras → OpenAI → Google basic | Sarvam → Edge | keep | on | on | yes |
| **My Hindi SRT → Voices** | detect | skipped | Edge | keep | on | off | no |

## Can it all be free? Yes

The **Free — Online** preset uses no paid service:

- Whisper, pyannote, Demucs and the speech check run on your PC.
- Translation uses free tiers. Their daily limits fall back to Google basic, which is flagged in the
  report.
- Voices use Edge. Your `hindi_voices.py` gives up to 14 male and 12 female distinct slots.

**Free — Local AI** removes the API limits for unlimited videos. It has caveats:

- **Edge-TTS** is free but unofficial. Microsoft can rate-limit it (403 errors are reported in
  2025–2026) or change it without notice.
- **Free tiers** have daily limits. Long videos can exhaust them.
- **Indic Parler-TTS** has 4 Hindi speakers: Rohit, Divya, Aman and Rani. The model card states only
  Rohit's gender; the others are inferred from the names. "low"/"high" style variants of the same
  speaker are reported as reuse.
  - It needs a GPU for usable speed; one report says it is not real-time even on a GPU.
  - It has not been run in this repository's CI.
  - The model card's licence is Apache-2.0. Open discussions question commercial use of its outputs
    because of its training data.
- **IndicTrans2** translates line by line, so it can't use dialogue context such as gendered verbs or
  formality across lines. It breaks on transformers 5.x, and IndicTransToolkit has no Windows wheels.
  On Windows use Ollama (e.g. `ollama pull gemma3:12b`), or IndicTrans2 inside WSL.
- **Local model library conflicts:** Indic Parler pins `transformers==4.46.1`, while IndicTrans2 needs
  ≥ 4.51. Each runs in its own persistent worker process, optionally under its own venv (see below).

### Other free/open-source dubbing tools (checked 2026-10-01)

| Tool | Licence | Multi-speaker | Keeps background | Hindi | Notes |
| --- | --- | --- | --- | --- | --- |
| [SoniTranslate](https://github.com/R3gm/SoniTranslate) | Apache-2.0 (model weights may restrict commercial use) | yes (pyannote) | yes (UVR MDX) | yes | Edge, XTTS, Piper, …; last commit 2026-08 |
| [pyVideoTrans](https://github.com/jianchang512/pyvideotrans) | **GPL-3.0** | yes | yes | yes | very active; many TTS engines; Ollama translation |
| [VideoLingo](https://github.com/Huanshere/VideoLingo) | Apache-2.0 | **no** per-speaker voices | yes (Demucs) | via Edge | active |
| [open-dubbing](https://github.com/Softcatala/open-dubbing) | Apache-2.0 | yes (pyannote) | yes (Demucs) | yes | experimental; default MT (NLLB) is non-commercial |

This app's own path is built from the same free building blocks, and adds what those tools lack:
per-turn identity checks, honest draft statuses, and this module matrix.

## Local AI setup (optional)

1. Accept the model terms on Hugging Face with the account whose `HF_TOKEN` you use:
   - `ai4bharat/indic-parler-tts`
   - `ai4bharat/indictrans2-en-indic-dist-200M` (only for IndicTrans2)
2. **Voices:** run `setup_local_ai.bat`. It creates `backend\.venvs\parler` and writes
   `INDIC_PARLER_PYTHON` to `backend\.env`. It was written without a Windows PC; if a step fails, send
   the error.
3. **Translation:** install Ollama, run `ollama pull gemma3:12b` (about 8 GB, fits a 12 GB GPU) and
   set `OLLAMA_MODEL=gemma3:12b` in `backend\.env`. If no model is set, the app picks your first pulled
   model and says so.
4. Check with `python -m dubbing.dialogue modules --preset free-local`.

Background separation runs first, before any local model is loaded, and frees its GPU memory when
it finishes. The local models load once per job, and their worker processes are closed before the
final mix.

## Verification status

- **Checked here:** all resolver rules, presets and the UI/API wiring (unit and API tests); the
  worker protocol (with a stand-in worker); provider binding.
- **Not run here:** the real Indic Parler-TTS, IndicTrans2 and Ollama models (no GPU or weights in
  the cloud build environment); `setup_local_ai.bat` on Windows; listening quality.
