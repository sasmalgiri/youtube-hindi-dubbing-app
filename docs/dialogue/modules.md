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
| | My Hindi SRT | free | – | an uploaded Hindi SRT | **blocks** the job (switch the input to SRT mode); when used, **speech-to-text and translation are skipped** |
| Speech-to-text | Automatic | free / free tier | local or cloud | `GROQ_API_KEY` **or** faster-whisper | → Whisper on this PC → Groq |
| | Whisper on this PC | free | local | faster-whisper (GPU recommended) | → Groq |
| | Groq Whisper | free tier | cloud | `GROQ_API_KEY` | → Whisper on this PC |
| Speakers | Detect speakers | free | local | pyannote.audio ≥ 4, torch, `HF_TOKEN` or `HUGGINGFACE_TOKEN` (GPU recommended) | → **Single voice** (reported) |
| | Single voice | free | – | – | – |
| Translation (chain) | Gemini | free tier | cloud | `GEMINI_API_KEY` | next in chain |
| | Groq LLM | free tier | cloud | `GROQ_API_KEY` | next in chain |
| | Cerebras | free tier | cloud | `CEREBRAS_API_KEY` | next in chain |
| | Ollama | free | local | Ollama running + a pulled instruct model named in `OLLAMA_MODEL` or the *Ollama model* option (never guessed) | next in chain |
| | IndicTrans2 | free | local | IndicTransToolkit + transformers 4.x (in the app's Python here; a separate venv via `INDICTRANS2_PYTHON` also works), `HF_TOKEN` with access to the model (`INDICTRANS2_MODEL`, default `ai4bharat/indictrans2-en-indic-1B`; checked on huggingface.co, a check that cannot finish is only a warning) | next in chain |
| | Google basic | free | cloud | deep-translator | next in chain |
| | OpenAI | **paid** | cloud | `OPENAI_API_KEY` + Allow paid | next in chain |
| Voices (chain) | Microsoft Edge | free | online | edge-tts | next provider |
| | Indic Parler-TTS | free | local | parler-tts (separate venv), `HF_TOKEN`, GPU strongly recommended | next provider (whole speaker) |
| | IndicF5 | free | local | curated authorised references in `backend/voices/indicf5/` | next provider |
| | Sarvam / ElevenLabs / Google | **paid** | cloud | key(s) + Allow paid | next provider |
| Sound like the original speaker (**experimental**) | Off (default) | free | – | – | – |
| | OpenVoice tone colour | free | local | OpenVoice in its own venv (`OPENVOICE_PYTHON`), GPU strongly recommended | → **Off** (reported) |
| Background | Keep background | free | local | audio-separator **or** demucs, torch | → **Hindi voice only** (reported) |
| | Hindi voice only | free | – | – | – |
| Speech check | Check speech | free | local | faster-whisper | → **Skip** (reported) |
| Shorten long lines | On | free | – | an **LLM** in the final translation chain | → **Off** |
| Subtitles | Embed / files only | free | – | – | – |

The fallback order for an empty stage is in each stage's `fallback_order` in `modules.py`.

Job options that are not modules pass straight through to the job when set in the
`params` of `dialogue_modules_json` (or the API's own job fields): `review_before_voice`,
`keep_original_audio`, `english_subtitles`, `burn_subtitles` (true/false) and `container`
(`mp4` or `mkv`; anything else means `mp4`). Left out, the job's defaults apply.

## Sound like the original speaker (experimental)

Off by default in every preset. When on, every Hindi clip keeps its words, pronunciation and
timing, but its voice **colour** (timbre) is changed toward the original speaker's own voice. The
Hindi still comes from the voice engine (Edge, Indic Parler, …): OpenVoice does not speak Hindi
itself, it only recolours what the engine said.

- **Engine:** the tone colour converter of [OpenVoice v2](https://github.com/myshell-ai/OpenVoice)
  (MIT), checkpoint `converter/` from
  [myshell-ai/OpenVoiceV2](https://huggingface.co/myshell-ai/OpenVoiceV2) (about 130 MB, downloaded
  into `backend/models/openvoice` on first use; to skip the download, copy `config.json` and
  `checkpoint.pth` from OpenVoice's `checkpoints_v2/converter` into
  `backend/models/openvoice/converter/`). The API calls follow OpenVoice's demo and SoniTranslate.
- **Own Python env:** OpenVoice pins numpy 1.22.0 and librosa 0.9.1, so it runs in a persistent
  worker under `OPENVOICE_PYTHON`, like Indic Parler-TTS. `setup_local_ai.bat` offers it at the end;
  `setup_local_ai.bat openvoice` sets up only this. It is never installed into the app's Python.
  The app does not require OpenVoice's demo UI and ASR pins (gradio, faster-whisper 0.9,
  whisper-timestamped): the converter never imports them, so the setup retries without them if
  they fail to install.
- **References:** each speaker's reference is cut from that speaker's own clean speech in the
  video (separated vocals when available). A speaker with **less than 3 s** of reference speech is
  left with the plain Hindi voice, and the report says so.
- **Timing never changes:** a converted clip keeps the clip's sample rate and exact length (the
  converter's output is at most about one 12 ms hop off, which is padded or trimmed at the end);
  anything else is refused and the clip keeps its Hindi voice. Any failure leaves the clip as the
  voice engine made it and is listed in the report; the job continues.
- **Speed:** **needs a GPU for usable speed.** On a 4-core CPU a clip took about as long as it
  lasts (a 2.7 s clip in about 2 s), so a long video on the CPU adds a lot of time.
- **Watermark:** OpenVoice adds its own inaudible watermark to the audio it converts.
- **Quality:** experimental. How close the result gets varies by speaker and by the Hindi voice it
  starts from; music or noise left in the reference makes it worse. Listen before you publish.
  Use it only for voices you have the right to imitate.

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
  formality across lines. It breaks on transformers 5.x. On this PC it runs in the app's own Python
  3.10 (IndicTransToolkit 1.1.1 + transformers 4.51.3), no separate venv needed.
- **Local model library conflicts:** Indic Parler pins `transformers==4.46.1`, while IndicTrans2 needs
  ≥ 4.51. Each runs in its own persistent worker process. Indic Parler is offered only when its Python
  meets those pins, so it shows as missing until `setup_local_ai.bat` has given it its own venv.

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
   - `ai4bharat/indictrans2-en-indic-1B` (only for IndicTrans2). Each checkpoint is gated on its own:
     if you set `INDICTRANS2_MODEL` to another one, accept that one's terms too.
2. **Voices:** run `setup_local_ai.bat`. It creates `backend\.venvs\parler` and writes
   `INDIC_PARLER_PYTHON` to `backend\.env`. It uses Python 3.10 and installs only into that venv, never
   into the app's own Python. It has not been run end to end yet; if a step fails, send the error.
3. **Translation:** install Ollama, run `ollama pull gemma3:12b` (about 8 GB, fits a 12 GB GPU) and
   set `OLLAMA_MODEL=gemma3:12b` in `backend\.env`, or pick the model under *Ollama model*. If none is
   set, Ollama is switched off and the preview shows that fix. The app never guesses from your pulled
   models: a small fine-tune that cannot return the translator's JSON would translate nothing.
4. **Sound like the original speaker (optional, experimental):** answer `y` when
   `setup_local_ai.bat` offers OpenVoice, or run `setup_local_ai.bat openvoice`. It creates
   `backend\.venvs\openvoice` and writes `OPENVOICE_PYTHON` to `backend\.env`. Then pick it under
   *Customise modules*. `python -m dubbing.dialogue doctor` shows whether it can run.
5. Check with `python -m dubbing.dialogue modules --preset free-local`.

Background separation runs first, before any local model is loaded, and frees its GPU memory when
it finishes. The local models load once per job, and their worker processes are closed before the
final mix.

## Verification status

- **Checked here:** all resolver rules, presets and the UI/API wiring (unit and API tests); the
  worker protocol (with a stand-in worker); provider binding.
- **OpenVoice:** run for real once in the cloud build environment (Linux, CPU, Python 3.10 venv
  with torch 2.4.1 and OpenVoice without its demo/ASR pins): checkpoint download, reference
  embedding with its cache, conversion of 24 kHz and 48 kHz clips with their exact length kept. The
  unit tests use a stand-in worker and the real worker script against stub libraries.
- **Not run here:** the real Indic Parler-TTS, IndicTrans2 and Ollama models (no GPU or weights in
  the cloud build environment); OpenVoice on a GPU and on real speech; `setup_local_ai.bat` on
  Windows; listening quality.
