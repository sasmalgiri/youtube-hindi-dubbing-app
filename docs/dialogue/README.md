# Hindi Dialogue profile

`pipeline_mode = "hindi_dialogue"` turns an English video (a link or a local file) into a Hindi-dubbed
MP4. Different speakers keep consistent voices, the video plays at its original speed, and each job
produces Hindi subtitles and a machine-readable quality report.

Use it for conversations, interviews, films and any video with more than one speaker. The older modes
(Classic, Hybrid, New, OneFlow, WordChunk, SRT Direct) are unchanged for single-narrator work.

## What it does

```
link / file ─► download ─► extract audio ─► text source ─► AUDIO diarization ─► word attribution
            ─► speaker turns ─► speaker registry (one stable voice per speaker)
            ─► contextual Hindi translation (JSON keyed by turn ID)
            ─► TTS via the voice resolver ─► fit (slack → faithful rewrite → ≤1.15× stretch)
            ─► verification (clip checks, Hindi re-ASR, coverage, identity)
            ─► multi-track mix (+ separated background if available) ─► MP4 + SRT/VTT + report
```

Guarantees, each enforced in code and covered by tests:

- **Speaker detection always runs.** Diarization runs whether the text came from Whisper, YouTube
  subtitles or an SRT you supplied. If diarization is unavailable, the report says so; it is never a
  silent single-voice dub.
- **Speakers never merge.** Two different speakers are never merged into one turn. Short replies such
  as "Yes." and interruptions stay separate turns.
- **Voices stay fixed per speaker.** Each speaker keeps one voice across retries, rewrites and
  fallbacks. When the voice pool is too small (Edge has one Hindi male and one Hindi female voice),
  speakers share a base voice with a distinct pitch variant, and the report lists this as reuse, not
  as unique voices.
- **Original timing is kept.** Source timing is never modified. The video speed is never changed and
  audio is never truncated. A turn that cannot fit is kept whole and reported, with its exact overflow
  in seconds.
- **Missing dialogue is reported.** Every required turn must have accepted audio, counted by unique
  turn ID. Otherwise the job is a **draft**, and the missing turn IDs and times are listed.
- **English is never used as background.** If background separation fails, the output is Hindi
  dialogue only.
- **Nothing is reused across jobs.** Each job starts from scratch. The legacy cross-job ASR cache is
  bypassed.

## Result statuses

| Status | Meaning |
| --- | --- |
| `completed` | Every required turn is voiced in its speaker's bound voice, with no unresolved warning. |
| `completed_with_warnings` | Output is complete, but something needs a listen: content mismatch, fallback provider, small timing overflow, diarization unavailable, unverified speech, and so on. |
| `draft_incomplete` | Something is missing or wrong: a missing turn, a voice/identity problem, a large timing overflow, or a speaker still voiced by two providers. The MP4 is kept for review. |
| `failed` / `cancelled` | Partial assets and `report.md` are kept in the job folder. |

A status reached by automated checks does **not** certify how natural the voices sound or how good
the Hindi is. Listen before publishing.

## Setup (Windows desktop, one time)

1. Run the normal setup (`setup.bat`) so the base app works.
2. Install the optional dialogue dependencies, preferably in a fresh venv first:
   `pip install -r backend\requirements-dialogue.txt`.
   Add `torch` with CUDA from pytorch.org for your GPU.
3. Speaker diarization needs a Hugging Face token:
   - Accept the conditions of `pyannote/speaker-diarization-community-1` on huggingface.co.
   - Put `HF_TOKEN=hf_...` in `backend\.env`.
4. Translation uses the LLM keys the app already reads (`GEMINI_API_KEY`, `GROQ_API_KEY`,
   `CEREBRAS_API_KEY`). If none is set, a non-contextual Google fallback is used and every affected
   turn is flagged.
5. Check readiness: `cd backend` then `python -m dubbing.dialogue doctor`. This reports only
   *whether* each credential is set, never its value.

## Running

- **Web / desktop app:** pick the **Hindi Dialogue** mode button, then paste a link or upload a
  file. When the job finishes, the job page shows the status and an **Open report** link. Uploading
  a Hindi SRT with "SRT provided" skips translation; speakers still come from the audio.
- **CLI:**

  ```
  cd backend
  python -m dubbing.dialogue dub "https://www.youtube.com/watch?v=..." --out ..\dubs\talk
  python -m dubbing.dialogue dub D:\clips\scene.mp4 --speakers 3
  python -m dubbing.dialogue dub scene.mp4 --srt-en scene.en.srt      # English text supplied
  python -m dubbing.dialogue dub scene.mp4 --srt-hi scene.hi.srt      # Hindi text supplied
  python -m dubbing.dialogue dub scene.mp4 --limit-seconds 180        # try the first 3 minutes
  ```

Outputs (in `--out`, and copied to `dubbed_outputs\<title> [HI Dialogue <status>] (<job>)` for UI
jobs):

- `dubbed_hi.mp4`: the original video stream, the Hindi mix and soft Hindi subtitles.
- `hindi_mix.wav`
- `dialogue_track_N.wav`: one stem per simultaneous-speech track.
- `subtitles_hi.srt` / `.vtt`: timed to the dubbed audio.
- `transcript_en_speakers.srt`
- `report.md` / `report.json`
- `turns.json`, `clips.json`, `speakers.json`

## Presets and modules

Pick a preset in the UI (Free — Online, Free — Local AI, Fast Draft, Single Narrator, Premium
Voices, My Hindi SRT) or use `--preset` on the CLI. Options that cannot run on your PC are switched
off automatically, with the reason and the fix shown. Fallbacks switch on automatically.

The full matrix, the dependency rules and the free / local-AI setup are in
[modules.md](modules.md). To see what a preset becomes on your PC, run:

```
cd backend
python -m dubbing.dialogue modules --preset free-local
```

## Options

| Option (API field / CLI flag) | Default | Notes |
| --- | --- | --- |
| `dialogue_tts_providers` / `--providers` | `edge` | Priority list. `sarvam`, `elevenlabs` and `google` are paid and used **only if listed**. ElevenLabs needs `ELEVENLABS_VOICES_MALE` and `ELEVENLABS_VOICES_FEMALE` (comma-separated voice IDs). Google needs `GOOGLE_TTS_API_KEY`. `indicf5` is experimental (see below). |
| `dialogue_translation_engines` / `--engines` | `gemini,groq,cerebras` | `openai` and `ollama` are also supported. |
| `dialogue_num_speakers` / `--speakers` | auto | Set it when you know the count; this helps diarization. |
| `dialogue_background` / `--background` | `auto` | `auto` means Demucs if installed, otherwise Hindi only. `demucs` makes a failure a warning. `none` means Hindi only. |
| `dialogue_verify` / `--verify` | `auto` | Hindi re-ASR of every clip (needs faster-whisper). `off` is recorded as a limitation. |
| `dub_duration` (minutes) / `--limit-seconds` | 0 | Dubs only the first part of the video. |

### IndicF5 (experimental, optional)

Put authorised Hindi reference clips at `backend\voices\indicf5\<male_like|female_like>\<id>.wav`,
each with an exact transcript in `<id>.txt`. Then run with `--providers indicf5,edge`.

- A speaker whose category has no reference is voiced entirely by the next provider, never mixed.
- Original-voice cloning is **not** enabled.
- This provider has not been run in this repository's CI (it needs a GPU and the gated weights).

## Owner acceptance (listening) procedure

Run these four samples. For each, keep `report.json` and note the input duration, wall time per stage
(`stages[].seconds`), peak RAM (`stages[].data.peak_rss_mb`), GPU, detected speakers, retries,
coverage, timing warnings and separation result.

1. A 2–5 minute mixed male/female dialogue.
2. A clip with two or more speakers of the same voice category (for example, two men).
3. A clip with noise or music under the speech.
4. A longer video (20+ minutes) to observe resource behaviour.

Listen for:

- speaker consistency
- Hindi meaning and grammar, including gendered verb forms
- interruptions
- names and numbers
- unnatural speed
- English leaking through
- missing speech

Compare against the turn list in `report.md`.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| "speaker diarization unavailable" | Run `doctor`. Check that `HF_TOKEN` is set and that the model conditions are accepted with that account. pyannote 4.x needs Python ≥ 3.10. |
| Everyone has the same voice | See the line above, or diarization found one speaker. Set `--speakers N`. |
| Two characters sound alike | Edge has only two Hindi voices, so shared voices get pitch variants. For distinct voices use `--providers sarvam,edge`, which is paid. |
| "The link could not be downloaded" | Private, age-restricted or members-only videos need `backend\cookies.txt` (export from a private window after logging in). Otherwise download the video yourself and upload the file. Some links cannot work at all. |
| `draft_incomplete` with `tts_failed` | Check your internet connection (Edge-TTS is online) and re-run. The report lists the exact turns. |
| Timing overflow warnings | The Hindi line is much longer than the English slot, even after a faithful rewrite and a 1.15× stretch. Listen to that turn; the video is never sped up or slowed to hide it. |
| Content mismatch warnings | The re-ASR heard different words after one regeneration. This can be the verifier's error, so listen to the turn. |

## Known limitations

- Overlapping speech is detected and voiced on separate tracks, but words spoken underneath another
  speaker may be missing from the ASR transcript. Such regions are listed as unresolved.
- Voice categories are audio-based suggestions, not identity. Uncertain speakers use the documented
  default (`unknown_voice_category = male_like`, Madhur) and are flagged `unknown`.
- Diarization is run on the whole file with global clustering (no chunking). `reconcile_chunks` is
  available and tested for chunked runs, but it is not used by default.
- There is no resume or checkpointing, in line with the owner's no-reuse preference. Every job starts
  from scratch.
