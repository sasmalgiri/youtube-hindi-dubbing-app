# Hindi Dialogue profile

`pipeline_mode = "hindi_dialogue"` turns an English video (a link or a local file) into a Hindi-dubbed
MP4. Different speakers keep consistent voices, the video plays at its original speed, and each job
produces Hindi subtitles and a machine-readable quality report.

Use it for conversations, interviews, films and any video with more than one speaker. The older modes
(Classic, Hybrid, New, OneFlow, WordChunk, SRT Direct) are unchanged for single-narrator work.

## What it does

```
link / file ─► download ─► extract audio ─► separate (vocals + background, if installed)
            ─► text source ─► AUDIO diarization on the vocals ─► word attribution
            ─► speaker turns ─► speaker registry (one stable voice per speaker)
            ─► story brief (whole transcript: genders, aap/tum, names)
            ─► contextual Hindi translation (JSON keyed by turn ID)
            ─► TTS via the voice resolver ─► fit (slack → faithful rewrite → native rate → stretch, ≤1.15× total)
            ─► verification (clip checks, Hindi re-ASR, coverage, identity)
            ─► multi-track mix (levelled clips + separated background) ─► MP4 + SRT/VTT + report
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
  bypassed. Re-voicing a job reuses only that job's own checkpoint and caches (see
  [Review, voices and re-voicing](#review-voices-and-re-voicing)).

## Result statuses

| Status | Meaning |
| --- | --- |
| `completed` | Every required turn is voiced in its speaker's bound voice, with no unresolved warning. |
| `completed_with_warnings` | Output is complete, but something needs a listen: content mismatch, fallback provider, small timing overflow, diarization unavailable, unverified speech, and so on. |
| `draft_incomplete` | Something is missing or wrong: a missing turn, a voice/identity problem, a large timing overflow, or a speaker still voiced by two providers. The MP4 is kept for review. |
| `failed` / `cancelled` | Partial assets and `report.md` are kept in the job folder. |

A status reached by automated checks does **not** certify how natural the voices sound or how good
the Hindi is. Listen before publishing.

## Setup (Windows desktop)

On the owner's PC everything this profile needs is **already installed** in the main Python 3.10:
CUDA torch 2.4.1+cu121 for the RTX 3060, pyannote.audio 4.0.4, faster-whisper 1.1.1,
audio-separator, Demucs, IndicTransToolkit and Parler-TTS. Do **not** run
`pip install -r backend\requirements.txt` or `pip install -r backend\requirements-dialogue.txt`
there. whisperx and pyannote.audio declare torch 2.8 and Demucs accepts any torch, so pip would
replace the CUDA torch with a CPU-only torch 2.8. faster-whisper's `onnxruntime` dependency would
also overwrite `onnxruntime-gpu`.

1. Check readiness: `cd backend` then `python -m dubbing.dialogue doctor`. This reports only
   *whether* each credential is set, never its value.
2. Speaker diarization needs a Hugging Face token:
   - Accept the conditions of `pyannote/speaker-diarization-community-1` on huggingface.co.
   - Put `HF_TOKEN=hf_...` in `backend\.env`.
3. Translation uses the LLM keys the app already reads (`GEMINI_API_KEY`, `GROQ_API_KEY`,
   `CEREBRAS_API_KEY`). If none is set, a non-contextual Google fallback is used and every affected
   turn is flagged. Google's free endpoint answers HTTP 429 on this PC, so keep `GROQ_API_KEY` set.
4. Install any new package with `backend\constraints.txt`. It pins torch, torchaudio, numpy,
   transformers, pyannote.audio, faster-whisper and onnxruntime-gpu (plus torchvision and
   CTranslate2) to the working versions, so pip stops with a conflict instead of replacing them.
   The first command only shows what pip would change; the second installs:

   ```
   python -m pip install --dry-run -c backend\constraints.txt <package>
   python -m pip install -c backend\constraints.txt <package>
   ```

   If pip reports a conflict, or the dry run lists `torch`, `numpy` or `onnxruntime`, install the
   package with `--no-deps` instead and add its missing dependencies one at a time (with `-c`).

On a new PC, `setup.bat` installs `backend\requirements.txt` with the constraints and the CUDA 12.1
torch. Then add the dialogue packages one at a time as in step 4. On torch 2.4.1, pyannote.audio 4.x
needs `--no-deps`.

## Running

For the practical character-review workflow and supplied speaker labels, see
[Character workflow](CHARACTER_WORKFLOW.md).

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
  python -m dubbing.dialogue dub scene.mp4 --out ..\dubs\scene --review              # check lines first
  python -m dubbing.dialogue dub scene.mp4 --out ..\dubs\scene --resume --edits e.json  # re-voice
  ```

Outputs (in `--out`, and copied to `dubbed_outputs\<title> [HI Dialogue <status>] (<job>)` for UI
jobs):

- `dubbed_hi.mp4` (or `dubbed_hi.mkv`): the original video stream, the Hindi mix (first, default
  audio track) and soft Hindi subtitles, plus the options below.
- `hindi_mix.wav`
- `dialogue_track_N.wav`: one stem per simultaneous-speech track.
- `subtitles_hi.srt` / `.vtt`: timed to the dubbed audio.
- `subtitles_en.srt`: the English text on the original timing, without speaker labels.
- `transcript_en_speakers.srt`
- `report.md` / `report.json`
- `turns.json`, `clips.json`, `speakers.json`
- `review.json` and `clips/`: the after-run review packet, each accepted Hindi line as
  `clips/<turn_id>.wav` and each speaker's original-voice reference as `clips/ref_<speaker>.wav`.

### Output options

| Option (API field / CLI flag) | Default | Notes |
| --- | --- | --- |
| `dialogue_keep_original_audio` / `--keep-original-audio` | off | Adds the original audio as a second, non-default track ("Original"). Hindi stays the first, default track. |
| `dialogue_english_subtitles` / `--no-english-subs` | on | Writes `subtitles_en.srt` and, when subtitles are embedded, adds it as a second subtitle stream after the Hindi one. |
| `dialogue_burn_subtitles` / `--burn-subs` | off | Burns the Hindi subtitles into the picture (the video is re-encoded with libx264 CRF 18). Otherwise the video stream is copied untouched. |
| `dialogue_container` / `--container` | `mp4` | `mkv` writes `dubbed_hi.mkv` (SRT subtitle streams instead of mov_text). |

## Review, voices and re-voicing

**Review before voicing.** Tick *Review before voicing* in the UI (`dialogue_review`, CLI
`--review`). After translation the job pauses ("Review the Hindi lines and voices, then
continue") and shows every line with its speaker, English and Hindi, and every speaker with the
voice it got and a short clip of the original speaker. You can:

- correct a Hindi line (it is marked `edited` and voiced exactly as written),
- move a line to another existing speaker,
- delete a line (marked `deleted_by_user`; it is not voiced and does not count as missing),
- merge two speakers that diarization split (all lines take the target speaker's voice; if the two
  had different voice categories the report warns that gendered Hindi forms may need checking),
- choose another voice for a speaker (a specific voice and pitch, or "male-like" / "female-like",
  which picks a voice no other speaker uses whenever the pool has one).

Then continue (or cancel). On the CLI the packet is `<work>/review.json`; write your changes to
`<work>/review_edits.json` as `{"turn_edits": {...}, "voice_overrides": {...},
"speaker_merges": {...}}` and press Enter. Every edit, applied or ignored with its reason, is listed
in the report.

**Re-voicing a finished job.** Each job saves a checkpoint after every stage up to translation in
its own work folder (`work/checkpoint/state.json`, written atomically). Re-voicing (the job page's
re-voice action, or `--resume` with the same `--out`, optionally with `--edits edits.json`) restores
that checkpoint, applies the new edits and runs only synthesize, fit, verify and mix again. Lines
whose text and voice did not change reuse their audio from the job's TTS cache (`work/tts_cache/`),
and shortened lines come from `work/rewrite_cache.json`, so only what changed is voiced again. The
report lists the stages taken from the checkpoint (`resumed_from_checkpoint`) and every edit
(`applied_edits`). A checkpoint is used only for the same source (same input, length limit and
SRTs); otherwise every stage runs again and the report says why. Nothing is ever shared between
jobs, and a fresh run clears any earlier checkpoint and cache in its work folder.

**Sound like the original speaker (experimental).** The `voice_match` module (`openvoice`) converts
each Hindi clip toward the original speaker's voice, learned from up to three clean clips per
speaker (≤ 12 s each, from the separated vocals when available). It is off by default. If it is not
installed, cannot learn a speaker, or fails on a clip, the stock voice is kept and the report says
so; the job never fails because of it.

## Presets and modules

Pick a preset in the UI (Free — Online, Free — Local AI, Fast Draft, Single Narrator, Premium
Voices, My Hindi SRT) or use `--preset` on the CLI. Options that cannot run on your PC are switched
off automatically, with the reason and the fix shown. Fallbacks switch on automatically.
My Hindi SRT needs a Hindi SRT, so for a link or an uploaded video it is shown disabled: upload the
SRT in the **SRT Dub** tab instead (Hindi Dialogue mode picks it up there).

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
| `dialogue_background` / `--background` | `auto` | `auto` separates with audio-separator (BS-Roformer, then UVR MDX Inst HQ 4) or Demucs if installed, otherwise Hindi only. `demucs` (any separator) makes a failure a warning. `none` means Hindi only. |
| `dialogue_verify` / `--verify` | `auto` | Hindi re-ASR of every clip (needs faster-whisper). `off` is recorded as a limitation. |
| `dub_duration` (minutes) / `--limit-seconds` | 0 | Dubs only the first part of the video. |

### IndicF5 (experimental, optional)

Put authorised Hindi reference clips at `backend\voices\indicf5\<male_like|female_like>\<id>.wav`,
each with an exact transcript in `<id>.txt`. Then run with `--providers indicf5,edge`.

- A speaker whose category has no reference is voiced entirely by the next provider, never mixed.
- Original-voice cloning is **not** enabled.
- This provider has not been run in this repository's CI (it needs a GPU and the gated weights).

## Ideas adopted from SoniTranslate and pyVideoTrans

Both projects were read for ideas (no code was copied). Adopted, each covered by a test:

| Idea | Where | Default |
| --- | --- | --- |
| Separate first; the vocals stem feeds diarization, gender/pitch analysis and speaker reference clips (music no longer creates false speakers). The background is reused by the mix. | `orchestrator` (`separate` stage) | on (`analysis_audio="vocals"`; `"mix"` restores the old behaviour; `asr_on_vocals` is off) |
| Better separator: audio-separator BS-Roformer / UVR MDX Inst HQ 4, Demucs as fallback | `mix.separate_background` | when installed |
| Gentle second background duck keyed on the original English voice (hides separation residue) | `mix.final_mix` | when a vocals stem exists |
| Per-clip loudness levelling (active speech toward −20 dBFS, ±9 dB) so voices sit at an even level | `mix.render_dialogue` | on |
| Speed up with Edge's own speaking rate before stretching; total speed-up still ≤ `max_stretch` | `fit.fit_all`, `tts` | on (`native_rate`) |
| If the previous Hindi clip overran, start up to 0.4 s later instead of talking over it (bounded: no drift) | `fit.fit_all` | on |
| The last turn may start up to 1.5 s early so the end of the video does not cut it | `fit.fit_all` | on |
| Whisper beam search 5, no conditioning on previous text, punctuation prompt (Groq gets the prompt too) | `pipeline._whisper_child_worker` | dialogue profile only (`asr_decode="accurate"`) |
| Story brief: one LLM call over the whole transcript for speaker genders, aap/tum register and names; a gender that disagrees with the voice analysis is reported | `translation.build_brief` | on when an LLM is available |
| Clean spoken text (stage directions, markdown, symbols, ellipses) before TTS; subtitles unchanged | `tts.tts_sanitize` | on |
| Edge retries with backoff and jitter, 45 s timeout per request | `tts` | on |
| Gentler silence trim (−50 dB, 40 ms head / 120 ms tail, short fades) | `audio.trim_silence` | on |
| Turns at most 10 s, clause-aware splitting, numbers never split ("3." + "5") | `turns` | on |

Not adopted: changing the video speed, speed-ups beyond 1.15×, a subtitle-level speaker vote (word
attribution is more precise), using the original English audio as background, and a cross-job TTS
cache (jobs start from scratch).

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
- A job resumes only from its own checkpoint, and only the stages up to translation are skipped:
  synthesize, fit, verify and mix always run again. Nothing is reused across jobs.
