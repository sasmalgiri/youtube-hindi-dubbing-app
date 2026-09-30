# Capability map (grounding, baseline `3377f4e`)

HEAD at the start of this work was `3377f4e6191e3ef38f18b962477fadd4244d2e7f`, the same commit the
directive reviewed. The findings below were checked against the code, not assumed.

## Entry points

| Entry | Route | Orchestrator(s) reached |
| --- | --- | --- |
| URL job | `POST /api/jobs` -> `_run_job` | by `pipeline_mode`: classic `Pipeline.run()`, hybrid (inline in `_run_job`), new (`DubbingRunner` or falls back to `Pipeline.run()` when a URL has YT subs), `run_oneflow`, `run_wordchunk`, `run_srtdub` |
| Local upload | `POST /api/jobs/upload` saves `work/source.<ext>`, then `_run_job` | same as URL |
| English SRT supplied | `transcript_srt_content` -> `Pipeline.run_from_source_srt` (classic/new); srtdub translates first via `_translate_srt_content` | classic `_run_from_step4` |
| Translated SRT supplied | `POST /api/jobs/with-srt` -> `_run_job_with_srt` -> `Pipeline.run_from_srt` | TTS + assembly only |
| Resume | `POST /api/jobs/{id}/resume-with-srt` -> `_run_resume` | `run_from_srt` |
| Split long video | `_run_job_split` (classic/hybrid/new when `split_duration>0`) | per-part `Pipeline.run()` |

`_run_job` forces **Edge-TTS only** for every mode (`req.use_* = False`), and it
replaces the default voice with Madhur (male).

## Stage map

| Stage | Implementation | Modes | Config | Failure behaviour (baseline) |
| --- | --- | --- | --- | --- |
| Acquire | `Pipeline._ingest_source` (yt-dlp, cookies) | all | `download_mode` | raises |
| Extract | `Pipeline._extract_audio` -> `audio_raw.wav` (48k stereo) + `audio_16k.wav` | all | – | raises |
| ASR | `Pipeline._transcribe`: Groq (segment granularity only, **no word times**) -> local faster-whisper child (word times) -> optional WhisperX align | classic/hybrid/new-fallback | `asr_model`, `use_whisperx` | Groq failure falls to local; **cross-job ASR cache** by audio hash (`cache.py`) |
| YT subs | `_fetch_youtube_subtitles`, `_fetch_youtube_translated_subs` | classic/new | `prefer_youtube_subs`, `use_yt_translate` | fast path `_run_from_step4` **skips diarization and clears `_voice_map`** |
| Diarization | `_diarize` (pyannote 3.1, `use_auth_token`) | classic only when `multi_speaker` | `HF_TOKEN` | returns `{}`; single voice |
| Gender | `_detect_speaker_genders`: F0 autocorrelation, 165 Hz threshold; <0.5 s of audio -> "female" | classic | – | never "unknown" |
| Speaker->segment | `_assign_speaker_to_segments`: segment-level max overlap; default `SPEAKER_00` | classic | – | uncertain -> SPEAKER_00 |
| Voice map | `_assign_voices_to_speakers`, `VOICE_POOL['hi']` = 1 female + 1 male | classic | – | silent reuse |
| Sentence merge | `_merge_broken_sentences`, `_combine_sentence_group`, `_group_sentences_by_count` | classic, YT fast path | – | **merges across speakers, keeps first speaker** |
| Gap close | `_close_segment_gaps` mutates `end` in place | classic, run_from_srt | – | source timing rewritten |
| Translation | `_translate_segments` -> `_dispatch_translation_engine` (Google default; Gemini/Groq/Cerebras/OpenAI/… numbered text parsed by `_parse_numbered_translations`) | classic/hybrid/new | `translation_engine` | numbered text, no ID validation. `dubbing/translation.py` only attaches hints — it does not call a model itself |
| YT Hindi distribution | inline in `run()`: flattens YT Hindi words and distributes them **by English word count** | classic | `yt_text_correction` | text detached from source turns |
| TTS | `_generate_tts_natural` -> `_tts_edge(voice_map)`; Edge failure -> `_sarvam_tts_single_mp3` with fixed `"shubh"` | all Pipeline modes | engine lock -> Edge | fallback silently changes voice |
| Other TTS adapters | `_tts_elevenlabs`, `_tts_sarvam_bulbul`, `_tts_google_cloud`, … use `cfg.tts_voice`/fixed speakers | (locked off) | – | no per-speaker binding |
| Word verify | `_post_tts_word_match_verify` -> `_retry_tts_segment` uses `cfg.tts_voice` | all Pipeline modes | `tts_word_match_verify` | **retry drops speaker voice**; compares word counts only |
| Completeness | `_verify_tts_completeness` logs only | all Pipeline modes | – | export proceeds; job "done" |
| Assembly | `_assemble_video_adapts_to_audio` (video slowed per segment) | classic/hybrid/new | `audio_priority` … | video speed changes |
| Background | `_separate_background` / `_demucs_single` -> returns `audio_raw` on failure; `_mix_audio` | only if `mix_original` (hard-disabled in `Pipeline.__init__`) | – | **raw English labelled as background** |
| Segment cache | `_load_segments_cache` always returns None (owner: no reuse) | – | – | – |
| Error handling | `_run_job` deletes `OUTPUTS/<job>` on any exception | all | – | partial assets lost |

## Mode constraints (baseline)

* `new`: forces `multi_speaker=False` silently.
* `hybrid`: builds `Word` objects without `speaker`, so `cue.speaker` is always None.
* `oneflow` / `wordchunk` / `srtdub`: receive a single `tts_voice`; `multi_speaker` is ignored silently.

## Corrections to the directive's findings

All 15 findings reproduce on HEAD. There are three additions:
1. Groq ASR (the default `asr_model`) requests `segment` granularity only, so the default route has no word
   timestamps at all. Word-level attribution needs either local faster-whisper or Groq `word` granularity.
2. `cache.py` performs **cross-job ASR reuse** keyed by audio hash, even though the owner prefers no reuse.
   The dialogue profile bypasses it and leaves legacy behaviour unchanged.
3. `_run_job` hard-locks Edge-TTS for all modes, so the provider adapters are effectively unused.
