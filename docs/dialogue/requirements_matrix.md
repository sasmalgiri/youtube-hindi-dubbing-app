# Requirement matrix

Baseline: `3377f4e`. All paths are under `backend/` unless noted, and tests live under
`backend/tests_dialogue/`.

**Runtime evidence** here means what was actually executed in the cloud build environment: Linux,
CPU only, no GPU, no model weights, no provider keys, and a proxy without WebSocket support.
Synthetic media plus mocked model backends prove orchestration, identity, coverage and rendering.
They do **not** prove dubbing quality.

## Baseline defects (directive §3)

| Finding | Fix | Files | Tests | Runtime evidence | Remaining limitation |
| --- | --- | --- | --- | --- | --- |
| YT-subs fast path skips diarization and clears voice map | Dialogue profile always diarizes. The legacy fast path diarizes when `multi_speaker` is set. | `dubbing/dialogue/orchestrator.py`, `pipeline.py` (`run`) | `test_youtube_subtitles_still_run_audio_diarization`, `test_subtitle_text_source_is_attributed_by_audio` | E2E with synthetic media + mocks | Legacy path attributes speakers per cue, not per word |
| New mode disables `multi_speaker` | Visible note on the job for new/oneflow/wordchunk/srtdub. `hindi_dialogue` is the shared path. | `app.py` (`_run_job`) | `test_upload_routes_to_dialogue_profile` (routing) | API test | Legacy modes stay single-voice by design |
| Hybrid omits speaker on `Word` | Hybrid diarizes when `multi_speaker` is set and carries `speaker` on every `Word` | `app.py` (hybrid block) | — (the block is inline in the route) | not run | Hybrid still uses segment-level attribution |
| OneFlow/WordChunk/SRT Direct single voice | Explicit incompatibility note. Use `hindi_dialogue`. | `app.py` | — | — | not converted |
| pyannote 3.1 + segment max-overlap | Community-1 adapter (pyannote 4 `token=`, `DiarizeOutput`), 3.1 fallback, word-level attribution on the exclusive track, overlap flags | `dubbing/dialogue/diarization.py` | `test_one_asr_segment_spanning_two_speakers…`, `test_uncertain_words_stay_unknown…`, `test_derive_exclusive…`, `test_chunk_reconciliation…` | unit tests only | **pyannote was not run** (no weights/GPU) |
| 165 Hz gender threshold | Aggregated F0 with confidence and explicit `unknown` / `child_like` states | `dubbing/dialogue/voice_analysis.py` | `test_classify_keeps_unknown_states`, `test_analyze_speaker_on_synthetic_voices` | synthetic harmonic tones | Thresholds are tuning values; not validated on real voices |
| One Hindi Edge voice per category | Registry binds stable (voice, pitch variant) pairs and reports reuse. Multi-voice pools for Sarvam/Google/ElevenLabs. | `dubbing/dialogue/speaker_registry.py` | `test_two_male_two_female_four_identities…`, `test_main_speaker_gets_base_voice…` | unit | Edge still has only 2 Hindi voices (verified live list) |
| Sentence merge keeps first speaker | Merge/group never crosses a known speaker change. Unknown segments adopt the first known label. Combining mixed speakers raises. | `pipeline.py` | `test_merge_*`, `test_grouping_respects_speakers…`, `test_combining_different_speakers_is_refused` | unit | — |
| TTS adapters use one global voice | `TTSRouter` resolves every call through `resolve_voice`. The legacy Edge→Sarvam fallback now picks a register-matched speaker and records the substitution. | `dubbing/dialogue/tts.py`, `pipeline.py` | `test_provider_fallback_preserves_speaker…`, `test_sarvam_fallback_speaker_matches_voice_register` | mock providers | Paid providers not called (not authorised) |
| Word-verify retry uses global voice | Retry uses `_voice_for_segment(seg)` | `pipeline.py` | `test_retry_voice_is_the_segment_speaker_voice`, `test_female_turn_failing_initially_retries_in_female_voice` | unit/mock | — |
| Gap closing rewrites source timing | `Turn.source_start/end` are immutable (enforced). Scheduling lives on `Clip`. The profile never calls `_close_segment_gaps`. | `dubbing/dialogue/contracts.py`, `fit.py` | `test_turn_source_timing_is_immutable`, `test_windows_use_immutable_source_times…` | E2E | Legacy modes unchanged |
| Background fallback returns raw English | Dialogue: failure means Hindi only, reported. Legacy: `_separate_background` returns None and `_mix_audio` skips the bed. | `dubbing/dialogue/mix.py`, `pipeline.py` | `test_failed_separation_never_returns_original_as_background`, E2E `separation.status == failed` | E2E | **Demucs not run** |
| Verification compares word counts | Normalised token diff: omissions/insertions/substitutions (substantive vs spelling), repetitions, numbers, negation, protected terms | `dubbing/dialogue/text_checks.py`, `verify.py` | `test_same_word_count_different_words_is_detected`, `test_wrong_speech_same_word_count_is_regenerated_then_reported` | mock re-ASR | **Whisper Hindi re-ASR not run** |
| Completeness warns and exports | Coverage gate by unique turn ID and `draft_incomplete` status. Legacy: counts unique segments, marks draft, message says DRAFT. | `verify.py`, `report.py`, `pipeline.py`, `app.py` | `test_missing_turn_plus_duplicate_clip_is_detected`, `test_missing_turn_makes_honest_draft`, `test_completeness_counts_unique_segments_not_files` | E2E | — |
| Cache loading disabled | Preserved. The dialogue profile bypasses the cross-job ASR cache as well. No resume was added. | `orchestrator.py` | — | — | Legacy `cache.py` ASR reuse unchanged for legacy modes |

## Acceptance matrix (directive §14)

| Scenario | Test(s) | Evidence level |
| --- | --- | --- |
| Male and female alternating turns | `test_alternating_male_female_are_stable`, `test_mixed_dialogue_end_to_end` | mock TTS + real FFmpeg |
| Two male + two female | `test_two_male_two_female_four_identities_with_reported_reuse` | unit |
| One ASR segment spans two speakers | `test_one_asr_segment_spanning_two_speakers_splits_into_two_turns`, `test_subtitle_text_source_is_attributed_by_audio` | unit + E2E |
| Short reply, laugh, interruption | `test_short_replies_and_interruption_are_not_merged`, `test_nonlexical_turn_is_not_required` | unit. Nonlexical tokens (`[laughter]`) are not dubbed and are not required. |
| YouTube subtitles available | `test_youtube_subtitles_still_run_audio_diarization` | E2E (download mocked) |
| SRT without speaker labels | `test_subtitle_text_source_is_attributed_by_audio`, `test_translated_srt_input_uses_audio_speakers`, `test_translated_cues_take_speakers_from_audio…` | E2E |
| Legacy mode requested with dialogue profile | `test_upload_routes_to_dialogue_profile` (note + diarization despite `multi_speaker=false`) | API |
| Female turn fails initial TTS | `test_female_turn_failing_initially_retries_in_female_voice` | mock |
| Provider fallback | `test_provider_fallback_preserves_speaker_and_reroutes_whole_speaker` | mock |
| Wrong words, same count | `test_same_word_count_different_words_is_detected`, `test_wrong_speech_same_word_count…` | unit + E2E |
| Missing turn + duplicate clip | `test_missing_turn_plus_duplicate_clip_is_detected` | unit |
| Overlong Hindi turn | `test_overlong_turn_gets_faithful_rewrite_before_stretch`, `test_unfittable_turn_is_kept_whole_and_reported`, `test_rewrite_shorter_rejects_dropping_facts` | mock |
| Pauses and overlaps | `test_overlapping_speech_keeps_both_turns_and_flags_overlap`, `test_windows_use_immutable_source_times_and_skip_overlap_partners` | unit |
| Separation unavailable | E2E `separation.status == failed` with no background, `test_failed_separation_never_returns…` | E2E |
| Long job / OOM / cancel | `test_cancel_keeps_partial_assets`. OOM: verifier downgrade and diarization CPU rerun (code paths, not exercised). | cancel: E2E. **OOM and long-video runs not exercised.** |
| Resume / config change | No resume exists, so nothing is reused (owner preference). | by construction |

## Checks NOT performed in this environment

- Any real model: faster-whisper, pyannote (community-1 or 3.1), Demucs, IndicF5, and LLM translation
  (no keys; paid/free-tier calls not made).
- Real Edge-TTS synthesis. The container's proxy does not support the WebSocket upgrade Edge-TTS
  needs, and `edge-tts` pins certifi's CA bundle.
- GPU/VRAM behaviour, long-video memory behaviour, and wall-time per stage on the owner's PC.
- Listening tests of speaker consistency, Hindi naturalness, names, numbers and timing.
- Windows-specific paths (developed and tested on Linux).
