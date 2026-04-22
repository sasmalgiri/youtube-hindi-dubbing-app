"""Built-in quality presets — curated flag combinations for common use cases.

These are read-only (clients can't delete or overwrite them). Sits alongside
the user-saved presets in backend/dubbing/presets.py — not a replacement.

Each preset is a tuple of (slug, name, description, settings). The settings
dict mirrors the DubbingSettings shape in web/src/components/SettingsPanel.tsx
and the PipelineConfig dataclass fields in backend/pipeline.py. Only flags
that are actually wired up are used — no aspirational features.
"""
from __future__ import annotations

from typing import List, Dict, Optional


_STUDIO_QUALITY = {
    "slug": "studio-quality",
    "name": "Studio Quality",
    "description": (
        "Best Hindi quality. Sarvam Bulbul v3 primary with Edge-TTS fallback. "
        "All verifications on, slot auto-fix, simplify+noun-preserve. Slowest, highest fidelity."
    ),
    "settings": {
        "pipeline_mode": "classic",
        # TTS chain: Sarvam primary, Edge fallback, Sarvam salvages stragglers
        "use_sarvam_bulbul": True,
        "use_edge_tts": True,
        "use_google_tts": False,
        "use_elevenlabs": False,
        "use_coqui_xtts": False,
        "use_cosyvoice": False,
        "use_chatterbox": False,
        "use_fish_speech": False,
        # Naturalness
        "simplify_english": True,
        "keep_subject_english": True,
        # AV sync: strict
        "slot_verify": "auto_fix",
        "av_sync_mode": "original",
        "audio_priority": True,
        "video_slow_to_match": True,
        "tts_no_time_pressure": True,
        # Rate: strict ceiling, auto-compute
        "tts_rate_mode": "auto",
        "tts_rate_ceiling": "+25%",
        "tts_rate_target_wpm": 130,
        # Verification: all on
        "tts_word_match_verify": True,
        "tts_word_match_tolerance": 0.15,
        "tts_word_match_model": "auto",
        "tts_truncation_threshold": 0.30,
        "enable_tts_verify_retry": True,
        # Audio quality
        "audio_bitrate": "256k",
        "post_tts_level": "minimal",
        "audio_quality_mode": "quality",
        # ASR: best
        "use_whisperx": True,
        "prefer_youtube_subs": True,
        "use_yt_translate": True,
    },
}


_BALANCED = {
    "slug": "balanced",
    "name": "Balanced",
    "description": (
        "Recommended default. Google Neural2 (multi-account pool) primary with "
        "Sarvam salvage. Auto-disables expensive verification above 1000 cues."
    ),
    "settings": {
        "pipeline_mode": "classic",
        "use_google_tts": True,
        "use_edge_tts": True,
        "use_sarvam_bulbul": False,
        "use_elevenlabs": False,
        "use_coqui_xtts": False,
        "use_cosyvoice": False,
        "use_chatterbox": False,
        "simplify_english": True,
        "keep_subject_english": True,
        "slot_verify": "dry_run",
        "av_sync_mode": "original",
        "audio_priority": True,
        "video_slow_to_match": True,
        "tts_no_time_pressure": True,
        "tts_rate_mode": "auto",
        "tts_rate_ceiling": "+25%",
        "tts_rate_target_wpm": 130,
        "tts_word_match_verify": True,
        "tts_word_match_tolerance": 0.15,
        "tts_truncation_threshold": 0.30,
        "audio_bitrate": "192k",
        "post_tts_level": "minimal",
        "audio_quality_mode": "fast",
        "use_whisperx": True,
        "prefer_youtube_subs": True,
        "use_yt_translate": True,
    },
}


_FAST_PREVIEW = {
    "slug": "fast-preview",
    "name": "Fast Preview",
    "description": (
        "OneFlow + Edge-TTS. Fastest path — no LLM, no verification. "
        "Use to sanity-check a video before committing to Studio/Balanced."
    ),
    "settings": {
        "pipeline_mode": "oneflow",
        "use_edge_tts": True,
        "use_google_tts": False,
        "use_sarvam_bulbul": False,
        "use_coqui_xtts": False,
        "use_elevenlabs": False,
        "simplify_english": False,
        "keep_subject_english": False,
        "slot_verify": "off",
        "av_sync_mode": "original",
        "audio_priority": True,
        "video_slow_to_match": True,
        "tts_rate_mode": "auto",
        "tts_rate_ceiling": "+50%",
        "tts_word_match_verify": False,
        "enable_tts_verify_retry": False,
        "tts_truncation_threshold": 0.30,
        "audio_bitrate": "128k",
        "post_tts_level": "minimal",
        "audio_quality_mode": "fast",
        "use_whisperx": False,
        "prefer_youtube_subs": True,
        "use_yt_translate": True,
    },
}


_YOUTUBE_PASSTHROUGH = {
    "slug": "youtube-hindi-passthrough",
    "name": "YouTube Hindi Pass-through",
    "description": (
        "Uses YouTube's own captions (or auto-translated Hindi) directly — "
        "skips Whisper entirely. WordChunk mode, fast, good when YouTube already has decent subs."
    ),
    "settings": {
        "pipeline_mode": "wordchunk",
        "prefer_youtube_subs": True,
        "use_yt_translate": True,
        "yt_transcript_mode": "yt_timeline",
        "use_edge_tts": True,
        "use_google_tts": False,
        "use_sarvam_bulbul": False,
        "simplify_english": False,
        "keep_subject_english": False,
        "slot_verify": "off",
        "audio_priority": True,
        "video_slow_to_match": True,
        "tts_rate_mode": "auto",
        "tts_rate_ceiling": "+35%",
        "tts_word_match_verify": False,
        "audio_bitrate": "192k",
        "use_whisperx": False,
    },
}


_SRT_DIRECT = {
    "slug": "srt-direct",
    "name": "SRT Direct",
    "description": (
        "You provide the translated SRT. TTS each cue verbatim via Sarvam Bulbul v3, "
        "concat zero-gap, stretch video to fit. No translation, no transcription."
    ),
    "settings": {
        "pipeline_mode": "srtdub",
        "use_sarvam_bulbul": True,
        "use_edge_tts": True,
        "use_google_tts": False,
        "simplify_english": False,
        "keep_subject_english": False,
        "audio_priority": True,
        "video_slow_to_match": True,
        "tts_rate_mode": "manual",
        "tts_rate": "+0%",
        "tts_word_match_verify": False,
        "audio_bitrate": "256k",
        "post_tts_level": "minimal",
        "use_whisperx": False,
    },
}


_VOICE_CLONE_SAME = {
    "slug": "voice-clone-same-lang",
    "name": "Voice Clone (Same-Lang)",
    "description": (
        "Re-voice with XTTS v2 clone of the original speaker. Input language = "
        "output language. Audio stays untouchable; post-processing disabled."
    ),
    "settings": {
        "pipeline_mode": "classic",
        "use_coqui_xtts": True,
        "use_edge_tts": False,
        "use_sarvam_bulbul": False,
        "use_google_tts": False,
        "use_cosyvoice": False,
        "use_chatterbox": False,
        "audio_untouchable": True,
        "post_tts_level": "none",
        "audio_priority": True,
        "video_slow_to_match": True,
        "tts_rate_mode": "manual",
        "tts_rate": "+0%",
        "simplify_english": False,
        "keep_subject_english": False,
        "tts_word_match_verify": False,
        "enable_manual_review": False,
        "audio_bitrate": "256k",
    },
}


_HINDI_REVOICE_CLONE = {
    "slug": "hindi-revoice-clone",
    "name": "Hindi Re-Voice (Clone)",
    "description": (
        "Re-dub an existing Hindi video with a clone of the original speaker. "
        "Whisper transcribes Hindi → translation skipped (same language) → "
        "Coqui XTTS v2 re-speaks in the cloned voice → per-cue video stretch "
        "(audio drives, video scales to fit each cue's ffmpeg setpts) → "
        "Wav2Lip re-syncs lips to the new audio. Audio and video match by "
        "construction: every cue's video slice is stretched to equal its "
        "TTS audio duration. To use a different voice instead of cloning, "
        "switch the TTS engine in Advanced Settings — Sarvam Bulbul v3 and "
        "Google Neural2 both support Hindi natively."
    ),
    "settings": {
        "source_language": "hi",
        "target_language": "hi",
        "pipeline_mode": "classic",
        # Cloning engine: XTTS v2 (supports Hindi, uses reference from source).
        # Other Hindi-capable engines explicitly disabled so the Voice Clone
        # preset doesn't accidentally fall through to a non-cloning path.
        "use_coqui_xtts": True,
        "use_edge_tts": False,
        "use_sarvam_bulbul": False,
        "use_google_tts": False,
        "use_cosyvoice": False,
        "use_chatterbox": False,
        "use_elevenlabs": False,
        # Preserve the cloned audio: no post-processing, no duration coercion.
        "audio_untouchable": True,
        "post_tts_level": "none",
        "tts_rate_mode": "manual",
        "tts_rate": "+0%",
        "tts_word_match_verify": False,
        "enable_manual_review": False,
        # AV match: audio sets the pace, video stretches to match, Wav2Lip
        # re-syncs the speaker's mouth to the re-voiced audio.
        "audio_priority": True,
        "video_slow_to_match": True,
        "use_wav2lip": True,
        # Hindi source — don't touch English-source-specific flags.
        "simplify_english": False,
        "keep_subject_english": False,
        # Force Whisper transcription of the actual Hindi audio. Skipping
        # YouTube subs here because captions are often edited/mistimed vs
        # the real speech we need to re-voice.
        "prefer_youtube_subs": False,
        "use_yt_translate": False,
        "use_whisperx": True,
        "audio_bitrate": "256k",
    },
}


_BUDGET = {
    "slug": "budget",
    "name": "Budget (Free-Tier Max)",
    "description": (
        "Stretches Google's 1M-char free tier across as many videos as possible. "
        "Minimal verification, Edge-TTS overflow, Sarvam salvage if key present."
    ),
    "settings": {
        "pipeline_mode": "classic",
        "use_google_tts": True,
        "use_edge_tts": True,
        "use_sarvam_bulbul": False,
        "use_coqui_xtts": False,
        "use_elevenlabs": False,
        "simplify_english": True,
        "keep_subject_english": True,
        "slot_verify": "off",
        "av_sync_mode": "original",
        "audio_priority": True,
        "video_slow_to_match": True,
        "tts_no_time_pressure": True,
        "tts_rate_mode": "auto",
        "tts_rate_ceiling": "+25%",
        "tts_word_match_verify": False,
        "tts_truncation_threshold": 0.30,
        "audio_bitrate": "192k",
        "post_tts_level": "minimal",
        "audio_quality_mode": "fast",
        "use_whisperx": True,
        "prefer_youtube_subs": True,
        "use_yt_translate": True,
    },
}


_BUILTIN_PRESETS: List[Dict] = [
    _BALANCED,
    _STUDIO_QUALITY,
    _FAST_PREVIEW,
    _YOUTUBE_PASSTHROUGH,
    _SRT_DIRECT,
    _VOICE_CLONE_SAME,
    _HINDI_REVOICE_CLONE,
    _BUDGET,
]


def list_builtin_presets() -> List[Dict]:
    """List built-in presets (name + slug + description, no full settings)."""
    return [
        {"name": p["name"], "slug": p["slug"], "description": p["description"]}
        for p in _BUILTIN_PRESETS
    ]


def get_builtin_preset(slug: str) -> Optional[Dict]:
    """Return a built-in preset with full settings by slug, or None."""
    for p in _BUILTIN_PRESETS:
        if p["slug"] == slug:
            return {
                "name": p["name"],
                "slug": p["slug"],
                "description": p["description"],
                "settings": p["settings"],
                "builtin": True,
            }
    return None
