"""Speaker registry and the single voice resolver.

Every synthesis path (first pass, retries, chunked synthesis, provider
fallback, re-synthesis after a rewrite) must obtain its voice through
``SpeakerRegistry.resolve_voice(speaker_id, provider, policy)``. Bindings are
made once per job and never change afterwards, so a speaker keeps the same
voice across every turn and every retry.

Voice pools are curated per provider. When a pool has fewer voices than there
are speakers of a category (Edge-TTS has one Hindi male and one Hindi female
voice), speakers share a base voice with a distinct, stable pitch variant and
the reuse is reported -- they are never described as unique voices.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..hindi_voices import FEMALE_SLOTS, MALE_SLOTS, split_slot
from .contracts import (CATEGORY_CHILD, CATEGORY_FEMALE, CATEGORY_MALE,
                        CATEGORY_UNKNOWN, UNKNOWN_SPEAKER, SpeakerRecord)

# Pitch variants used (in order) when several speakers share a base voice.


def _env_list(name: str) -> List[str]:
    return [v.strip() for v in os.environ.get(name, "").split(",") if v.strip()]


INDICF5_REF_ROOT = Path(__file__).resolve().parents[2] / "voices" / "indicf5"


def indicf5_pool(root: Path = INDICF5_REF_ROOT) -> Dict[str, Any]:
    """Curated IndicF5 references: <root>/<category>/<id>.wav + <id>.txt
    (exact Hindi transcript). Only authorised voices belong there."""
    pool: Dict[str, Any] = {"model": "ai4bharat/IndicF5", "supports_pitch": False, "refs": {},
                            CATEGORY_MALE: [], CATEGORY_FEMALE: [], CATEGORY_CHILD: []}
    if Path(root).exists():
        for cat_dir in sorted(p for p in Path(root).iterdir() if p.is_dir()):
            ids = []
            for wav_p in sorted(cat_dir.glob("*.wav")):
                if wav_p.with_suffix(".txt").exists():
                    ids.append(wav_p.stem)
                    pool["refs"][wav_p.stem] = {"ref_audio": str(wav_p),
                                                "ref_text": str(wav_p.with_suffix(".txt"))}
            pool[cat_dir.name] = ids
    return pool


def curated_pools() -> Dict[str, Dict[str, Any]]:
    """Provider -> {category: [voice, ...], "model": ..., "supports_pitch": bool}.

    Only voices whose register is documented by the provider are listed.
    Edge: live `edge-tts --list-voices` shows exactly two hi-IN voices.
    Sarvam bulbul:v2: the documented female/male speaker split.
    Google Cloud: hi-IN WaveNet voices A/D (female), B/C (male).
    ElevenLabs: owner-supplied voice IDs via env (no default IDs are assumed).
    """
    return {
        "edge": {
            "model": "edge-neural",
            "supports_pitch": True,
            # One slot per character, most distinct first, shared with the
            # classic pipeline (dubbing/hindi_voices.py): native + Multilingual
            # voices, then +/-20 Hz variants ("voice|pitch").
            CATEGORY_MALE: list(MALE_SLOTS),
            CATEGORY_FEMALE: list(FEMALE_SLOTS),
            # Raised-pitch voices no adult slot uses, so a child never sounds
            # exactly like the lead woman.
            CATEGORY_CHILD: ["hi-IN-SwaraNeural|+25Hz", "pt-BR-ThalitaMultilingualNeural|+25Hz",
                             "en-US-EmmaMultilingualNeural|+25Hz"],
        },
        "sarvam": {
            "model": "bulbul:v2",
            "supports_pitch": False,
            CATEGORY_MALE: ["abhilash", "karun", "hitesh"],
            CATEGORY_FEMALE: ["anushka", "manisha", "vidya", "arya"],
            CATEGORY_CHILD: [],
        },
        "google": {
            "model": "wavenet",
            "supports_pitch": True,
            CATEGORY_MALE: ["hi-IN-Wavenet-B", "hi-IN-Wavenet-C", "hi-IN-Wavenet-B|+20Hz",
                            "hi-IN-Wavenet-C|-20Hz", "hi-IN-Wavenet-B|-20Hz", "hi-IN-Wavenet-C|+20Hz"],
            CATEGORY_FEMALE: ["hi-IN-Wavenet-A", "hi-IN-Wavenet-D", "hi-IN-Wavenet-A|+20Hz",
                              "hi-IN-Wavenet-D|-20Hz", "hi-IN-Wavenet-A|-20Hz", "hi-IN-Wavenet-D|+20Hz"],
            CATEGORY_CHILD: [],
        },
        "elevenlabs": {
            "model": os.environ.get("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2"),
            "supports_pitch": False,
            CATEGORY_MALE: _env_list("ELEVENLABS_VOICES_MALE"),
            CATEGORY_FEMALE: _env_list("ELEVENLABS_VOICES_FEMALE"),
            CATEGORY_CHILD: _env_list("ELEVENLABS_VOICES_CHILD"),
        },
        "indicf5": indicf5_pool(),
        # Indic Parler-TTS Hindi speakers from the model card (Rohit, Divya
        # recommended; Aman, Rani available). Only Rohit's gender is stated by
        # AI4Bharat; Divya/Rani (female) and Aman (male) are inferred from the
        # names. "|low"/"|high" change the voice description (style variants of
        # the same speaker, reported as reuse).
        "indic_parler": {
            "model": "ai4bharat/indic-parler-tts",
            "supports_pitch": False,
            CATEGORY_MALE: ["Rohit", "Aman", "Rohit|low", "Aman|low", "Rohit|high", "Aman|high"],
            CATEGORY_FEMALE: ["Divya", "Rani", "Divya|high", "Rani|high", "Divya|low", "Rani|low"],
            CATEGORY_CHILD: ["Divya|high"],
        },
        "mock": {
            "model": "mock",
            "supports_pitch": True,
            CATEGORY_MALE: ["mock-male-1", "mock-male-2"],
            CATEGORY_FEMALE: ["mock-female-1", "mock-female-2"],
            CATEGORY_CHILD: [],
        },
    }


class VoiceResolutionError(RuntimeError):
    pass


class SpeakerRegistry:
    """Per-job speaker records plus immutable provider voice bindings."""

    def __init__(self, unknown_default_category: str = CATEGORY_MALE,
                 pools: Optional[Dict[str, Dict[str, Any]]] = None):
        self.speakers: Dict[str, SpeakerRecord] = {}
        self.unknown_default_category = unknown_default_category
        self.pools = pools if pools is not None else curated_pools()
        self._bound_providers: set = set()

    # ── registration ────────────────────────────────────────────────────
    def register(self, speaker_id: str, **fields) -> SpeakerRecord:
        rec = self.speakers.get(speaker_id)
        if rec is None:
            rec = SpeakerRecord(speaker_id=speaker_id)
            self.speakers[speaker_id] = rec
        for k, v in fields.items():
            setattr(rec, k, v)
        rec.classification_unknown = rec.voice_category == CATEGORY_UNKNOWN
        return rec

    # ── binding ─────────────────────────────────────────────────────────
    def _binding_category(self, rec: SpeakerRecord, provider: str) -> str:
        pool = self.pools.get(provider, {})
        cat = rec.voice_category
        if cat == CATEGORY_UNKNOWN:
            return self.unknown_default_category
        if cat == CATEGORY_CHILD and not pool.get(CATEGORY_CHILD):
            return CATEGORY_FEMALE
        return cat

    def bind_provider(self, provider: str):
        """Assign every registered speaker a stable voice for ``provider``.

        Speakers with the most speech get the first (unvaried) voices. Speakers
        already bound keep their binding; new speakers are appended.
        """
        pool = self.pools.get(provider)
        if pool is None:
            raise VoiceResolutionError(f"No voice pool for provider '{provider}'")
        # Count existing usage so late-registered speakers continue the sequence.
        usage: Dict[str, int] = {}
        for rec in self.speakers.values():
            b = rec.provider_voices.get(provider)
            if b:
                usage[b["category_used"]] = usage.get(b["category_used"], 0) + 1
        order = sorted(self.speakers.values(),
                       key=lambda r: (-r.total_speech_s, r.speaker_id))
        for rec in order:
            if provider in rec.provider_voices:
                continue
            cat = self._binding_category(rec, provider)
            voices = list(pool.get(cat) or [])
            if not voices:
                # Provider cannot express this category: leave unbound. The router
                # must then route the whole speaker to another provider or degrade.
                continue
            n = usage.get(cat, 0)
            usage[cat] = n + 1
            slot = voices[n % len(voices)]
            round_idx = n // len(voices)
            voice, slot_pitch = split_slot(slot)
            binding: Dict[str, Any] = {
                "provider": provider,
                "voice": voice,
                "model": pool.get("model", ""),
                "category_used": cat,
                "pitch": slot_pitch,
                "variant": round_idx,
            }
            if pool.get("refs"):  # reference-based providers (IndicF5)
                binding.update(pool["refs"][voice])
                binding["reference_version"] = voice
            if pool.get("supports_pitch") and rec.voice_category == CATEGORY_CHILD \
                    and cat == CATEGORY_FEMALE:
                binding["pitch"] = "+25Hz"
            if round_idx > 0:
                # More characters than distinct slots: this exact voice is
                # already used by another character (reported).
                binding["indistinguishable_reuse"] = True
            rec.provider_voices[provider] = binding
            if not rec.mapping_origin:
                rec.mapping_origin = ("default_unknown" if rec.voice_category == CATEGORY_UNKNOWN
                                      else "auto_category")
        self._bound_providers.add(provider)

    def resolve_voice(self, speaker_id: str, provider: str,
                      policy: str = "strict") -> Dict[str, Any]:
        """Return the job-stable binding for (speaker, provider).

        policy="strict": unknown speaker or unbound provider raises.
        policy="register_unknown": an unseen speaker is registered with an
            unknown category, then bound (used for UNKNOWN_SPEAKER turns).
        """
        rec = self.speakers.get(speaker_id)
        if rec is None:
            if policy != "register_unknown":
                raise VoiceResolutionError(f"Speaker '{speaker_id}' is not registered")
            rec = self.register(speaker_id, voice_category=CATEGORY_UNKNOWN,
                                mapping_origin="default_unknown")
        if provider not in rec.provider_voices:
            # Binding is idempotent: already-bound speakers keep their voices.
            self.bind_provider(provider)
        b = rec.provider_voices.get(provider)
        if not b:
            raise VoiceResolutionError(
                f"Provider '{provider}' has no voice for speaker '{speaker_id}' "
                f"(category {rec.voice_category})")
        out = dict(b)
        out["speaker_id"] = speaker_id
        return out

    def record_fallback(self, speaker_id: str, from_provider: str, to_provider: str,
                        turn_id: str, reason: str):
        rec = self.speakers.get(speaker_id)
        if rec is not None:
            rec.fallback_history.append({
                "from": from_provider, "to": to_provider,
                "turn_id": turn_id, "reason": reason[:300],
            })

    # ── reporting ──────────────────────────────────────────────────────
    def voice_reuse(self, provider: str) -> Dict[str, List[str]]:
        """voice -> speakers sharing it (only entries with >1 speaker)."""
        by_voice: Dict[str, List[str]] = {}
        for rec in self.speakers.values():
            b = rec.provider_voices.get(provider)
            if b:
                label = rec.speaker_id + (f" (pitch {b['pitch']})" if b.get("pitch") else "")
                by_voice.setdefault(b["voice"], []).append(label)
        return {v: s for v, s in by_voice.items() if len(s) > 1}

    def to_dict(self) -> Dict[str, Any]:
        return {"unknown_default_category": self.unknown_default_category,
                "speakers": {k: v.to_dict() for k, v in self.speakers.items()}}

    def save(self, path: Path):
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
                              encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "SpeakerRegistry":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        reg = cls(unknown_default_category=data.get("unknown_default_category", CATEGORY_MALE))
        for sid, d in data.get("speakers", {}).items():
            reg.speakers[sid] = SpeakerRecord(**d)
            for p in d.get("provider_voices", {}):
                reg._bound_providers.add(p)
        return reg


def ensure_unknown_speaker(registry: SpeakerRegistry) -> SpeakerRecord:
    """Register the catch-all speaker used for unattributed turns."""
    rec = registry.speakers.get(UNKNOWN_SPEAKER)
    if rec is None:
        rec = registry.register(UNKNOWN_SPEAKER, voice_category=CATEGORY_UNKNOWN,
                                mapping_origin="default_unknown")
    return rec
