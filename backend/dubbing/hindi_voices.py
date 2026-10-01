"""Ordered Edge-TTS voice slots for Hindi characters (one slot per character).

Edge has only two native Hindi voices (Madhur, Swara). The Multilingual
voices read Devanagari about as clearly (Whisper word-match on a Hindi test
passage, 2026-10-01: native 79-85%, Multilingual 73-88%), and a +/-20 Hz pitch
shift of a voice is heard as a different person, while +/-10 Hz is not.

Order = greedy "most different from every earlier slot" using WeSpeaker
speaker-embedding cosine distance (the model pyannote diarization uses; one
person's own lines sit within ~0.25 of each other), with Hindi clarity
breaking ties. The most-talkative characters get the earliest, most distinct
slots.

    min distance to all earlier slots      male          female
      slot 2                                0.72          0.87
      slot 4                                0.64          0.59
      +/-20 Hz variants vs base voice       0.55-0.59     0.61-0.62
      +/-10 Hz variants vs base voice       0.21-0.26     0.24-0.33  (not used)

A slot is a voice name, optionally with "|<pitch>" (e.g.
"hi-IN-MadhurNeural|+20Hz"). The pitch rides inside the name so it flows
through every existing voice parameter; split_slot() takes it apart right
before an edge_tts.Communicate call.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

MALE_SLOTS: List[str] = [
    "hi-IN-MadhurNeural",
    "en-AU-WilliamMultilingualNeural",
    "en-US-AndrewMultilingualNeural",
    "it-IT-GiuseppeMultilingualNeural",
    "fr-FR-RemyMultilingualNeural",
    "hi-IN-MadhurNeural|+20Hz",
    "hi-IN-MadhurNeural|-20Hz",
    "en-US-BrianMultilingualNeural",
    "ko-KR-HyunsuMultilingualNeural",
    "de-DE-FlorianMultilingualNeural",
    "en-AU-WilliamMultilingualNeural|-20Hz",
    "en-US-AndrewMultilingualNeural|+20Hz",
    "it-IT-GiuseppeMultilingualNeural|-20Hz",
    "en-US-BrianMultilingualNeural|+20Hz",
]

FEMALE_SLOTS: List[str] = [
    "hi-IN-SwaraNeural",
    "pt-BR-ThalitaMultilingualNeural",
    "de-DE-SeraphinaMultilingualNeural",
    "en-US-AvaMultilingualNeural",
    "hi-IN-SwaraNeural|+20Hz",
    "hi-IN-SwaraNeural|-20Hz",
    "en-US-EmmaMultilingualNeural",
    "fr-FR-VivienneMultilingualNeural",
    "pt-BR-ThalitaMultilingualNeural|-20Hz",
    "de-DE-SeraphinaMultilingualNeural|+20Hz",
    "en-US-EmmaMultilingualNeural|+20Hz",
    "en-US-AvaMultilingualNeural|-20Hz",
]


def split_slot(slot: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """"hi-IN-MadhurNeural|+20Hz" -> ("hi-IN-MadhurNeural", "+20Hz")."""
    if not slot:
        return slot, None
    voice, _, pitch = slot.partition("|")
    return voice, (pitch or None)


def base_voice(slot: Optional[str]) -> Optional[str]:
    return split_slot(slot)[0]


def slot_label(slot: Optional[str]) -> str:
    """Short display name: "Madhur", "Madhur +20Hz", "William"."""
    voice, pitch = split_slot(slot)
    if not voice:
        return ""
    name = voice.split("-")[-1].replace("MultilingualNeural", "").replace("Neural", "")
    return f"{name} {pitch}" if pitch else name


def is_female_slot(slot: Optional[str]) -> bool:
    return base_voice(slot) in {base_voice(s) for s in FEMALE_SLOTS}


def edge_communicate(text: str, slot: str, rate: str = "+0%"):
    """edge_tts.Communicate for a voice slot (applies the slot's pitch)."""
    import edge_tts
    voice, pitch = split_slot(slot)
    if pitch:
        return edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
    return edge_tts.Communicate(text, voice, rate=rate)
