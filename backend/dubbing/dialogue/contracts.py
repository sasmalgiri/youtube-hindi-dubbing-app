"""Data contracts for the Hindi dialogue profile.

Invariants enforced here (not just documented):
  * A Turn's source_start / source_end never change after construction.
    Scheduling lives on the Clip (scheduled_start / scheduled_end).
  * Every Clip names both the turn_id and the speaker_id it was made for.
  * Unknown speaker attribution is represented explicitly (speaker_id=None on a
    word, UNKNOWN_SPEAKER on a turn) -- never silently defaulted to SPEAKER_00.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

UNKNOWN_SPEAKER = "UNKNOWN"

# Final job statuses (ordered from best to worst for "worst-of" merging).
STATUS_COMPLETED = "completed"
STATUS_COMPLETED_WITH_WARNINGS = "completed_with_warnings"
STATUS_DRAFT_INCOMPLETE = "draft_incomplete"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"
_STATUS_RANK = {
    STATUS_COMPLETED: 0,
    STATUS_COMPLETED_WITH_WARNINGS: 1,
    STATUS_DRAFT_INCOMPLETE: 2,
    STATUS_FAILED: 3,
    STATUS_CANCELLED: 4,
}


def worst_status(*statuses: str) -> str:
    return max(statuses, key=lambda s: _STATUS_RANK.get(s, 0))


# Voice categories are an audio-based *suggestion* about how a voice sounds,
# not a claim about a person's gender.
CATEGORY_MALE = "male_like"
CATEGORY_FEMALE = "female_like"
CATEGORY_CHILD = "child_like"
CATEGORY_UNKNOWN = "unknown"


@dataclass
class SpeakerRecord:
    speaker_id: str
    voice_category: str = CATEGORY_UNKNOWN
    category_confidence: Optional[float] = None
    category_evidence: Dict[str, Any] = field(default_factory=dict)
    classification_unknown: bool = True
    # provider -> binding dict {"voice": ..., "pitch": ..., "model": ...}
    provider_voices: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    mapping_origin: str = ""          # auto_category | default_unknown | srt_label | user
    fallback_history: List[Dict[str, Any]] = field(default_factory=list)
    reference_clip: Optional[str] = None
    reference_transcript: Optional[str] = None
    embedding_provenance: Optional[str] = None
    total_speech_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WordRecord:
    word_id: str
    text: str
    start: float
    end: float
    speaker_id: Optional[str] = None           # None == unknown attribution
    alignment_confidence: Optional[float] = None  # only if a provider supplied one
    overlap: bool = False
    timing_estimated: bool = False
    nonlexical: bool = False
    attribution: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_IMMUTABLE_TURN_FIELDS = ("turn_id", "source_start", "source_end")


@dataclass
class Turn:
    turn_id: str
    speaker_id: str
    source_start: float
    source_end: float
    word_ids: List[str] = field(default_factory=list)
    source_text: str = ""
    hi_raw: str = ""
    hi_fit: str = ""        # the text actually sent to TTS
    hi_display: str = ""    # subtitle text
    context_group: int = 0
    protected_terms: List[str] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)
    translation_attempts: List[Dict[str, Any]] = field(default_factory=list)
    budget_s: float = 0.0
    overlaps_with: List[str] = field(default_factory=list)
    required: bool = True
    voice_category_hint: str = CATEGORY_UNKNOWN

    def __post_init__(self):
        if self.source_end < self.source_start:
            raise ValueError(f"{self.turn_id}: source_end < source_start")
        object.__setattr__(self, "_frozen", True)

    def __setattr__(self, key, value):
        if key in _IMMUTABLE_TURN_FIELDS and getattr(self, "_frozen", False):
            if getattr(self, key) != value:
                raise AttributeError(
                    f"Turn.{key} is immutable (source timing/identity); "
                    f"use Clip scheduling fields instead")
        object.__setattr__(self, key, value)

    @property
    def source_duration(self) -> float:
        return self.source_end - self.source_start

    @property
    def speech_text(self) -> str:
        return (self.hi_fit or self.hi_raw).strip()

    def add_flag(self, flag: str):
        if flag not in self.flags:
            self.flags.append(flag)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("_frozen", None)
        return d


@dataclass
class Clip:
    clip_id: str
    turn_id: str
    speaker_id: str
    provider: str
    voice: str
    spoken_text: str
    path: str
    model: str = ""
    voice_params: Dict[str, Any] = field(default_factory=dict)
    reference_version: str = ""
    natural_duration: float = 0.0
    final_duration: float = 0.0
    stretch: float = 1.0
    scheduled_start: float = 0.0
    scheduled_end: float = 0.0
    track: int = 0
    degraded: bool = False
    verification: Dict[str, Any] = field(default_factory=dict)
    retry_history: List[Dict[str, Any]] = field(default_factory=list)
    accepted: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StageStatus:
    name: str
    status: str = "pending"   # pending | ok | degraded | skipped | failed
    seconds: float = 0.0
    detail: str = ""
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JobReport:
    input_identity: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    model_versions: Dict[str, Any] = field(default_factory=dict)
    stages: List[StageStatus] = field(default_factory=list)
    speakers: List[Dict[str, Any]] = field(default_factory=list)
    voice_reuse: Dict[str, List[str]] = field(default_factory=dict)
    required_turn_ids: List[str] = field(default_factory=list)
    generated_turn_ids: List[str] = field(default_factory=list)
    missing_turns: List[Dict[str, Any]] = field(default_factory=list)
    duplicate_clips: List[Dict[str, Any]] = field(default_factory=list)
    identity_violations: List[Dict[str, Any]] = field(default_factory=list)
    translation_warnings: List[Dict[str, Any]] = field(default_factory=list)
    content_warnings: List[Dict[str, Any]] = field(default_factory=list)
    timing_deviations: List[Dict[str, Any]] = field(default_factory=list)
    unresolved_overlaps: List[Dict[str, Any]] = field(default_factory=list)
    separation: Dict[str, Any] = field(default_factory=dict)
    unresolved_failures: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    # Stages a resumed run took from THIS job's checkpoint (never another
    # job's), and every review/re-voice edit applied to the lines and voices.
    resumed_from_checkpoint: List[str] = field(default_factory=list)
    applied_edits: List[Dict[str, Any]] = field(default_factory=list)
    environment: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, str] = field(default_factory=dict)
    final_status: str = STATUS_COMPLETED
    status_reasons: List[str] = field(default_factory=list)

    def stage(self, name: str) -> StageStatus:
        for s in self.stages:
            if s.name == name:
                return s
        s = StageStatus(name=name)
        self.stages.append(s)
        return s

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
