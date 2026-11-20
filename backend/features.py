"""Module on/off matrix (tracker X01, A06, L08, R07).

Every mode, engine and option the app already has is a *module* that can be
switched on or off. Nothing is deleted: a module that is off stays in the
code, is hidden in the UI, and is refused by the API with a message naming
the switch, so a request can never select it by hand-editing a field.

Two deployment profiles (env ``VOICEDUB_PROFILE``):

* ``local``  (default): the owner's desktop app. Every module keeps today's
  behaviour, so nothing changes for the personal workflow.
* ``hosted``: the public service. Only modules whose tracker rows allow a
  public surface start on; the rest start off (deny-by-default).

The owner's switch changes are stored per profile in
``<STATE_DIR>/features.json``. A switch can also pause one capability during
an incident (R07) without stopping unrelated work.

Gates:
  * ``fields``: a job request selects the module when one of these
    (field, value) pairs matches. ``True`` means "the boolean field is on".
  * ``routes``: (METHOD, path-prefix) pairs refused while the module is off.
  * ``input``: the job's input mode (url / upload / batch / srt / story /
    story_package) selects the module.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

PROFILES = ("local", "hosted")


@dataclass(frozen=True)
class Module:
    id: str
    group: str
    label: str
    desc: str
    local: bool = True                    # default in the local profile
    hosted: bool = False                  # default in the hosted profile
    fields: Tuple[Tuple[str, Any], ...] = ()
    routes: Tuple[Tuple[str, str], ...] = ()
    input: str = ""
    locked_hosted: bool = False           # cannot be switched on in hosted (security/legal: tracker row in `why`)
    why: str = ""                         # reason shown next to a hosted default/lock


# ── the matrix ───────────────────────────────────────────────────────────
# Order = display order. Groups:
GROUPS = (
    ("input", "Input modes"),
    ("pipeline", "Pipelines"),
    ("asr", "Speech-to-text"),
    ("mt", "Translation"),
    ("tts", "Voices (TTS)"),
    ("clone", "Voice cloning"),
    ("option", "Processing options"),
    ("visual", "Video editing"),
    ("surface", "App surfaces"),
    ("integration", "Integrations"),
)

_M: List[Module] = [
    # Input modes
    Module("input.url", "input", "Video link", "Start a job from a YouTube or other video link.",
           hosted=False, input="url", why="P01/L01: a link alone does not establish rights; hosted asks for the original media"),
    Module("input.upload", "input", "Upload a file", "Start a job from an uploaded video.", hosted=True, input="upload"),
    Module("input.batch", "input", "Batch", "Queue many links/files.", hosted=False, input="batch",
           why="S04: batches must be server records first"),
    Module("input.srt", "input", "SRT dub", "Dub from a supplied SRT plus the video.", hosted=True, input="srt"),
    Module("input.story", "input", "Story → video", "Write a script from a story and render clips on this PC's GPU.",
           hosted=False, input="story", routes=(("POST", "/api/jobs/story"), ("GET", "/api/jobs/{id}/story")),
           locked_hosted=True, why="local GPU feature"),
    Module("input.story_package", "input", "Story package import", "Import a prepared story package (ZIP).",
           hosted=False, input="story_package"),

    # Pipelines
    Module("pipeline.classic", "pipeline", "Classic", "The original sentence-level pipeline.", hosted=True,
           fields=(("pipeline_mode", "classic"),)),
    Module("pipeline.hindi_dialogue", "pipeline", "Hindi Dialogue", "Turn-based dialogue profile with speakers, review and revoice.",
           hosted=True, fields=(("pipeline_mode", "hindi_dialogue"),)),
    Module("pipeline.hybrid", "pipeline", "Hybrid", "Word-level hybrid pipeline.", fields=(("pipeline_mode", "hybrid"),)),
    Module("pipeline.new", "pipeline", "New (experimental)", "Modular pipeline, experimental.",
           fields=(("pipeline_mode", "new"), ("use_new_pipeline", True))),
    Module("pipeline.oneflow", "pipeline", "OneFlow", "Single-pass legacy flow (single voice).", fields=(("pipeline_mode", "oneflow"),)),
    Module("pipeline.wordchunk", "pipeline", "WordChunk", "Word-chunk legacy flow (single voice).", fields=(("pipeline_mode", "wordchunk"),)),
    Module("pipeline.srtdub", "pipeline", "SRT Direct", "SRT direct legacy flow (single voice).", fields=(("pipeline_mode", "srtdub"),)),

    # Speech-to-text
    Module("asr.groq-whisper", "asr", "Groq Whisper", "Cloud Whisper large-v3 (free tier).", hosted=True,
           fields=(("asr_model", "groq-whisper"), ("asr_model", "groq"))),
    Module("asr.large-v3", "asr", "Whisper large-v3 (this PC)", "faster-whisper on the GPU.", hosted=True, fields=(("asr_model", "large-v3"),)),
    Module("asr.large-v3-turbo", "asr", "Whisper turbo (this PC)", "faster-whisper turbo.", hosted=True, fields=(("asr_model", "large-v3-turbo"),)),
    Module("asr.medium", "asr", "Whisper medium", "Smaller local model.", fields=(("asr_model", "medium"),)),
    Module("asr.small", "asr", "Whisper small", "Smaller local model.", fields=(("asr_model", "small"),)),
    Module("asr.base", "asr", "Whisper base", "Smallest local model.", fields=(("asr_model", "base"),)),
    Module("asr.parakeet", "asr", "Parakeet", "NVIDIA Parakeet (English).", fields=(("asr_model", "parakeet"),)),
    Module("asr.whisperx", "asr", "WhisperX alignment", "Forced alignment for word timestamps.", hosted=True,
           fields=(("use_whisperx", True),)),

    # Translation
    Module("mt.groq", "mt", "Groq LLM", "LLM translation through Groq.", hosted=True, fields=(("translation_engine", "groq"),)),
    Module("mt.gemini", "mt", "Gemini", "Google Gemini translation.", hosted=True, fields=(("translation_engine", "gemini"),)),
    Module("mt.cerebras", "mt", "Cerebras", "Cerebras LLM translation.", fields=(("translation_engine", "cerebras"),)),
    Module("mt.sambanova", "mt", "SambaNova", "SambaNova LLM translation.", fields=(("translation_engine", "sambanova"),)),
    Module("mt.auto", "mt", "Auto", "Pick the first working engine.", fields=(("translation_engine", "auto"),)),
    Module("mt.turbo", "mt", "Turbo", "Parallel fast translation.", fields=(("translation_engine", "turbo"),)),
    Module("mt.gemma4", "mt", "Gemma 4", "Gemma 4 translation.", fields=(("translation_engine", "gemma4"),)),
    Module("mt.chain_dub", "mt", "Chain Dub", "Translate through an intermediate language.", fields=(("translation_engine", "chain_dub"),)),
    Module("mt.nllb", "mt", "IndicTrans2", "Local IndicTrans2.", fields=(("translation_engine", "nllb"),)),
    Module("mt.nllb_polish", "mt", "IndicTrans2 + polish", "IndicTrans2 then LLM polish.", fields=(("translation_engine", "nllb_polish"),)),
    Module("mt.google", "mt", "Google free translate", "Unofficial free endpoint (IP-blocked here).",
           fields=(("translation_engine", "google"),), locked_hosted=True, why="L06: no commercial terms for the free endpoint"),
    Module("mt.google_polish", "mt", "Google + polish", "Free Google then LLM polish.",
           fields=(("translation_engine", "google_polish"),), locked_hosted=True, why="L06"),
    Module("mt.ollama", "mt", "Ollama (this PC)", "Local LLM through Ollama.", fields=(("translation_engine", "ollama"),)),
    Module("mt.hinglish", "mt", "Hinglish AI", "Local Hinglish model.", fields=(("translation_engine", "hinglish"),)),
    Module("mt.seamless", "mt", "SeamlessM4T", "Meta SeamlessM4T.", fields=(("translation_engine", "seamless"),)),

    # Voices
    Module("tts.edge", "tts", "Edge TTS", "Microsoft Edge read-aloud voices (Madhur, Swara, Multilingual).",
           fields=(("use_edge_tts", True),), why="L06: consumer endpoint, no documented commercial rights"),
    Module("tts.google", "tts", "Google Cloud TTS", "Neural2 / Chirp 3 HD voices (needs a key).", hosted=True,
           fields=(("use_google_tts", True),)),
    Module("tts.sarvam", "tts", "Sarvam Bulbul", "Sarvam Hindi voices (paid; also the Edge fallback).", hosted=True,
           fields=(("use_sarvam_bulbul", True),)),
    Module("tts.elevenlabs", "tts", "ElevenLabs", "ElevenLabs voices (paid).", fields=(("use_elevenlabs", True),)),
    Module("tts.indic_parler", "tts", "Indic Parler", "Local Indic Parler-TTS.", fields=(("use_indic_parler", True),)),

    # Voice cloning (L02: needs speaker authority before any hosted use)
    Module("clone.coqui_xtts", "clone", "Coqui XTTS (Voice Clone)", "Clones the original speaker's voice.",
           fields=(("use_coqui_xtts", True),), locked_hosted=True, why="L02: speaker authority required"),
    Module("clone.cosyvoice", "clone", "CosyVoice", "Voice cloning TTS.", fields=(("use_cosyvoice", True),),
           locked_hosted=True, why="L02, L07 (gitlink licence)"),
    Module("clone.chatterbox", "clone", "Chatterbox", "Voice cloning TTS.", fields=(("use_chatterbox", True),),
           locked_hosted=True, why="L02"),
    Module("clone.fish_speech", "clone", "Fish Speech", "Voice cloning TTS.", fields=(("use_fish_speech", True),),
           locked_hosted=True, why="L02"),

    # Processing options
    Module("option.multi_speaker", "option", "Multi-speaker voices", "One voice per detected speaker.", hosted=True,
           fields=(("multi_speaker", True),)),
    Module("option.mix_original", "option", "Mix original background", "Keep the original music/effects under the dub.",
           hosted=True, fields=(("mix_original", True),)),
    Module("option.step_by_step", "option", "Step-by-step review", "Pause after transcription and translation.",
           hosted=True, fields=(("step_by_step", True),), routes=(("POST", "/api/jobs/{id}/continue"),)),
    Module("option.manual_review", "option", "Manual review", "Hold segments for manual review.",
           fields=(("enable_manual_review", True),)),
    Module("option.transcribe_only", "option", "Transcribe only", "Stop after the transcript.", hosted=True,
           fields=(("transcribe_only", True),)),
    Module("option.dub_chain", "option", "Dub chain", "Dub through several languages in sequence.",
           fields=(("dub_chain", True),)),
    Module("option.global_stretch", "option", "Global video stretch", "Slow the whole video to fit the dub.",
           fields=(("use_global_stretch", True),)),
    Module("option.compare_translations", "option", "Compare translations", "Side-by-side translation engines.",
           routes=(("POST", "/api/jobs/{id}/compare-translations"),)),
    Module("option.revoice", "option", "Review and revoice", "Edit lines and regenerate their audio.", hosted=True,
           routes=(("POST", "/api/jobs/{id}/dialogue/revoice"), ("POST", "/api/jobs/{id}/dialogue/review"))),

    # Video editing (L08: ordinary editing controls, not an evasion feature)
    Module("visual.transforms", "visual", "Visual adjustments", "Mirror, hue and zoom adjustments applied to the output video.",
           hosted=True, fields=(("visual_transforms", True),)),

    # App surfaces
    Module("surface.cookies", "surface", "YouTube cookie status", "Reads this PC's browser cookies for downloads.",
           routes=(("GET", "/api/cookies/status"),), locked_hosted=True, why="S02"),
    Module("surface.local_dub", "surface", "Local download-and-dub", "Next.js route that downloads on this PC and forwards the file.",
           locked_hosted=True, why="S02"),
    Module("surface.outputs", "surface", "Saved-output browser", "Lists folders on this PC's drives.",
           routes=(("GET", "/api/outputs"),), locked_hosted=True, why="S01/T04: exposes local paths"),
    Module("surface.cache_admin", "surface", "Cache admin", "View and purge model/source caches.",
           routes=(("GET", "/api/cache"), ("DELETE", "/api/cache"), ("POST", "/api/cache/purge")),
           locked_hosted=True, why="admin only"),
    Module("surface.static_files", "surface", "Direct job file URLs", "Serves every job's files at /static/jobs/<id>/... without a login.",
           routes=(("GET", "/static/jobs/*"), ("HEAD", "/static/jobs/*")),
           why="T04: needs ownership-checked signed URLs first"),
    Module("surface.global_lists", "surface", "Shared links, presets, glossary", "Saved links, presets and glossary (one shared list today).",
           routes=(("POST", "/api/links"), ("PATCH", "/api/links/{id}"), ("DELETE", "/api/links/{id}"), ("GET", "/api/links"),
                   ("POST", "/api/presets"), ("DELETE", "/api/presets/{id}"),
                   ("POST", "/api/glossary"), ("DELETE", "/api/glossary/{id}")),
           why="S01: per-account lists first"),
    Module("surface.job_list_all", "surface", "All-jobs list", "Lists every job on this machine.",
           routes=(("GET", "/api/jobs"),), why="S01: per-account history first"),

    # Integrations
    Module("integration.trainer", "integration", "Hinglish trainer forwarding", "Sends each job's source and translated SRT to the local trainer.",
           locked_hosted=True, why="S03/L04: customer scripts must never leave for training"),
    Module("integration.youtube_description", "integration", "YouTube description", "Generate a description for the result.",
           hosted=True),
]

MODULES: Dict[str, Module] = {m.id: m for m in _M}
assert len(MODULES) == len(_M), "duplicate module id"


# ── state ────────────────────────────────────────────────────────────────
def profile() -> str:
    p = (os.environ.get("VOICEDUB_PROFILE") or "local").strip().lower()
    return p if p in PROFILES else "local"


def _state_file() -> Path:
    base = os.environ.get("VOICEDUB_STATE_DIR") or str(Path(__file__).resolve().parent)
    return Path(base) / "features.json"


_lock = threading.Lock()
_cache: Dict[str, Any] = {"mtime": None, "data": {}}


def _overrides() -> Dict[str, Dict[str, bool]]:
    f = _state_file()
    try:
        mt = f.stat().st_mtime
    except OSError:
        return {}
    with _lock:
        if _cache["mtime"] != mt:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                _cache["data"] = data if isinstance(data, dict) else {}
            except (OSError, json.JSONDecodeError):
                _cache["data"] = {}
            _cache["mtime"] = mt
        return _cache["data"]


def default(mod: Module, prof: Optional[str] = None) -> bool:
    return mod.local if (prof or profile()) == "local" else mod.hosted


def is_on(module_id: str, prof: Optional[str] = None) -> bool:
    """Unknown ids are on (a new value nobody registered is not silently blocked)."""
    mod = MODULES.get(module_id)
    if mod is None:
        return True
    prof = prof or profile()
    if prof == "hosted" and mod.locked_hosted:
        return False
    ov = _overrides().get(prof, {})
    return bool(ov[module_id]) if module_id in ov else default(mod, prof)


def set_state(module_id: str, on: bool, prof: Optional[str] = None) -> bool:
    """Persist a switch (atomic write). Returns the effective state."""
    mod = MODULES.get(module_id)
    if mod is None:
        raise KeyError(module_id)
    prof = prof or profile()
    if on and prof == "hosted" and mod.locked_hosted:
        raise PermissionError(f"{mod.label} cannot be switched on in the hosted profile ({mod.why})")
    f = _state_file()
    with _lock:
        try:
            data = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
        except (OSError, json.JSONDecodeError):
            data = {}
        data.setdefault(prof, {})[module_id] = bool(on)
        tmp = f.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, f)
        _cache["mtime"] = None
    return is_on(module_id, prof)


def reset(prof: Optional[str] = None) -> None:
    """Back to the profile defaults."""
    prof = prof or profile()
    f = _state_file()
    with _lock:
        try:
            data = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
        except (OSError, json.JSONDecodeError):
            data = {}
        data.pop(prof, None)
        tmp = f.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, f)
        _cache["mtime"] = None


def describe(prof: Optional[str] = None) -> Dict[str, Any]:
    prof = prof or profile()
    ov = _overrides().get(prof, {})
    return {
        "profile": prof,
        "groups": [{"id": g, "label": label} for g, label in GROUPS],
        "modules": [{
            "id": m.id, "group": m.group, "label": m.label, "desc": m.desc,
            "on": is_on(m.id, prof), "default": default(m, prof) and not (prof == "hosted" and m.locked_hosted),
            "overridden": m.id in ov, "locked": prof == "hosted" and m.locked_hosted, "why": m.why,
        } for m in _M],
    }


# ── enforcement ──────────────────────────────────────────────────────────
def _value_matches(req: Any, fname: str, want: Any) -> bool:
    v = req.get(fname) if isinstance(req, dict) else getattr(req, fname, None)
    if want is True:
        if isinstance(v, (list, tuple)):
            return len(v) > 0
        return bool(v)
    return v == want


def selected_modules(req: Any, input_mode: Any = "") -> List[Module]:
    modes = (input_mode,) if isinstance(input_mode, str) else tuple(input_mode or ())
    out = []
    for m in _M:
        if m.input and m.input in modes:
            out.append(m)
        elif m.fields and any(_value_matches(req, f, w) for f, w in m.fields):
            out.append(m)
    return out


def job_violations(req: Any, input_mode: Any = "", prof: Optional[str] = None) -> List[Module]:
    return [m for m in selected_modules(req, input_mode) if not is_on(m.id, prof)]


def job_refusal(req: Any, input_mode: Any = "", prof: Optional[str] = None) -> Optional[str]:
    """None when allowed, otherwise the message for HTTP 400."""
    bad = job_violations(req, input_mode, prof)
    if not bad:
        return None
    names = ", ".join(f"{m.label} ({m.id})" for m in bad)
    return (f"Switched off: {names}. Choose another option, or switch the module on in "
            f"Settings → Modules.")


def _route_match(pattern: str, path: str) -> bool:
    if pattern.endswith("/*"):
        return path.rstrip("/").startswith(pattern[:-2]) and len(path.rstrip("/")) > len(pattern[:-2])
    pp, ap = pattern.strip("/").split("/"), path.strip("/").split("/")
    if len(pp) != len(ap):
        return False
    return all(a == b or (a.startswith("{") and a.endswith("}")) for a, b in zip(pp, ap))


def route_module(method: str, path: str) -> Optional[Module]:
    for m in _M:
        for meth, pat in m.routes:
            if meth == method.upper() and _route_match(pat, path):
                return m
    return None


def route_refusal(method: str, path: str, prof: Optional[str] = None) -> Optional[Tuple[int, str]]:
    """(status, detail) when the route is switched off. Hosted answers 404 so a
    switched-off surface is indistinguishable from one that does not exist."""
    mod = route_module(method, path)
    if mod is None or is_on(mod.id, prof):
        return None
    prof = prof or profile()
    if prof == "hosted":
        return 404, "Not Found"
    return 403, f"{mod.label} is switched off (Settings → Modules → {mod.id})"


def trainer_allowed() -> bool:
    """S03: forwarding is impossible in hosted regardless of env vars or overrides."""
    return profile() == "local" and is_on("integration.trainer")


def ids(group: Optional[str] = None) -> Iterable[str]:
    return [m.id for m in _M if group is None or m.group == group]
