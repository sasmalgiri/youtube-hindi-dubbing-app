"""Module matrix, presets and the automatic activate/deactivate resolver for
the Hindi dialogue profile.

Every option the profile has is a *choice* inside a *stage*. A choice
declares what it needs (Python packages, API keys, a running service, a GPU,
the internet, an input file) and what it costs (free / free tier / paid).

`resolve()` takes a preset + the user's overrides + what is actually
installed on this machine and returns the effective selection:
  * a choice whose hard requirement is missing is deactivated, with the
    reason and the exact fix; a requirement that cannot be checked (e.g.
    huggingface.co unreachable) is a warning, not a block;
  * a stage left empty falls back to the first workable choice (recorded as
    an automatic activation);
  * choices that depend on other stages follow them (e.g. a supplied Hindi
    SRT switches speech-to-text and translation off, asking for one without
    the file blocks; duration rewrites need an LLM translator);
  * paid choices are dropped unless "allow_paid" is on;
  * "local_only" drops every cloud AI service.
Nothing is changed silently: every change is listed with its reason.
"""
from __future__ import annotations

import importlib.util
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

FREE, FREE_TIER, PAID = "free", "free_tier", "paid"

# The IndicTrans2 checkpoint the worker loads (workers/indictrans2_worker.py
# reads the same env var with the same default). Not the distilled 200M repo:
# it is gated separately and answered 403 for this account, while the 1B repo
# is accessible.
INDICTRANS2_DEFAULT_MODEL = "ai4bharat/indictrans2-en-indic-1B"
# Gated Hugging Face checkpoints whose access is checked: req name -> (env var, default).
GATED_HF_MODELS = {"indictrans2": ("INDICTRANS2_MODEL", INDICTRANS2_DEFAULT_MODEL)}

# The Ollama translator needs an instruct model that follows the JSON prompt
# and writes Devanagari Hindi; it is named, never guessed (see _ollama_problem).
OLLAMA_SUGGESTED_MODEL = "qwen2.5:14b-instruct"
OLLAMA_MODEL_FIX = (f"set OLLAMA_MODEL in backend/.env to an instruct model that writes Hindi, "
                    f"e.g. {OLLAMA_SUGGESTED_MODEL} (or pick it under 'Ollama model')")

# Import names used to detect installed packages (no import is performed).
PKG = {
    "faster_whisper": "faster-whisper",
    "pyannote.audio": "pyannote.audio>=4",
    "torch": "torch (CUDA build from pytorch.org for GPU)",
    "demucs": "demucs",
    "audio_separator": "audio-separator[gpu]",
    "edge_tts": "edge-tts",
    "deep_translator": "deep-translator",
    "transformers": "transformers",
    "requests": "requests",
}


@dataclass(frozen=True)
class Req:
    kind: str          # package | env | env_any | gpu | internet | service | source | file | refs
                       # | runtime | hf_model
    name: str
    fix: str = ""
    soft: bool = False  # soft = warning only (e.g. GPU makes it faster)


@dataclass(frozen=True)
class Choice:
    id: str
    label: str
    description: str
    cost: str = FREE
    cloud: bool = False                 # uses a remote AI service (excluded by local_only)
    requires: Tuple[Req, ...] = ()
    tags: Tuple[str, ...] = ()          # e.g. ("llm",) for translators that can rewrite


@dataclass(frozen=True)
class Stage:
    id: str
    label: str
    description: str
    choices: Tuple[Choice, ...]
    default: Tuple[str, ...]
    multi: bool = False                 # ordered chain (first = primary, rest = fallbacks)
    required: bool = True               # must keep >= 1 working choice
    fallback_order: Tuple[str, ...] = ()

    def choice(self, cid: str) -> Optional[Choice]:
        return next((c for c in self.choices if c.id == cid), None)


def _pkg(name: str, soft: bool = False) -> Req:
    return Req("package", name, f"pip install {PKG.get(name, name)}", soft)


_GPU = Req("gpu", "cuda", "NVIDIA GPU + CUDA build of torch (works on CPU, much slower)", soft=True)
_NET = Req("internet", "internet", "needs an internet connection")

STAGES: Tuple[Stage, ...] = (
    Stage("text_source", "Text source", "Where the English (or Hindi) words come from. "
          "Speakers always come from the audio.",
          (
              Choice("asr", "Speech-to-text", "Transcribe the video's audio."),
              Choice("english_srt", "My English SRT", "Use an English SRT you supply.",
                     requires=(Req("file", "english_srt", "upload an English .srt"),)),
              Choice("hindi_srt", "My Hindi SRT", "Use a Hindi SRT you supply; translation "
                     "and speech-to-text are skipped.",
                     requires=(Req("file", "hindi_srt", "upload a Hindi .srt"),)),
          ), default=("asr",), fallback_order=("asr",)),

    Stage("asr", "Speech-to-text", "Turns English speech into words with timestamps.",
          (
              Choice("auto", "Automatic", "Groq Whisper when GROQ_API_KEY is set, else local Whisper.",
                     requires=(Req("env_any", "GROQ_API_KEY|faster_whisper",
                                   "set GROQ_API_KEY or pip install faster-whisper"),)),
              Choice("whisper_local", "Whisper on this PC", "faster-whisper large-v3, unlimited.",
                     requires=(_pkg("faster_whisper"), _GPU)),
              Choice("groq", "Groq Whisper (cloud)", "Fast cloud Whisper, free tier with limits.",
                     cost=FREE_TIER, cloud=True,
                     requires=(Req("env", "GROQ_API_KEY", "set GROQ_API_KEY in backend/.env"), _NET)),
          ), default=("auto",), fallback_order=("whisper_local", "groq")),

    Stage("speakers", "Speakers (multi-speaker)", "Detects who speaks when, so every "
          "character keeps its own voice.",
          (
              Choice("pyannote", "Detect speakers", "pyannote community-1 diarization "
                     "(runs locally; model download needs a free Hugging Face token once).",
                     requires=(_pkg("pyannote.audio"), _pkg("torch"),
                               # either name: diarization.hf_token_from_env() reads both
                               Req("env_any", "HF_TOKEN|HUGGINGFACE_TOKEN",
                                   "free token from huggingface.co in HF_TOKEN (backend/.env) + "
                                   "accept the pyannote/speaker-diarization-community-1 terms"), _GPU)),
              Choice("off", "Single voice", "Everything voiced by one voice (narration)."),
          ), default=("pyannote",), fallback_order=("off",)),

    Stage("translation", "Translation", "English to Hindi. First working engine is used; "
          "the rest are fallbacks.",
          (
              Choice("gemini", "Gemini (cloud LLM)", "Contextual, free tier with daily limits.",
                     cost=FREE_TIER, cloud=True, tags=("llm",),
                     requires=(Req("env", "GEMINI_API_KEY", "free key from aistudio.google.com"), _NET)),
              Choice("groq", "Groq (cloud LLM)", "Contextual, free tier with limits.",
                     cost=FREE_TIER, cloud=True, tags=("llm",),
                     requires=(Req("env", "GROQ_API_KEY", "free key from console.groq.com"), _NET)),
              Choice("cerebras", "Cerebras (cloud LLM)", "Contextual, free tier with limits.",
                     cost=FREE_TIER, cloud=True, tags=("llm",),
                     requires=(Req("env", "CEREBRAS_API_KEY", "key from cloud.cerebras.ai"), _NET)),
              Choice("ollama", "Ollama (local LLM)", "Contextual, unlimited, runs on this PC.",
                     tags=("llm",),
                     requires=(Req("service", "ollama",
                                   f"install Ollama (ollama.com), `ollama pull {OLLAMA_SUGGESTED_MODEL}` "
                                   f"and set OLLAMA_MODEL={OLLAMA_SUGGESTED_MODEL} in backend/.env"),
                               _GPU)),
              Choice("indictrans2", "IndicTrans2 (local MT)", "AI4Bharat English->Hindi model, "
                     "unlimited and offline; translates line by line (no dialogue context).",
                     requires=(Req("runtime", "indictrans2",
                                   "python -m pip install -c backend/constraints.txt indictranstoolkit "
                                   "(runs in the app's own Python 3.10 on Windows too), or install it "
                                   "in a separate venv and set INDICTRANS2_PYTHON"),
                               Req("env", "HF_TOKEN", "accept the ai4bharat/indictrans2 model terms on "
                                   "huggingface.co and set HF_TOKEN"),
                               # the checkpoint itself is gated per repo: a refused one shows
                               # up here instead of failing when the job reaches translation
                               Req("hf_model", "indictrans2", "accept the model's terms on "
                                   "huggingface.co with the account whose token is in HF_TOKEN"),
                               _GPU)),
              Choice("google_basic", "Google Translate (basic)", "Free, no key; line by line, "
                     "no context (flagged in the report).", cloud=True,
                     requires=(_pkg("deep_translator"), _NET)),
              Choice("openai", "OpenAI (paid LLM)", "Contextual, paid per use.", cost=PAID,
                     cloud=True, tags=("llm",),
                     requires=(Req("env", "OPENAI_API_KEY", "set OPENAI_API_KEY"), _NET)),
          ), default=("gemini", "groq", "cerebras", "google_basic"), multi=True,
          fallback_order=("ollama", "indictrans2", "google_basic", "gemini", "groq", "cerebras")),

    Stage("voices", "Hindi voices", "Text-to-speech. Every speaker is bound to one voice of "
          "the first working provider; later providers are fallbacks.",
          (
              Choice("edge", "Microsoft Edge voices", "Free, online. 2 native Hindi + "
                     "multilingual voices: up to 14 male / 12 female distinct slots.",
                     cloud=True, requires=(_pkg("edge_tts"), _NET)),
              Choice("indic_parler", "Indic Parler-TTS (local)", "AI4Bharat model, free and "
                     "offline. 4 Hindi speakers (Rohit, Aman, Divya, Rani) + style variants. "
                     "Needs a GPU for usable speed.",
                     requires=(Req("runtime", "parler",
                                   "run setup_local_ai.bat: parler-tts pins transformers 4.46.1, so "
                                   "it gets its own Python env (INDIC_PARLER_PYTHON). Never pip "
                                   "install parler-tts into the app's own Python"),
                               Req("env", "HF_TOKEN", "accept the ai4bharat/indic-parler-tts terms on "
                                   "huggingface.co and set HF_TOKEN"), _GPU)),
              Choice("indicf5", "IndicF5 (local, experimental)", "Clones curated, authorised "
                     "Hindi reference voices you place in backend/voices/indicf5/.",
                     requires=(_pkg("transformers"), _pkg("torch"),
                               Req("refs", "indicf5", "add <category>/<id>.wav + .txt references"), _GPU)),
              Choice("sarvam", "Sarvam Bulbul (paid)", "Natural Indian voices, 7 documented "
                     "male/female speakers.", cost=PAID, cloud=True,
                     requires=(Req("env", "SARVAM_API_KEY", "set SARVAM_API_KEY"), _NET)),
              Choice("elevenlabs", "ElevenLabs (paid)", "Your chosen voice IDs per category.",
                     cost=PAID, cloud=True,
                     requires=(Req("env", "ELEVENLABS_API_KEY", "set ELEVENLABS_API_KEY"),
                               Req("env_any", "ELEVENLABS_VOICES_MALE|ELEVENLABS_VOICES_FEMALE",
                                   "list voice IDs in ELEVENLABS_VOICES_MALE / _FEMALE"), _NET)),
              Choice("google", "Google Cloud TTS (paid)", "WaveNet Hindi voices.", cost=PAID,
                     cloud=True, requires=(Req("env", "GOOGLE_TTS_API_KEY", "set GOOGLE_TTS_API_KEY"), _NET)),
          ), default=("edge",), multi=True, fallback_order=("edge", "indic_parler")),

    Stage("voice_match", "Sound like the original speaker", "EXPERIMENTAL. Changes the colour "
          "(timbre) of each speaker's Hindi voice toward that speaker's own voice in the video. "
          "Words, pronunciation and timing stay the Hindi voice's.",
          (
              Choice("off", "Off", "Keep the Hindi voices as the voice engine made them."),
              Choice("openvoice", "OpenVoice tone colour (local, experimental)", "OpenVoice v2 "
                     "tone colour converter (MIT), free and offline after a one-time download. "
                     "Needs at least 3 s of clean speech per speaker (shorter ones keep their "
                     "Hindi voice); a clip it cannot convert keeps its Hindi voice. Needs a GPU "
                     "for usable speed. Listen to check: results vary.",
                     requires=(Req("runtime", "openvoice",
                                   "run setup_local_ai.bat: OpenVoice pins numpy 1.22 and librosa "
                                   "0.9.1, so it gets its own Python env (OPENVOICE_PYTHON). Never "
                                   "pip install it into the app's own Python"), _GPU)),
          ), default=("off",), fallback_order=("off",)),

    Stage("background", "Background sound", "Keep the video's music/effects under the Hindi voice.",
          (
              Choice("keep", "Keep background", "Removes the English voice and keeps "
                     "music/effects (BS-Roformer / UVR MDX via audio-separator, else Demucs; an "
                     "estimate, listen to check). The separated voice also improves speaker "
                     "detection.",
                     requires=(Req("env_any", "audio_separator|demucs",
                                   'pip install "audio-separator[gpu]" (or pip install demucs)'),
                               _pkg("torch"), _GPU)),
              Choice("none", "Hindi voice only", "No background bed."),
          ), default=("keep",), fallback_order=("none",)),

    Stage("verify", "Speech check", "Re-listens to every Hindi clip with Whisper and "
          "regenerates wrong ones once.",
          (
              Choice("whisper", "Check speech", "faster-whisper Hindi re-ASR.",
                     requires=(_pkg("faster_whisper"), _GPU)),
              Choice("off", "Skip", "Faster; spoken words are not verified (reported)."),
          ), default=("whisper",), fallback_order=("off",)),

    Stage("duration_rewrite", "Shorten long lines", "Asks the LLM for a shorter, faithful "
          "Hindi line when it does not fit the original timing.",
          (
              Choice("on", "On", "Needs an LLM translator (Gemini, Groq, Cerebras, Ollama or OpenAI)."),
              Choice("off", "Off", "Long lines are only sped up (max 1.15x) and reported."),
          ), default=("on",), fallback_order=("off",)),

    Stage("subtitles", "Subtitles", "Hindi subtitles timed to the dubbed audio.",
          (
              Choice("embed", "Embed in MP4 + files", "Soft subtitles inside the video, plus SRT/VTT."),
              Choice("files", "Files only", "SRT/VTT files only."),
          ), default=("embed",), fallback_order=("files",)),
)
STAGE_BY_ID = {s.id: s for s in STAGES}

PARAMS = {
    "num_speakers": {"label": "Number of speakers", "type": "int", "default": 0,
                     "help": "0 = detect automatically"},
    "max_stretch": {"label": "Max speed-up of a line", "type": "float", "default": 1.15,
                    "min": 1.0, "max": 1.3},
    "ollama_model": {"label": "Ollama model", "type": "str", "default": "",
                     "help": f"an instruct model that writes Hindi, e.g. {OLLAMA_SUGGESTED_MODEL}; "
                             "empty = OLLAMA_MODEL from backend/.env"},
    "allow_paid": {"label": "Allow paid services", "type": "bool", "default": False},
    "local_only": {"label": "Local AI only (no cloud AI)", "type": "bool", "default": False},
}


# ── presets ─────────────────────────────────────────────────────────────────
PRESETS: Tuple[Dict[str, Any], ...] = (
    {
        "id": "free-online", "name": "Free — Online (recommended)",
        "description": "All free. Cloud LLM free tiers for translation, Edge voices, "
                       "multi-speaker, background kept, speech checked.",
        "selections": {"text_source": ["asr"], "asr": ["auto"], "speakers": ["pyannote"],
                       "translation": ["gemini", "groq", "cerebras", "google_basic"],
                       "voices": ["edge"], "background": ["keep"], "verify": ["whisper"],
                       "duration_rewrite": ["on"], "subtitles": ["embed"]},
        "params": {"allow_paid": False, "local_only": False},
    },
    {
        "id": "free-local", "name": "Free — Local AI (unlimited)",
        "description": "No API limits: Whisper, speaker detection, Ollama/IndicTrans2 "
                       "translation, Indic Parler voices and Demucs all run on your PC "
                       "(GPU strongly recommended). Edge voices only if local voices are missing.",
        "selections": {"text_source": ["asr"], "asr": ["whisper_local"], "speakers": ["pyannote"],
                       "translation": ["ollama", "indictrans2"],
                       "voices": ["indic_parler", "edge"], "background": ["keep"],
                       "verify": ["whisper"], "duration_rewrite": ["on"], "subtitles": ["embed"]},
        "params": {"allow_paid": False, "local_only": True},
    },
    {
        "id": "fast-draft", "name": "Fast Draft (free)",
        "description": "Quick preview: multi-speaker and Edge voices, but no background "
                       "separation, no speech check and no line shortening.",
        "selections": {"text_source": ["asr"], "asr": ["auto"], "speakers": ["pyannote"],
                       "translation": ["gemini", "groq", "cerebras", "google_basic"],
                       "voices": ["edge"], "background": ["none"], "verify": ["off"],
                       "duration_rewrite": ["off"], "subtitles": ["files"]},
        "params": {"allow_paid": False, "local_only": False},
    },
    {
        "id": "single-narrator", "name": "Single Narrator (free)",
        "description": "One voice for everything (documentaries, tutorials); background kept.",
        "selections": {"text_source": ["asr"], "asr": ["auto"], "speakers": ["off"],
                       "translation": ["gemini", "groq", "cerebras", "google_basic"],
                       "voices": ["edge"], "background": ["keep"], "verify": ["whisper"],
                       "duration_rewrite": ["on"], "subtitles": ["embed"]},
        "params": {"allow_paid": False, "local_only": False},
    },
    {
        "id": "premium-voices", "name": "Premium Voices (paid)",
        "description": "Sarvam Bulbul voices with Edge fallback; otherwise like Free — Online. "
                       "Costs money per character.",
        "selections": {"text_source": ["asr"], "asr": ["auto"], "speakers": ["pyannote"],
                       "translation": ["gemini", "groq", "cerebras", "openai", "google_basic"],
                       "voices": ["sarvam", "edge"], "background": ["keep"], "verify": ["whisper"],
                       "duration_rewrite": ["on"], "subtitles": ["embed"]},
        "params": {"allow_paid": True, "local_only": False},
    },
    {
        "id": "hindi-srt-revoice", "name": "My Hindi SRT → Voices (free)",
        "description": "You supply the Hindi SRT; speakers still come from the video audio.",
        "selections": {"text_source": ["hindi_srt"], "asr": ["auto"], "speakers": ["pyannote"],
                       "translation": [], "voices": ["edge"], "background": ["keep"],
                       "verify": ["whisper"], "duration_rewrite": ["off"], "subtitles": ["embed"]},
        "params": {"allow_paid": False, "local_only": False},
    },
)
PRESET_BY_ID = {p["id"]: p for p in PRESETS}
DEFAULT_PRESET = "free-online"


# ── environment probe ──────────────────────────────────────────────────────
@dataclass
class Probe:
    packages: Dict[str, bool] = field(default_factory=dict)
    env: Dict[str, bool] = field(default_factory=dict)
    gpu: Optional[bool] = None
    ollama: Optional[bool] = None
    ollama_models: List[str] = field(default_factory=list)
    indicf5_refs: bool = False
    runtimes: Dict[str, bool] = field(default_factory=dict)
    internet: bool = True     # assumed; failures surface at run time
    # Hugging Face model access: repo id -> (True / False / None = unknown, why).
    # Pre-filled entries are used as-is (tests inject them, nothing goes online);
    # others are asked of huggingface.co only when check_hf is on (detect()).
    hf_models: Dict[str, Tuple[Optional[bool], str]] = field(default_factory=dict)
    check_hf: bool = False

    def has_pkg(self, name: str) -> bool:
        if name not in self.packages:
            try:
                self.packages[name] = importlib.util.find_spec(name) is not None
            except (ImportError, ValueError):
                self.packages[name] = False
        return self.packages[name]

    def has_runtime(self, name: str) -> bool:
        if name not in self.runtimes:
            from .local_workers import runtime_available
            self.runtimes[name] = runtime_available(name)
        return self.runtimes[name]

    def env_value(self, name: str) -> str:
        return os.environ.get(name, "").strip()

    def has_env(self, name: str) -> bool:
        if name not in self.env:
            self.env[name] = bool(self.env_value(name))
        return self.env[name]

    def hf_access(self, repo_id: str) -> Tuple[Optional[bool], str]:
        """Can this PC's token download `repo_id`? (None, "") = not checked."""
        if repo_id not in self.hf_models:
            self.hf_models[repo_id] = hf_model_access(repo_id) if self.check_hf else (None, "")
        return self.hf_models[repo_id]

    @classmethod
    def detect(cls, check_gpu: bool = True, check_ollama: bool = True,
               check_hf: bool = True) -> "Probe":
        p = cls(check_hf=check_hf)
        if check_gpu and p.has_pkg("torch"):
            try:
                import torch
                p.gpu = bool(torch.cuda.is_available())
            except Exception:
                p.gpu = False
        if check_ollama:
            try:
                import requests
                host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
                if not host.startswith("http"):
                    host = "http://" + host
                r = requests.get(f"{host}/api/tags", timeout=0.6)
                p.ollama = r.status_code == 200
                if p.ollama:
                    p.ollama_models = [m.get("name", "") for m in r.json().get("models", [])]
            except Exception:
                p.ollama = False
        try:
            from .speaker_registry import indicf5_pool
            pool = indicf5_pool()
            p.indicf5_refs = bool(pool.get("refs"))
        except Exception:
            p.indicf5_refs = False
        return p


def hf_model_access(repo_id: str, timeout: float = 5.0) -> Tuple[Optional[bool], str]:
    """Whether this PC's Hugging Face token may download `repo_id`.

    (True, "") it may; (False, why) the Hub refuses it (gated terms not
    accepted, token rejected, no such repo) and no copy is cached; (None, why)
    it cannot be told (offline, Hub down or slow, huggingface_hub missing), or
    a cached copy will be used despite the refusal. Unknown is reported, never
    treated as blocked. Same token as the workers: HF_TOKEN, else the saved
    `huggingface-cli login` token (token=None).
    """
    if os.path.isdir(repo_id):
        return True, ""                      # a checkpoint folder on this PC
    box: Dict[str, Tuple[Optional[bool], str]] = {}

    def check():
        try:
            from huggingface_hub import auth_check
            auth_check(repo_id, token=os.environ.get("HF_TOKEN", "").strip() or None)
            box["result"] = (True, "")
        except Exception as e:
            box["result"] = _hf_refusal(repo_id, e)

    # auth_check has no timeout of its own: a stalled connection must not hang
    # the preview, nor the job start (which resolves the same way).
    t = threading.Thread(target=check, name="hf-auth-check", daemon=True)
    t.start()
    t.join(timeout)
    return box.get("result", (None, f"could not confirm access to {repo_id}: huggingface.co "
                                    f"did not answer within {timeout:g}s; it is tried when the "
                                    f"job runs"))


def _hf_refusal(repo_id: str, err: Exception) -> Tuple[Optional[bool], str]:
    """Classify an auth_check error: refused (False) or unknown (None)."""
    status = getattr(getattr(err, "response", None), "status_code", None)
    kinds = {c.__name__ for c in type(err).__mro__}     # no huggingface_hub import needed
    if "GatedRepoError" in kinds:        # 403: terms not accepted; 401: no/invalid token
        why = (f"gated model: accept its terms at https://huggingface.co/{repo_id} with the "
               f"account whose read token is in HF_TOKEN"
               + (" (the token was missing or invalid)" if status == 401 else ""))
    elif "RepositoryNotFoundError" in kinds or status == 404:
        why = f"no model {repo_id} on huggingface.co that this token can see (check the model id)"
    elif status in (401, 403):
        why = f"huggingface.co refused this PC's token for {repo_id} (HTTP {status}): check HF_TOKEN"
    else:
        detail = (str(err).strip().splitlines() or [""])[0][:120]
        return None, (f"could not confirm access to {repo_id} on huggingface.co "
                      f"({type(err).__name__}: {detail}); it is tried when the job runs")
    # The Hub refusing does not stop a copy that is already downloaded:
    # hf_hub_download falls back to the cache on HTTP errors.
    if _hf_cached(repo_id):
        return None, f"{why}; the copy already in the local Hugging Face cache will be used"
    return False, why


def _hf_cached(repo_id: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
        return isinstance(try_to_load_from_cache(repo_id, "config.json"), str)
    except Exception:
        return False


# ── resolver ───────────────────────────────────────────────────────────────
@dataclass
class Resolution:
    preset: str
    selections: Dict[str, List[str]]
    params: Dict[str, Any]
    changes: List[Dict[str, str]]
    warnings: List[str]
    blocking: List[str]
    config: Dict[str, Any]

    @property
    def ok(self) -> bool:
        return not self.blocking

    def to_dict(self) -> Dict[str, Any]:
        return {"preset": self.preset, "selections": self.selections, "params": self.params,
                "changes": self.changes, "warnings": self.warnings, "blocking": self.blocking,
                "ok": self.ok, "config": self.config}


def _ollama_problem(req: Req, probe: Probe, params: Dict[str, Any]) -> Optional[Req]:
    """What stops the Ollama translator here, as the specific requirement (None = nothing).

    The model is never guessed from the pulled list: the first one pulled here
    was a 1.5B "Translate to Hindi:" fine-tune that cannot return the JSON the
    dialogue translator needs, so the preview said "will run" and nothing got
    translated.
    """
    if not probe.ollama:
        return req                                    # not installed / not running
    model = str(params.get("ollama_model") or "").strip() or probe.env_value("OLLAMA_MODEL")
    if not model:
        return Req("env", "OLLAMA_MODEL", OLLAMA_MODEL_FIX)
    # Ollama reads 'qwen2.5' as 'qwen2.5:latest' and ignores case in names.
    name = model.lower()
    want = {name} if ":" in name else {name, name + ":latest"}
    if not any(m.lower() in want for m in probe.ollama_models):
        return Req("service", f"{model} (not pulled in Ollama)",
                   f"`ollama pull {model}`, or set OLLAMA_MODEL to a pulled instruct model "
                   f"that writes Hindi")
    return None


def _hf_model_problem(req: Req, probe: Probe) -> Optional[Req]:
    """A refused checkpoint as a hard requirement, an unconfirmed one as a soft
    one (a warning), None when accessible or not checked."""
    env, default = GATED_HF_MODELS[req.name]
    repo = probe.env_value(env) or default
    access, why = probe.hf_access(repo)
    if access is False:
        return Req("hf_model", repo, why or req.fix)
    if access is None and why:
        return Req("hf_model", repo, why, soft=True)
    return None


def _unmet(choice: Choice, probe: Probe, ctx: Dict[str, Any], params: Dict[str, Any]
           ) -> Tuple[List[Req], List[Req]]:
    hard, soft = [], []
    for r in choice.requires:
        ok = True
        if r.kind == "package":
            ok = probe.has_pkg(r.name)
        elif r.kind == "env":
            ok = probe.has_env(r.name)
        elif r.kind == "env_any":
            ok = any(probe.has_env(n) or probe.has_pkg(n) for n in r.name.split("|"))
        elif r.kind == "gpu":
            ok = bool(probe.gpu)
        elif r.kind == "internet":
            ok = probe.internet
        elif r.kind == "service" and r.name == "ollama":
            problem = _ollama_problem(r, probe, params)
            ok, r = problem is None, problem or r      # report the specific missing piece
        elif r.kind == "hf_model":
            problem = _hf_model_problem(r, probe)
            ok, r = problem is None, problem or r
        elif r.kind == "source":
            ok = ctx.get("source_kind") == r.name
        elif r.kind == "file":
            ok = bool(ctx.get("files", {}).get(r.name))
        elif r.kind == "refs":
            ok = probe.indicf5_refs
        elif r.kind == "runtime":
            ok = probe.has_runtime(r.name)
        if not ok:
            (soft if r.soft else hard).append(r)
    return hard, soft


def describe_matrix(probe: Optional[Probe] = None, ctx: Optional[Dict[str, Any]] = None
                    ) -> Dict[str, Any]:
    """The full matrix with per-choice availability on this machine."""
    probe = probe or Probe.detect()
    ctx = ctx or {}
    stages = []
    for s in STAGES:
        choices = []
        for c in s.choices:
            hard, soft = _unmet(c, probe, ctx, {})
            choices.append({
                "id": c.id, "label": c.label, "description": c.description, "cost": c.cost,
                "cloud": c.cloud, "tags": list(c.tags),
                "requires": [{"kind": r.kind, "name": r.name, "fix": r.fix, "soft": r.soft}
                             for r in c.requires],
                "available": not hard,
                "missing": [{"name": r.name, "fix": r.fix} for r in hard],
                "notes": [r.fix for r in soft],
            })
        stages.append({"id": s.id, "label": s.label, "description": s.description,
                       "multi": s.multi, "required": s.required, "default": list(s.default),
                       "choices": choices})
    return {"stages": stages, "params": PARAMS,
            "presets": [{k: p[k] for k in ("id", "name", "description")} for p in PRESETS],
            "default_preset": DEFAULT_PRESET,
            "environment": {"gpu": probe.gpu, "ollama": probe.ollama,
                            "ollama_models": probe.ollama_models}}


def resolve(preset: str = DEFAULT_PRESET, overrides: Optional[Dict[str, Any]] = None,
            ctx: Optional[Dict[str, Any]] = None, probe: Optional[Probe] = None) -> Resolution:
    """Effective module selection for this job on this machine.

    overrides: {"<stage>": [choice ids] | "choice", "params": {...}}
    ctx: {"source_kind": "url"|"file", "files": {"english_srt": bool, "hindi_srt": bool}}
    """
    ctx = ctx or {}
    probe = probe or Probe.detect()
    p = PRESET_BY_ID.get(preset) or PRESET_BY_ID[DEFAULT_PRESET]
    sel: Dict[str, List[str]] = {s.id: list(s.default) for s in STAGES}
    sel.update({k: list(v) for k, v in p["selections"].items()})
    params = {k: v["default"] for k, v in PARAMS.items()}
    params.update(p.get("params", {}))
    overrides = dict(overrides or {})
    params.update(overrides.pop("params", {}) or {})
    for k, v in overrides.items():
        if k in STAGE_BY_ID:
            sel[k] = [v] if isinstance(v, str) else list(v)
    changes: List[Dict[str, str]] = []
    warnings: List[str] = []
    blocking: List[str] = []

    def change(stage, action, choice, reason, fix=""):
        changes.append({"stage": stage, "action": action, "choice": choice,
                        "reason": reason, "fix": fix})

    # Inputs decide the text source first.
    files = ctx.get("files", {})
    if files.get("hindi_srt") and sel["text_source"] != ["hindi_srt"]:
        change("text_source", "activated", "hindi_srt", "a Hindi SRT was supplied")
        sel["text_source"] = ["hindi_srt"]
    elif files.get("english_srt") and sel["text_source"] not in (["english_srt"], ["hindi_srt"]):
        change("text_source", "activated", "english_srt", "an English SRT was supplied")
        sel["text_source"] = ["english_srt"]

    # A Hindi SRT asked for but not supplied: there is no Hindi text to voice.
    # Skipping speech-to-text + translation would dub nothing, and transcribing
    # instead would quietly ignore the choice -> block with the way out.
    if sel["text_source"] == ["hindi_srt"] and not files.get("hindi_srt"):
        if p["id"] == "hindi-srt-revoice":
            blocking.append("My Hindi SRT → Voices needs a Hindi .srt: switch the input to SRT "
                            "mode, or pick 'Free — Online'")
        else:
            blocking.append("Text source 'My Hindi SRT' needs a Hindi .srt: switch the input to "
                            "SRT mode, or choose 'Speech-to-text'")

    cpu_only: List[str] = []
    # Stages a Hindi SRT makes unnecessary, decided from the RESOLVED text source
    # (text_source is the first stage, so the loop settles it before these): only
    # a supplied file switches them off, never the request alone.
    hindi_srt_skips = {"asr": "Hindi SRT supplied: speech-to-text not needed",
                       "translation": "Hindi SRT supplied: translation not needed"}
    if sel["speakers"] == ["off"]:
        pass  # single voice: nothing depends on it

    def note_soft(stage, c, soft):
        # also for fallback activations: an unconfirmed requirement is never dropped silently
        for r in soft:
            if r.kind == "gpu":
                cpu_only.append(c.label)
            else:
                warnings.append(f"{stage.label} — {c.label}: {r.fix}")

    for stage in STAGES:
        sid = stage.id
        if sid in hindi_srt_skips and sel["text_source"] == ["hindi_srt"]:
            if sel.get(sid):
                change(sid, "skipped", ",".join(sel[sid]), hindi_srt_skips[sid])
            sel[sid] = []
            continue
        kept: List[str] = []
        for cid in sel.get(sid, []):
            c = stage.choice(cid)
            if c is None:
                change(sid, "deactivated", cid, "unknown option")
                continue
            if c.cost == PAID and not params.get("allow_paid"):
                change(sid, "deactivated", cid, "paid service and 'Allow paid services' is off",
                       "turn on 'Allow paid services' to use it")
                continue
            if c.cloud and params.get("local_only") and c.id != "edge":
                change(sid, "deactivated", cid, "cloud AI service and 'Local AI only' is on")
                continue
            hard, soft = _unmet(c, probe, ctx, params)
            if hard:
                change(sid, "deactivated", cid, "missing: " + ", ".join(r.name for r in hard),
                       "; ".join(r.fix for r in hard if r.fix))
                continue
            note_soft(stage, c, soft)
            kept.append(cid)
        if not stage.multi:
            kept = kept[:1]
        if params.get("local_only") and sid == "voices" and len(kept) > 1 and "edge" in kept \
                and kept[0] != "edge":
            kept = [k for k in kept if k != "edge"] + ["edge"]   # Edge only as last resort
        if not kept:
            for cid in stage.fallback_order:
                c = stage.choice(cid)
                if c is None or (c.cost == PAID and not params.get("allow_paid")):
                    continue
                if c.cloud and params.get("local_only") and c.id != "edge":
                    continue
                hard, soft = _unmet(c, probe, ctx, params)
                if not hard:
                    kept = [cid]
                    change(sid, "activated", cid,
                           f"nothing selected for '{stage.label}' could run here; using the "
                           f"first workable option")
                    note_soft(stage, c, soft)
                    break
        if not kept and stage.required:
            blocking.append(f"{stage.label}: no option can run on this machine — "
                            + "; ".join(r.fix for c in stage.choices for r in
                                        _unmet(c, probe, ctx, params)[0] if r.fix)[:400])
        sel[sid] = kept

    if cpu_only:
        warnings.append("No NVIDIA GPU detected: " + ", ".join(dict.fromkeys(cpu_only))
                        + " will run on the CPU (works, but much slower).")

    # Cross-stage dependencies.
    llms = [c for c in sel.get("translation", [])
            if "llm" in (STAGE_BY_ID["translation"].choice(c).tags if STAGE_BY_ID["translation"].choice(c) else ())]
    if sel["duration_rewrite"] == ["on"] and not llms:
        change("duration_rewrite", "deactivated", "on",
               "needs an LLM translator (Gemini, Groq, Cerebras, Ollama or OpenAI)")
        sel["duration_rewrite"] = ["off"]
    if sel.get("speakers") == ["off"]:
        warnings.append("Single voice: every line uses one voice (multi-speaker is off).")
    if "edge" in sel.get("voices", []) and params.get("local_only"):
        warnings.append("Edge voices are online; used only if the local voices cannot run.")
    if sel.get("translation") and sel["translation"][0] in ("google_basic", "indictrans2"):
        warnings.append("Primary translator works line by line (no dialogue context); "
                        "gender/formality may be less consistent.")
    if sel.get("voice_match") == ["openvoice"]:
        warnings.append("Sound like the original speaker is experimental: only the voice colour "
                        "changes, and how close it gets varies by speaker. Listen to check.")

    return Resolution(preset=p["id"], selections=sel, params=params, changes=changes,
                      warnings=warnings, blocking=blocking, config=to_config(sel, params))


def _as_bool(v: Any) -> bool:
    """A job option as a bool; form/JSON strings like "false" or "0" are False."""
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(v)


# Job options passed through to DialogueConfig as-is when the caller sets them
# (they are not modules: nothing on this PC can make them unavailable).
PASSTHROUGH_BOOLS = ("review_before_voice", "keep_original_audio", "english_subtitles",
                     "burn_subtitles")
CONTAINERS = ("mp4", "mkv")


def to_config(sel: Dict[str, List[str]], params: Dict[str, Any]) -> Dict[str, Any]:
    """DialogueConfig keyword arguments for a resolved selection."""
    asr = (sel.get("asr") or ["auto"])[0]
    translation = sel.get("translation", [])
    llm_names = [t for t in translation if t in ("gemini", "groq", "cerebras", "openai", "ollama")]
    mt_names = [t for t in translation if t in ("indictrans2", "google_basic")]
    num = int(params.get("num_speakers") or 0)
    extra: Dict[str, Any] = {k: _as_bool(params[k]) for k in PASSTHROUGH_BOOLS if k in params}
    if "container" in params:
        container = str(params["container"] or "").strip().lower()
        extra["container"] = container if container in CONTAINERS else "mp4"
    return {
        "use_youtube_subs": False,   # YouTube-subtitle input was removed (092e758)
        "asr": {"whisper_local": "local", "groq": "groq"}.get(asr, "auto"),
        "diarization": (sel.get("speakers") or ["off"])[0] == "pyannote",
        "num_speakers": num or None,
        "translation_engines": llm_names,
        "mt_engines": mt_names,
        "allow_basic_translation_fallback": "google_basic" in mt_names,
        "ollama_model": params.get("ollama_model") or "",
        "tts_providers": list(sel.get("voices") or ["edge"]),
        "background": "demucs" if (sel.get("background") or ["none"])[0] == "keep" else "none",
        "content_verify": "on" if (sel.get("verify") or ["off"])[0] == "whisper" else "off",
        "duration_rewrite": (sel.get("duration_rewrite") or ["off"])[0] == "on",
        "embed_subtitles": (sel.get("subtitles") or ["embed"])[0] == "embed",
        "max_stretch": float(params.get("max_stretch") or 1.15),
        "voice_match": "openvoice" if (sel.get("voice_match") or ["off"])[0] == "openvoice"
                       else "off",
        **extra,
    }
