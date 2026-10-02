"""Module matrix: automatic activation/deactivation, dependencies and presets."""
import sys
import threading
import types

from dubbing.dialogue import modules
from dubbing.dialogue.modules import (INDICTRANS2_DEFAULT_MODEL, PRESETS, STAGES, Probe,
                                      describe_matrix, hf_model_access, resolve, to_config)

ALL_PKGS = {"faster_whisper", "pyannote.audio", "torch", "demucs", "edge_tts",
            "deep_translator", "transformers", "parler_tts", "IndicTransToolkit", "requests",
            "openvoice"}
QWEN = "qwen2.5:14b-instruct"


class FakeProbe(Probe):
    """Offline probe: env names (or {name: value}), pulled Ollama models and Hugging
    Face access results are injected; nothing reads os.environ or the network."""

    def __init__(self, pkgs=ALL_PKGS, env=("HF_TOKEN", "GEMINI_API_KEY", "GROQ_API_KEY"),
                 gpu=True, ollama=False, refs=False, ollama_models=(), hf=None):
        super().__init__(gpu=gpu, ollama=ollama, indicf5_refs=refs,
                         ollama_models=list(ollama_models), hf_models=dict(hf or {}))
        self._pkgs = set(pkgs)
        self._env = dict(env) if isinstance(env, dict) else {n: "set" for n in env}

    def has_pkg(self, name):
        return name in self._pkgs

    def has_env(self, name):
        return bool(self._env.get(name))

    def env_value(self, name):
        return self._env.get(name, "")

    def has_runtime(self, name):
        return {"parler": "parler_tts", "indictrans2": "IndicTransToolkit",
                "openvoice": "openvoice"}[name] in self._pkgs


def _changes(res, stage):
    return [(c["action"], c["choice"]) for c in res.changes if c["stage"] == stage]


def test_every_preset_resolves_on_a_fully_equipped_pc():
    env = {k: "set" for k in ("HF_TOKEN", "GEMINI_API_KEY", "GROQ_API_KEY", "CEREBRAS_API_KEY",
                              "SARVAM_API_KEY", "OPENAI_API_KEY")}
    probe = FakeProbe(env=dict(env, OLLAMA_MODEL=QWEN), ollama=True, ollama_models=[QWEN],
                      hf={INDICTRANS2_DEFAULT_MODEL: (True, "")})
    for p in PRESETS:
        ctx = {"source_kind": "url", "files": {"hindi_srt": p["id"] == "hindi-srt-revoice"}}
        res = resolve(p["id"], ctx=ctx, probe=probe)
        assert res.ok, (p["id"], res.blocking)
        assert res.selections["voices"], p["id"]


def test_free_online_defaults():
    res = resolve("free-online", ctx={"source_kind": "url"}, probe=FakeProbe())
    assert res.selections["speakers"] == ["pyannote"]
    assert res.selections["background"] == ["keep"]
    assert res.selections["voices"] == ["edge"]
    assert res.selections["translation"][0] == "gemini"
    cfg = res.config
    assert cfg["diarization"] and cfg["background"] == "demucs" and cfg["tts_providers"] == ["edge"]


def test_missing_hf_token_deactivates_speakers_with_fix():
    res = resolve("free-online", probe=FakeProbe(env=("GEMINI_API_KEY",)))
    assert ("deactivated", "pyannote") in _changes(res, "speakers")
    assert ("activated", "off") in _changes(res, "speakers")
    fix = [c["fix"] for c in res.changes if c["choice"] == "pyannote"][0]
    assert "HF_TOKEN" in fix or "huggingface" in fix
    assert res.config["diarization"] is False


def test_missing_demucs_switches_background_off():
    probe = FakeProbe(pkgs=ALL_PKGS - {"demucs"})
    res = resolve("free-online", probe=probe)
    assert res.selections["background"] == ["none"]
    assert res.config["background"] == "none"


def test_paid_voices_need_allow_paid():
    res = resolve("free-online", overrides={"voices": ["sarvam", "edge"]},
                  probe=FakeProbe(env=("HF_TOKEN", "SARVAM_API_KEY", "GEMINI_API_KEY")))
    assert res.selections["voices"] == ["edge"]
    assert any("paid" in c["reason"] for c in res.changes)
    res2 = resolve("free-online", overrides={"voices": ["sarvam", "edge"], "params": {"allow_paid": True}},
                   probe=FakeProbe(env=("HF_TOKEN", "SARVAM_API_KEY", "GEMINI_API_KEY")))
    assert res2.selections["voices"] == ["sarvam", "edge"]


def test_hindi_srt_skips_asr_and_translation_and_rewrite():
    res = resolve("free-online", ctx={"files": {"hindi_srt": True}}, probe=FakeProbe())
    assert res.selections["text_source"] == ["hindi_srt"]
    assert res.selections["asr"] == [] and res.selections["translation"] == []
    assert res.selections["duration_rewrite"] == ["off"]
    assert res.ok


def test_rewrite_needs_an_llm():
    res = resolve("free-online", overrides={"translation": ["google_basic"]}, probe=FakeProbe())
    assert res.selections["duration_rewrite"] == ["off"]
    assert ("deactivated", "on") in _changes(res, "duration_rewrite")


def test_no_llm_keys_fall_back_to_basic_translation():
    res = resolve("free-online", probe=FakeProbe(env=("HF_TOKEN",)))
    assert res.selections["translation"] == ["google_basic"]
    assert res.config["translation_engines"] == [] and res.config["mt_engines"] == ["google_basic"]
    assert any("line by line" in w for w in res.warnings)


def test_local_preset_drops_cloud_ai_and_keeps_edge_last():
    probe = FakeProbe(env={"HF_TOKEN": "x", "OLLAMA_MODEL": QWEN, "GEMINI_API_KEY": "x"},
                      ollama=True, ollama_models=[QWEN])
    res = resolve("free-local", overrides={"translation": ["gemini", "ollama"]}, probe=probe)
    assert res.selections["translation"] == ["ollama"]
    assert ("deactivated", "gemini") in _changes(res, "translation")
    assert res.selections["voices"] == ["indic_parler", "edge"]
    assert res.selections["asr"] == ["whisper_local"]


def test_local_preset_without_ollama_uses_indictrans2_and_no_rewrite():
    probe = FakeProbe(env=("HF_TOKEN",), ollama=False)
    res = resolve("free-local", probe=probe)
    assert res.selections["translation"] == ["indictrans2"]
    assert res.selections["duration_rewrite"] == ["off"]


def test_local_voices_missing_fall_back_to_edge():
    probe = FakeProbe(pkgs=ALL_PKGS - {"parler_tts"}, env=("HF_TOKEN",))
    res = resolve("free-local", probe=probe)
    assert res.selections["voices"] == ["edge"]
    assert ("deactivated", "indic_parler") in _changes(res, "voices")


def test_no_gpu_is_a_warning_not_a_block():
    res = resolve("free-online", probe=FakeProbe(gpu=False))
    assert res.ok and any("GPU" in w for w in res.warnings)


def test_nothing_workable_blocks_with_fixes():
    res = resolve("free-online", probe=FakeProbe(pkgs=set(), env=()))
    assert not res.ok
    assert any("Hindi voices" in b for b in res.blocking)


def test_matrix_describes_every_choice_with_availability():
    m = describe_matrix(probe=FakeProbe(pkgs=ALL_PKGS - {"demucs"}))
    stages = {s["id"]: s for s in m["stages"]}
    assert set(stages) == {s.id for s in STAGES}
    keep = [c for c in stages["background"]["choices"] if c["id"] == "keep"][0]
    assert keep["available"] is False and keep["missing"][0]["name"] == "audio_separator|demucs"
    assert len(m["presets"]) == len(PRESETS)


def test_audio_separator_alone_keeps_background():
    probe = FakeProbe(pkgs=(ALL_PKGS - {"demucs"}) | {"audio_separator"})
    res = resolve("free-online", probe=probe)
    assert res.selections["background"] == ["keep"] and res.config["background"] == "demucs"


def test_single_voice_preset():
    res = resolve("single-narrator", probe=FakeProbe())
    assert res.config["diarization"] is False
    assert any("Single voice" in w for w in res.warnings)


def test_to_config_maps_chain_order():
    cfg = to_config({"translation": ["ollama", "indictrans2", "google_basic"], "voices": ["indic_parler", "edge"]},
                    {"ollama_model": "m"})
    assert cfg["translation_engines"] == ["ollama"]
    assert cfg["mt_engines"] == ["indictrans2", "google_basic"]
    assert cfg["tts_providers"] == ["indic_parler", "edge"] and cfg["ollama_model"] == "m"


# ── Ollama: the model is named (OLLAMA_MODEL / option), never guessed ─────────
def test_free_local_never_guesses_the_ollama_model():
    # The first pulled model here was a 1.5B "Translate to Hindi:" fine-tune that
    # cannot return the translator's JSON: picking it said "will run" and
    # translated nothing.
    probe = FakeProbe(env=("HF_TOKEN",), ollama=True,
                      ollama_models=["hinglish-translator:latest", QWEN])
    res = resolve("free-local", probe=probe)
    assert ("deactivated", "ollama") in _changes(res, "translation")
    fix = [c["fix"] for c in res.changes if c["choice"] == "ollama"][0]
    assert "OLLAMA_MODEL" in fix and "backend/.env" in fix and QWEN in fix
    assert res.params["ollama_model"] == "" and res.config["ollama_model"] == ""
    assert not any("hinglish" in str(c) for c in res.changes)
    assert res.selections["translation"] == ["indictrans2"]
    assert res.selections["duration_rewrite"] == ["off"]     # no LLM left to shorten lines


def test_ollama_model_from_env_or_option():
    pulled = ["hinglish-translator:latest", QWEN, "gemma3:latest"]
    from_env = resolve("free-local", probe=FakeProbe(env={"HF_TOKEN": "x", "OLLAMA_MODEL": QWEN},
                                                    ollama=True, ollama_models=pulled))
    assert from_env.selections["translation"][0] == "ollama"
    from_option = resolve("free-local", overrides={"params": {"ollama_model": "Gemma3"}},
                          probe=FakeProbe(env=("HF_TOKEN",), ollama=True, ollama_models=pulled))
    assert from_option.selections["translation"][0] == "ollama"      # 'gemma3' = 'gemma3:latest'
    assert from_option.config["ollama_model"] == "Gemma3"


def test_ollama_model_that_is_not_pulled_is_deactivated():
    probe = FakeProbe(env={"HF_TOKEN": "x", "OLLAMA_MODEL": "llama3.1:8b"}, ollama=True,
                      ollama_models=[QWEN])
    res = resolve("free-local", probe=probe)
    ch = [c for c in res.changes if c["choice"] == "ollama"][0]
    assert ch["action"] == "deactivated" and "llama3.1:8b" in ch["reason"]
    assert "ollama pull llama3.1:8b" in ch["fix"]


def test_ollama_not_running_says_how_to_set_it_up():
    res = resolve("free-local", probe=FakeProbe(env=("HF_TOKEN",), ollama=False))
    ch = [c for c in res.changes if c["choice"] == "ollama"][0]
    assert ch["reason"] == "missing: ollama" and "OLLAMA_MODEL" in ch["fix"]


# ── IndicTrans2: the 1B checkpoint, access checked before the job ─────────────
def test_indictrans2_access_refused_shows_before_the_job():
    assert INDICTRANS2_DEFAULT_MODEL == "ai4bharat/indictrans2-en-indic-1B"
    why = "gated model: accept its terms at https://huggingface.co/x"
    probe = FakeProbe(env=("HF_TOKEN",), hf={INDICTRANS2_DEFAULT_MODEL: (False, why)})
    res = resolve("free-local", probe=probe)
    ch = [c for c in res.changes if c["choice"] == "indictrans2"][0]
    assert ch["action"] == "deactivated" and INDICTRANS2_DEFAULT_MODEL in ch["reason"]
    assert why in ch["fix"]
    # local-only and nothing else can translate: blocked in the preview, not mid-job
    assert not res.ok and any(why in b for b in res.blocking)


def test_indictrans2_model_follows_env():
    dist = "ai4bharat/indictrans2-en-indic-dist-200M"
    probe = FakeProbe(env={"HF_TOKEN": "x", "INDICTRANS2_MODEL": dist},
                      hf={dist: (False, "gated"), INDICTRANS2_DEFAULT_MODEL: (True, "")})
    res = resolve("free-local", probe=probe)
    ch = [c for c in res.changes if c["choice"] == "indictrans2"][0]
    assert ch["action"] == "deactivated" and dist in ch["reason"]
    ok = resolve("free-local", probe=FakeProbe(env=("HF_TOKEN",), hf=probe.hf_models))
    assert ok.selections["translation"] == ["indictrans2"]


def test_unconfirmed_hf_access_is_a_warning_not_a_block():
    why = "could not confirm access to the model on huggingface.co (ConnectionError: offline)"
    res = resolve("free-local", probe=FakeProbe(env=("HF_TOKEN",),
                                                hf={INDICTRANS2_DEFAULT_MODEL: (None, why)}))
    assert res.ok and res.selections["translation"] == ["indictrans2"]
    assert any(why in w for w in res.warnings)
    # also when IndicTrans2 is only reached as the automatic fallback
    res2 = resolve("free-online", overrides={"translation": ["gemini"]},
                   probe=FakeProbe(env=("HF_TOKEN",), hf={INDICTRANS2_DEFAULT_MODEL: (None, why)}))
    assert ("activated", "indictrans2") in _changes(res2, "translation")
    assert any(why in w for w in res2.warnings)


def test_plain_probe_never_goes_online(monkeypatch):
    def boom(repo_id, timeout=5.0):
        raise AssertionError("network check without check_hf")
    monkeypatch.setattr(modules, "hf_model_access", boom)
    assert Probe().hf_access(INDICTRANS2_DEFAULT_MODEL) == (None, "")
    assert Probe(check_hf=True, hf_models={"r": (True, "")}).hf_access("r") == (True, "")


def _hub_error(name, status):
    err = type(name, (Exception,), {})(f"{status} Client Error.")
    err.response = types.SimpleNamespace(status_code=status)
    return err


def _fake_hub(monkeypatch, outcome=None, cached=None):
    """A stand-in huggingface_hub: auth_check runs `outcome` (raise / wait)."""
    calls = []
    hub = types.ModuleType("huggingface_hub")

    def auth_check(repo_id, token=None):
        calls.append((repo_id, token))
        if outcome:
            outcome()
    hub.auth_check = auth_check
    hub.try_to_load_from_cache = lambda repo_id, filename: cached
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    return calls


def _raise(err):
    def go():
        raise err
    return go


def test_hf_model_access_classifies_hub_answers(monkeypatch, tmp_path):
    repo = INDICTRANS2_DEFAULT_MODEL
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    calls = _fake_hub(monkeypatch)
    assert hf_model_access(repo) == (True, "")
    assert calls == [(repo, "hf_test_token")]                # the token the worker uses

    _fake_hub(monkeypatch, _raise(_hub_error("GatedRepoError", 403)))
    access, why = hf_model_access(repo)
    assert access is False and f"https://huggingface.co/{repo}" in why
    _fake_hub(monkeypatch, _raise(_hub_error("GatedRepoError", 401)))
    assert "missing or invalid" in hf_model_access(repo)[1]
    _fake_hub(monkeypatch, _raise(_hub_error("RepositoryNotFoundError", 404)))
    assert hf_model_access("ai4bharat/nope")[0] is False

    # refused, but a downloaded copy is cached: transformers loads it -> not blocked
    _fake_hub(monkeypatch, _raise(_hub_error("GatedRepoError", 403)), cached="C:/hf/config.json")
    access, why = hf_model_access(repo)
    assert access is None and "cache" in why

    # cannot be told: unknown, never blocked
    _fake_hub(monkeypatch, _raise(ConnectionError("offline")))
    access, why = hf_model_access(repo)
    assert access is None and "could not confirm" in why
    release = threading.Event()
    _fake_hub(monkeypatch, lambda: release.wait(5))
    access, why = hf_model_access(repo, timeout=0.2)
    release.set()
    assert access is None and "did not answer" in why

    # a checkpoint folder on this PC needs no Hub at all
    calls = _fake_hub(monkeypatch, _raise(AssertionError("asked the Hub")))
    assert hf_model_access(str(tmp_path)) == (True, "") and calls == []


# ── Hindi SRT preset without a Hindi SRT ──────────────────────────────────────
def test_hindi_srt_preset_without_srt_blocks_instead_of_dubbing_nothing():
    for files in ({}, {"english_srt": True}):
        res = resolve("hindi-srt-revoice", ctx={"source_kind": "url", "files": files},
                      probe=FakeProbe())
        assert not res.ok
        assert ("My Hindi SRT → Voices needs a Hindi .srt: switch the input to SRT mode, "
                "or pick 'Free — Online'") in res.blocking
        # speech-to-text / translation are not switched off for an SRT that is not there
        assert not [c for c in res.changes if c["action"] == "skipped"]
        assert res.selections["text_source"] != ["hindi_srt"]
    res = resolve("free-online", overrides={"text_source": ["hindi_srt"]},
                  ctx={"source_kind": "file"}, probe=FakeProbe())
    assert not res.ok and any("needs a Hindi .srt" in b for b in res.blocking)


def test_hindi_srt_preset_with_srt_skips_asr_and_translation():
    res = resolve("hindi-srt-revoice", ctx={"source_kind": "url", "files": {"hindi_srt": True}},
                  probe=FakeProbe())
    assert res.ok and res.selections["text_source"] == ["hindi_srt"]
    assert res.selections["asr"] == [] and res.selections["translation"] == []
    assert ("skipped", "auto") in _changes(res, "asr")


# ── HF token under either name (diarization.hf_token_from_env reads both) ─────
def test_huggingface_token_name_enables_speaker_detection():
    res = resolve("free-online", probe=FakeProbe(env=("HUGGINGFACE_TOKEN", "GEMINI_API_KEY")))
    assert res.selections["speakers"] == ["pyannote"] and res.config["diarization"] is True


# ── "Sound like the original speaker" (experimental, off by default) ─────────
def test_voice_match_is_off_in_every_preset():
    for p in PRESETS:
        ctx = {"source_kind": "url", "files": {"hindi_srt": p["id"] == "hindi-srt-revoice"}}
        res = resolve(p["id"], ctx=ctx, probe=FakeProbe())
        assert res.selections["voice_match"] == ["off"], p["id"]
        assert res.config["voice_match"] == "off"
        assert not _changes(res, "voice_match")
    stage = modules.STAGE_BY_ID["voice_match"]
    assert stage.label == "Sound like the original speaker" and "EXPERIMENTAL" in stage.description


def test_voice_match_openvoice_runs_when_its_runtime_does():
    res = resolve("free-online", overrides={"voice_match": ["openvoice"]}, probe=FakeProbe())
    assert res.ok and res.selections["voice_match"] == ["openvoice"]
    assert res.config["voice_match"] == "openvoice"
    assert any("experimental" in w for w in res.warnings)
    no_gpu = resolve("free-online", overrides={"voice_match": "openvoice"},
                     probe=FakeProbe(gpu=False))
    assert no_gpu.config["voice_match"] == "openvoice"             # GPU is soft: slower, allowed
    assert any("OpenVoice" in w and "GPU" in w for w in no_gpu.warnings)


def test_voice_match_without_its_runtime_falls_back_to_off_with_the_fix():
    res = resolve("free-online", overrides={"voice_match": ["openvoice"]},
                  probe=FakeProbe(pkgs=ALL_PKGS - {"openvoice"}))
    assert res.ok and res.config["voice_match"] == "off"
    assert _changes(res, "voice_match") == [("deactivated", "openvoice"), ("activated", "off")]
    fix = [c["fix"] for c in res.changes if c["choice"] == "openvoice"][0]
    assert "OPENVOICE_PYTHON" in fix and "setup_local_ai.bat" in fix
    m = describe_matrix(probe=FakeProbe(pkgs=ALL_PKGS - {"openvoice"}))
    vm = [s for s in m["stages"] if s["id"] == "voice_match"][0]
    ov = [c for c in vm["choices"] if c["id"] == "openvoice"][0]
    assert vm["default"] == ["off"] and ov["available"] is False
    assert ov["missing"][0]["name"] == "openvoice"


def test_probe_checks_the_openvoice_runtime(monkeypatch, tmp_path):
    from dubbing.dialogue import local_workers as lw
    monkeypatch.setattr(lw, "_RUNTIME_CACHE", {})
    monkeypatch.setenv("OPENVOICE_PYTHON", str(tmp_path / "missing" / "python.exe"))
    assert Probe().has_runtime("openvoice") is False
    asked = []
    monkeypatch.setattr(lw, "runtime_available", lambda name: asked.append(name) or True)
    assert Probe().has_runtime("openvoice") is True and asked == ["openvoice"]


def test_to_config_passes_job_options_through():
    base = to_config({}, {})
    for k in ("review_before_voice", "keep_original_audio", "english_subtitles",
              "burn_subtitles", "container"):
        assert k not in base                                   # DialogueConfig defaults apply
    cfg = to_config({}, {"review_before_voice": True, "keep_original_audio": "false",
                         "english_subtitles": 0, "burn_subtitles": "1", "container": " MKV "})
    assert cfg["review_before_voice"] is True and cfg["keep_original_audio"] is False
    assert cfg["english_subtitles"] is False and cfg["burn_subtitles"] is True
    assert cfg["container"] == "mkv"
    for bad in ("avi", "", None, 3):
        assert to_config({}, {"container": bad})["container"] == "mp4"
    res = resolve("free-online", overrides={"params": {"review_before_voice": True,
                                                       "keep_original_audio": True,
                                                       "container": "mkv"}}, probe=FakeProbe())
    assert res.config["review_before_voice"] is True and res.config["keep_original_audio"] is True
    assert res.config["container"] == "mkv" and "burn_subtitles" not in res.config
