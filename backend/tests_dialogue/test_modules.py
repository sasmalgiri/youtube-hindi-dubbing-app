"""Module matrix: automatic activation/deactivation, dependencies and presets."""
from dubbing.dialogue.modules import (PRESETS, STAGES, Probe, describe_matrix,
                                      resolve, to_config)

ALL_PKGS = {"faster_whisper", "pyannote.audio", "torch", "demucs", "edge_tts",
            "deep_translator", "transformers", "parler_tts", "IndicTransToolkit", "requests"}


class FakeProbe(Probe):
    def __init__(self, pkgs=ALL_PKGS, env=("HF_TOKEN", "GEMINI_API_KEY", "GROQ_API_KEY"),
                 gpu=True, ollama=False, refs=False):
        super().__init__(gpu=gpu, ollama=ollama, indicf5_refs=refs)
        self._pkgs, self._env = set(pkgs), set(env)

    def has_pkg(self, name):
        return name in self._pkgs

    def has_env(self, name):
        return name in self._env

    def has_runtime(self, name):
        return {"parler": "parler_tts", "indictrans2": "IndicTransToolkit"}[name] in self._pkgs


def _changes(res, stage):
    return [(c["action"], c["choice"]) for c in res.changes if c["stage"] == stage]


def test_every_preset_resolves_on_a_fully_equipped_pc():
    probe = FakeProbe(env=("HF_TOKEN", "GEMINI_API_KEY", "GROQ_API_KEY", "CEREBRAS_API_KEY",
                           "SARVAM_API_KEY", "OPENAI_API_KEY", "OLLAMA_MODEL"), ollama=True)
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
    probe = FakeProbe(env=("HF_TOKEN", "OLLAMA_MODEL", "GEMINI_API_KEY"), ollama=True)
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
    assert keep["available"] is False and keep["missing"][0]["name"] == "demucs"
    assert len(m["presets"]) == len(PRESETS)


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
