"""Character identity across subtitle import, review and resume; no live APIs."""
import json
from pathlib import Path

import pytest

from dubbing.dialogue.contracts import Turn
from dubbing.dialogue.diarization import DiarizationResult
from dubbing.dialogue.orchestrator import DialogueConfig, DialogueOrchestrator
from dubbing.dialogue.speaker_registry import SpeakerRegistry
from dubbing.dialogue.turns import align_text_to_words, turns_from_translated_cues, words_from_cues
from dubbing.dialogue.tts import MockProvider
from test_review_resume import _orch, needs_ffmpeg


def test_supplied_hindi_labels_win_without_becoming_spoken_text(tmp_path):
    from srt_utils import parse_srt
    path = tmp_path / 'hi.srt'
    path.write_text('1\n00:00:00,000 --> 00:00:02,000\n[SPEAKER_07] नमस्ते।\n', encoding='utf-8')
    cues = parse_srt(path)
    diar = DiarizationResult([(0, 2, 'SPEAKER_00')], [(0, 2, 'SPEAKER_00')])
    automatic = turns_from_translated_cues(cues, diar)
    supplied = turns_from_translated_cues(cues, diar, prefer_labels=True)
    assert automatic[0].speaker_id == 'SPEAKER_00'
    assert supplied[0].speaker_id == 'SPEAKER_07'
    assert supplied[0].speech_text == 'नमस्ते।'


def test_labelled_cue_spanning_audio_speakers_is_still_flagged():
    cues = [{'start': 0, 'end': 2, 'text': 'हाँ। नहीं।', 'speaker_id': 'SPEAKER_07'}]
    diar = DiarizationResult([(0, 1, 'A'), (1, 2, 'B')], [(0, 1, 'A'), (1, 2, 'B')])
    t = turns_from_translated_cues(cues, diar, prefer_labels=True)[0]
    assert t.speaker_id == 'SPEAKER_07' and 'multi_speaker_cue' in t.flags


def test_english_words_keep_cue_identity_with_and_without_alignment():
    cues = [{'start': 0, 'end': 1, 'text': 'Hello.'}, {'start': 2, 'end': 3, 'text': 'Hi.'}]
    words = words_from_cues(cues)
    for result in (words, align_text_to_words(cues, words), align_text_to_words(cues, [])):
        assert [w.attribution['cue_index'] for w in result] == [0, 1]


@needs_ffmpeg
def test_review_character_name_survives_resume_and_clear(tmp_path):
    prov = MockProvider()
    def review(packet):
        return {'speaker_names': {'SPEAKER_00': 'Male lead'}}
    first, _ = _orch(tmp_path, cfg={'review_before_voice': True}, review=review, provider=prov)
    first.run()
    assert first.registry.speakers['SPEAKER_00'].display_name == 'Male lead'
    restored = SpeakerRegistry.load(tmp_path / 'out' / 'speakers.json')
    assert restored.speakers['SPEAKER_00'].display_name == 'Male lead'
    second, calls = _orch(tmp_path, cfg={'resume': True}, provider=prov)
    second.run()
    assert not calls.get('diarize')
    packet = second.build_review_packet()
    assert next(s for s in packet['speakers'] if s['speaker_id'] == 'SPEAKER_00')['display_name'] == 'Male lead'
    assert second.apply_edits({'speaker_names': {'SPEAKER_00': ''}}) == 1
    assert second.registry.speakers['SPEAKER_00'].display_name == ''


@needs_ffmpeg
def test_review_failure_never_generates_speech(tmp_path):
    provider = MockProvider()
    def broken_review(packet):
        raise RuntimeError('connection lost')
    orch, _ = _orch(tmp_path, cfg={'review_before_voice': True}, review=broken_review, provider=provider)
    result = orch.run()
    assert result.status == 'failed'
    assert not provider.calls
    assert (tmp_path / 'work' / 'checkpoint' / 'state.json').exists()


@needs_ffmpeg
def test_reviewed_hindi_cannot_be_overwritten_by_duration_rewrite(tmp_path):
    orch, _ = _orch(tmp_path)
    t = Turn('t0001', 'SPEAKER_00', 0, 1, hi_raw='मैं यहाँ हूँ।', flags=['edited'])
    calls = []
    rewrite = orch._cached_rewrite(lambda *args: calls.append(args) or 'यहाँ।')
    assert rewrite(t, t.speech_text, 0.5) is None
    assert not calls and t.speech_text == 'मैं यहाँ हूँ।'


@needs_ffmpeg
@pytest.mark.parametrize('language', ['english', 'hindi'])
def test_supplied_labels_drive_pipeline_and_reference_ranges(tmp_path, language):
    # Use IDs which deliberately contradict the audio diarizer's numeric IDs.
    p = tmp_path / 'input.srt'
    text = 'नमस्ते।' if language == 'hindi' else 'Hello.'
    p.write_text(f'1\n00:00:00,100 --> 00:00:01,000\n[SPEAKER_01] {text}\n\n'
                 f'2\n00:00:02,000 --> 00:00:03,000\n[SPEAKER_00] {text}\n', encoding='utf-8')
    cfg = {'speaker_label_policy': 'supplied',
           'translated_srt' if language == 'hindi' else 'source_srt': p}
    orch, _ = _orch(tmp_path, cfg=cfg)
    orch.run()
    assert [t.speaker_id for t in orch.turns] == ['SPEAKER_01', 'SPEAKER_00']
    for t in orch.turns:
        if 'multi_speaker_cue' not in t.flags:
            assert (t.source_start, t.source_end) in orch.speaker_ranges[t.speaker_id]
        else:
            assert not orch.speaker_ranges[t.speaker_id]  # do not sample mixed voices
        assert '[SPEAKER_' not in t.speech_text


@needs_ffmpeg
def test_partial_labels_rejected_before_generation(tmp_path):
    p = tmp_path / 'hi.srt'
    p.write_text('1\n00:00:00,000 --> 00:00:01,000\nनमस्ते।\n', encoding='utf-8')
    provider = MockProvider()
    orch, _ = _orch(tmp_path, cfg={'speaker_label_policy': 'supplied', 'translated_srt': p}, provider=provider)
    result = orch.run()
    assert result.status == 'failed' and not provider.calls
    assert any('every cue' in r for r in result.reasons)


def test_api_names_validate_and_accumulate():
    from app import _clean_dialogue_edits, _merge_dialogue_edits, DialogueEditsRequest, _edits_from
    edits = _edits_from(DialogueEditsRequest(speaker_names={'SPEAKER_00': ' Lead '}))
    assert edits == {'speaker_names': {'SPEAKER_00': 'Lead'}}
    assert _merge_dialogue_edits(edits, {'speaker_names': {'SPEAKER_00': ''}}) == {'speaker_names': {'SPEAKER_00': ''}}
    for value in ('x' * 81, 'line\nbreak', 123):
        with pytest.raises(ValueError):
            _clean_dialogue_edits({'speaker_names': {'SPEAKER_00': value}})

@needs_ffmpeg
def test_changed_label_policy_cannot_reuse_old_speaker_checkpoint(tmp_path):
    first, _ = _orch(tmp_path)
    first.run()
    second, calls = _orch(tmp_path, cfg={'resume': True, 'speaker_label_policy': 'supplied'})
    result = second.run()
    assert result.status == 'failed'
    assert any('policy changed' in why for why in result.reasons)
    assert not calls.get('diarize')


def test_module_setting_reaches_dialogue_config():
    from dubbing.dialogue.modules import to_config
    assert to_config({}, {'speaker_label_policy': 'supplied'})['speaker_label_policy'] == 'supplied'
    with pytest.raises(ValueError):
        to_config({}, {'speaker_label_policy': 'guess'})
