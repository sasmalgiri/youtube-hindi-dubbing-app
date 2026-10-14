// Tests for the review panel's pure logic (no test framework needed).
// Run from web/:  node --experimental-strip-types --test src/lib/dialogueReview.test.mjs
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
    EMPTY_DRAFT, applyReplace, buildEdits, countMatches, formatTime, isFlagged, isTurnEdited, overflowLabel,
    parseVoiceKey, previewReplace, replaceAllIn, resolveMerges, summarizeEdits, voiceGroups, voiceKey,
    voiceProviders, withDeleted, withHindi, withMerge, withSpeaker, withVoice,
} from './dialogueReview.ts';

const turn = (id, speaker, hindi, extra = {}) => ({
    turn_id: id, speaker_id: speaker, start: 0, end: 1, english: 'x', hindi,
    flags: [], required: true, clip: null, overflow_s: null, ...extra,
});
const speaker = (id, voice, pitch = null, extra = {}) => ({
    speaker_id: id, voice_category: 'male_like', category_confidence: 0.9, total_speech_s: 10, turns: 2,
    provider: 'edge', voice, pitch, reference_clip: null, override: false, ...extra,
});
const packet = {
    job_stage: 'before_voice',
    turns: [turn('t1', 'S0', 'नमस्ते दोस्त'), turn('t2', 'S1', 'दोस्त कहाँ हो?'), turn('t3', 'S0', 'ठीक है')],
    speakers: [speaker('S0', 'hi-IN-MadhurNeural'), speaker('S1', 'hi-IN-SwaraNeural', null, { voice_category: 'female_like' }), speaker('S2', 'hi-IN-MadhurNeural', '+20Hz')],
    voice_options: {
        edge: {
            male_like: [{ voice: 'hi-IN-MadhurNeural', pitch: null, label: 'Madhur' }, { voice: 'hi-IN-MadhurNeural', pitch: '+20Hz', label: 'Madhur +20Hz' }],
            female_like: [{ voice: 'hi-IN-SwaraNeural', pitch: null, label: 'Swara' }],
        },
        sarvam: { paid: true },
    },
    media_duration: 10,
};

test('no changes -> empty body', () => {
    assert.deepEqual(buildEdits(packet, EMPTY_DRAFT), {});
    assert.equal(summarizeEdits({}).total, 0);
});

test('only changed turn fields are sent; typing the original back drops the edit', () => {
    let d = withHindi(EMPTY_DRAFT, packet.turns[0], 'नमस्ते भाई');
    d = withSpeaker(d, packet.turns[1], 'S0');
    d = withSpeaker(d, packet.turns[2], 'S0');          // same as current: not an edit
    assert.deepEqual(buildEdits(packet, d), {
        turn_edits: { t1: { hi: 'नमस्ते भाई' }, t2: { speaker_id: 'S0' } },
    });
    d = withHindi(d, packet.turns[0], 'नमस्ते दोस्त');
    assert.deepEqual(buildEdits(packet, d), { turn_edits: { t2: { speaker_id: 'S0' } } });
    assert.equal(isTurnEdited(packet.turns[0], d), false);
    assert.equal(isTurnEdited(packet.turns[1], d), true);
});

test('a deleted line sends only delete, and can be restored', () => {
    let d = withHindi(EMPTY_DRAFT, packet.turns[2], 'बदला');
    d = withDeleted(d, 't3', true);
    assert.deepEqual(buildEdits(packet, d).turn_edits, { t3: { delete: true } });
    d = withDeleted(d, 't3', false);
    assert.deepEqual(buildEdits(packet, d).turn_edits, { t3: { hi: 'बदला' } });
});

test('voice overrides only when different from the current voice; merged speakers skip theirs', () => {
    let d = withVoice(EMPTY_DRAFT, packet.speakers[0], { provider: 'edge', voice: 'hi-IN-MadhurNeural', pitch: null });
    assert.deepEqual(buildEdits(packet, d), {});
    d = withVoice(d, packet.speakers[0], { provider: 'edge', voice: 'hi-IN-MadhurNeural', pitch: '+20Hz' });
    d = withVoice(d, packet.speakers[2], parseVoiceKey('edge', voiceKey('hi-IN-SwaraNeural', null)));
    assert.deepEqual(buildEdits(packet, d).voice_overrides, {
        S0: { provider: 'edge', voice: 'hi-IN-MadhurNeural', pitch: '+20Hz' },
        S2: { provider: 'edge', voice: 'hi-IN-SwaraNeural', pitch: null },
    });
    d = withMerge(d, 'S2', 'S0');
    const e = buildEdits(packet, d);
    assert.deepEqual(e.speaker_merges, { S2: 'S0' });
    assert.deepEqual(Object.keys(e.voice_overrides), ['S0']);
    assert.equal(summarizeEdits(e).total, 2);
});

test('merges: chains resolve, loops and self merges are dropped', () => {
    assert.deepEqual(resolveMerges({ A: 'B', B: 'C' }), { A: 'C', B: 'C' });
    assert.deepEqual(resolveMerges({ A: 'B', B: 'A' }), {});
    assert.deepEqual(resolveMerges({ A: 'A' }), {});
    let d = withMerge(EMPTY_DRAFT, 'B', 'A');
    d = withMerge(d, 'A', 'B');                         // newer choice wins
    assert.deepEqual(d.merges, { A: 'B' });
    d = withMerge(d, 'C', 'A');                         // A already merged into B
    assert.deepEqual(resolveMerges(d.merges), { A: 'B', C: 'B' });
    d = withMerge(d, 'A', '');
    assert.deepEqual(d.merges, { C: 'A' });
});

test('search & replace: literal text, preview count, deleted lines untouched', () => {
    assert.equal(countMatches('a.b a.b axb', 'a.b'), 2);
    assert.equal(countMatches('Dost dost', 'dost', false), 2);
    assert.equal(replaceAllIn('पैसा $1', '$1', '$&'), 'पैसा $&');
    let d = withDeleted(EMPTY_DRAFT, 't2', true);
    assert.deepEqual(previewReplace(packet.turns, d, 'दोस्त'), { matches: 1, lines: 1 });
    d = applyReplace(packet.turns, d, 'दोस्त', 'यार');
    assert.deepEqual(d.hi, { t1: 'नमस्ते यार' });
    d = applyReplace(packet.turns, d, 'यार', 'दोस्त');  // back to the original: no edit left
    assert.deepEqual(d.hi, {});
});

test('voice groups: current first, own category next, no duplicates; paid marker ignored', () => {
    const g = voiceGroups(packet.voice_options, 'edge', 'female_like', { voice: 'hi-IN-MadhurNeural', pitch: '+20Hz' });
    assert.deepEqual(g.map((x) => x.label), ['Current', 'Female-like voices', 'Male-like voices']);
    assert.equal(g[0].items[0].label, 'Madhur +20Hz');
    assert.deepEqual(g[2].items.map((o) => voiceKey(o.voice, o.pitch)), ['hi-IN-MadhurNeural|']);
    assert.deepEqual(voiceGroups(packet.voice_options, 'missing', 'male_like', { voice: 'hi-IN-AaravNeural', pitch: null })
        .map((x) => [x.label, x.items[0].label]), [['Current', 'Aarav']]);
    assert.deepEqual(voiceProviders(packet.voice_options), ['edge']);
});

test('display helpers', () => {
    assert.equal(formatTime(75.34), '1:15.3');
    assert.equal(formatTime(3725), '1:02:05.0');
    assert.equal(overflowLabel(0.81), '+0.8 s too long');
    assert.equal(overflowLabel(0.01), null);
    assert.equal(overflowLabel(null), null);
    assert.equal(isFlagged(turn('a', 'S', 'x', { overflow_s: 0.3 })), true);
    assert.equal(isFlagged(turn('a', 'S', 'x', { flags: ['nonlexical'] })), true);
    assert.equal(isFlagged(turn('a', 'S', 'x')), false);
});

test('character names are editable labels; merges discard obsolete names', async () => {
    const { withName } = await import('./dialogueReview.ts');
    let d = withName(EMPTY_DRAFT, packet.speakers[0], 'Male lead');
    assert.deepEqual(buildEdits(packet, d), { speaker_names: { S0: 'Male lead' } });
    assert.equal(summarizeEdits(buildEdits(packet, d)).total, 1);
    d = withMerge(d, 'S0', 'S1');
    assert.equal(buildEdits(packet, d).speaker_names, undefined);
    const namedPacket = { ...packet, speakers: [ { ...packet.speakers[0], display_name: 'Lead' } ] };
    const cleared = withName(EMPTY_DRAFT, namedPacket.speakers[0], '');
    assert.deepEqual(buildEdits(namedPacket, cleared), { speaker_names: { S0: '' } });
});
