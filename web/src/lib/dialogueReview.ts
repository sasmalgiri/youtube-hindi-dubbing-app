/**
 * Hindi Dialogue review — the pure logic behind DialogueReviewPanel.
 *
 * The panel keeps the user's changes as a draft next to the packet the
 * backend sent; buildEdits() turns that draft into the request body, with
 * ONLY what differs from the packet (a line typed back to its original text,
 * or a voice set back to the current one, is not sent).
 *
 * Kept free of React and of runtime imports so it can be tested with plain
 * Node:  node --experimental-strip-types --test src/lib/dialogueReview.test.mjs
 */
import type {
    DialogueEdits, ReviewPacket, ReviewSpeaker, ReviewTurn, TurnEdit, VoiceOption, VoiceOptions, VoiceOverride,
} from './api';

export interface ReviewDraft {
    names?: Record<string, string>;         // display labels, never speaker identities
    hi: Record<string, string>;            // turn_id -> edited Hindi line
    speaker: Record<string, string>;       // turn_id -> other existing speaker
    deleted: Record<string, boolean>;      // turn_id -> drop this line
    voices: Record<string, VoiceOverride>; // speaker_id -> chosen voice
    merges: Record<string, string>;        // from speaker_id -> into speaker_id
}

export const EMPTY_DRAFT: ReviewDraft = { hi: {}, speaker: {}, deleted: {}, voices: {}, merges: {} };

const samePitch = (a: string | null | undefined, b: string | null | undefined) => (a || '') === (b || '');

export function sameVoice(v: VoiceOverride, s: Pick<ReviewSpeaker, 'provider' | 'voice' | 'pitch'>): boolean {
    return v.provider === s.provider && v.voice === s.voice && samePitch(v.pitch, s.pitch);
}

/** The Hindi line as it stands in the draft. */
export function currentHindi(t: ReviewTurn, d: ReviewDraft): string {
    return d.hi[t.turn_id] ?? t.hindi;
}

export function isTurnEdited(t: ReviewTurn, d: ReviewDraft): boolean {
    return Boolean(d.deleted[t.turn_id])
        || (d.hi[t.turn_id] !== undefined && d.hi[t.turn_id] !== t.hindi)
        || (d.speaker[t.turn_id] !== undefined && d.speaker[t.turn_id] !== t.speaker_id);
}

/** Draft with one turn's Hindi set; typing the original back removes the edit. */
export function withHindi(d: ReviewDraft, t: ReviewTurn, value: string): ReviewDraft {
    const hi = { ...d.hi };
    if (value === t.hindi) delete hi[t.turn_id];
    else hi[t.turn_id] = value;
    return { ...d, hi };
}

export function withSpeaker(d: ReviewDraft, t: ReviewTurn, speakerId: string): ReviewDraft {
    const speaker = { ...d.speaker };
    if (!speakerId || speakerId === t.speaker_id) delete speaker[t.turn_id];
    else speaker[t.turn_id] = speakerId;
    return { ...d, speaker };
}

export function withDeleted(d: ReviewDraft, turnId: string, on: boolean): ReviewDraft {
    const deleted = { ...d.deleted };
    if (on) deleted[turnId] = true;
    else delete deleted[turnId];
    return { ...d, deleted };
}

export function withVoice(d: ReviewDraft, s: ReviewSpeaker, v: VoiceOverride | null): ReviewDraft {
    const voices = { ...d.voices };
    if (!v || sameVoice(v, s)) delete voices[s.speaker_id];
    else voices[s.speaker_id] = v;
    return { ...d, voices };
}

export function withMerge(d: ReviewDraft, from: string, into: string): ReviewDraft {
    const merges = { ...d.merges };
    if (!into || into === from) {
        delete merges[from];
        return { ...d, merges };
    }
    merges[from] = into;
    for (const k of Object.keys(merges)) {
        if (merges[k] !== from) continue;
        if (k === into) delete merges[k];   // `into` was merged into `from`: the newer choice wins, no loop
        else merges[k] = into;              // speakers merged into `from` follow it
    }
    return { ...d, merges };
}

/** Follow merge chains (A -> B, B -> C gives A -> C); loops and self-merges are dropped. */
export function resolveMerges(merges: Record<string, string>): Record<string, string> {
    const out: Record<string, string> = {};
    for (const from of Object.keys(merges)) {
        let into = merges[from];
        const seen = new Set([from]);
        while (into && merges[into] && !seen.has(into)) {
            seen.add(into);
            into = merges[into];
        }
        if (into && !seen.has(into)) out[from] = into;
    }
    return out;
}

export function withName(d: ReviewDraft, s: ReviewSpeaker, name: string): ReviewDraft {
    const names = { ...d.names };
    if (name === (s.display_name || '')) delete names[s.speaker_id];
    else names[s.speaker_id] = name;
    return { ...d, names };
}

/** Request body: only the fields the user changed, empty groups left out. */
export function buildEdits(packet: ReviewPacket, d: ReviewDraft): DialogueEdits {
    const turn_edits: Record<string, TurnEdit> = {};
    for (const t of packet.turns) {
        if (d.deleted[t.turn_id]) {
            turn_edits[t.turn_id] = { delete: true };
            continue;
        }
        const e: TurnEdit = {};
        const hi = d.hi[t.turn_id];
        if (hi !== undefined && hi !== t.hindi) e.hi = hi;
        const sp = d.speaker[t.turn_id];
        if (sp !== undefined && sp !== t.speaker_id) e.speaker_id = sp;
        if (Object.keys(e).length) turn_edits[t.turn_id] = e;
    }
    const merges = resolveMerges(d.merges);
    const voice_overrides: Record<string, VoiceOverride> = {};
    for (const s of packet.speakers) {
        const v = d.voices[s.speaker_id];
        if (!v || merges[s.speaker_id] || sameVoice(v, s)) continue;   // a merged speaker's voice is moot
        voice_overrides[s.speaker_id] = { provider: v.provider, voice: v.voice, pitch: v.pitch || null };
    }
    const edits: DialogueEdits = {};
    const names: Record<string, string> = {};
    for (const s of packet.speakers) {
        const name = d.names?.[s.speaker_id];
        if (name !== undefined && !merges[s.speaker_id] && name.trim() !== (s.display_name || '')) {
            names[s.speaker_id] = name.trim();
        }
    }
    if (Object.keys(names).length) edits.speaker_names = names;
    if (Object.keys(turn_edits).length) edits.turn_edits = turn_edits;
    if (Object.keys(voice_overrides).length) edits.voice_overrides = voice_overrides;
    if (Object.keys(merges).length) edits.speaker_merges = merges;
    return edits;
}

export interface EditSummary { lines: number; deleted: number; relabelled: number; voices: number; merges: number; names: number; total: number; }

export function summarizeEdits(e: DialogueEdits): EditSummary {
    const te = Object.values(e.turn_edits || {});
    const s = {
        lines: te.filter((x) => x.hi !== undefined).length,
        deleted: te.filter((x) => x.delete).length,
        relabelled: te.filter((x) => x.speaker_id !== undefined).length,
        voices: Object.keys(e.voice_overrides || {}).length,
        merges: Object.keys(e.speaker_merges || {}).length,
        names: Object.keys(e.speaker_names || {}).length,
    };
    return { ...s, total: s.lines + s.deleted + s.relabelled + s.voices + s.merges + s.names };
}

// ── Search & replace across the Hindi lines ──────────────────────────────

const escapeRe = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

export function countMatches(text: string, find: string, caseSensitive = true): number {
    if (!find) return 0;
    return (text.match(new RegExp(escapeRe(find), caseSensitive ? 'g' : 'gi')) || []).length;
}

export function replaceAllIn(text: string, find: string, replacement: string, caseSensitive = true): string {
    if (!find) return text;
    // A function replacement keeps "$&" / "$1" in the user's text literal.
    return text.replace(new RegExp(escapeRe(find), caseSensitive ? 'g' : 'gi'), () => replacement);
}

/** Preview count over the lines that are not deleted. */
export function previewReplace(turns: ReviewTurn[], d: ReviewDraft, find: string, caseSensitive = true) {
    let matches = 0;
    let lines = 0;
    for (const t of turns) {
        if (d.deleted[t.turn_id]) continue;
        const n = countMatches(currentHindi(t, d), find, caseSensitive);
        matches += n;
        if (n) lines += 1;
    }
    return { matches, lines };
}

export function applyReplace(turns: ReviewTurn[], d: ReviewDraft, find: string, replacement: string,
    caseSensitive = true): ReviewDraft {
    let next = d;
    for (const t of turns) {
        if (d.deleted[t.turn_id]) continue;
        const cur = currentHindi(t, d);
        const out = replaceAllIn(cur, find, replacement, caseSensitive);
        if (out !== cur) next = withHindi(next, t, out);
    }
    return next;
}

// ── Display helpers ──────────────────────────────────────────────────────

/** 75.34 -> "1:15.3" (h:mm:ss.s past an hour). */
export function formatTime(s: number): string {
    const t = Math.max(0, Number(s) || 0);
    const h = Math.floor(t / 3600);
    const m = Math.floor((t % 3600) / 60);
    const sec = (t % 60).toFixed(1).padStart(4, '0');
    return h ? `${h}:${String(m).padStart(2, '0')}:${sec}` : `${m}:${sec}`;
}

/** "+0.8 s too long" when the voiced line overruns its slot by 0.05 s or more. */
export function overflowLabel(overflow: number | null | undefined): string | null {
    const o = Number(overflow) || 0;
    return o >= 0.05 ? `+${o.toFixed(1)} s too long` : null;
}

export function isFlagged(t: ReviewTurn): boolean {
    return (t.flags || []).length > 0 || overflowLabel(t.overflow_s) !== null;
}

// Flags that mean the line probably sounds wrong (the rest are informational).
export const SERIOUS_FLAGS = new Set([
    'multi_speaker_cue', 'translation_failed', 'tts_failed', 'timing_overflow', 'speaker_unknown', 'critical_token_warning', 'translation_uncertain',
]);

export const flagLabel = (f: string) => f.replace(/_/g, ' ');

// ── Voices ────────────────────────────────────────────────────────────────

export const voiceKey = (voice: string, pitch: string | null | undefined) => `${voice}|${pitch || ''}`;

export function parseVoiceKey(provider: string, key: string): VoiceOverride {
    const i = key.lastIndexOf('|');
    const voice = i >= 0 ? key.slice(0, i) : key;
    const pitch = i >= 0 ? key.slice(i + 1) : '';
    return { provider, voice, pitch: pitch || null };
}

/** "hi-IN-MadhurNeural" + "+20Hz" -> "Madhur +20Hz" (when the backend sent no label). */
export function shortVoiceLabel(voice: string, pitch?: string | null): string {
    const name = (voice || '?').split('-').pop()!.replace('MultilingualNeural', '').replace('Neural', '') || voice;
    return pitch ? `${name} ${pitch}` : name;
}

export const CATEGORY_LABEL: Record<string, string> = {
    male_like: 'Male-like voices', female_like: 'Female-like voices',
};

/**
 * Voice choices for one speaker: the current voice first, then the voices of
 * the speaker's own category, then the other category. Duplicates dropped.
 */
export function voiceGroups(options: VoiceOptions, provider: string, category: string,
    current: { voice: string; pitch: string | null } | null): { label: string; items: VoiceOption[] }[] {
    const p = options?.[provider] || {};
    const order = category === 'female_like' ? ['female_like', 'male_like'] : ['male_like', 'female_like'];
    const seen = new Set<string>();
    const groups: { label: string; items: VoiceOption[] }[] = [];
    if (current?.voice) {
        const all = [...(p.male_like || []), ...(p.female_like || [])];
        const known = all.find((o) => o.voice === current.voice && samePitch(o.pitch, current.pitch));
        groups.push({ label: 'Current', items: [{ voice: current.voice, pitch: current.pitch, label: known?.label || shortVoiceLabel(current.voice, current.pitch) }] });
        seen.add(voiceKey(current.voice, current.pitch));
    }
    for (const cat of order) {
        const list = (p as Record<string, unknown>)[cat];
        if (!Array.isArray(list)) continue;
        const items = (list as VoiceOption[]).filter((o) => {
            const k = voiceKey(o.voice, o.pitch);
            if (!o.voice || seen.has(k)) return false;
            seen.add(k);
            return true;
        });
        if (items.length) groups.push({ label: CATEGORY_LABEL[cat] || cat, items });
    }
    return groups;
}

/** Providers that have voices to pick from (the `paid` marker is not a category). */
export function voiceProviders(options: VoiceOptions): string[] {
    return Object.keys(options || {}).filter((p) => {
        const o = options[p] || {};
        return (o.male_like || []).length + (o.female_like || []).length > 0;
    });
}
