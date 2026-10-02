'use client';

/**
 * Hindi Dialogue — review the Hindi lines and the voices.
 *
 * mode "before_voice": the job is paused after translation (state
 * review_translation) and nothing is voiced yet. The user fixes lines,
 * relabels or merges speakers and picks voices, then continues with (or
 * without) the changes.
 * mode "after_run": the job has finished. The same editor re-voices the
 * changed lines from this job's own checkpoint (unchanged lines reuse their
 * audio), with the chosen output options.
 *
 * Only what changed is sent (lib/dialogueReview.buildEdits). After a submit
 * the job page's own polling shows the progress.
 */
import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
    continueJob, dialogueClipUrl, fetchDialogueVoices, getDialogueReview, originalVideoUrl, previewVoice,
    revoiceDialogue, submitDialogueReview,
    type DialogueOutputOptions, type ReviewPacket, type ReviewSpeaker, type ReviewTurn, type VoiceOptions,
    type VoiceOverride,
} from '@/lib/api';
import {
    EMPTY_DRAFT, SERIOUS_FLAGS, applyReplace, buildEdits, countMatches, currentHindi, flagLabel, formatTime,
    isFlagged, isTurnEdited, overflowLabel, parseVoiceKey, previewReplace, resolveMerges, shortVoiceLabel,
    summarizeEdits, voiceGroups, voiceKey, voiceProviders, withDeleted, withHindi, withMerge, withSpeaker,
    withVoice, type ReviewDraft,
} from '@/lib/dialogueReview';

interface Props {
    jobId: string;
    mode: 'before_voice' | 'after_run';
    // after_run: the output options of the last run, when known
    initialOutput?: Partial<DialogueOutputOptions>;
    // called once the backend accepted the continue / re-voice
    onSubmitted?: () => void;
}

const DEFAULT_OUTPUT: DialogueOutputOptions = {
    keep_original_audio: false, english_subtitles: true, burn_subtitles: false, container: 'mp4',
};
// Long videos have thousands of lines: past this many the list is paged.
const PAGINATE_ABOVE = 200;
const PAGE_SIZE = 100;

const errText = (e: unknown) => (e instanceof Error ? e.message : String(e));
const SELECT = 'rounded bg-white/5 border border-border px-1 py-0.5 text-[11px] text-text-primary';
const SMALL_BTN = 'text-[11px] px-2 py-0.5 rounded-md bg-white/5 hover:bg-white/10 text-text-secondary transition-colors disabled:opacity-40 disabled:cursor-not-allowed';

function categoryPill(cat: string): { cls: string; short: string; label: string } {
    if (cat === 'female_like') return { cls: 'bg-pink-500/20 text-pink-300', short: 'F', label: 'female-like' };
    if (cat === 'male_like') return { cls: 'bg-blue-500/20 text-blue-300', short: 'M', label: 'male-like' };
    if (cat === 'child_like') return { cls: 'bg-teal-500/20 text-teal-300', short: 'C', label: 'child-like' };
    return { cls: 'bg-zinc-500/20 text-zinc-300', short: '?', label: 'unknown' };
}

// ── One line of the table (memoised: typing in one line re-renders only that line) ──
interface LineRowProps {
    t: ReviewTurn;
    index: number;
    hindi: string;
    speakerId: string;
    speakerChoices: { id: string; label: string }[];
    edited: boolean;
    hiEdited: boolean;
    deleted: boolean;
    match: boolean;
    clipUrl: string | null;
    showClip: boolean;
    playing: 'clip' | 'orig' | null;
    originalMissing: boolean;
    onHindi: (t: ReviewTurn, value: string) => void;
    onSpeaker: (t: ReviewTurn, speakerId: string) => void;
    onDelete: (turnId: string, on: boolean) => void;
    onPlayClip: (key: string, url: string) => void;
    onPlayOriginal: (key: string, start: number, end: number) => void;
}

const LineRow = memo(function LineRow({
    t, index, hindi, speakerId, speakerChoices, edited, hiEdited, deleted, match, clipUrl, showClip, playing,
    originalMissing, onHindi, onSpeaker, onDelete, onPlayClip, onPlayOriginal,
}: LineRowProps) {
    const overflow = overflowLabel(t.overflow_s);
    return (
        <div className={`rounded-lg border px-3 py-2 ${deleted ? 'border-red-500/30 bg-red-500/5 opacity-60'
            : edited ? 'border-amber-500/40 bg-amber-500/5' : 'border-border bg-white/[0.02]'}`}>
            <div className="flex flex-wrap items-center gap-1.5 text-[11px]">
                <span className="font-mono tabular-nums text-text-muted" title={t.turn_id}>
                    #{index + 1} · {formatTime(t.start)}–{formatTime(t.end)}
                </span>
                <select value={speakerId} disabled={deleted} className={SELECT}
                    title="Who says this line (relabel to another speaker)"
                    onChange={(e) => onSpeaker(t, e.target.value)}>
                    {speakerChoices.map((s) => <option key={s.id} value={s.id}>{s.label}</option>)}
                </select>
                {(t.flags || []).map((f) => (
                    <span key={f} className={`px-1.5 rounded ${SERIOUS_FLAGS.has(f) ? 'bg-red-500/15 text-red-400' : 'bg-zinc-500/15 text-zinc-400'}`}>
                        {flagLabel(f)}
                    </span>
                ))}
                {overflow && <span className="px-1.5 rounded bg-amber-500/15 text-amber-400">{overflow}</span>}
                {!t.required && <span className="px-1.5 rounded bg-zinc-500/15 text-zinc-400" title="May be left unvoiced">optional</span>}
                <div className="ml-auto flex items-center gap-1">
                    {showClip && (
                        <button type="button" className={SMALL_BTN} disabled={!clipUrl}
                            title={clipUrl ? 'Play the Hindi clip' : 'No voiced clip for this line'}
                            onClick={() => clipUrl && onPlayClip(`clip:${t.turn_id}`, clipUrl)}>
                            {playing === 'clip' ? '■ Stop' : '▶ Hindi'}
                        </button>
                    )}
                    <button type="button" className={SMALL_BTN} disabled={originalMissing}
                        title={originalMissing ? 'The original video is not available for this job' : 'Play this moment of the original video'}
                        onClick={() => onPlayOriginal(`orig:${t.turn_id}`, t.start, t.end)}>
                        {playing === 'orig' ? '■ Stop' : '▶ Original'}
                    </button>
                    <button type="button" onClick={() => onDelete(t.turn_id, !deleted)}
                        className={`text-[11px] px-2 py-0.5 rounded-md transition-colors ${deleted
                            ? 'bg-red-500/20 text-red-300 hover:bg-red-500/30' : 'bg-white/5 text-text-muted hover:bg-red-500/15 hover:text-red-400'}`}
                        title={deleted ? 'Keep this line' : 'Leave this line out of the dub'}>
                        {deleted ? 'Restore' : 'Delete'}
                    </button>
                </div>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2 mt-1.5">
                <p className={`text-xs text-text-secondary leading-relaxed whitespace-pre-wrap ${deleted ? 'line-through' : ''}`}>
                    {t.english}
                </p>
                <textarea lang="hi" value={hindi} disabled={deleted} spellCheck={false}
                    rows={Math.min(4, Math.max(2, Math.ceil(hindi.length / 60)))}
                    onChange={(e) => onHindi(t, e.target.value)}
                    className={`w-full resize-y rounded-lg border px-2 py-1 text-sm text-text-primary outline-none focus:border-primary disabled:cursor-not-allowed ${hiEdited
                        ? 'border-amber-500/60 bg-amber-500/5' : 'border-border bg-white/5'} ${match ? 'ring-1 ring-sky-500/60' : ''}`} />
            </div>
        </div>
    );
});

export default function DialogueReviewPanel({ jobId, mode, initialOutput, onSubmitted }: Props) {
    const [packet, setPacket] = useState<ReviewPacket | null>(null);
    const [loading, setLoading] = useState(true);
    const [loadError, setLoadError] = useState<string | null>(null);
    const [reloadKey, setReloadKey] = useState(0);
    const [loadedAt, setLoadedAt] = useState(0);   // clip cache buster: a re-voice rewrites the clips
    const [draft, setDraft] = useState<ReviewDraft>(EMPTY_DRAFT);
    const [open, setOpen] = useState(mode === 'before_voice');
    const [filter, setFilter] = useState('all');
    const [page, setPage] = useState(0);
    const [find, setFind] = useState('');
    const [replacement, setReplacement] = useState('');
    const [caseSensitive, setCaseSensitive] = useState(true);
    const [output, setOutput] = useState<DialogueOutputOptions>(() => {
        const o = { ...DEFAULT_OUTPUT };
        for (const [k, v] of Object.entries(initialOutput || {})) if (v !== undefined && v !== null) (o as any)[k] = v;
        return o;
    });
    const [submitting, setSubmitting] = useState<null | 'edits' | 'plain'>(null);
    const [submitError, setSubmitError] = useState<string | null>(null);
    const [sent, setSent] = useState<string | null>(null);
    const [extraVoices, setExtraVoices] = useState<VoiceOptions>({});
    const [playing, setPlaying] = useState<string | null>(null);
    const [playError, setPlayError] = useState<string | null>(null);
    const [originalMissing, setOriginalMissing] = useState(false);
    const [previewBusy, setPreviewBusy] = useState<string | null>(null);

    // ── Load the packet ──
    useEffect(() => {
        let alive = true;
        setLoading(true);
        setLoadError(null);
        getDialogueReview(jobId)
            .then((p) => {
                if (!alive) return;
                setPacket(p);
                setDraft(EMPTY_DRAFT);
                setLoadedAt(Date.now());
            })
            .catch((e) => { if (alive) setLoadError(errText(e)); })
            .finally(() => { if (alive) setLoading(false); });
        return () => { alive = false; };
    }, [jobId, reloadKey]);

    // A speaker voiced by a provider the packet has no voice list for: ask the backend.
    useEffect(() => {
        if (!packet) return;
        const have = new Set(voiceProviders(packet.voice_options || {}));
        if ((packet.speakers || []).every((s) => !s.provider || have.has(s.provider))) return;
        let alive = true;
        fetchDialogueVoices().then((v) => { if (alive) setExtraVoices(v || {}); }).catch(() => { });
        return () => { alive = false; };
    }, [packet]);

    const voiceOptions = useMemo<VoiceOptions>(() => {
        const out: VoiceOptions = { ...extraVoices };
        for (const [p, o] of Object.entries(packet?.voice_options || {})) {
            if (voiceProviders({ [p]: o }).length) out[p] = o;
        }
        return out;
    }, [packet, extraVoices]);
    // Providers offered to switch to: the ones this job has voices for.
    const jobProviders = useMemo(() => voiceProviders(packet?.voice_options || {}), [packet]);

    // ── Playback: one <audio> for clips/previews, a hidden <video> for original moments ──
    const audioRef = useRef<HTMLAudioElement | null>(null);
    const videoRef = useRef<HTMLVideoElement | null>(null);
    const rafRef = useRef<number | null>(null);
    const playingRef = useRef<string | null>(null);
    const tokenRef = useRef(0);
    const previewUrls = useRef<Map<string, string>>(new Map());

    const setPlayingKey = useCallback((k: string | null) => {
        playingRef.current = k;
        setPlaying(k);
    }, []);

    const stop = useCallback(() => {
        tokenRef.current += 1;
        if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
        audioRef.current?.pause();
        videoRef.current?.pause();
        setPlayingKey(null);
    }, [setPlayingKey]);

    useEffect(() => {
        const urls = previewUrls.current;
        return () => {
            if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
            audioRef.current?.pause();
            urls.forEach((u) => URL.revokeObjectURL(u));
            urls.clear();
        };
    }, []);

    const playUrl = useCallback(async (key: string, url: string) => {
        if (playingRef.current === key) return stop();
        stop();
        setPlayError(null);
        const token = tokenRef.current;
        const a = audioRef.current || (audioRef.current = new Audio());
        a.onended = () => { if (playingRef.current === key) setPlayingKey(null); };
        a.src = url;
        setPlayingKey(key);
        try {
            await a.play();
        } catch (e) {
            if (tokenRef.current !== token) return;   // stopped meanwhile
            setPlayingKey(null);
            setPlayError(`Could not play this audio: ${errText(e)}`);
        }
    }, [stop, setPlayingKey]);

    const playOriginal = useCallback(async (key: string, start: number, end: number) => {
        if (playingRef.current === key) return stop();
        stop();
        setPlayError(null);
        const token = tokenRef.current;
        const v = videoRef.current;
        if (!v) return;
        setPlayingKey(key);
        try {
            if (v.readyState < 1) {
                await new Promise<void>((resolve, reject) => {
                    const done = (ok: boolean) => () => {
                        v.removeEventListener('loadedmetadata', onOk);
                        v.removeEventListener('error', onErr);
                        if (ok) resolve(); else reject(new Error('the original video could not be loaded'));
                    };
                    const onOk = done(true);
                    const onErr = done(false);
                    v.addEventListener('loadedmetadata', onOk);
                    v.addEventListener('error', onErr);
                    v.preload = 'auto';
                    v.load();
                });
            }
            if (tokenRef.current !== token) return;
            v.currentTime = Math.max(0, start);
            await v.play();
            const tick = () => {
                if (tokenRef.current !== token) return;
                if (v.paused || v.currentTime >= end) {
                    v.pause();
                    rafRef.current = null;
                    if (playingRef.current === key) setPlayingKey(null);
                    return;
                }
                rafRef.current = requestAnimationFrame(tick);
            };
            rafRef.current = requestAnimationFrame(tick);
        } catch (e) {
            if (tokenRef.current !== token) return;
            setPlayingKey(null);
            if (v.error) {
                setOriginalMissing(true);
                setPlayError('The original video is not available for this job.');
            } else {
                setPlayError(`Could not play the original: ${errText(e)}`);
            }
        }
    }, [stop, setPlayingKey]);

    const preview = useCallback(async (s: ReviewSpeaker, v: VoiceOverride) => {
        const key = `pv:${s.speaker_id}`;
        if (playingRef.current === key) return stop();
        const cacheKey = `${v.provider}|${voiceKey(v.voice, v.pitch)}`;
        let url = previewUrls.current.get(cacheKey);
        if (!url) {
            setPreviewBusy(s.speaker_id);
            setPlayError(null);
            try {
                url = await previewVoice({ provider: v.provider, voice: v.voice, pitch: v.pitch });
                previewUrls.current.set(cacheKey, url);
            } catch (e) {
                setPlayError(`Voice preview failed: ${errText(e)}`);
                return;
            } finally {
                setPreviewBusy(null);
            }
        }
        playUrl(key, url);
    }, [stop, playUrl]);

    // ── Draft edits (stable callbacks for the memoised rows) ──
    const onHindi = useCallback((t: ReviewTurn, value: string) => setDraft((d) => withHindi(d, t, value)), []);
    const onSpeaker = useCallback((t: ReviewTurn, sid: string) => setDraft((d) => withSpeaker(d, t, sid)), []);
    const onDelete = useCallback((id: string, on: boolean) => setDraft((d) => withDeleted(d, id, on)), []);

    const turns = useMemo(() => packet?.turns || [], [packet]);
    const speakers = useMemo(() => packet?.speakers || [], [packet]);
    const merges = useMemo(() => resolveMerges(draft.merges), [draft.merges]);

    // Every speaker a line can be relabelled to (incl. ids only seen on lines, e.g. UNKNOWN).
    const speakerIds = useMemo(() => {
        const ids = speakers.map((s) => s.speaker_id);
        for (const t of turns) if (!ids.includes(t.speaker_id)) ids.push(t.speaker_id);
        return ids;
    }, [speakers, turns]);
    const speakerCounts = useMemo(() => {
        const n = new Map<string, number>();
        for (const t of turns) {
            const id = draft.speaker[t.turn_id] ?? t.speaker_id;
            n.set(id, (n.get(id) || 0) + 1);
        }
        return n;
    }, [turns, draft.speaker]);
    // "Play original" for a speaker without a reference clip: their longest line.
    const longestTurn = useMemo(() => {
        const m = new Map<string, ReviewTurn>();
        for (const t of turns) {
            const cur = m.get(t.speaker_id);
            if (!cur || t.end - t.start > cur.end - cur.start) m.set(t.speaker_id, t);
        }
        return m;
    }, [turns]);
    const speakerChoices = useMemo(() => speakerIds.map((id) => ({
        id, label: merges[id] ? `${id} → ${merges[id]}` : id,
    })), [speakerIds, merges]);

    const edits = useMemo(() => (packet ? buildEdits(packet, draft) : {}), [packet, draft]);
    const summary = useMemo(() => summarizeEdits(edits), [edits]);
    const flaggedCount = useMemo(() => turns.filter(isFlagged).length, [turns]);
    const editedCount = useMemo(() => turns.filter((t) => isTurnEdited(t, draft)).length, [turns, draft]);
    const search = useMemo(() => previewReplace(turns, draft, find, caseSensitive), [turns, draft, find, caseSensitive]);

    const rows = useMemo(() => turns.filter((t) => {
        if (filter === 'flagged') return isFlagged(t);
        if (filter === 'edited') return isTurnEdited(t, draft);
        if (filter === 'matches') return !draft.deleted[t.turn_id] && countMatches(currentHindi(t, draft), find, caseSensitive) > 0;
        if (filter.startsWith('speaker:')) return (draft.speaker[t.turn_id] ?? t.speaker_id) === filter.slice(8);
        return true;
    }), [turns, draft, filter, find, caseSensitive]);
    const paged = rows.length > PAGINATE_ABOVE;
    const pageCount = paged ? Math.ceil(rows.length / PAGE_SIZE) : 1;
    const curPage = Math.min(page, pageCount - 1);
    const shown = paged ? rows.slice(curPage * PAGE_SIZE, (curPage + 1) * PAGE_SIZE) : rows;
    const indexOf = useMemo(() => new Map(turns.map((t, i) => [t.turn_id, i])), [turns]);

    useEffect(() => { setPage(0); }, [filter]);
    useEffect(() => { if (filter === 'matches' && !find) setFilter('all'); }, [filter, find]);

    // ── Submit ──
    const submit = async (withChanges: boolean) => {
        if (mode === 'before_voice' && !withChanges && summary.total > 0
            && !confirm(`Discard your ${summary.total} change${summary.total === 1 ? '' : 's'} and continue?`)) return;
        setSubmitting(withChanges ? 'edits' : 'plain');
        setSubmitError(null);
        try {
            if (mode === 'before_voice') {
                if (withChanges && packet) await submitDialogueReview(jobId, edits);
                else await continueJob(jobId);
                setSent(withChanges && summary.total ? 'Sent — voicing continues with your changes.' : 'Voicing continues.');
            } else {
                await revoiceDialogue(jobId, { ...edits, ...output });
                setSent('Re-voice started — progress is shown above.');
            }
            stop();
            onSubmitted?.();
        } catch (e) {
            setSubmitError(errText(e));
        } finally {
            setSubmitting(null);
        }
    };

    const busy = submitting !== null || sent !== null;

    // ── Render ──
    if (mode === 'after_run' && !packet) {
        // Nothing to review (classic job, failed run, older job): stay out of the way.
        if (loading || !loadError) return null;
        return <div className="glass-card p-3 text-xs text-red-400">Line review unavailable: {loadError}</div>;
    }

    const changesText = summary.total === 0 ? 'No changes yet' : [
        summary.lines && `${summary.lines} line${summary.lines === 1 ? '' : 's'} edited`,
        summary.deleted && `${summary.deleted} deleted`,
        summary.relabelled && `${summary.relabelled} relabelled`,
        summary.voices && `${summary.voices} voice${summary.voices === 1 ? '' : 's'} changed`,
        summary.merges && `${summary.merges} speaker merge${summary.merges === 1 ? '' : 's'}`,
    ].filter(Boolean).join(' · ');

    const actions = (
        <div className="flex flex-wrap items-center gap-2">
            {mode === 'before_voice' ? (
                <>
                    <button type="button" onClick={() => submit(false)} disabled={busy}
                        className="text-sm px-4 py-2 rounded-lg border border-border bg-white/5 text-text-secondary hover:bg-white/10 transition-colors disabled:opacity-50">
                        {submitting === 'plain' ? 'Continuing...' : 'Continue without changes'}
                    </button>
                    <button type="button" onClick={() => submit(true)} disabled={busy || !packet}
                        className="px-4 py-2 rounded-lg bg-primary text-white text-sm font-medium hover:bg-primary/80 transition-colors disabled:opacity-50">
                        {submitting === 'edits' ? 'Sending...' : 'Continue with these changes'}
                    </button>
                </>
            ) : (
                <button type="button" onClick={() => submit(true)} disabled={busy}
                    title="Unchanged lines reuse their audio; changed lines and voices are voiced again"
                    className="px-4 py-2 rounded-lg bg-primary text-white text-sm font-medium hover:bg-primary/80 transition-colors disabled:opacity-50">
                    {submitting ? 'Starting...' : 'Re-voice changed lines'}
                </button>
            )}
        </div>
    );

    return (
        <div className="glass-card p-5 space-y-4 animate-slide-up">
            {/* Header */}
            {mode === 'before_voice' ? (
                <div className="p-4 rounded-xl bg-yellow-400/10 border border-yellow-400/30">
                    <div className="flex flex-wrap items-center justify-between gap-3">
                        <h3 className="text-lg font-semibold text-yellow-400">Review the Hindi lines and voices</h3>
                        {actions}
                    </div>
                    <p className="text-sm text-text-muted mt-2">
                        Nothing has been voiced yet. Fix a line, move it to another speaker or pick a voice per
                        character, then continue. Only what you change is sent.
                    </p>
                </div>
            ) : (
                <button type="button" onClick={() => setOpen(!open)}
                    className="w-full flex items-center justify-between gap-3 text-left">
                    <span className="text-sm font-medium text-text-primary">
                        {open ? '▾' : '▸'} Review lines &amp; voices
                        <span className="ml-2 text-xs font-normal text-text-muted">
                            {turns.length} lines · {speakers.length} characters{flaggedCount ? ` · ${flaggedCount} flagged` : ''}
                        </span>
                    </span>
                    <span className="text-xs text-text-muted">fix and re-voice without starting over</span>
                </button>
            )}

            {loading && <p className="text-xs text-text-muted">Loading the lines…</p>}
            {loadError && (
                <div className="rounded-lg border border-red-500/40 bg-red-500/10 px-3 py-2 text-xs text-red-400 flex items-center justify-between gap-3">
                    <span>Could not load the review: {loadError}</span>
                    <button type="button" onClick={() => setReloadKey((k) => k + 1)} className="underline">Retry</button>
                </div>
            )}
            {mode === 'before_voice' && !loading && !loadError && !packet && (
                <div className="rounded-lg border border-border px-3 py-2 text-xs text-text-muted flex items-center justify-between gap-3">
                    <span>The lines are not ready to review yet.</span>
                    <button type="button" onClick={() => setReloadKey((k) => k + 1)} className="underline">Reload</button>
                </div>
            )}

            {packet && open && (
                <>
                    {/* Characters */}
                    <div className="space-y-2">
                        <p className="text-sm font-medium text-text-primary">Characters ({speakers.length})</p>
                        {speakers.map((s) => {
                            const cat = categoryPill(s.voice_category);
                            const chosen: VoiceOverride = draft.voices[s.speaker_id]
                                ?? { provider: s.provider, voice: s.voice, pitch: s.pitch };
                            const groups = voiceGroups(voiceOptions, chosen.provider, s.voice_category,
                                chosen.provider === s.provider ? { voice: s.voice, pitch: s.pitch } : null);
                            const providers = jobProviders.includes(s.provider) || !s.provider ? jobProviders : [s.provider, ...jobProviders];
                            const into = merges[s.speaker_id];
                            const changed = Boolean(draft.voices[s.speaker_id]);
                            const longest = longestTurn.get(s.speaker_id);
                            const refKey = `ref:${s.speaker_id}`;
                            const pickProvider = (p: string) => {
                                if (p === s.provider) return setDraft((d) => withVoice(d, s, null));
                                const g = voiceGroups(voiceOptions, p, s.voice_category, null);
                                const first = g[0]?.items[0];
                                if (first) setDraft((d) => withVoice(d, s, { provider: p, voice: first.voice, pitch: first.pitch }));
                            };
                            return (
                                <div key={s.speaker_id} className={`rounded-lg border px-3 py-2 text-xs ${into ? 'border-border opacity-60' : changed ? 'border-amber-500/40 bg-amber-500/5' : 'border-border bg-white/[0.02]'}`}>
                                    <div className="flex flex-wrap items-center gap-2">
                                        <span className={`px-1.5 py-0.5 rounded font-medium ${cat.cls}`}>{cat.short}</span>
                                        <span className="font-medium text-text-primary">{s.speaker_id}</span>
                                        <span className="text-text-muted">
                                            {cat.label}{s.category_confidence != null ? ` ${Math.round(s.category_confidence * 100)}%` : ''}
                                            {' · '}{s.turns} line{s.turns === 1 ? '' : 's'} · {Math.round(s.total_speech_s || 0)}s
                                        </span>
                                        {s.override && <span className="px-1.5 rounded bg-violet-500/15 text-violet-300">your voice</span>}
                                        {into && <span className="text-amber-400">merged into {into}</span>}
                                    </div>
                                    <div className="flex flex-wrap items-center gap-2 mt-1.5">
                                        {providers.length > 1 && (
                                            <select value={chosen.provider} disabled={!!into || busy} className={SELECT}
                                                title="TTS provider" onChange={(e) => pickProvider(e.target.value)}>
                                                {providers.map((p) => (
                                                    <option key={p} value={p}>{p}{voiceOptions[p]?.paid ? ' (paid)' : ''}</option>
                                                ))}
                                            </select>
                                        )}
                                        {groups.length > 0 ? (
                                            <select value={voiceKey(chosen.voice, chosen.pitch)} disabled={!!into || busy} className={SELECT}
                                                title="Hindi voice for this character"
                                                onChange={(e) => setDraft((d) => withVoice(d, s, parseVoiceKey(chosen.provider, e.target.value)))}>
                                                {!chosen.voice && <option value={voiceKey('', null)} disabled>(choose a voice)</option>}
                                                {groups.map((g) => (
                                                    <optgroup key={g.label} label={g.label}>
                                                        {g.items.map((o) => (
                                                            <option key={voiceKey(o.voice, o.pitch)} value={voiceKey(o.voice, o.pitch)}>
                                                                {o.label || shortVoiceLabel(o.voice, o.pitch)}
                                                            </option>
                                                        ))}
                                                    </optgroup>
                                                ))}
                                            </select>
                                        ) : (
                                            <span className="text-text-secondary">{chosen.voice ? shortVoiceLabel(chosen.voice, chosen.pitch) : 'no voice'}</span>
                                        )}
                                        <span className="text-text-muted">pitch {chosen.pitch || 'default'}</span>
                                        <button type="button" className={SMALL_BTN} disabled={!chosen.voice || previewBusy === s.speaker_id}
                                            title="Hear a short Hindi sample in this voice"
                                            onClick={() => preview(s, chosen)}>
                                            {previewBusy === s.speaker_id ? 'Loading…' : playing === `pv:${s.speaker_id}` ? '■ Stop' : '▶ Preview'}
                                        </button>
                                        <button type="button" className={SMALL_BTN}
                                            disabled={!s.reference_clip && (!longest || originalMissing)}
                                            title={s.reference_clip ? 'Hear this speaker in the original' : 'Play this speaker\'s longest line from the original video'}
                                            onClick={() => s.reference_clip
                                                ? playUrl(refKey, dialogueClipUrl(jobId, s.reference_clip, loadedAt))
                                                : longest && playOriginal(refKey, longest.start, longest.end)}>
                                            {playing === refKey ? '■ Stop' : '▶ Original'}
                                        </button>
                                        <label className="flex items-center gap-1 text-text-muted ml-auto">
                                            Merge into
                                            <select value={draft.merges[s.speaker_id] || ''} disabled={busy} className={SELECT}
                                                onChange={(e) => setDraft((d) => withMerge(d, s.speaker_id, e.target.value))}>
                                                <option value="">(keep separate)</option>
                                                {speakers.filter((o) => o.speaker_id !== s.speaker_id && (!merges[o.speaker_id] || draft.merges[s.speaker_id] === o.speaker_id))
                                                    .map((o) => <option key={o.speaker_id} value={o.speaker_id}>{o.speaker_id}</option>)}
                                            </select>
                                        </label>
                                    </div>
                                </div>
                            );
                        })}
                    </div>

                    {/* Search & replace across the Hindi lines */}
                    <div className="rounded-lg border border-border px-3 py-2 space-y-1.5">
                        <div className="flex flex-wrap items-center gap-2 text-xs">
                            <input value={find} onChange={(e) => setFind(e.target.value)} placeholder="Find in Hindi lines" lang="hi"
                                className="flex-1 min-w-[10rem] rounded bg-white/5 border border-border px-2 py-1 text-text-primary outline-none focus:border-primary" />
                            <input value={replacement} onChange={(e) => setReplacement(e.target.value)} placeholder="Replace with" lang="hi"
                                className="flex-1 min-w-[10rem] rounded bg-white/5 border border-border px-2 py-1 text-text-primary outline-none focus:border-primary" />
                            <label className="flex items-center gap-1 text-text-muted">
                                <input type="checkbox" checked={caseSensitive} onChange={(e) => setCaseSensitive(e.target.checked)} />
                                Match case
                            </label>
                            <button type="button" className={SMALL_BTN} disabled={!find || search.matches === 0 || busy}
                                onClick={() => setDraft((d) => applyReplace(turns, d, find, replacement, caseSensitive))}>
                                Replace all
                            </button>
                        </div>
                        {find && (
                            <p className="text-[11px] text-text-muted">
                                {search.matches === 0 ? 'No matches' : `${search.matches} match${search.matches === 1 ? '' : 'es'} in ${search.lines} line${search.lines === 1 ? '' : 's'}`}
                                {search.matches > 0 && filter !== 'matches' && (
                                    <button type="button" onClick={() => setFilter('matches')} className="ml-2 underline">show them</button>
                                )}
                            </p>
                        )}
                    </div>

                    {/* Filter + paging */}
                    <div className="flex flex-wrap items-center gap-2 text-xs text-text-muted">
                        <span>Show</span>
                        <select value={filter} onChange={(e) => setFilter(e.target.value)} className={SELECT}>
                            <option value="all">All lines ({turns.length})</option>
                            <option value="flagged">Flagged ({flaggedCount})</option>
                            <option value="edited">Edited ({editedCount})</option>
                            {find && <option value="matches">Matching “{find}” ({search.lines})</option>}
                            {speakerIds.map((id) => (
                                <option key={id} value={`speaker:${id}`}>
                                    {id} ({speakerCounts.get(id) || 0})
                                </option>
                            ))}
                        </select>
                        <span>{rows.length} shown</span>
                        {paged && (
                            <span className="ml-auto flex items-center gap-1">
                                <button type="button" className={SMALL_BTN} disabled={curPage === 0} onClick={() => setPage(curPage - 1)}>‹ Prev</button>
                                <span className="tabular-nums">page {curPage + 1} / {pageCount}</span>
                                <button type="button" className={SMALL_BTN} disabled={curPage >= pageCount - 1} onClick={() => setPage(curPage + 1)}>Next ›</button>
                            </span>
                        )}
                    </div>

                    {/* Lines */}
                    <div className="space-y-1.5">
                        {shown.length === 0 && <p className="text-xs text-text-muted py-4 text-center">No lines match this filter.</p>}
                        {shown.map((t) => {
                            const hi = currentHindi(t, draft);
                            const sp = draft.speaker[t.turn_id] ?? t.speaker_id;
                            return (
                                <LineRow key={t.turn_id} t={t} index={indexOf.get(t.turn_id) ?? 0} hindi={hi} speakerId={sp}
                                    speakerChoices={speakerChoices} edited={isTurnEdited(t, draft)} hiEdited={hi !== t.hindi}
                                    deleted={Boolean(draft.deleted[t.turn_id])}
                                    match={Boolean(find) && countMatches(hi, find, caseSensitive) > 0}
                                    clipUrl={t.clip ? dialogueClipUrl(jobId, t.clip, loadedAt) : null}
                                    showClip={mode === 'after_run'}
                                    playing={playing === `clip:${t.turn_id}` ? 'clip' : playing === `orig:${t.turn_id}` ? 'orig' : null}
                                    originalMissing={originalMissing}
                                    onHindi={onHindi} onSpeaker={onSpeaker} onDelete={onDelete}
                                    onPlayClip={playUrl} onPlayOriginal={playOriginal} />
                            );
                        })}
                    </div>
                    {paged && (
                        <div className="flex items-center justify-end gap-1 text-xs text-text-muted">
                            <button type="button" className={SMALL_BTN} disabled={curPage === 0} onClick={() => setPage(curPage - 1)}>‹ Prev</button>
                            <span className="tabular-nums">page {curPage + 1} / {pageCount}</span>
                            <button type="button" className={SMALL_BTN} disabled={curPage >= pageCount - 1} onClick={() => setPage(curPage + 1)}>Next ›</button>
                        </div>
                    )}

                    {/* Output options for the re-voice */}
                    {mode === 'after_run' && (
                        <div className="rounded-lg border border-border px-3 py-2">
                            <p className="text-xs font-semibold text-text-primary mb-1.5">Output</p>
                            <div className="grid grid-cols-1 sm:grid-cols-2 gap-1.5 text-xs text-text-secondary">
                                <label className="flex items-center gap-2">
                                    <input type="checkbox" checked={output.keep_original_audio} disabled={busy}
                                        onChange={(e) => setOutput((o) => ({ ...o, keep_original_audio: e.target.checked }))} />
                                    Keep the English audio as a 2nd track
                                </label>
                                <label className="flex items-center gap-2">
                                    <input type="checkbox" checked={output.english_subtitles} disabled={busy}
                                        onChange={(e) => setOutput((o) => ({ ...o, english_subtitles: e.target.checked }))} />
                                    English subtitles track
                                </label>
                                <label className="flex items-center gap-2">
                                    <input type="checkbox" checked={output.burn_subtitles} disabled={busy}
                                        onChange={(e) => setOutput((o) => ({ ...o, burn_subtitles: e.target.checked }))} />
                                    Burn the Hindi subtitles into the picture <span className="text-text-muted">(re-encodes, slower)</span>
                                </label>
                                <label className="flex items-center gap-2">
                                    Container
                                    <select value={output.container} disabled={busy} className={SELECT}
                                        onChange={(e) => setOutput((o) => ({ ...o, container: e.target.value === 'mkv' ? 'mkv' : 'mp4' }))}>
                                        <option value="mp4">MP4 (plays everywhere)</option>
                                        <option value="mkv">MKV</option>
                                    </select>
                                </label>
                            </div>
                        </div>
                    )}
                </>
            )}

            {/* Status + actions */}
            {(playError || submitError || sent) && (
                <div className="space-y-1 text-xs">
                    {playError && <p className="text-amber-400">{playError}</p>}
                    {submitError && <p className="rounded-lg border border-red-500/40 bg-red-500/10 px-3 py-2 text-red-400">{submitError}</p>}
                    {sent && <p className="text-green-400">{sent}</p>}
                </div>
            )}
            {packet && (open || mode === 'before_voice') && (
                <div className="flex flex-wrap items-center justify-between gap-3 border-t border-border pt-3">
                    <div className="flex items-center gap-3 text-xs text-text-muted">
                        <span className={summary.total ? 'text-amber-400' : ''}>{changesText}</span>
                        {summary.total > 0 && !busy && (
                            <button type="button" onClick={() => setDraft(EMPTY_DRAFT)} className="underline hover:text-text-primary">
                                Discard changes
                            </button>
                        )}
                    </div>
                    {actions}
                </div>
            )}

            {/* Hidden player for "play the original moment" (loaded on first use) */}
            <video ref={videoRef} src={originalVideoUrl(jobId)} preload="none" playsInline hidden
                onError={() => setOriginalMissing(true)} />
        </div>
    );
}
