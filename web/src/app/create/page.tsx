'use client';

import { useEffect, useRef, useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { getDraft, getQuote, getRights, submitDraft, uploadDraft, type DraftInfo, type QuoteInfo } from '@/lib/api';
import { useAuth } from '@/hooks/useAuth';

const DRAFT_KEY = 'voicedub.create.draft';

function fmtDur(s: number) {
    const t = Math.round(s);
    const h = Math.floor(t / 3600), m = Math.floor((t % 3600) / 60), sec = t % 60;
    return h ? `${h}h ${String(m).padStart(2, '0')}m ${String(sec).padStart(2, '0')}s` : `${m}m ${String(sec).padStart(2, '0')}s`;
}

function newKey() {
    return (typeof crypto !== 'undefined' && 'randomUUID' in crypto) ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`;
}

export default function CreatePage() {
    const router = useRouter();
    const auth = useAuth();
    const [draft, setDraft] = useState<DraftInfo | null>(null);
    const [uploading, setUploading] = useState(false);
    const [mode, setMode] = useState<'full' | 'first'>('full');
    const [minutes, setMinutes] = useState(10);
    const [quote, setQuote] = useState<QuoteInfo | null>(null);
    const [rights, setRights] = useState<{ version: string; text: string } | null>(null);
    const [accepted, setAccepted] = useState(false);
    const [busy, setBusy] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const submitKey = useRef<string>('');           // one key per quote: a double click starts one job

    useEffect(() => {
        if (auth.loaded && auth.config?.required && !auth.user) router.push('/signin');
    }, [auth.loaded, auth.config?.required, auth.user, router]);

    useEffect(() => { getRights().then(setRights).catch(() => {}); }, []);

    // restore the draft after a refresh or a sign-in round trip
    useEffect(() => {
        let saved: { id: string; mode: 'full' | 'first'; minutes: number } | null = null;
        try { saved = JSON.parse(localStorage.getItem(DRAFT_KEY) || 'null'); } catch { /* storage off */ }
        if (saved?.id) {
            getDraft(saved.id).then((d) => { setDraft(d); setMode(saved!.mode); setMinutes(saved!.minutes); }).catch(() => {
                try { localStorage.removeItem(DRAFT_KEY); } catch { /* ignore */ }
            });
        }
    }, []);
    useEffect(() => {
        if (!draft) return;
        try { localStorage.setItem(DRAFT_KEY, JSON.stringify({ id: draft.id, mode, minutes })); } catch { /* ignore */ }
    }, [draft, mode, minutes]);

    // any change to the choices invalidates the price shown
    useEffect(() => { setQuote(null); setAccepted(false); }, [draft?.id, mode, minutes]);

    const onFile = async (f: File | null) => {
        if (!f) return;
        setUploading(true); setError(null); setDraft(null);
        try { setDraft(await uploadDraft(f)); }
        catch (e) { setError(e instanceof Error ? e.message : 'Upload failed'); }
        finally { setUploading(false); }
    };

    const price = async () => {
        if (!draft) return;
        setBusy(true); setError(null);
        try {
            const q = await getQuote({ draft_id: draft.id, languages: ['hi'], minutes: mode === 'full' ? 0 : minutes, style: 'dialogue', quality: 'standard' });
            submitKey.current = newKey();
            setQuote(q);
        } catch (e) { setError(e instanceof Error ? e.message : 'Could not price this job'); }
        finally { setBusy(false); }
    };

    const start = async () => {
        if (!draft || !quote || !rights || busy) return;
        setBusy(true); setError(null);
        try {
            const r = await submitDraft(draft.id, { quote_id: quote.id, idempotency_key: submitKey.current, rights_accepted: accepted, rights_version: rights.version });
            try { localStorage.removeItem(DRAFT_KEY); } catch { /* ignore */ }
            router.push(`/jobs/${r.id}`);
        } catch (e) {
            setError(e instanceof Error ? e.message : 'Could not start the job');
            setBusy(false);
        }
    };

    const cashDue = !!quote && quote.cash_minor > 0;
    const expired = !!quote && Date.now() / 1000 >= quote.expires_at;

    return (
        <main className="max-w-2xl mx-auto px-4 py-8 space-y-6">
            <div>
                <Link href="/" className="text-xs text-text-muted hover:text-text-primary">← Back</Link>
                <h1 className="text-2xl font-semibold text-text-primary mt-1">New dub</h1>
                <p className="text-sm text-text-muted mt-1">Upload your own video, see the exact price, then start. English → Hindi, one voice.</p>
                <Link href="/create/batch" className="text-xs text-primary hover:underline">Several videos? Use a Premium batch →</Link>
            </div>

            {/* 1. upload */}
            <section className="glass-card p-4 space-y-3" aria-labelledby="s1">
                <h2 id="s1" className="text-sm font-semibold text-text-primary">1 · Your video</h2>
                <label className="block cursor-pointer border-2 border-dashed rounded-xl p-5 text-center border-border hover:border-primary/50 hover:bg-primary/5">
                    <input type="file" accept=".mp4,.mov,.mkv,.webm,.m4v,.avi,.mp3,.wav,.m4a" className="sr-only"
                        onChange={(e) => onFile(e.target.files?.[0] || null)} disabled={uploading} aria-label="Choose a video file" />
                    <span className="text-sm text-text-muted">{uploading ? 'Uploading and checking the file…' : draft ? 'Choose a different file' : 'Choose a video or audio file'}</span>
                </label>
                {draft && (
                    <div className="text-sm text-text-primary space-y-1" data-testid="draft">
                        <p><strong>{draft.filename}</strong> · {fmtDur(draft.duration)} · {(draft.size / 2 ** 20).toFixed(1)} MB</p>
                        <p className="text-xs text-text-muted">
                            {draft.probe.video[0] ? `${draft.probe.video[0].width}×${draft.probe.video[0].height} ${draft.probe.video[0].codec}, ` : 'audio only, '}
                            audio {draft.probe.audio[0]?.codec} {draft.probe.audio[0]?.channels}ch
                        </p>
                        {draft.probe.warnings.map((w) => <p key={w} className="text-xs text-amber-300">{w}</p>)}
                    </div>
                )}
            </section>

            {/* 2. choices */}
            <section className={`glass-card p-4 space-y-3 ${draft ? '' : 'opacity-40 pointer-events-none'}`} aria-labelledby="s2">
                <h2 id="s2" className="text-sm font-semibold text-text-primary">2 · What to make</h2>
                <fieldset className="space-y-2">
                    <legend className="text-xs text-text-muted">Length</legend>
                    <label className="flex items-center gap-2 text-sm text-text-primary">
                        <input type="radio" name="len" checked={mode === 'full'} onChange={() => setMode('full')} /> Full video{draft ? ` (${fmtDur(draft.duration)})` : ''}
                    </label>
                    <label className="flex items-center gap-2 text-sm text-text-primary">
                        <input type="radio" name="len" checked={mode === 'first'} onChange={() => setMode('first')} /> First
                        <input type="number" min={1} max={600} value={minutes} aria-label="minutes"
                            onChange={(e) => setMinutes(Math.max(1, Math.min(600, Number(e.target.value) || 1)))}
                            className="input-field w-20 px-2 py-1 text-sm" disabled={mode !== 'first'} /> minutes
                    </label>
                </fieldset>
                <div className="grid grid-cols-2 gap-3 text-sm">
                    <div><p className="text-xs text-text-muted">Language</p><p className="text-text-primary">English → Hindi</p></div>
                    <div><p className="text-xs text-text-muted">Style · Quality</p><p className="text-text-primary">Single-voice dialogue · Standard</p>
                        <p className="text-[11px] text-text-muted">Storytelling, multi-speaker and Highest Quality are not on sale yet.</p></div>
                </div>
                <button type="button" onClick={price} disabled={!draft || busy} className="btn-primary px-5 py-2 text-sm">Get price</button>
            </section>

            {/* 3. price + rights + start */}
            {quote && (
                <section className="glass-card p-4 space-y-3" aria-labelledby="s3" data-testid="quote">
                    <h2 id="s3" className="text-sm font-semibold text-text-primary">3 · Price</h2>
                    <dl className="grid grid-cols-2 gap-y-1 text-sm">
                        <dt className="text-text-muted">Time billed</dt><dd className="text-text-primary">{quote.display.billed}</dd>
                        <dt className="text-text-muted">From Standard time</dt><dd className="text-text-primary">{quote.display.standard}</dd>
                        <dt className="text-text-muted">From Premium time</dt><dd className="text-text-primary">{quote.display.premium}</dd>
                        <dt className="text-text-muted">To pay now</dt><dd className="text-text-primary font-semibold">{quote.display.cash} <span className="text-xs text-text-muted font-normal">(incl. GST {quote.display.tax})</span></dd>
                    </dl>
                    {cashDue && (
                        <p className="text-sm text-amber-300">
                            Your balance does not cover this job. Buying prepaid time opens with the payment step (coming next); nothing is charged here.
                        </p>
                    )}
                    {rights && (
                        <label className="flex items-start gap-2 text-sm text-text-primary">
                            <input type="checkbox" checked={accepted} onChange={(e) => setAccepted(e.target.checked)} className="mt-1" />
                            <span>{rights.text}</span>
                        </label>
                    )}
                    {expired && <p className="text-sm text-amber-300">This price has expired. Get a new price.</p>}
                    <button type="button" onClick={start} disabled={!accepted || busy || cashDue || expired} className="btn-primary w-full py-3 text-base font-semibold">
                        {busy ? 'Starting…' : 'Start dubbing'}
                    </button>
                </section>
            )}

            {error && <p role="status" aria-live="assertive" className="text-sm text-red-300" data-testid="error">{error}</p>}
        </main>
    );
}
