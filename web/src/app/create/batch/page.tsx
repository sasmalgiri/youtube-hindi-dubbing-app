'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import {
    createBatch, getBatch, getRights, retryBatchFailures, submitBatch, uploadDraft,
    type BatchInfo, type DraftInfo,
} from '@/lib/api';
import { useAuth } from '@/hooks/useAuth';

const BATCH_KEY = 'voicedub.create.batch';

function newKey() {
    return (typeof crypto !== 'undefined' && 'randomUUID' in crypto) ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`;
}

const STATE: Record<string, string> = {
    queued: 'Queued', running: 'In progress', done: 'Complete', error: 'Failed', 'not started': 'Not started',
};

/** W8: several uploads in one server-side batch, funded only by Premium time. */
export default function BatchPage() {
    const router = useRouter();
    const auth = useAuth();
    const [drafts, setDrafts] = useState<DraftInfo[]>([]);
    const [uploading, setUploading] = useState(0);
    const [minutes, setMinutes] = useState(0);
    const [batch, setBatch] = useState<BatchInfo | null>(null);
    const [rights, setRights] = useState<{ version: string; text: string } | null>(null);
    const [accepted, setAccepted] = useState(false);
    const [busy, setBusy] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const key = useRef('');

    useEffect(() => {
        if (auth.loaded && auth.config?.required && !auth.user) router.push('/signin');
    }, [auth.loaded, auth.config?.required, auth.user, router]);
    useEffect(() => { getRights().then(setRights).catch(() => {}); }, []);

    // a batch is a server record: reopen it after a refresh or a sign-in round trip
    useEffect(() => {
        let id: string | null = null;
        try { id = localStorage.getItem(BATCH_KEY); } catch { /* storage off */ }
        if (id) getBatch(id).then(setBatch).catch(() => { try { localStorage.removeItem(BATCH_KEY); } catch { /* ignore */ } });
    }, []);

    const refresh = useCallback(async () => {
        if (batch?.id) setBatch(await getBatch(batch.id));
    }, [batch?.id]);
    useEffect(() => {
        if (batch?.status !== 'submitted') return;
        const t = setInterval(() => { refresh().catch(() => {}); }, 3000);
        return () => clearInterval(t);
    }, [batch?.status, refresh]);

    const onFiles = async (files: FileList | null) => {
        if (!files?.length) return;
        setError(null); setBatch(null); setUploading(files.length);
        for (const f of Array.from(files)) {
            try { const d = await uploadDraft(f); setDrafts((x) => [...x, d]); }
            catch (e) { setError(`${f.name}: ${e instanceof Error ? e.message : 'upload failed'}`); }
            finally { setUploading((n) => n - 1); }
        }
    };

    const price = async () => {
        setBusy(true); setError(null);
        try {
            const b = await createBatch({ draft_ids: drafts.map((d) => d.id), minutes });
            key.current = newKey();
            setBatch(b); setAccepted(false);
            try { localStorage.setItem(BATCH_KEY, b.id); } catch { /* ignore */ }
        } catch (e) { setError(e instanceof Error ? e.message : 'Could not price the batch'); }
        finally { setBusy(false); }
    };

    const start = async () => {
        if (!batch || !rights) return;
        setBusy(true); setError(null);
        try {
            await submitBatch(batch.id, { idempotency_key: key.current, rights_accepted: accepted, rights_version: rights.version });
            await refresh();
        } catch (e) { setError(e instanceof Error ? e.message : 'Could not start the batch'); }
        finally { setBusy(false); }
    };

    const retry = async () => {
        if (!batch || !rights) return;
        setBusy(true); setError(null);
        try {
            await retryBatchFailures(batch.id, { idempotency_key: newKey(), rights_accepted: true, rights_version: rights.version });
            await refresh();
        } catch (e) { setError(e instanceof Error ? e.message : 'Could not retry'); }
        finally { setBusy(false); }
    };

    const failedReturned = batch?.children.filter((c) => c.state === 'error' && c.closed === 'released').length || 0;

    return (
        <main className="max-w-2xl mx-auto px-4 py-8 space-y-6">
            <div>
                <Link href="/create" className="text-xs text-text-muted hover:text-text-primary">← Single dub</Link>
                <h1 className="text-2xl font-semibold text-text-primary mt-1">Premium batch</h1>
                <p className="text-sm text-text-muted mt-1">Several uploads in one batch, paid from Premium time only. If an item fails, only its time is returned and you can retry just that item.</p>
            </div>

            {batch?.status !== 'submitted' && (
                <section className="glass-card p-4 space-y-3">
                    <h2 className="text-sm font-semibold text-text-primary">1 · Your videos</h2>
                    <label className="block cursor-pointer border-2 border-dashed rounded-xl p-5 text-center border-border hover:border-primary/50">
                        <input type="file" multiple accept=".mp4,.mov,.mkv,.webm,.m4v,.avi,.mp3,.wav,.m4a" className="sr-only"
                            aria-label="Choose video files" disabled={uploading > 0} onChange={(e) => onFiles(e.target.files)} />
                        <span className="text-sm text-text-muted">{uploading ? `Uploading and checking ${uploading} file(s)…` : 'Add videos'}</span>
                    </label>
                    <ul className="text-sm text-text-primary space-y-1" data-testid="batch-drafts">
                        {drafts.map((d) => (
                            <li key={d.id} className="flex justify-between">
                                <span className="truncate">{d.filename}</span>
                                <button type="button" className="text-xs text-text-muted hover:text-red-300"
                                    onClick={() => { setDrafts((x) => x.filter((y) => y.id !== d.id)); setBatch(null); }}>Remove</button>
                            </li>
                        ))}
                    </ul>
                    <label className="flex items-center gap-2 text-sm text-text-primary">
                        Length: <select className="input-field px-2 py-1 text-sm" value={minutes} onChange={(e) => { setMinutes(Number(e.target.value)); setBatch(null); }}>
                            <option value={0}>Full videos</option>
                            {[1, 5, 10, 30, 60].map((m) => <option key={m} value={m}>First {m} min of each</option>)}
                        </select>
                    </label>
                    <button type="button" className="btn-primary px-5 py-2 text-sm" disabled={!drafts.length || busy || uploading > 0} onClick={price}>Price the batch</button>
                </section>
            )}

            {batch && (
                <section className="glass-card p-4 space-y-3" data-testid="batch">
                    <h2 className="text-sm font-semibold text-text-primary">{batch.status === 'quoted' ? '2 · Price' : 'Progress'}</h2>
                    <table className="w-full text-sm">
                        <thead><tr className="text-xs text-text-muted text-left"><th>#</th><th>File</th><th>Premium time</th>{batch.status === 'submitted' && <th>Status</th>}</tr></thead>
                        <tbody>
                            {batch.children.map((c) => (
                                <tr key={c.n} className="text-text-primary">
                                    <td>{c.n}</td>
                                    <td className="truncate max-w-[14rem]">{c.job_id && batch.status === 'submitted' ? <Link className="underline" href={`/jobs/${c.job_id}`}>{c.filename}</Link> : c.filename}</td>
                                    <td>{Math.floor(c.premium_seconds / 60)}m {String(c.premium_seconds % 60).padStart(2, '0')}s</td>
                                    {batch.status === 'submitted' && <td>{STATE[c.state || ''] || c.state}{c.attempts.length ? ` (retry ${c.attempts.length})` : ''}</td>}
                                </tr>
                            ))}
                        </tbody>
                    </table>
                    <p className="text-sm text-text-primary">Total from Premium time: <strong>{batch.display.premium}</strong> · nothing else is charged</p>

                    {batch.status === 'quoted' && rights && (
                        <>
                            <label className="flex items-start gap-2 text-sm text-text-primary">
                                <input type="checkbox" checked={accepted} onChange={(e) => setAccepted(e.target.checked)} className="mt-1" />
                                <span>{rights.text} (applies to every file in this batch)</span>
                            </label>
                            <button type="button" className="btn-primary w-full py-3" disabled={!accepted || busy} onClick={start}>
                                {busy ? 'Starting…' : 'Start the batch'}
                            </button>
                        </>
                    )}
                    {batch.status === 'submitted' && failedReturned > 0 && (
                        <button type="button" className="btn-primary px-5 py-2 text-sm" disabled={busy} onClick={retry}>
                            Retry {failedReturned} failed item{failedReturned > 1 ? 's' : ''}
                        </button>
                    )}
                    {batch.status === 'submitted' && (
                        <button type="button" className="text-xs text-text-muted underline" onClick={() => {
                            try { localStorage.removeItem(BATCH_KEY); } catch { /* ignore */ }
                            setBatch(null); setDrafts([]);
                        }}>New batch</button>
                    )}
                </section>
            )}

            {error && <p role="status" aria-live="assertive" className="text-sm text-red-300">{error}</p>}
        </main>
    );
}
