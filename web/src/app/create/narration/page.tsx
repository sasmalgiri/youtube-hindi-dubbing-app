'use client';

import { useMemo, useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { approveNarration, draftNarration, dubNarration, type NarrationScene } from '@/lib/api';
import { useFeatures } from '@/hooks/useFeatures';

function t(s: number) {
    const m = Math.floor(s / 60);
    return `${m}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
}
const spoken = (hi: string) => hi.replace(/\s+/g, '').length;

type Row = NarrationScene & { selected: boolean };

/** P04: a Hindi narrator retells the scenes in order at their original times. */
export default function NarrationPage() {
    const router = useRouter();
    const { isOn } = useFeatures();
    const [srt, setSrt] = useState('');
    const [rows, setRows] = useState<Row[] | null>(null);
    const [busy, setBusy] = useState<string | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [approved, setApproved] = useState<{ srt: string; warnings: string[] } | null>(null);
    const [videoUrl, setVideoUrl] = useState('');
    const [videoFile, setVideoFile] = useState<File | null>(null);

    const chosen = useMemo(() => (rows || []).filter((r) => r.selected).length, [rows]);

    if (!isOn('input.narration')) {
        return (
            <main className="max-w-3xl mx-auto px-4 py-8 space-y-3">
                <Link href="/" className="text-xs text-text-muted hover:text-text-primary">← Back</Link>
                <h1 className="text-2xl font-semibold text-text-primary">Narration</h1>
                <p className="text-sm text-text-muted">Narration is switched off on this server.</p>
            </main>
        );
    }

    const run = async (label: string, f: () => Promise<void>) => {
        setBusy(label); setError(null);
        try { await f(); } catch (e) { setError(e instanceof Error ? e.message : String(e)); } finally { setBusy(null); }
    };
    const edit = (id: string, patch: Partial<Row>) => {
        setApproved(null);
        setRows((rs) => (rs || []).map((r) => (r.id === id ? { ...r, ...patch } : r)));
    };
    const download = (text: string) => {
        const a = document.createElement('a');
        a.href = URL.createObjectURL(new Blob([text], { type: 'text/plain;charset=utf-8' }));
        a.download = 'narration_hi.srt';
        a.click();
    };

    return (
        <main className="max-w-4xl mx-auto px-4 py-8 space-y-6">
            <div>
                <Link href="/" className="text-xs text-text-muted hover:text-text-primary">← Back</Link>
                <h1 className="text-2xl font-semibold text-text-primary mt-1">Narration</h1>
                <p className="text-sm text-text-muted mt-1">
                    A Hindi narrator retells the video scene by scene, each passage inside the scene&apos;s original time.
                    Choose scenes, edit the Hindi, approve, then voice exactly what you approved.
                </p>
            </div>

            <section className="glass-card p-4 space-y-3">
                <label className="block text-sm text-text-primary" htmlFor="srt">English subtitles (SRT) of the video</label>
                <textarea id="srt" value={srt} onChange={(e) => setSrt(e.target.value)} rows={6} spellCheck={false}
                    className="w-full rounded-lg bg-white/5 border border-white/10 px-3 py-2 text-xs font-mono text-text-primary"
                    placeholder={'1\n00:00:01,000 --> 00:00:04,000\nThe old king called his three sons.'} />
                <div className="flex items-center gap-3">
                    <input type="file" accept=".srt,text/plain" aria-label="SRT file"
                        onChange={async (e) => { const f = e.target.files?.[0]; if (f) setSrt(await f.text()); }}
                        className="text-xs text-text-muted" />
                    <button type="button" className="btn-primary px-4 py-1.5 text-sm ml-auto" disabled={!srt.trim() || !!busy}
                        onClick={() => run('draft', async () => {
                            const d = await draftNarration(srt);
                            setRows(d.scenes.map((s) => ({ ...s, selected: true })));
                            setApproved(null);
                        })}>
                        {busy === 'draft' ? 'Writing narration…' : rows ? 'Write again' : 'Write narration'}
                    </button>
                </div>
            </section>

            {error && <p role="alert" className="text-sm text-red-300">{error}</p>}

            {rows && (
                <section className="space-y-2" data-testid="scenes">
                    <p className="text-xs text-text-muted">{rows.length} scenes · {chosen} chosen</p>
                    {rows.map((r) => {
                        const over = spoken(r.hi) > r.budget_chars;
                        return (
                            <div key={r.id} className={`glass-card p-3 space-y-2 ${r.selected ? '' : 'opacity-50'}`}>
                                <div className="flex items-center gap-2 text-xs">
                                    <input type="checkbox" checked={r.selected} aria-label={`Use scene ${r.id}`}
                                        onChange={(e) => edit(r.id, { selected: e.target.checked })} />
                                    <span className="font-mono text-text-muted">{r.id} · {t(r.start)}–{t(r.end)}</span>
                                    <span className="text-text-muted" title="Subtitle lines this passage covers">cues {r.cue_ids.join(', ')}</span>
                                    <span className={`ml-auto font-mono ${over ? 'text-amber-300' : 'text-text-muted'}`}
                                        title="Spoken characters / what fits in the scene">
                                        {spoken(r.hi)}/{r.budget_chars}
                                    </span>
                                </div>
                                <p className="text-xs text-text-secondary">{r.english}</p>
                                <textarea lang="hi" value={r.hi} rows={2} aria-label={`Narration for ${r.id}`}
                                    onChange={(e) => edit(r.id, { hi: e.target.value })}
                                    className={`w-full rounded-lg border px-2 py-1 text-sm text-text-primary bg-white/5 ${over ? 'border-amber-500/50' : 'border-white/10'}`} />
                                {r.problems.length > 0 && <p className="text-[11px] text-amber-300">{r.problems.join('; ')}</p>}
                            </div>
                        );
                    })}
                    <div className="flex gap-2">
                        <button type="button" className="btn-primary px-4 py-1.5 text-sm" disabled={!chosen || !!busy}
                            onClick={() => run('approve', async () => {
                                const a = await approveNarration(rows);
                                setApproved({ srt: a.srt, warnings: a.warnings });
                            })}>
                            {busy === 'approve' ? 'Checking…' : `Approve ${chosen} scene${chosen === 1 ? '' : 's'}`}
                        </button>
                    </div>
                </section>
            )}

            {approved && (
                <section className="glass-card p-4 space-y-3" data-testid="approved">
                    <p className="text-sm text-text-primary font-medium">Approved narration</p>
                    {approved.warnings.length > 0 && (
                        <ul className="list-disc ml-5 text-xs text-amber-300">{approved.warnings.map((w) => <li key={w}>{w}</li>)}</ul>
                    )}
                    <pre className="text-xs whitespace-pre-wrap bg-white/5 rounded-lg p-2 max-h-48 overflow-auto">{approved.srt}</pre>
                    <div className="flex flex-wrap items-center gap-2">
                        <button type="button" className="px-3 py-1.5 text-xs rounded-lg border border-white/10 text-text-primary hover:bg-white/5"
                            onClick={() => download(approved.srt)}>Download SRT</button>
                        <input value={videoUrl} onChange={(e) => setVideoUrl(e.target.value)} placeholder="Video link"
                            aria-label="Video link" className="flex-1 min-w-[12rem] px-3 py-1.5 rounded-lg bg-white/5 border border-white/10 text-xs text-text-primary" />
                        <input type="file" accept="video/*,audio/*" aria-label="Video file" className="text-xs text-text-muted"
                            onChange={(e) => setVideoFile(e.target.files?.[0] || null)} />
                        <button type="button" className="btn-primary px-4 py-1.5 text-sm" disabled={(!videoUrl.trim() && !videoFile) || !!busy}
                            onClick={() => run('dub', async () => {
                                const j = await dubNarration(approved.srt, videoFile ? { file: videoFile } : { url: videoUrl.trim() });
                                router.push(`/jobs/${j.id}`);
                            })}>
                            {busy === 'dub' ? 'Starting…' : 'Voice this narration'}
                        </button>
                    </div>
                </section>
            )}
        </main>
    );
}
