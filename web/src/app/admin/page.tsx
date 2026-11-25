'use client';

import { useCallback, useEffect, useState } from 'react';
import Link from 'next/link';
import { getAdminOverview, type AdminOverview } from '@/lib/api';

const KEY = 'voicedub_admin_token';

function read(): string {
    try { return sessionStorage.getItem(KEY) || ''; } catch { return ''; }
}

function Card({ title, children }: { title: string; children: React.ReactNode }) {
    return (
        <section className="rounded-xl border border-border bg-card/40 p-4 space-y-2">
            <h2 className="text-xs uppercase tracking-wider text-text-muted">{title}</h2>
            {children}
        </section>
    );
}

function money(minor: number, cur: string) {
    try { return new Intl.NumberFormat(undefined, { style: 'currency', currency: cur }).format(minor / 100); }
    catch { return `${(minor / 100).toFixed(2)} ${cur}`; }
}

/** Owner-only overview. Hosted: needs VOICEDUB_ADMIN_TOKEN; kept for this browser tab only. */
export default function AdminPage() {
    const [token, setToken] = useState('');
    const [draft, setDraft] = useState('');
    const [data, setData] = useState<AdminOverview | null>(null);
    const [err, setErr] = useState('');

    const load = useCallback((t: string) => {
        getAdminOverview(t).then((d) => { setData(d); setErr(''); }).catch((e) => { setData(null); setErr(String(e.message || e)); });
    }, []);

    useEffect(() => { const t = read(); setToken(t); load(t); }, [load]);
    useEffect(() => {
        const i = setInterval(() => load(token), 30000);
        return () => clearInterval(i);
    }, [token, load]);

    const save = () => {
        try { sessionStorage.setItem(KEY, draft); } catch { /* private mode: keep in memory only */ }
        setToken(draft); load(draft);
    };

    const s = data?.snapshot;
    const problems = data ? Object.entries(data.reconcile).flatMap(([k, v]) => v.map((x) => `${k}: ${x}`)) : [];

    return (
        <main className="max-w-5xl mx-auto px-4 py-8 space-y-6 text-sm text-text-secondary">
            <div className="flex items-center justify-between">
                <div>
                    <Link href="/" className="text-xs text-text-muted hover:text-text-primary">← Back</Link>
                    <h1 className="text-2xl font-semibold text-text-primary">Service overview</h1>
                </div>
                {data && <button onClick={() => load(token)} className="text-xs px-3 py-1.5 rounded-lg border border-white/10 hover:bg-white/5">Refresh</button>}
            </div>

            {!data && (
                <form onSubmit={(e) => { e.preventDefault(); save(); }} className="flex gap-2 items-end">
                    <label className="flex-1 text-xs">Admin token
                        <input type="password" value={draft} onChange={(e) => setDraft(e.target.value)} autoComplete="off"
                               className="mt-1 w-full bg-card border border-border rounded px-2 py-1.5 text-sm text-text-primary" />
                    </label>
                    <button className="btn-primary text-sm px-4 py-1.5">Open</button>
                </form>
            )}
            {err && <p role="alert" className="text-error">{err}</p>}

            {data && s && (
                <>
                    {data.alerts.length > 0 ? (
                        <ul className="space-y-1" aria-label="Active alerts">
                            {data.alerts.map((a) => (
                                <li key={a.key + a.text} className={`rounded-lg p-2 border ${a.level === 'critical' ? 'border-error/50 bg-error/10 text-error' : 'border-amber-500/40 bg-amber-500/10 text-amber-200'}`}>
                                    {a.text}
                                </li>
                            ))}
                        </ul>
                    ) : <p className="rounded-lg p-2 border border-success/40 bg-success/10 text-success">No active alerts.</p>}

                    <div className="grid sm:grid-cols-3 gap-3">
                        <Card title="Queue">
                            {Object.keys(s.queue.by_state).length === 0 ? <p>Empty</p> :
                                Object.entries(s.queue.by_state).map(([k, v]) => <p key={k}>{k}: <b className="text-text-primary">{v}</b></p>)}
                            {s.queue.oldest_queued_s > 0 && <p className="text-xs">Oldest waiting {Math.round(s.queue.oldest_queued_s / 60)} min</p>}
                        </Card>
                        <Card title="Dubs, last 24 h">
                            <p>{s.jobs_24h.total} started · <span className="text-success">{s.jobs_24h.done} done</span> · <span className={s.jobs_24h.failed ? 'text-error' : ''}>{s.jobs_24h.failed} failed</span></p>
                        </Card>
                        <Card title="Disk and backup">
                            <p>{s.disk.free_gb} GB free of {s.disk.total_gb} GB</p>
                            <p>Last backup: {s.backup.age_h === null ? 'none' : `${s.backup.age_h} h ago`}</p>
                        </Card>
                    </div>

                    <div className="grid sm:grid-cols-2 gap-3">
                        <Card title="Orders, last 30 days">
                            {Object.keys(s.orders_30d).length === 0 ? <p>None</p> :
                                Object.entries(s.orders_30d).map(([k, v]) => <p key={k}>{k}: <b className="text-text-primary">{v.count}</b></p>)}
                            {s.paid_by_currency_30d.map((c) => <p key={c.currency} className="text-xs">Paid in {c.currency}: {money(c.amount_minor, c.currency)} ({c.paid})</p>)}
                        </Card>
                        <Card title="Billing reconciliation">
                            {problems.length === 0 ? <p className="text-success">0 problems</p> :
                                <ul className="list-disc pl-4 text-error">{problems.slice(0, 20).map((p) => <li key={p}>{p}</li>)}</ul>}
                        </Card>
                    </div>

                    <Card title="Recent failures">
                        {s.jobs_24h.recent_failures.length === 0 ? <p>None</p> : (
                            <ul className="space-y-1">{s.jobs_24h.recent_failures.map((f) => (
                                <li key={f.id}><Link href={`/jobs/${f.id}`} className="underline">{f.id}</Link> — {f.error || 'no message'}</li>
                            ))}</ul>
                        )}
                    </Card>

                    <Card title="Alert history">
                        {data.recent_alerts.length === 0 ? <p>None yet</p> : (
                            <ul className="space-y-1 text-xs">{data.recent_alerts.map((a, i) => (
                                <li key={i}>{a.at ? new Date(a.at * 1000).toLocaleString() : ''} · {a.level} · {a.text}{a.emailed ? ' · emailed' : ''}</li>
                            ))}</ul>
                        )}
                    </Card>
                </>
            )}
        </main>
    );
}
