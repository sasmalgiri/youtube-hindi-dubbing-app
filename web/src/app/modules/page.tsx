'use client';

import { useState } from 'react';
import Link from 'next/link';
import { resetFeatures, setFeature, type FeatureModule } from '@/lib/api';
import { refreshFeatures, useFeatures } from '@/hooks/useFeatures';

export default function ModulesPage() {
    const { matrix } = useFeatures();
    const [busy, setBusy] = useState<string | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [token, setToken] = useState('');
    const [filter, setFilter] = useState<'all' | 'on' | 'off'>('all');

    const hosted = matrix?.profile === 'hosted';

    const toggle = async (m: FeatureModule) => {
        setBusy(m.id);
        setError(null);
        try {
            await setFeature(m.id, !m.on, hosted ? token : undefined);
            await refreshFeatures();
        } catch (e) {
            setError(e instanceof Error ? e.message : 'Could not change the switch');
        } finally {
            setBusy(null);
        }
    };

    const reset = async () => {
        if (!confirm('Put every module back to its default for this profile?')) return;
        setBusy('reset');
        setError(null);
        try {
            await resetFeatures(hosted ? token : undefined);
            await refreshFeatures();
        } catch (e) {
            setError(e instanceof Error ? e.message : 'Could not reset');
        } finally {
            setBusy(null);
        }
    };

    if (!matrix) {
        return (
            <main className="max-w-4xl mx-auto px-4 py-10">
                <p className="text-text-muted text-sm">Loading modules… (is the backend running?)</p>
            </main>
        );
    }

    const onCount = matrix.modules.filter((m) => m.on).length;

    return (
        <main className="max-w-4xl mx-auto px-4 py-8 space-y-6">
            <div className="flex flex-wrap items-center justify-between gap-3">
                <div>
                    <Link href="/" className="text-xs text-text-muted hover:text-text-primary">← Back</Link>
                    <h1 className="text-2xl font-semibold text-text-primary mt-1">Modules</h1>
                    <p className="text-sm text-text-muted mt-1">
                        Switch any mode, engine or option on or off. A module that is off stays in the app, is hidden from
                        the forms, and is refused by the server. Nothing is deleted.
                    </p>
                </div>
                <div className="flex items-center gap-2">
                    <span className={`px-3 py-1 rounded-full text-xs font-medium border ${hosted
                        ? 'bg-amber-500/10 border-amber-500/30 text-amber-300'
                        : 'bg-primary/10 border-primary/30 text-primary-light'}`}>
                        Profile: {matrix.profile}
                    </span>
                    <span className="text-xs text-text-muted">{onCount}/{matrix.modules.length} on</span>
                </div>
            </div>

            {hosted && (
                <div className="glass-card p-4 space-y-2">
                    <label className="block text-xs font-medium text-text-muted" htmlFor="admin-token">Admin token</label>
                    <input id="admin-token" type="password" value={token} onChange={(e) => setToken(e.target.value)}
                        className="input-field w-full px-3 py-2 text-sm" autoComplete="off" />
                    <p className="text-[11px] text-text-muted">Hosted switches need VOICEDUB_ADMIN_TOKEN. Locked modules cannot be switched on in hosted.</p>
                </div>
            )}

            <div className="flex flex-wrap items-center gap-2">
                {(['all', 'on', 'off'] as const).map((f) => (
                    <button key={f} type="button" onClick={() => setFilter(f)}
                        aria-pressed={filter === f}
                        className={`px-3 py-1.5 rounded-lg text-xs font-medium border ${filter === f
                            ? 'bg-primary/20 border-primary text-primary-light' : 'bg-white/5 border-white/10 text-text-muted hover:bg-white/10'}`}>
                        {f === 'all' ? 'All' : f === 'on' ? 'On' : 'Off'}
                    </button>
                ))}
                <button type="button" onClick={reset} disabled={busy !== null}
                    className="ml-auto px-3 py-1.5 rounded-lg text-xs font-medium border bg-white/5 border-white/10 text-text-muted hover:bg-white/10">
                    Reset to defaults
                </button>
            </div>

            {error && (
                <div role="alert" className="p-3 rounded-lg bg-red-500/10 border border-red-500/30 text-sm text-red-300">{error}</div>
            )}

            {matrix.groups.map((g) => {
                const mods = matrix.modules.filter((m) => m.group === g.id && (filter === 'all' || (filter === 'on') === m.on));
                if (mods.length === 0) return null;
                return (
                    <section key={g.id} className="glass-card p-4">
                        <h2 className="text-sm font-semibold text-text-primary mb-3">{g.label}</h2>
                        <ul className="divide-y divide-white/5">
                            {mods.map((m) => (
                                <li key={m.id} className="py-2.5 flex items-start gap-3">
                                    <div className="flex-1 min-w-0">
                                        <div className="flex flex-wrap items-center gap-2">
                                            <span className="text-sm text-text-primary">{m.label}</span>
                                            <code className="text-[10px] text-text-muted">{m.id}</code>
                                            {m.overridden && <span className="text-[10px] px-1.5 rounded bg-white/10 text-text-muted">changed</span>}
                                            {m.locked && <span className="text-[10px] px-1.5 rounded bg-amber-500/15 text-amber-300">locked</span>}
                                        </div>
                                        <p className="text-xs text-text-muted mt-0.5">{m.desc}</p>
                                        {m.why && <p className="text-[11px] text-text-muted/80 mt-0.5">Hosted: {m.why}</p>}
                                    </div>
                                    <button
                                        type="button"
                                        role="switch"
                                        aria-checked={m.on}
                                        aria-label={`${m.label}: ${m.on ? 'on' : 'off'}`}
                                        disabled={busy !== null || (m.locked && !m.on)}
                                        onClick={() => toggle(m)}
                                        className={`relative inline-flex h-6 w-11 shrink-0 items-center rounded-full transition-colors
                                            ${m.on ? 'bg-primary' : 'bg-white/10'} ${(m.locked && !m.on) ? 'opacity-40 cursor-not-allowed' : ''}`}
                                    >
                                        <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${m.on ? 'translate-x-6' : 'translate-x-1'}`} />
                                    </button>
                                </li>
                            ))}
                        </ul>
                    </section>
                );
            })}
        </main>
    );
}
