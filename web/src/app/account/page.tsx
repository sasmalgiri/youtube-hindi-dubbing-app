'use client';

import { useCallback, useEffect, useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { getBalance, getSessions, revokeSession, signOut, signOutEverywhere, type CreditBalance, type SessionInfo } from '@/lib/api';
import { refreshAuth, useAuth } from '@/hooks/useAuth';

function fmtTime(seconds: number) {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = seconds % 60;
    return h ? `${h}h ${String(m).padStart(2, '0')}m` : `${m}m ${String(s).padStart(2, '0')}s`;
}

export default function AccountPage() {
    const router = useRouter();
    const { user, loaded } = useAuth();
    const [balance, setBalance] = useState<Record<'standard' | 'premium', CreditBalance> | null>(null);
    const [sessions, setSessions] = useState<{ current: string; sessions: SessionInfo[] } | null>(null);
    const [error, setError] = useState<string | null>(null);

    const load = useCallback(async () => {
        try {
            const [b, s] = await Promise.all([getBalance(), getSessions()]);
            setBalance(b); setSessions(s);
        } catch (e) {
            setError(e instanceof Error ? e.message : 'Could not load the account');
        }
    }, []);

    useEffect(() => { if (user) load(); }, [user, load]);

    if (loaded && !user) {
        return (
            <main className="max-w-md mx-auto px-4 py-16 text-center space-y-4">
                <p className="text-text-muted">You are not signed in.</p>
                <Link href="/signin" className="btn-primary inline-block px-6 py-2">Sign in</Link>
            </main>
        );
    }

    const out = async (everywhere: boolean) => {
        if (everywhere && !confirm('Sign out on every device, including this one?')) return;
        await (everywhere ? signOutEverywhere() : signOut());
        await refreshAuth();
        router.push('/signin');
    };

    return (
        <main className="max-w-2xl mx-auto px-4 py-8 space-y-6">
            <div>
                <Link href="/" className="text-xs text-text-muted hover:text-text-primary">← Back</Link>
                <h1 className="text-2xl font-semibold text-text-primary mt-1">Account</h1>
                <p className="text-sm text-text-muted mt-1">{user?.email}</p>
            </div>

            {error && <p role="alert" className="text-sm text-red-300">{error}</p>}

            <section className="glass-card p-4">
                <h2 className="text-sm font-semibold text-text-primary mb-3">Prepaid time</h2>
                <p className="text-xs text-text-muted mb-3">Time never expires. Standard time is used first for single jobs; batches use Premium time only.</p>
                <div className="grid grid-cols-2 gap-3">
                    {(['standard', 'premium'] as const).map((t) => (
                        <div key={t} className="rounded-lg border border-white/10 p-3">
                            <p className="text-xs text-text-muted capitalize">{t}</p>
                            <p className="text-xl font-semibold text-text-primary">{balance ? fmtTime(balance[t].spendable_seconds) : '…'}</p>
                            {balance && balance[t].reserved_seconds > 0 && (
                                <p className="text-[11px] text-text-muted">{fmtTime(balance[t].reserved_seconds)} reserved by running jobs</p>
                            )}
                            {balance && balance[t].debt_seconds > 0 && (
                                <p className="text-[11px] text-amber-300">{fmtTime(balance[t].debt_seconds)} owed after a disputed payment</p>
                            )}
                        </div>
                    ))}
                </div>
            </section>

            <section className="glass-card p-4">
                <h2 className="text-sm font-semibold text-text-primary mb-3">Signed-in devices</h2>
                <ul className="divide-y divide-white/5">
                    {(sessions?.sessions || []).map((s) => (
                        <li key={s.id} className="py-2 flex items-center gap-3">
                            <div className="flex-1 min-w-0">
                                <p className="text-sm text-text-primary truncate">{s.user_agent || 'Unknown device'}</p>
                                <p className="text-[11px] text-text-muted">
                                    {s.method} · last active {new Date(s.last_seen * 1000).toLocaleString()}
                                    {s.id === sessions?.current ? ' · this device' : ''}
                                </p>
                            </div>
                            {s.id !== sessions?.current && (
                                <button type="button" onClick={async () => { await revokeSession(s.id); load(); }}
                                    className="text-xs px-2 py-1 rounded border border-white/10 text-text-muted hover:bg-white/5">Sign out</button>
                            )}
                        </li>
                    ))}
                </ul>
                <div className="flex gap-2 mt-3">
                    <button type="button" onClick={() => out(false)} className="text-xs px-3 py-1.5 rounded-lg border border-white/10 text-text-muted hover:bg-white/5">Sign out</button>
                    <button type="button" onClick={() => out(true)} className="text-xs px-3 py-1.5 rounded-lg border border-red-500/30 text-red-300 hover:bg-red-500/10">Sign out everywhere</button>
                </div>
            </section>
        </main>
    );
}
