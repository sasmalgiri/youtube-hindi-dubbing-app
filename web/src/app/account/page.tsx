'use client';

import { useCallback, useEffect, useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { deleteAccount, getDeletionPlan, type DeletionPlan, getBalance, getCatalog, getOrders, getSessions, receiptUrl, revokeSession, signOut, signOutEverywhere, startCheckout, verifyCheckout, type CatalogInfo, type CreditBalance, type OrderInfo, type SessionInfo } from '@/lib/api';
import { refreshAuth, useAuth } from '@/hooks/useAuth';

type RzpResponse = { razorpay_order_id: string; razorpay_payment_id: string; razorpay_signature: string };
type RzpInstance = { open: () => void; on: (e: string, f: (r: { error?: { description?: string } }) => void) => void };
declare global { interface Window { Razorpay?: new (o: Record<string, unknown>) => RzpInstance } }

function loadRazorpay(src = 'https://checkout.razorpay.com/v1/checkout.js'): Promise<void> {
    if (typeof window !== 'undefined' && window.Razorpay) return Promise.resolve();
    return new Promise((ok, bad) => {
        const s = document.createElement('script');
        s.src = src;
        s.onload = () => ok();
        s.onerror = () => bad(new Error('Could not load the payment window'));
        document.body.appendChild(s);
    });
}

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
    const [catalog, setCatalog] = useState<CatalogInfo | null>(null);
    const [orders, setOrders] = useState<OrderInfo[]>([]);
    const [buying, setBuying] = useState<string | null>(null);
    const [notice, setNotice] = useState<string | null>(null);
    const [plan, setPlan] = useState<DeletionPlan | null>(null);
    const [confirmEmail, setConfirmEmail] = useState('');
    const [delError, setDelError] = useState<string | null>(null);
    const [deleting, setDeleting] = useState(false);

    const load = useCallback(async () => {
        try {
            const [b, s, c, o] = await Promise.all([getBalance(), getSessions(), getCatalog(), getOrders()]);
            setBalance(b); setSessions(s); setCatalog(c); setOrders(o);
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

    const buy = async (packId: string) => {
        setError(null); setNotice(null); setBuying(packId);
        try {
            const o = await startCheckout(packId);
            await loadRazorpay(catalog?.checkout.script_url);
            const rz = new window.Razorpay!({
                key: o.key_id, order_id: o.razorpay_order_id, amount: o.amount_minor, currency: o.currency,
                name: 'VoiceDub', description: `${o.display.time} prepaid time`, prefill: o.prefill,
                handler: async (r: RzpResponse) => {
                    try {
                        const done = await verifyCheckout(r);
                        setNotice(`Payment received. ${done.display.time} added. Receipt ${done.receipt_no}.`);
                        await load();
                    } catch (e) {
                        setError(`Payment taken but not yet confirmed (${e instanceof Error ? e.message : 'error'}). It is added automatically when the payment provider confirms it.`);
                    } finally { setBuying(null); }
                },
                modal: { ondismiss: () => setBuying(null) },
            });
            rz.on('payment.failed', (r) => { setError(`Payment failed: ${r.error?.description || 'declined'}. Nothing was charged.`); setBuying(null); });
            rz.open();
        } catch (e) {
            setError(e instanceof Error ? e.message : 'Could not start checkout');
            setBuying(null);
        }
    };

    const showPlan = async () => {
        setDelError(null);
        try { setPlan(await getDeletionPlan()); } catch (e) { setDelError(e instanceof Error ? e.message : 'Could not load'); }
    };
    const doDelete = async () => {
        setDeleting(true); setDelError(null);
        try {
            await deleteAccount(confirmEmail);
            await refreshAuth();
            router.push('/signin');
        } catch (e) {
            setDelError(e instanceof Error ? e.message : 'Could not delete the account');
        } finally { setDeleting(false); }
    };

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

            <section className="glass-card p-4" id="buy" data-testid="buy">
                <h2 className="text-sm font-semibold text-text-primary mb-3">Buy prepaid time</h2>
                {catalog && !catalog.checkout.configured && <p className="text-xs text-amber-300 mb-2">Checkout is not switched on yet.</p>}
                {catalog?.checkout.test_mode && <p className="text-xs text-amber-300 mb-2">{catalog.checkout.simulated
                    ? 'Simulated payments: a local stand-in for Razorpay; no real money moves.'
                    : 'Test mode: use Razorpay test cards; no real money moves.'}</p>}
                <div className="grid sm:grid-cols-3 gap-3">
                    {(catalog?.packs || []).map((p) => (
                        <div key={p.id} className="rounded-lg border border-white/10 p-3 space-y-1">
                            <p className="text-xs text-text-muted capitalize">{p.credit_type}</p>
                            <p className="text-lg font-semibold text-text-primary">{p.display.time}</p>
                            <p className="text-sm text-text-primary">{p.display.price} <span className="text-[11px] text-text-muted">incl. GST {p.display.tax}</span></p>
                            <button type="button" className="btn-primary w-full py-1.5 text-sm" disabled={!catalog?.checkout.configured || !!buying}
                                onClick={() => buy(p.id)}>{buying === p.id ? 'Opening…' : 'Buy'}</button>
                        </div>
                    ))}
                </div>
                {notice && <p role="status" className="text-sm text-emerald-300 mt-3">{notice}</p>}
            </section>

            {orders.length > 0 && (
                <section className="glass-card p-4" data-testid="receipts">
                    <h2 className="text-sm font-semibold text-text-primary mb-3">Payments and receipts</h2>
                    <ul className="divide-y divide-white/5 text-sm">
                        {orders.map((o) => (
                            <li key={o.id} className="py-2 flex items-center gap-3">
                                <span className="flex-1 text-text-primary">{o.display.time} {o.credit_type} · {o.display.amount}
                                    <span className="text-[11px] text-text-muted"> · {o.paid_at ? new Date(o.paid_at * 1000).toLocaleDateString() : ''} · {o.status}</span></span>
                                {o.receipt_no && <a className="text-xs underline text-text-muted" href={receiptUrl(o.id)} target="_blank" rel="noreferrer">Receipt {o.receipt_no}</a>}
                            </li>
                        ))}
                    </ul>
                </section>
            )}

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

            <section className="glass-card p-4 border border-red-500/20" data-testid="delete-account">
                <h2 className="text-sm font-semibold text-red-300 mb-2">Delete account</h2>
                {!plan ? (
                    <button type="button" onClick={showPlan} className="text-xs px-3 py-1.5 rounded-lg border border-red-500/30 text-red-300 hover:bg-red-500/10">See what will be deleted</button>
                ) : (
                    <div className="space-y-3 text-sm">
                        <p className="text-text-secondary">This removes your {plan.jobs} dub(s) and their files, {plan.drafts} upload(s), {plan.batches} batch(es), your email and your sign-ins. It cannot be undone.</p>
                        {Object.entries(plan.unused_seconds).some(([, s]) => s > 0) && (
                            <p className="text-amber-300">Unused prepaid time is lost: {Object.entries(plan.unused_seconds).filter(([, s]) => s > 0).map(([t, s]) => `${fmtTime(s)} ${t}`).join(', ')}.</p>
                        )}
                        {plan.active_jobs.length > 0 && (
                            <p className="text-red-300">{plan.active_jobs.length} dub(s) are still running. Cancel them or wait until they finish.</p>
                        )}
                        <div>
                            <p className="text-xs text-text-muted mb-1">Kept:</p>
                            <ul className="list-disc ml-5 text-xs text-text-muted">{plan.kept.map((k) => <li key={k}>{k}</li>)}</ul>
                        </div>
                        <label className="block text-xs text-text-muted">Type your email to confirm
                            <input value={confirmEmail} onChange={(e) => setConfirmEmail(e.target.value)} aria-label="Confirm email"
                                className="mt-1 w-full px-3 py-1.5 rounded-lg bg-white/5 border border-white/10 text-text-primary" />
                        </label>
                        <button type="button" onClick={doDelete}
                            disabled={deleting || plan.active_jobs.length > 0 || confirmEmail.trim().toLowerCase() !== (user?.email || '').toLowerCase()}
                            className="text-xs px-3 py-1.5 rounded-lg bg-red-500/20 border border-red-500/40 text-red-200 disabled:opacity-40">
                            {deleting ? 'Deleting…' : 'Delete my account'}
                        </button>
                    </div>
                )}
                {delError && <p role="alert" className="text-xs text-red-300 mt-2">{delError}</p>}
            </section>
        </main>
    );
}
