'use client';

import { useEffect, useRef, useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { emailStart, emailVerify, googleSignIn } from '@/lib/api';
import { refreshAuth, useAuth } from '@/hooks/useAuth';

declare global {
    interface Window {
        google?: {
            accounts: {
                id: {
                    initialize: (o: { client_id: string; callback: (r: { credential: string }) => void }) => void;
                    renderButton: (el: HTMLElement, o: Record<string, unknown>) => void;
                };
            };
        };
    }
}

export default function SignInPage() {
    const router = useRouter();
    const { user, config, loaded } = useAuth();
    const [email, setEmail] = useState('');
    const [code, setCode] = useState('');
    const [stage, setStage] = useState<'email' | 'code'>('email');
    const [busy, setBusy] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [info, setInfo] = useState<string | null>(null);
    const googleRef = useRef<HTMLDivElement>(null);

    const done = async () => {
        await refreshAuth();
        router.push('/');
    };

    // Google Identity Services button, only when the server has a client id
    useEffect(() => {
        const cid = config?.google_client_id;
        if (!cid || !googleRef.current) return;
        const render = () => {
            if (!window.google || !googleRef.current) return;
            window.google.accounts.id.initialize({
                client_id: cid,
                callback: async (r) => {
                    setBusy(true); setError(null);
                    try { await googleSignIn(r.credential); await done(); }
                    catch (e) { setError(e instanceof Error ? e.message : 'Google sign-in failed'); }
                    finally { setBusy(false); }
                },
            });
            window.google.accounts.id.renderButton(googleRef.current, { theme: 'filled_black', size: 'large', width: 320, text: 'continue_with' });
        };
        if (window.google) { render(); return; }
        const s = document.createElement('script');
        s.src = 'https://accounts.google.com/gsi/client';
        s.async = true;
        s.onload = render;
        document.head.appendChild(s);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [config?.google_client_id]);

    const sendCode = async (e: React.FormEvent) => {
        e.preventDefault();
        setBusy(true); setError(null); setInfo(null);
        try {
            await emailStart(email);
            setStage('code');
            setInfo(config?.email_is_local_log
                ? 'Desktop mode: the code is printed in the backend window/log (no email is sent).'
                : `We sent a 6-digit code to ${email}. It expires in 10 minutes.`);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Could not send the code');
        } finally { setBusy(false); }
    };

    const verify = async (e: React.FormEvent) => {
        e.preventDefault();
        setBusy(true); setError(null);
        try { await emailVerify(email, code); await done(); }
        catch (err) { setError(err instanceof Error ? err.message : 'Sign-in failed'); }
        finally { setBusy(false); }
    };

    if (loaded && user) {
        return (
            <main className="max-w-md mx-auto px-4 py-16 text-center space-y-4">
                <p className="text-text-primary">Signed in as <strong>{user.email}</strong>.</p>
                <Link href="/" className="btn-primary inline-block px-6 py-2">Continue</Link>
            </main>
        );
    }

    return (
        <main className="max-w-md mx-auto px-4 py-16 space-y-6">
            <div>
                <h1 className="text-2xl font-semibold text-text-primary">Sign in</h1>
                <p className="text-sm text-text-muted mt-1">
                    {config?.required ? 'Sign in to dub videos, see your jobs and your balance.' : 'Optional on this desktop app: your jobs stay on this PC either way.'}
                </p>
            </div>

            {config?.google_client_id && (
                <div className="glass-card p-4 flex justify-center">
                    <div ref={googleRef} aria-label="Continue with Google" />
                </div>
            )}

            {config?.email ? (
                <div className="glass-card p-4 space-y-3">
                    {stage === 'email' ? (
                        <form onSubmit={sendCode} className="space-y-3">
                            <label htmlFor="email" className="block text-xs font-medium text-text-muted">Email</label>
                            <input id="email" type="email" required autoComplete="email" value={email}
                                onChange={(e) => setEmail(e.target.value)} className="input-field w-full px-3 py-2 text-sm" />
                            <button type="submit" disabled={busy || !email} className="btn-primary w-full py-2.5 text-sm">
                                {busy ? 'Sending…' : 'Email me a code'}
                            </button>
                        </form>
                    ) : (
                        <form onSubmit={verify} className="space-y-3">
                            <label htmlFor="code" className="block text-xs font-medium text-text-muted">6-digit code for {email}</label>
                            <input id="code" inputMode="numeric" pattern="[0-9]{6}" maxLength={6} required autoComplete="one-time-code"
                                value={code} onChange={(e) => setCode(e.target.value.replace(/\D/g, ''))}
                                className="input-field w-full px-3 py-2 text-lg tracking-[0.4em] text-center" />
                            <button type="submit" disabled={busy || code.length !== 6} className="btn-primary w-full py-2.5 text-sm">
                                {busy ? 'Checking…' : 'Sign in'}
                            </button>
                            <button type="button" onClick={() => { setStage('email'); setCode(''); setInfo(null); }}
                                className="w-full text-xs text-text-muted hover:text-text-primary">Use a different email / send a new code</button>
                        </form>
                    )}
                </div>
            ) : loaded && !config?.google_client_id ? (
                <p className="text-sm text-text-muted">Sign-in is not configured on this server.</p>
            ) : null}

            {info && <p role="status" className="text-sm text-text-muted">{info}</p>}
            {error && <p role="alert" className="text-sm text-red-300">{error}</p>}
        </main>
    );
}
