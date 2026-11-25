'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { BRAND } from '@/lib/brand';
import { useAuth } from '@/hooks/useAuth';
import { getCatalog, getLanguages, COUNTRIES, type CatalogInfo, type LanguageInfo } from '@/lib/api';

// Optional sample clips (set once a real before/after dub exists).
const SAMPLE_BEFORE = process.env.NEXT_PUBLIC_SAMPLE_BEFORE_URL || '';
const SAMPLE_AFTER = process.env.NEXT_PUBLIC_SAMPLE_AFTER_URL || '';

const FAQ: [string, string][] = [
    ['What do I pay for?', 'Prepaid time. One hour covers one hour of source video dubbed into one language. Time does not expire.'],
    ['What if a dub fails?', 'It costs nothing. Time is held when a job starts and only taken once the finished dub passes our delivery check; otherwise it goes back to your balance automatically.'],
    ['Which languages?', 'The ones marked "Available" below are fully reviewed. "Preview" languages work but have not yet been signed off by a native reviewer.'],
    ['Can I fix a line?', 'Yes. You can review the script, change a line and re-voice the dub a few times at no extra charge.'],
    ['What can I upload?', 'Only videos you own or have the rights to dub. You confirm this before every job.'],
    ['Do you train on my videos?', 'No. Your content is used only to make your dubs. You can delete your account and files at any time.'],
];

export default function PricingPage() {
    const { user } = useAuth();
    const [country, setCountry] = useState('');
    const [cat, setCat] = useState<CatalogInfo | null>(null);
    const [langs, setLangs] = useState<LanguageInfo[]>([]);
    const [err, setErr] = useState('');

    useEffect(() => { getLanguages().then((r) => setLangs(r.languages)).catch(() => {}); }, []);
    useEffect(() => {
        getCatalog(country || undefined).then((c) => { setCat(c); if (!country) setCountry(c.country); setErr(''); })
            .catch(() => setErr('Prices could not be loaded. Try again shortly.'));
    }, [country]);

    const shown = langs.filter((l) => l.available);
    const start = user ? '/create' : '/signin';

    return (
        <main className="max-w-4xl mx-auto px-4 py-10 space-y-12 text-text-secondary">
            <section className="text-center space-y-4">
                <h1 className="text-3xl sm:text-4xl font-semibold text-text-primary">Dub your videos into other languages</h1>
                <p className="max-w-2xl mx-auto">{BRAND} transcribes your video, translates it to fit the timing, voices every speaker and gives you
                    the dubbed video and subtitles. You review it before you publish.</p>
                <div className="flex justify-center gap-3">
                    <Link href={start} className="btn-primary px-5 py-2 text-sm">{user ? 'Start a dub' : 'Sign in to start'}</Link>
                    <a href="#prices" className="px-5 py-2 text-sm rounded-lg border border-white/10 hover:bg-white/5">See prices</a>
                </div>
            </section>

            {SAMPLE_BEFORE && SAMPLE_AFTER && (
                <section className="grid sm:grid-cols-2 gap-4" aria-label="Sample">
                    <figure><video src={SAMPLE_BEFORE} controls className="w-full rounded-lg" /><figcaption className="text-xs mt-1">Original</figcaption></figure>
                    <figure><video src={SAMPLE_AFTER} controls className="w-full rounded-lg" /><figcaption className="text-xs mt-1">Dubbed</figcaption></figure>
                </section>
            )}

            <section id="prices" className="space-y-4">
                <div className="flex flex-wrap items-end justify-between gap-3">
                    <h2 className="text-xl font-semibold text-text-primary">Prices</h2>
                    <label className="text-xs flex items-center gap-2">Billing country
                        <select value={country} onChange={(e) => setCountry(e.target.value)}
                                className="bg-card border border-border rounded px-2 py-1 text-sm text-text-primary">
                            {COUNTRIES.map(([cc, name]) => <option key={cc} value={cc}>{name}</option>)}
                        </select>
                    </label>
                </div>
                {err && <p role="alert" className="text-error text-sm">{err}</p>}
                {cat && !cat.on_sale && (
                    <p className="rounded-lg border border-amber-500/40 bg-amber-500/10 p-3 text-sm text-amber-200">
                        Not on sale in this country yet. We sell in: {cat.currencies.join(', ') || 'no currency yet'}.
                    </p>
                )}
                {cat?.on_sale && (
                    <>
                        <div className="grid sm:grid-cols-3 gap-3">
                            {cat.packs.filter((p) => p.display).map((p) => (
                                <div key={p.id} className="rounded-xl border border-border bg-card/40 p-4 space-y-1">
                                    <p className="text-xs uppercase tracking-wider text-text-muted">{p.credit_type}</p>
                                    <p className="text-2xl font-semibold text-text-primary">{p.display!.price}</p>
                                    <p className="text-sm">{p.display!.time} of video</p>
                                    <p className="text-[11px] text-text-muted">{p.display!.tax_label}: {p.display!.tax}</p>
                                </div>
                            ))}
                        </div>
                        <p className="text-xs text-text-muted">{cat.tax.note}</p>
                    </>
                )}
            </section>

            <section className="space-y-3">
                <h2 className="text-xl font-semibold text-text-primary">Languages</h2>
                <p className="text-sm">From English into:</p>
                <ul className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-sm">
                    {shown.map((l) => (
                        <li key={l.code} className="rounded-lg border border-border px-3 py-2 flex items-center justify-between gap-2">
                            <span><span className="text-text-primary">{l.name}</span> <span className="text-text-muted" dir={l.rtl ? 'rtl' : undefined}>{l.native}</span></span>
                            <span className={`text-[10px] px-1.5 py-0.5 rounded-full ${l.sellable ? 'bg-success/20 text-success' : 'bg-warning/20 text-warning'}`}>
                                {l.sellable ? 'Available' : 'Preview'}
                            </span>
                        </li>
                    ))}
                </ul>
            </section>

            <section className="space-y-3">
                <h2 className="text-xl font-semibold text-text-primary">Questions</h2>
                <dl className="space-y-3 text-sm">
                    {FAQ.map(([q, a]) => (<div key={q}><dt className="text-text-primary font-medium">{q}</dt><dd>{a}</dd></div>))}
                </dl>
            </section>

            <footer className="text-xs text-text-muted border-t border-border pt-4 flex gap-4">
                <Link href="/legal">Terms · Refunds · Privacy</Link>
            </footer>
        </main>
    );
}
