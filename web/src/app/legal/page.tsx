import Link from 'next/link';

export const metadata = { title: 'Terms, refunds and privacy (draft)' };

function Todo({ children }: { children: React.ReactNode }) {
    return <mark className="bg-amber-500/20 text-amber-200 px-1 rounded">[OWNER: {children}]</mark>;
}

/** Drafts written from what the service actually does (2026-11-25). Not legal advice; counsel must review (tracker L01–L06). */
export default function LegalPage() {
    return (
        <main className="max-w-3xl mx-auto px-4 py-8 space-y-8 text-sm text-text-secondary leading-relaxed">
            <div>
                <Link href="/" className="text-xs text-text-muted hover:text-text-primary">← Back</Link>
                <h1 className="text-2xl font-semibold text-text-primary mt-1">Terms, refunds and privacy</h1>
                <p role="note" className="mt-3 rounded-lg border border-amber-500/40 bg-amber-500/10 p-3 text-amber-200">
                    DRAFT for review by a lawyer and an accountant. It describes how the service works today and is not yet in force.
                    Items marked [OWNER] need a decision.
                </p>
                <nav className="mt-3 flex gap-4 text-xs"><a href="#terms" className="underline">Terms</a><a href="#refunds" className="underline">Refunds</a><a href="#privacy" className="underline">Privacy</a></nav>
            </div>

            <section id="terms" className="space-y-3">
                <h2 className="text-lg font-semibold text-text-primary">Terms of service</h2>
                <p><strong>Who we are.</strong> <Todo>legal name, address, contact email</Todo> (&quot;we&quot;) provide an AI dubbing service that
                    turns videos you upload into dubbed versions in the target language you choose.</p>
                <p><strong>Your content.</strong> You may only upload material you own or have the rights to dub and use as intended. Before each job
                    you confirm this; we keep that confirmation with its date. You keep all rights in your content and in the dubs we make for you.
                    We use your content only to make your dubs, never to train models.</p>
                <p><strong>Not allowed.</strong> Content you have no rights to; impersonating a real person&apos;s voice without their permission;
                    unlawful, hateful or sexual content involving minors. We may refuse or remove jobs that break these rules and close repeat accounts.</p>
                <p><strong>Machine output.</strong> Translations and voices are produced by AI and may contain mistakes. Review a dub before you
                    publish it; you are responsible for what you publish.</p>
                <p><strong>Prepaid time.</strong> You buy prepaid time in packs. One hour covers one hour of source video in one target language,
                    Standard quality. Time does not expire. Prices, the currency and any tax are shown before you pay and on your receipt.</p>
                <p><strong>Availability.</strong> We aim to keep the service running but do not guarantee it. Jobs may wait in a queue; the queue
                    position is shown before you start.</p>
                <p><strong>Liability.</strong> <Todo>liability cap and exclusions, per counsel</Todo></p>
                <p><strong>Law and disputes.</strong> <Todo>governing law and courts, e.g. India, Kolkata</Todo></p>
                <p><strong>Changes.</strong> We will give notice before changing these terms; time already bought keeps the price you paid.</p>
            </section>

            <section id="refunds" className="space-y-3">
                <h2 className="text-lg font-semibold text-text-primary">Refunds</h2>
                <p><strong>Failed dubs cost nothing.</strong> Time is held when a job starts and only taken when a finished dub passes our delivery
                    check (it plays, has sound, is the length you paid for, its subtitles are valid). If a job fails, is cancelled before it starts,
                    or fails the check, the held time returns to your balance automatically.</p>
                <p><strong>Unused packs.</strong> You can ask for a refund of a pack you have not used at all within <Todo>number of days, e.g. 14</Todo> days
                    of buying it. It is returned to your original payment method; the time is removed from your balance.</p>
                <p><strong>Partly used packs.</strong> <Todo>pro-rata refund or none</Todo></p>
                <p><strong>Chargebacks.</strong> If a payment is disputed with your bank, the time from that payment is removed; time already used
                    shows as owed until the dispute is resolved.</p>
                <p><strong>How to ask.</strong> Email <Todo>support address</Todo> with your receipt number (VD/…).</p>
            </section>

            <section id="privacy" className="space-y-3">
                <h2 className="text-lg font-semibold text-text-primary">Privacy</h2>
                <p><strong>What we keep.</strong> Your email (and name if you sign in with Google); your sign-in sessions; the videos you upload and the
                    dubs, subtitles and job reports we make; your billing country; payment orders and receipts; your rights confirmations.</p>
                <p><strong>Who processes it.</strong> To make a dub, parts of your content go to: a speech-to-text service (<em>Groq</em> or this
                    server&apos;s own Whisper), a translation model (<em>Groq</em>, <em>Google Gemini</em> or similar), and a voice service
                    (<em>Google Cloud Text-to-Speech</em>, with <em>Sarvam AI</em> as fallback). Payments are handled by <em>Razorpay</em>; we never
                    see your card. Finished files may be stored in <Todo>storage provider, e.g. Cloudflare R2</Todo>. Email codes are sent by
                    <Todo>email provider</Todo>.</p>
                <p><strong>How long.</strong> Uploads not used for a job: <Todo>days</Todo>. Finished dubs and subtitles: <Todo>days, or until you delete them</Todo>.
                    Payment records and receipts: as long as tax law requires <Todo>years</Todo>. Backups: <Todo>days</Todo>, after which deleted data is gone from them too.</p>
                <p><strong>Deleting your account.</strong> From your account page. We delete your dubs, files, uploads and batches, your email, name and
                    Google link, and sign you out everywhere. We keep payment orders, receipts and the prepaid-time ledger (financial records the
                    law requires) and your rights confirmations, linked to an anonymous id. Unused prepaid time is lost.</p>
                <p><strong>Your rights.</strong> You can see, download and delete your data. Where data-protection law gives you more rights (India&apos;s
                    DPDP Act, the EU/UK GDPR and others), email <Todo>privacy contact / grievance officer</Todo>.</p>
                <p><strong>Cookies.</strong> One sign-in cookie, needed for the service to work. No advertising or tracking cookies.</p>
            </section>
        </main>
    );
}
