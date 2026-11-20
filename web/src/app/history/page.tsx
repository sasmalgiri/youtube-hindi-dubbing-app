'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { getJobs, resultSrtUrl, resultVideoUrl, type JobStatus } from '@/lib/api';
import { useAuth } from '@/hooks/useAuth';

const LABEL: Record<string, string> = {
    done: 'Complete', error: 'Failed', running: 'In progress', queued: 'Queued',
    cancelled: 'Cancelled', waiting_for_srt: 'Waiting for subtitles', review_translation: 'Waiting for review',
};

function when(ts?: number) {
    if (!ts) return '';
    return new Date(ts * 1000).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' });
}

/** W7: the signed-in account's own jobs (the backend filters by owner). */
export default function HistoryPage() {
    const router = useRouter();
    const auth = useAuth();
    const [jobs, setJobs] = useState<JobStatus[] | null>(null);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        if (auth.loaded && auth.config?.required && !auth.user) router.push('/signin');
    }, [auth.loaded, auth.config?.required, auth.user, router]);

    useEffect(() => {
        if (!auth.loaded) return;
        getJobs().then(setJobs).catch((e) => setError(e instanceof Error ? e.message : 'Could not load history'));
    }, [auth.loaded, auth.user?.id]);

    return (
        <main className="max-w-3xl mx-auto px-4 py-8 space-y-6">
            <div>
                <Link href="/" className="text-xs text-text-muted hover:text-text-primary">← Back</Link>
                <h1 className="text-2xl font-semibold text-text-primary mt-1">History</h1>
                <p className="text-sm text-text-muted mt-1">Your dubs, newest first. Only you can see these.</p>
            </div>
            {error && <p className="text-sm text-red-300">{error}</p>}
            {jobs === null && !error && <p className="text-sm text-text-muted">Loading…</p>}
            {jobs?.length === 0 && (
                <p className="text-sm text-text-muted">No dubs yet. <Link href="/create" className="underline">Start one</Link>.</p>
            )}
            <ul className="space-y-2" data-testid="history">
                {jobs?.map((j) => (
                    <li key={j.id} className="glass-card p-4 flex flex-wrap items-center gap-3">
                        <div className="flex-1 min-w-0">
                            <Link href={`/jobs/${j.id}`} className="text-sm font-medium text-text-primary hover:underline block truncate">
                                {j.video_title || j.source_url || j.id}
                            </Link>
                            <p className="text-xs text-text-muted">
                                {LABEL[j.state] || j.state} · {when(j.created_at)}
                                {j.state === 'error' && j.error ? ` · ${j.error}` : ''}
                            </p>
                        </div>
                        {j.state === 'done' && (
                            <div className="flex gap-2">
                                <a href={resultVideoUrl(j.id)} className="btn-primary px-3 py-1.5 text-xs">Download video</a>
                                {j.subtitles_path && <a href={resultSrtUrl(j.id)} className="px-3 py-1.5 text-xs rounded-lg border border-white/10 text-text-primary hover:bg-white/5">SRT</a>}
                            </div>
                        )}
                    </li>
                ))}
            </ul>
        </main>
    );
}
