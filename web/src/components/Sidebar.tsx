'use client';

import { useFeatures } from '@/hooks/useFeatures';
import { useAuth } from '@/hooks/useAuth';
import { useEffect, useState } from 'react';
import Link from 'next/link';
import { getJobs, type JobStatus } from '@/lib/api';
import { formatTimeAgo } from '@/lib/utils';

interface SidebarProps {
    onNewDubbing?: () => void;
}

export default function Sidebar({ onNewDubbing }: SidebarProps) {
    const [jobs, setJobs] = useState<JobStatus[]>([]);
    const { isOn } = useFeatures();

    useEffect(() => {
        const load = () => {
            getJobs().then(setJobs).catch(() => {});
        };
        load();
        const interval = setInterval(load, 5000);
        return () => clearInterval(interval);
    }, []);

    const stateColors: Record<string, string> = {
        queued: 'bg-warning/20 text-warning',
        running: 'bg-primary/20 text-primary-light',
        done: 'bg-success/20 text-success',
        error: 'bg-error/20 text-error',
        waiting_for_srt: 'bg-warning/20 text-warning',
    };

    return (
        <aside className="w-64 h-screen border-r border-border bg-card/30 flex flex-col fixed left-0 top-0">
            {/* Logo */}
            <div className="h-16 flex items-center px-5 border-b border-border">
                <div className="w-8 h-8 rounded-lg bg-primary flex items-center justify-center mr-3">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z" />
                        <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
                        <line x1="12" x2="12" y1="19" y2="22" />
                    </svg>
                </div>
                <span className="text-lg font-semibold text-text-primary">
                    Hindi<span className="text-primary">Dub</span>
                </span>
            </div>

            {/* New Dubbing Button */}
            <div className="p-4">
                <Link
                    href="/"
                    onClick={onNewDubbing}
                    className="btn-primary w-full flex items-center justify-center gap-2 text-sm"
                >
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M5 12h14" />
                        <path d="M12 5v14" />
                    </svg>
                    New Dubbing
                </Link>
                <Link
                    href="/modules"
                    className="mt-2 w-full flex items-center justify-center gap-2 text-xs text-text-muted hover:text-text-primary py-1.5 rounded-lg border border-white/10 hover:bg-white/5"
                >
                    Modules (on/off)
                </Link>
                <Link
                    href="/create"
                    className="mt-2 w-full flex items-center justify-center gap-2 text-xs text-text-muted hover:text-text-primary py-1.5 rounded-lg border border-white/10 hover:bg-white/5"
                >
                    New dub (prepaid flow)
                </Link>
                <Link
                    href="/history"
                    className="mt-2 w-full flex items-center justify-center gap-2 text-xs text-text-muted hover:text-text-primary py-1.5 rounded-lg border border-white/10 hover:bg-white/5"
                >
                    History
                </Link>
                {isOn('input.narration') && (
                    <Link
                        href="/create/narration"
                        className="mt-2 w-full flex items-center justify-center gap-2 text-xs text-text-muted hover:text-text-primary py-1.5 rounded-lg border border-white/10 hover:bg-white/5"
                    >
                        Narration
                    </Link>
                )}
                <AccountLink />
            </div>

            {/* Recent Jobs */}
            <div className="flex-1 overflow-y-auto px-3">
                <p className="text-xs font-medium text-text-muted uppercase tracking-wider px-2 mb-2">
                    Recent Jobs
                </p>
                {jobs.length === 0 && (
                    <p className="text-sm text-text-muted px-2 py-4">No jobs yet</p>
                )}
                <div className="space-y-1">
                    {jobs.slice(0, 20).map((job) => (
                        <Link
                            key={job.id}
                            href={`/jobs/${job.id}`}
                            className="block px-3 py-2.5 rounded-lg hover:bg-white/5 transition-colors group"
                        >
                            <div className="flex items-center gap-2 mb-1">
                                <span className={`text-[10px] px-1.5 py-0.5 rounded-full font-medium ${stateColors[job.state] || ''}`}>
                                    {job.state}
                                </span>
                                <span className="text-[10px] text-text-muted ml-auto">
                                    {formatTimeAgo(job.created_at)}
                                </span>
                            </div>
                            <p className="text-xs text-text-secondary truncate group-hover:text-text-primary transition-colors">
                                {job.video_title || job.source_url || job.id}
                            </p>
                            {job.state === 'running' && (
                                <div className="mt-1.5 h-1 bg-white/5 rounded-full overflow-hidden">
                                    <div
                                        className="h-full bg-primary rounded-full transition-all duration-500"
                                        style={{ width: `${(job.overall_progress || 0) * 100}%` }}
                                    />
                                </div>
                            )}
                        </Link>
                    ))}
                </div>
            </div>

            {/* Footer */}
            <div className="p-4 border-t border-border">
                <p className="text-[10px] text-text-muted text-center">
                    Powered by Whisper + Chatterbox AI
                </p>
            </div>
        </aside>
    );
}


function AccountLink() {
    const { user, loaded } = useAuth();
    if (!loaded) return null;
    return (
        <Link
            href={user ? '/account' : '/signin'}
            className="mt-2 w-full flex items-center justify-center gap-2 text-xs text-text-muted hover:text-text-primary py-1.5 rounded-lg border border-white/10 hover:bg-white/5 truncate"
            title={user ? user.email : 'Sign in'}
        >
            {user ? user.email : 'Sign in'}
        </Link>
    );
}
