'use client';

import Link from 'next/link';
import { type JobStatus, deleteJob } from '@/lib/api';
import { formatTimeAgo, cn } from '@/lib/utils';

interface JobCardProps {
    job: JobStatus;
    onDelete?: () => void;
}

const stateConfig: Record<string, { color: string; bg: string; label: string }> = {
    queued: { color: 'text-warning', bg: 'bg-warning/10', label: 'Queued' },
    running: { color: 'text-primary-light', bg: 'bg-primary/10', label: 'Running' },
    done: { color: 'text-success', bg: 'bg-success/10', label: 'Complete' },
    error: { color: 'text-error', bg: 'bg-error/10', label: 'Failed' },
    waiting_for_srt: { color: 'text-warning', bg: 'bg-warning/10', label: 'Awaiting SRT' },
};

export default function JobCard({ job, onDelete }: JobCardProps) {
    const state = stateConfig[job.state] || stateConfig.queued;

    const handleDelete = async (e: React.MouseEvent) => {
        e.preventDefault();
        e.stopPropagation();
        try {
            await deleteJob(job.id);
            onDelete?.();
        } catch {
            // ignore
        }
    };

    return (
        <Link href={`/jobs/${job.id}`} className="block">
            <div className="glass-card-hover p-5 group">
                <div className="flex items-start justify-between mb-3">
                    <span className={cn('text-xs font-medium px-2.5 py-1 rounded-full', state.color, state.bg)}>
                        {state.label}
                    </span>
                    <div className="flex items-center gap-2">
                        <span className="text-xs text-text-muted">
                            {formatTimeAgo(job.created_at)}
                        </span>
                        <button
                            onClick={handleDelete}
                            className="opacity-0 group-hover:opacity-100 transition-opacity text-text-muted hover:text-error"
                        >
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <path d="M3 6h18" /><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6" /><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2" />
                            </svg>
                        </button>
                    </div>
                </div>

                <h3 className="text-sm font-medium text-text-primary truncate mb-1">
                    {job.video_title || 'Untitled Video'}
                </h3>
                <p className="text-xs text-text-muted truncate mb-3">
                    {job.source_url}
                </p>

                {/* Progress bar for running jobs */}
                {job.state === 'running' && (
                    <div>
                        <div className="flex items-center justify-between mb-1">
                            <span className="text-[10px] text-text-muted capitalize">{job.current_step || 'Starting...'}</span>
                            <span className="text-[10px] text-text-muted">{Math.round(job.overall_progress * 100)}%</span>
                        </div>
                        <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
                            <div
                                className="h-full bg-gradient-to-r from-primary to-accent rounded-full transition-all duration-500"
                                style={{ width: `${job.overall_progress * 100}%` }}
                            />
                        </div>
                    </div>
                )}

                {job.state === 'error' && job.error && (
                    <p className="text-xs text-error truncate">{job.error}</p>
                )}
            </div>
        </Link>
    );
}
