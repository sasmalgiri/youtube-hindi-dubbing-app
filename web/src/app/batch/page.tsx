'use client';

import { useEffect } from 'react';
import Link from 'next/link';
import { useBatchManager } from '@/hooks/useBatchManager';
import BatchItemCard from '@/components/BatchItemCard';

export default function BatchPage() {
    const {
        items,
        isRunning,
        completedCount,
        overallProgress,
        autoDownload,
        setAutoDownload,
        start,
    } = useBatchManager();

    // Read batch data from sessionStorage and start processing
    useEffect(() => {
        if (items.length > 0) return; // Already started

        const stored = sessionStorage.getItem('batch_pending');
        if (!stored) return;

        sessionStorage.removeItem('batch_pending');
        try {
            const { urls, settings } = JSON.parse(stored);
            if (urls?.length > 0) {
                start(urls, settings);
            }
        } catch {}
    }, [items.length, start]);

    const totalCount = items.length;
    const errorCount = items.filter((i) => i.state === 'error').length;

    return (
        <div className="min-h-screen">
            {/* Top bar */}
            <div className="border-b border-border px-8 py-4 flex items-center justify-between">
                <div className="flex items-center gap-4">
                    <Link
                        href="/"
                        className="text-text-muted hover:text-text-primary transition-colors flex items-center gap-1"
                    >
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <path d="m12 19-7-7 7-7" />
                            <path d="M19 12H5" />
                        </svg>
                        Back
                    </Link>
                    <div className="h-4 w-px bg-border" />
                    <h1 className="text-lg font-semibold text-text-primary">
                        Batch Dubbing
                    </h1>
                    {totalCount > 0 && (
                        <span className="text-sm text-text-muted">
                            {completedCount}/{totalCount} complete
                            {errorCount > 0 && ` (${errorCount} failed)`}
                        </span>
                    )}
                </div>
                <div className="flex items-center gap-4">
                    {/* Auto-download toggle */}
                    <label className="flex items-center gap-2 cursor-pointer text-sm text-text-secondary">
                        <input
                            type="checkbox"
                            checked={autoDownload}
                            onChange={(e) => setAutoDownload(e.target.checked)}
                            className="w-4 h-4 rounded accent-primary"
                        />
                        Auto-download
                    </label>
                </div>
            </div>

            {/* Content */}
            <div className="max-w-4xl mx-auto px-8 py-8 space-y-6">
                {/* Overall progress */}
                {totalCount > 0 && (
                    <div className="glass-card p-5 space-y-3">
                        <div className="flex items-center justify-between">
                            <span className="text-sm font-medium text-text-primary">
                                Overall Progress
                            </span>
                            <span className="text-sm text-text-secondary">{overallProgress}%</span>
                        </div>
                        <div className="w-full h-2.5 bg-border rounded-full overflow-hidden">
                            <div
                                className="h-full bg-primary rounded-full transition-all duration-500"
                                style={{ width: `${overallProgress}%` }}
                            />
                        </div>
                        <div className="flex items-center gap-4 text-xs text-text-muted">
                            {isRunning && (
                                <span className="flex items-center gap-1.5">
                                    <svg className="animate-spin" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                        <path d="M21 12a9 9 0 1 1-6.219-8.56" />
                                    </svg>
                                    Processing...
                                </span>
                            )}
                            {!isRunning && completedCount === totalCount && totalCount > 0 && (
                                <span className="text-green-400 font-medium">All done!</span>
                            )}
                        </div>
                    </div>
                )}

                {/* Tip about auto-download */}
                {autoDownload && totalCount > 0 && (
                    <div className="flex items-start gap-2 text-xs text-text-muted px-1">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="flex-shrink-0 mt-0.5">
                            <circle cx="12" cy="12" r="10" />
                            <path d="M12 16v-4" /><path d="M12 8h.01" />
                        </svg>
                        Videos will auto-download to your browser&apos;s Downloads folder when complete.
                    </div>
                )}

                {/* Batch items */}
                {items.length > 0 ? (
                    <div className="space-y-3">
                        {items.map((item, index) => (
                            <BatchItemCard key={item.url + index} item={item} index={index} />
                        ))}
                    </div>
                ) : (
                    <div className="text-center py-16">
                        <p className="text-text-muted text-sm">
                            No batch jobs yet. Go to the home page and use the &quot;Batch URLs&quot; tab to start.
                        </p>
                        <Link
                            href="/"
                            className="inline-flex items-center gap-2 mt-4 btn-primary text-sm px-6 py-2.5"
                        >
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <path d="m12 19-7-7 7-7" />
                                <path d="M19 12H5" />
                            </svg>
                            Start Batch Dubbing
                        </Link>
                    </div>
                )}
            </div>
        </div>
    );
}
