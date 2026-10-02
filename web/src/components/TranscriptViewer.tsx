'use client';

import { useEffect, useState } from 'react';
import { getTranscript, resultSrtUrl, type TranscriptSegment } from '@/lib/api';
import { formatTime, getLanguageName } from '@/lib/utils';

interface TranscriptViewerProps {
    jobId: string;
    targetLanguage?: string;
    // Only when the backend has a subtitle file for the job (the job page's
    // hasSubtitles); an always-on link 404ed after the SRT moved to the saved folder.
    showSrtLink?: boolean;
}

export default function TranscriptViewer({ jobId, targetLanguage = 'hi', showSrtLink = false }: TranscriptViewerProps) {
    const [segments, setSegments] = useState<TranscriptSegment[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        getTranscript(jobId)
            .then((data) => {
                setSegments(data.segments);
                setLoading(false);
            })
            .catch(() => setLoading(false));
    }, [jobId]);

    if (loading) {
        return (
            <div className="glass-card p-6">
                <p className="text-sm text-text-muted text-center">Loading transcript...</p>
            </div>
        );
    }

    if (segments.length === 0) {
        return (
            <div className="glass-card p-6">
                <p className="text-sm text-text-muted text-center">No transcript available</p>
            </div>
        );
    }

    return (
        <div className="glass-card overflow-hidden">
            {/* Header */}
            <div className="flex items-center justify-between px-5 py-3 border-b border-border">
                <h3 className="text-sm font-medium text-text-primary">Transcript</h3>
                {showSrtLink && <a
                    href={resultSrtUrl(jobId)}
                    download
                    className="text-xs text-primary hover:text-primary-light transition-colors flex items-center gap-1"
                >
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                        <polyline points="7 10 12 15 17 10" />
                        <line x1="12" x2="12" y1="15" y2="3" />
                    </svg>
                    Download SRT
                </a>}
            </div>

            {/* Column headers */}
            <div className="grid grid-cols-[60px_1fr_1fr] gap-4 px-5 py-2 border-b border-border bg-white/[0.02]">
                <span className="text-[10px] font-medium text-text-muted uppercase">Time</span>
                <span className="text-[10px] font-medium text-text-muted uppercase">Original</span>
                <span className="text-[10px] font-medium text-text-muted uppercase">Translated ({getLanguageName(targetLanguage)})</span>
            </div>

            {/* Segments */}
            <div className="max-h-[400px] overflow-y-auto">
                {segments.filter(s => s.text?.trim()).map((seg, i) => (
                    <div
                        key={i}
                        className="grid grid-cols-[60px_1fr_1fr] gap-4 px-5 py-3 border-b border-border/50 hover:bg-white/[0.02] transition-colors"
                    >
                        <span className="text-xs text-text-muted font-mono">
                            {formatTime(seg.start)}
                        </span>
                        <div className="text-sm text-text-secondary">
                            {seg.text_original && seg.text_original !== seg.text ? (
                                <>
                                    <p className="text-text-muted line-through text-xs mb-1">{seg.text_original}</p>
                                    <p>{seg.text}</p>
                                </>
                            ) : (
                                <p>{seg.text}</p>
                            )}
                        </div>
                        <p className="text-sm text-text-primary">
                            {seg.text_translated}
                        </p>
                    </div>
                ))}
            </div>

            {/* Footer */}
            <div className="px-5 py-2 border-t border-border bg-white/[0.02]">
                <p className="text-[10px] text-text-muted">
                    {segments.filter(s => s.text?.trim()).length} segments
                </p>
            </div>
        </div>
    );
}
