'use client';

import { useState, useRef } from 'react';
import { useParams, useRouter } from 'next/navigation';
import Link from 'next/link';
import { useJobProgress } from '@/hooks/useJobProgress';
import ProgressPipeline from '@/components/ProgressPipeline';
import VideoPlayer from '@/components/VideoPlayer';
import TranscriptViewer from '@/components/TranscriptViewer';
import { resultVideoUrl, originalVideoUrl, resultSrtUrl, sourceSrtUrl, uploadTranslatedSrt, deleteJob } from '@/lib/api';

export default function JobPage() {
    const params = useParams();
    const router = useRouter();
    const jobId = params.id as string;
    const [cancelling, setCancelling] = useState(false);
    const [uploading, setUploading] = useState(false);
    const [uploadError, setUploadError] = useState<string | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleCancel = async () => {
        if (!confirm('Cancel this dubbing job?')) return;
        setCancelling(true);
        try {
            await deleteJob(jobId);
            router.push('/');
        } catch {
            setCancelling(false);
        }
    };

    const {
        status,
        step,
        stepProgress,
        overallProgress,
        message,
        isComplete,
        isError,
        isWaitingForSrt,
        error,
        restart,
    } = useJobProgress(jobId);

    const handleSrtUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;
        setUploading(true);
        setUploadError(null);
        try {
            await uploadTranslatedSrt(jobId, file);
            restart();
        } catch (err: any) {
            setUploadError(err.message || 'Upload failed');
        } finally {
            setUploading(false);
            if (fileInputRef.current) fileInputRef.current.value = '';
        }
    };

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
                        {status?.video_title || 'Dubbing Job'}
                    </h1>
                </div>
                <div className="flex items-center gap-3">
                    {/* Cancel button - shown while running or on error */}
                    {(!isComplete && !isWaitingForSrt || isError) && (
                        <button
                            onClick={handleCancel}
                            disabled={cancelling}
                            className="text-sm px-4 py-2 rounded-lg border border-error/30 text-error hover:bg-error/10 transition-colors flex items-center gap-2 disabled:opacity-50"
                        >
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <circle cx="12" cy="12" r="10" />
                                <path d="m15 9-6 6" />
                                <path d="m9 9 6 6" />
                            </svg>
                            {cancelling ? 'Cancelling...' : 'Cancel'}
                        </button>
                    )}
                    {/* Download buttons - shown when complete */}
                    {isComplete && (
                        <>
                            <a
                                href={resultSrtUrl(jobId)}
                                download
                                className="btn-secondary text-sm flex items-center gap-2"
                            >
                                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                    <path d="M17 6.1H3" /><path d="M21 12.1H3" /><path d="M15.1 18H3" />
                                </svg>
                                Subtitles
                            </a>
                            <a
                                href={resultVideoUrl(jobId)}
                                download
                                className="btn-primary text-sm flex items-center gap-2"
                            >
                                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                                    <polyline points="7 10 12 15 17 10" />
                                    <line x1="12" x2="12" y1="15" y2="3" />
                                </svg>
                                Download Video
                            </a>
                        </>
                    )}
                </div>
            </div>

            {/* Content */}
            <div className="max-w-5xl mx-auto px-8 py-8 space-y-8">
                {/* Progress Pipeline */}
                <ProgressPipeline
                    currentStep={step}
                    stepProgress={stepProgress}
                    overallProgress={overallProgress}
                    message={message}
                    isComplete={isComplete}
                    isError={isError}
                />

                {/* Error message */}
                {isError && error && (
                    <div className="glass-card p-5 border-error/30">
                        <div className="flex items-start gap-3">
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#ef4444" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="flex-shrink-0 mt-0.5">
                                <circle cx="12" cy="12" r="10" />
                                <line x1="12" x2="12" y1="8" y2="12" />
                                <line x1="12" x2="12.01" y1="16" y2="16" />
                            </svg>
                            <div>
                                <p className="text-sm font-medium text-error mb-1">Dubbing Failed</p>
                                <p className="text-sm text-text-secondary">{error}</p>
                            </div>
                        </div>
                    </div>
                )}

                {/* Waiting for SRT - manual translation workflow */}
                {isWaitingForSrt && (
                    <div className="glass-card p-6 space-y-5 animate-slide-up">
                        <div className="flex items-start gap-3">
                            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-primary flex-shrink-0 mt-0.5">
                                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                                <polyline points="14 2 14 8 20 8" />
                                <line x1="16" x2="8" y1="13" y2="13" />
                                <line x1="16" x2="8" y1="17" y2="17" />
                            </svg>
                            <div>
                                <h3 className="text-base font-semibold text-text-primary mb-1">Transcription Complete</h3>
                                <p className="text-sm text-text-secondary">
                                    Download the source SRT below, translate it (e.g. with Claude or any tool), then upload the translated SRT to continue dubbing.
                                </p>
                            </div>
                        </div>

                        {/* Step 1: Download source SRT */}
                        <div className="flex items-center gap-4">
                            <span className="flex items-center justify-center w-7 h-7 rounded-full bg-primary/20 text-primary text-sm font-bold">1</span>
                            <a
                                href={sourceSrtUrl(jobId)}
                                download
                                className="btn-secondary text-sm flex items-center gap-2"
                            >
                                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                                    <polyline points="7 10 12 15 17 10" />
                                    <line x1="12" x2="12" y1="15" y2="3" />
                                </svg>
                                Download Source SRT
                            </a>
                        </div>

                        {/* Step 2: Upload translated SRT */}
                        <div className="flex items-center gap-4">
                            <span className="flex items-center justify-center w-7 h-7 rounded-full bg-primary/20 text-primary text-sm font-bold">2</span>
                            <div className="flex-1">
                                <label className="btn-primary text-sm inline-flex items-center gap-2 cursor-pointer">
                                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                                        <polyline points="17 8 12 3 7 8" />
                                        <line x1="12" x2="12" y1="3" y2="15" />
                                    </svg>
                                    {uploading ? 'Uploading...' : 'Upload Translated SRT'}
                                    <input
                                        ref={fileInputRef}
                                        type="file"
                                        accept=".srt"
                                        onChange={handleSrtUpload}
                                        disabled={uploading}
                                        className="hidden"
                                    />
                                </label>
                            </div>
                        </div>

                        {uploadError && (
                            <p className="text-sm text-error ml-11">{uploadError}</p>
                        )}
                    </div>
                )}

                {/* Results - shown when complete */}
                {isComplete && (
                    <div className="space-y-8 animate-slide-up">
                        {/* Video Player */}
                        <VideoPlayer
                            originalUrl={originalVideoUrl(jobId)}
                            dubbedUrl={resultVideoUrl(jobId)}
                            targetLanguage={status?.target_language}
                        />

                        {/* Transcript */}
                        <TranscriptViewer jobId={jobId} targetLanguage={status?.target_language} />
                    </div>
                )}

                {/* Loading state */}
                {!isComplete && !isError && !isWaitingForSrt && (
                    <div className="text-center py-16">
                        <div className="inline-flex items-center gap-3 px-6 py-3 rounded-2xl bg-card border border-border">
                            <svg className="animate-spin text-primary" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <path d="M21 12a9 9 0 1 1-6.219-8.56" />
                            </svg>
                            <span className="text-sm text-text-secondary">
                                Processing your video... This may take a few minutes.
                            </span>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
