'use client';

import { useState, useRef } from 'react';
import { useParams, useRouter } from 'next/navigation';
import Link from 'next/link';
import { useJobProgress } from '@/hooks/useJobProgress';
import ProgressPipeline from '@/components/ProgressPipeline';
import VideoPlayer from '@/components/VideoPlayer';
import TranscriptViewer from '@/components/TranscriptViewer';
import { resultVideoUrl, originalVideoUrl, resultSrtUrl, sourceSrtUrl, uploadTranslatedSrt, deleteJob, retryJob } from '@/lib/api';

export default function JobPage() {
    const params = useParams();
    const router = useRouter();
    const jobId = params.id as string;
    const [cancelling, setCancelling] = useState(false);
    const [retrying, setRetrying] = useState(false);
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
        eta,
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
                    eta={eta}
                />

                {/* Resource pills */}
                {status?.config && Object.keys(status.config).length > 0 && (
                    <div className="flex flex-wrap gap-2">
                        {status.config.asr_model && (
                            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium bg-blue-500/10 text-blue-400 border border-blue-500/20">
                                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z" /><path d="M19 10v2a7 7 0 0 1-14 0v-2" /></svg>
                                Whisper {status.config.asr_model}
                            </span>
                        )}
                        {status.config.translation_engine && (
                            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium bg-purple-500/10 text-purple-400 border border-purple-500/20">
                                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m5 8 6 6" /><path d="m4 14 6-6 2-3" /><path d="M2 5h12" /><path d="M7 2h1" /></svg>
                                {status.config.translation_engine}
                            </span>
                        )}
                        {status.config.tts_engine && (
                            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium bg-green-500/10 text-green-400 border border-green-500/20">
                                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M2 10v3" /><path d="M6 6v11" /><path d="M10 3v18" /><path d="M14 8v7" /><path d="M18 5v13" /><path d="M22 10v3" /></svg>
                                {status.config.tts_engine}
                            </span>
                        )}
                        {status.config.audio_priority && (
                            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium bg-amber-500/10 text-amber-400 border border-amber-500/20">
                                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" /></svg>
                                Audio Priority
                            </span>
                        )}
                        {status.config.audio_bitrate && (
                            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
                                {status.config.audio_bitrate}
                            </span>
                        )}
                        {status.config.encode_preset && status.config.encode_preset !== 'veryfast' && (
                            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium bg-rose-500/10 text-rose-400 border border-rose-500/20">
                                {status.config.encode_preset}
                            </span>
                        )}
                    </div>
                )}

                {/* Error message */}
                {isError && error && (
                    <div className="glass-card p-5 border-error/30">
                        <div className="flex items-start gap-3">
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#ef4444" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="flex-shrink-0 mt-0.5">
                                <circle cx="12" cy="12" r="10" />
                                <line x1="12" x2="12" y1="8" y2="12" />
                                <line x1="12" x2="12.01" y1="16" y2="16" />
                            </svg>
                            <div className="flex-1">
                                <p className="text-sm font-medium text-error mb-1">Dubbing Failed</p>
                                <p className="text-sm text-text-secondary mb-3">{error}</p>
                                <button
                                    onClick={async () => {
                                        setRetrying(true);
                                        try {
                                            await retryJob(jobId);
                                            restart();
                                        } catch (e) {
                                            alert(e instanceof Error ? e.message : 'Retry failed');
                                        } finally {
                                            setRetrying(false);
                                        }
                                    }}
                                    disabled={retrying}
                                    className="px-4 py-2 rounded-lg bg-primary text-white text-sm font-medium hover:bg-primary/90 transition-colors disabled:opacity-50 flex items-center gap-2"
                                >
                                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                        <path d="M21 2v6h-6" />
                                        <path d="M3 12a9 9 0 0 1 15-6.7L21 8" />
                                        <path d="M3 22v-6h6" />
                                        <path d="M21 12a9 9 0 0 1-15 6.7L3 16" />
                                    </svg>
                                    {retrying ? 'Retrying...' : 'Retry Job'}
                                </button>
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
