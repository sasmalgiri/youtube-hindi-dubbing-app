'use client';

import { useState, useRef, useCallback, useEffect } from 'react';
import { useParams, useRouter } from 'next/navigation';
import Link from 'next/link';
import { useJobProgress } from '@/hooks/useJobProgress';
import ProgressPipeline from '@/components/ProgressPipeline';
import VideoPlayer from '@/components/VideoPlayer';
import TranscriptViewer from '@/components/TranscriptViewer';
import { resultVideoUrl, originalVideoUrl, resultSrtUrl, sourceSrtUrl, uploadTranslatedSrt, deleteJob, continueJob, dialogueReportUrl, getJob, type JobStatus } from '@/lib/api';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || '';

export default function JobPage() {
    const params = useParams();
    const router = useRouter();
    const jobId = params.id as string;
    const [cancelling, setCancelling] = useState(false);
    const [uploading, setUploading] = useState(false);
    const [uploadError, setUploadError] = useState<string | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const [qaReport, setQaReport] = useState<string | null>(null);
    const [qaOpen, setQaOpen] = useState(false);

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
        isReviewing,
        reviewStep,
        error,
        eta,
        restart,
    } = useJobProgress(jobId);
    const [continuing, setContinuing] = useState(false);

    // The Subtitles button is shown only when the backend has a subtitle file
    // for this job (the old always-on button 404ed). Split classic jobs move
    // their subtitles into the saved folder just after the final "done" event,
    // so the hook's last status read can miss subtitles_path: read once more.
    const [lateStatus, setLateStatus] = useState<JobStatus | null>(null);
    useEffect(() => {
        if (!isComplete || !status || status.subtitles_path) return;
        const t = setTimeout(() => { getJob(jobId).then(setLateStatus).catch(() => { }); }, 3000);
        return () => clearTimeout(t);
    }, [isComplete, status, jobId]);
    const hasSubtitles = Boolean(status?.subtitles_path || lateStatus?.subtitles_path);

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
                            {hasSubtitles && (
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
                            )}
                            <a
                                href={resultVideoUrl(jobId)}
                                download={status?.video_title ? `${status.video_title} - Dubbed.mp4` : `dubbed_${jobId}.mp4`}
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
                    stepTimes={status?.step_times}
                />

                {/* Resource pills */}
                {status?.config && Object.keys(status.config).length > 0 && (
                    <div className="space-y-3">
                        {(() => {
                            const c = status.config;
                            const ON = 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20';
                            const OFF = 'bg-zinc-500/10 text-zinc-500 border border-zinc-500/20 opacity-60';
                            const VAL = 'bg-blue-500/10 text-blue-400 border border-blue-500/20';
                            const pillCls = 'inline-flex items-center px-2.5 py-0.5 rounded-full text-[10px] font-medium';
                            const B = (label: string, cls: string) => (
                                <span key={label} className={`${pillCls} ${cls}`}>{label}</span>
                            );
                            const BoolPill = (label: string, val: boolean | undefined) =>
                                B(`${label}: ${val ? 'ON' : 'OFF'}`, val ? ON : OFF);

                            return (
                                <>
                                    {/* Core Settings */}
                                    <div>
                                        <p className="text-[10px] uppercase tracking-wider text-text-muted mb-1.5 font-semibold">Core</p>
                                        <div className="flex flex-wrap gap-1.5">
                                            {c.asr_model && B(c.asr_model, 'bg-blue-500/10 text-blue-400 border border-blue-500/20')}
                                            {c.translation_engine && B(c.translation_engine, 'bg-purple-500/10 text-purple-400 border border-purple-500/20')}
                                            {c.tts_engine && B(c.tts_engine, 'bg-green-500/10 text-green-400 border border-green-500/20')}
                                            {c.source_language && B(`Source: ${c.source_language}`, VAL)}
                                            {c.target_language && B(`Target: ${c.target_language}`, VAL)}
                                            {c.voice && B(`Voice: ${c.voice}`, VAL)}
                                            {c.preset_name && B(`Preset: ${c.preset_name}`, 'bg-violet-500/10 text-violet-400 border border-violet-500/20')}
                                            {c.pipeline_mode && B(`Pipeline: ${c.pipeline_mode}`, VAL)}
                                        </div>
                                    </div>

                                    {/* Audio & Encoding */}
                                    <div>
                                        <p className="text-[10px] uppercase tracking-wider text-text-muted mb-1.5 font-semibold">Audio & Encoding</p>
                                        <div className="flex flex-wrap gap-1.5">
                                            {c.tts_rate && B(`Rate: ${c.tts_rate}`, 'bg-teal-500/10 text-teal-400 border border-teal-500/20')}
                                            {c.audio_bitrate && B(`Bitrate: ${c.audio_bitrate}`, 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/20')}
                                            {c.encode_preset && B(`Encode: ${c.encode_preset}`, 'bg-rose-500/10 text-rose-400 border border-rose-500/20')}
                                            {c.post_tts_level && B(`Post-TTS: ${c.post_tts_level}`, 'bg-indigo-500/10 text-indigo-400 border border-indigo-500/20')}
                                            {c.audio_quality_mode && B(`Quality: ${c.audio_quality_mode}`, VAL)}
                                            {c.download_mode && B(`Download: ${c.download_mode}`, VAL)}
                                            {c.original_volume != null && B(`BG Vol: ${c.original_volume}`, VAL)}
                                            {c.split_duration != null && B(`Split: ${c.split_duration}s`, VAL)}
                                            {c.dub_duration != null && c.dub_duration > 0 && B(`Dub Limit: ${c.dub_duration >= 60 ? `${Math.floor(c.dub_duration / 60)}h${c.dub_duration % 60 ? ` ${c.dub_duration % 60}m` : ''}` : `${c.dub_duration}m`}`, VAL)}
                                        </div>
                                    </div>

                                    {/* Boolean Toggles */}
                                    <div>
                                        <p className="text-[10px] uppercase tracking-wider text-text-muted mb-1.5 font-semibold">Options</p>
                                        <div className="flex flex-wrap gap-1.5">
                                            {BoolPill('Audio Priority', c.audio_priority)}
                                            {BoolPill('Untouchable', c.audio_untouchable)}
                                            {BoolPill('Video Sync', c.video_slow_to_match)}
                                            {BoolPill('Mix BG Music', c.mix_original)}
                                            {BoolPill('Multi-Speaker', c.multi_speaker)}
                                            {BoolPill('Fast Assemble', c.fast_assemble)}
                                            {BoolPill('Sentence Gap', c.enable_sentence_gap)}
                                            {BoolPill('Duration Fit', c.enable_duration_fit)}
                                            {BoolPill('WhisperX', c.use_whisperx)}
                                            {BoolPill('Simplify EN', c.simplify_english)}
                                            {BoolPill('Manual Review', c.enable_manual_review)}
                                            {BoolPill('Transcribe Only', c.transcribe_only)}
                                            {BoolPill('Sarvam Bulbul', c.use_sarvam_bulbul)}
                                            {BoolPill('TTS Verify Retry', c.enable_tts_verify_retry)}
                                            {BoolPill('Keep Subj EN', c.keep_subject_english)}
                                            {BoolPill('Word Match', c.tts_word_match_verify)}
                                            {BoolPill('Seg Trace', c.long_segment_trace)}
                                            {BoolPill('No Time Pressure', c.tts_no_time_pressure)}
                                            {BoolPill('Dynamic Workers', c.tts_dynamic_workers)}
                                            {BoolPill('Purge on URL', c.purge_on_new_url)}
                                            {BoolPill('Step-by-Step', c.step_by_step)}
                                            {BoolPill('New Pipeline', c.use_new_pipeline)}
                                            {BoolPill('SRT Translate', c.srt_needs_translation)}
                                            {c.av_sync_mode && c.av_sync_mode !== 'original' && B(`AV Sync: ${c.av_sync_mode}`, 'bg-orange-500/10 text-orange-400 border border-orange-500/20')}
                                            {c.use_global_stretch && B('Global Stretch', 'bg-orange-500/10 text-orange-400 border border-orange-500/20')}
                                            {c.segmenter && c.segmenter !== 'dp' && B(`Segmenter: ${c.segmenter}`, 'bg-orange-500/10 text-orange-400 border border-orange-500/20')}
                                            {/* Detect Voice Clone mode */}
                                            {c.audio_untouchable && c.post_tts_level === 'none' && B('Voice Clone', 'bg-violet-500/10 text-violet-400 border border-violet-500/20')}
                                        </div>
                                    </div>

                                    {/* Thresholds */}
                                    <div>
                                        <p className="text-[10px] uppercase tracking-wider text-text-muted mb-1.5 font-semibold">Thresholds</p>
                                        <div className="flex flex-wrap gap-1.5">
                                            {c.tts_truncation_threshold != null && B(`Truncation: ${c.tts_truncation_threshold}`, VAL)}
                                            {c.tts_word_match_tolerance != null && B(`Match Tol: ${c.tts_word_match_tolerance}`, VAL)}
                                            {c.tts_word_match_model && B(`Match Model: ${c.tts_word_match_model}`, VAL)}
                                            {c.long_segment_threshold_words != null && B(`Long Seg: ${c.long_segment_threshold_words}w`, VAL)}
                                            {c.tts_dynamic_min != null && B(`Workers: ${c.tts_dynamic_min}-${c.tts_dynamic_max ?? '?'} (start ${c.tts_dynamic_start ?? '?'})`, VAL)}
                                        </div>
                                    </div>
                                </>
                            );
                        })()}
                    </div>
                )}

                {/* Multi-speaker outcome: who spoke, detected gender, Hindi voice */}
                {status?.speaker_warning && (
                    <div className="glass-card p-4 border border-amber-500/30">
                        <p className="text-sm font-medium text-amber-400">Multi-speaker did not run</p>
                        <p className="text-xs text-text-muted mt-1">{status.speaker_warning}</p>
                    </div>
                )}
                {status?.speakers && status.speakers.length > 0 && (() => {
                    // One row per character (split jobs report each part; the
                    // character bank keeps labels and voices stable across parts).
                    // Dialogue jobs fill this list from speakers.json (voice
                    // categories male_like / female_like / unknown): a missing
                    // field must not crash the whole job page.
                    const chars = new Map<string, { gender: string; voice: string; label: string; seconds: number; parts: number[]; reused: boolean }>();
                    for (const s of status.speakers!) {
                        const speaker = String(s.speaker ?? '?');
                        const voice = String(s.voice ?? '');
                        // Job-wide CHARxx labels merge across parts; older split jobs
                        // reused SPEAKER_xx per part, so keep those rows per part.
                        const key = s.part != null && !speaker.startsWith('CHAR') ? `Part ${s.part} · ${speaker}` : speaker;
                        const c = chars.get(key) ?? { gender: String(s.gender ?? '').toLowerCase(), voice, label: s.voice_label || voice.split('-').pop()?.replace('Neural', '') || voice || '?', seconds: 0, parts: [], reused: false };
                        c.seconds += Number(s.seconds) || 0;
                        if (s.part != null && !c.parts.includes(s.part)) c.parts.push(s.part);
                        c.reused = c.reused || !!s.reused;
                        chars.set(key, c);
                    }
                    const rows = [...chars.entries()].sort((a, b) => b[1].seconds - a[1].seconds);
                    return (
                        <div className="glass-card p-4">
                            <p className="text-sm font-medium text-text-primary mb-2">
                                Characters ({rows.length}) · {new Set(rows.map(([, c]) => c.voice)).size} distinct voices
                            </p>
                            <div className="space-y-1.5">
                                {rows.map(([name, c]) => (
                                    <div key={name} className="flex items-center justify-between text-xs">
                                        <div className="flex items-center gap-2">
                                            <span className={`px-1.5 py-0.5 rounded font-medium ${c.gender.startsWith('female') ? 'bg-pink-500/20 text-pink-300' : c.gender.startsWith('male') ? 'bg-blue-500/20 text-blue-300' : 'bg-zinc-500/20 text-zinc-300'}`}>
                                                {c.gender.startsWith('female') ? 'F' : c.gender.startsWith('male') ? 'M' : '?'}
                                            </span>
                                            <span className="text-text-secondary">{name}</span>
                                            <span className="text-text-muted">
                                                {Math.round(c.seconds)}s{c.parts.length > 1 ? ` · parts ${c.parts.sort((x, y) => x - y).join(',')}` : ''}
                                            </span>
                                        </div>
                                        <span className="text-text-primary">
                                            {c.label}{c.reused && <span className="text-amber-400" title="More characters than distinct voices"> (reused)</span>}
                                        </span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    );
                })()}

                {/* Honest result status (any mode); the report link only when a
                    report exists — classic jobs with warnings have none (404) */}
                {status?.result_status && (
                    <div className="glass-card p-4">
                        <div className="flex items-center justify-between gap-3">
                            <div>
                                <span className={`inline-block px-2 py-0.5 rounded text-xs font-semibold ${
                                    status.result_status === 'completed' ? 'bg-green-500/20 text-green-400' :
                                    status.result_status === 'completed_with_warnings' ? 'bg-amber-500/20 text-amber-400' :
                                    'bg-red-500/20 text-red-400'}`}>
                                    {status.result_status.replace(/_/g, ' ')}
                                </span>
                                {(status.status_reasons || []).length > 0 && (
                                    <ul className="mt-2 text-xs text-text-muted list-disc pl-4 space-y-0.5">
                                        {(status.status_reasons || []).slice(0, 6).map((r, i) => <li key={i}>{r}</li>)}
                                    </ul>
                                )}
                            </div>
                            {status.report_path && (
                                <a href={dialogueReportUrl(jobId)} target="_blank" rel="noreferrer"
                                   className="text-xs px-3 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 text-text-primary whitespace-nowrap">
                                    Open report
                                </a>
                            )}
                        </div>
                    </div>
                )}

                {/* QA Score Badge */}
                {status?.qa_score != null && (
                    <div className="glass-card p-4">
                        <div className="flex items-center justify-between">
                            <div className="flex items-center gap-3">
                                <div className={`
                                    w-10 h-10 rounded-full flex items-center justify-center text-sm font-bold
                                    ${status.qa_score >= 0.7 ? 'bg-green-500/20 text-green-400' :
                                        status.qa_score >= 0.4 ? 'bg-amber-500/20 text-amber-400' :
                                            'bg-red-500/20 text-red-400'}
                                `}>
                                    {Math.round(status.qa_score * 100)}%
                                </div>
                                <div>
                                    <p className="text-sm font-medium text-text-primary">
                                        Translation QA Score
                                    </p>
                                    <p className="text-xs text-text-muted">
                                        {status.qa_score >= 0.7 ? 'Good match with reference subtitles' :
                                            status.qa_score >= 0.4 ? 'Partial match — some segments may differ' :
                                                'Low match — auto-corrected using reference subs'}
                                    </p>
                                </div>
                            </div>
                            <button
                                onClick={async () => {
                                    if (!qaReport) {
                                        try {
                                            const res = await fetch(`${API_BASE}/api/jobs/${jobId}/qa`);
                                            if (res.ok) {
                                                const data = await res.json();
                                                setQaReport(data.report);
                                            }
                                        } catch { }
                                    }
                                    setQaOpen(!qaOpen);
                                }}
                                className="text-xs text-primary hover:text-primary-light transition-colors"
                            >
                                {qaOpen ? 'Hide Report' : 'View Report'}
                            </button>
                        </div>
                        {qaOpen && qaReport && (
                            <pre className="mt-3 p-3 rounded-lg bg-black/30 text-xs text-text-secondary overflow-x-auto max-h-80 overflow-y-auto whitespace-pre-wrap font-mono">
                                {qaReport}
                            </pre>
                        )}
                    </div>
                )}

                {/* Word Count Card — populated by _pretts_word_budget after TTS finishes */}
                {status && (status.total_words ?? 0) > 0 && (
                    <div className="glass-card p-4">
                        <div className="flex items-center gap-3 mb-3">
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-primary-light">
                                <path d="M17 6.1H3" />
                                <path d="M21 12.1H3" />
                                <path d="M15.1 18H3" />
                            </svg>
                            <p className="text-sm font-medium text-text-primary">Word Count</p>
                        </div>
                        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                            <div>
                                <p className="text-2xl font-bold text-primary-light tabular-nums">
                                    {(status.total_words ?? 0).toLocaleString()}
                                </p>
                                <p className="text-[10px] text-text-muted uppercase tracking-wide mt-0.5">
                                    Total Words
                                </p>
                            </div>
                            <div>
                                <p className="text-2xl font-bold text-text-primary tabular-nums">
                                    {(status.total_sentences ?? 0).toLocaleString()}
                                </p>
                                <p className="text-[10px] text-text-muted uppercase tracking-wide mt-0.5">
                                    Sentences
                                </p>
                            </div>
                            <div>
                                <p className="text-2xl font-bold text-text-primary tabular-nums">
                                    {(status.avg_words_per_sent ?? 0).toFixed(1)}
                                </p>
                                <p className="text-[10px] text-text-muted uppercase tracking-wide mt-0.5">
                                    Avg / Sentence
                                </p>
                            </div>
                            <div>
                                <p className="text-2xl font-bold text-text-primary tabular-nums">
                                    {status.max_sent_words ?? 0}
                                </p>
                                <p className="text-[10px] text-text-muted uppercase tracking-wide mt-0.5">
                                    Longest Sentence
                                </p>
                            </div>
                        </div>
                        {(status.max_seg_words ?? 0) > 0 && (
                            <p className="text-[10px] text-text-muted mt-3">
                                Largest segment: <span className="font-mono text-text-secondary">{status.max_seg_words} words</span>
                                {' · '}
                                Counted before TTS so the truncation guard knows what to expect.
                            </p>
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
                        {/* Saved Location */}
                        {status?.saved_folder && (
                            <div className="glass-card p-4 flex items-center gap-3">
                                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-green-400 flex-shrink-0">
                                    <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
                                    <path d="m9 15 3-3 3 3" /><path d="M12 12v6" />
                                </svg>
                                <div className="flex-1 min-w-0">
                                    <p className="text-xs text-text-muted mb-0.5">Saved to</p>
                                    <p className="text-sm text-text-primary font-mono truncate">{status.saved_folder}</p>
                                </div>
                            </div>
                        )}

                        {/* YouTube Description */}
                        {status?.description && (
                            <div className="glass-card p-5 space-y-3">
                                <div className="flex items-center justify-between">
                                    <h3 className="text-sm font-semibold text-text-primary flex items-center gap-2">
                                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                                            <polyline points="14 2 14 8 20 8" />
                                            <line x1="16" x2="8" y1="13" y2="13" />
                                            <line x1="16" x2="8" y1="17" y2="17" />
                                        </svg>
                                        YouTube Description
                                    </h3>
                                    <button
                                        onClick={() => {
                                            navigator.clipboard.writeText(status.description || '');
                                        }}
                                        className="text-xs px-3 py-1.5 rounded-lg bg-primary/10 text-primary hover:bg-primary/20 transition-colors flex items-center gap-1.5"
                                    >
                                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                            <rect width="14" height="14" x="8" y="8" rx="2" ry="2" />
                                            <path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2" />
                                        </svg>
                                        Copy
                                    </button>
                                </div>
                                <pre className="text-xs text-text-secondary whitespace-pre-wrap leading-relaxed bg-bg/50 rounded-lg p-3 max-h-60 overflow-y-auto">
                                    {status.description}
                                </pre>
                            </div>
                        )}

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

                {/* Step-by-step review panel */}
                {isReviewing && (
                    <div className="space-y-4">
                        <div className="p-4 rounded-xl bg-yellow-400/10 border border-yellow-400/30">
                            <div className="flex items-center justify-between mb-3">
                                <h3 className="text-lg font-semibold text-yellow-400">
                                    Review {reviewStep === 'transcription' ? 'Transcription' : 'Translation'}
                                </h3>
                                <button
                                    type="button"
                                    onClick={async () => {
                                        setContinuing(true);
                                        try {
                                            await continueJob(jobId);
                                        } catch {
                                            alert('Failed to continue job');
                                        }
                                        setContinuing(false);
                                    }}
                                    disabled={continuing}
                                    className="px-4 py-2 rounded-lg bg-primary text-white font-medium hover:bg-primary/80 transition-colors disabled:opacity-50"
                                >
                                    {continuing ? 'Continuing...' : `Continue to ${reviewStep === 'transcription' ? 'Translation' : 'TTS Synthesis'}`}
                                </button>
                            </div>
                            <p className="text-sm text-text-muted mb-3">
                                {reviewStep === 'transcription'
                                    ? 'Review the transcribed text below. If it looks good, click Continue to proceed to translation.'
                                    : 'Review the translated Hindi text below. If it looks good, click Continue to proceed to TTS synthesis.'}
                            </p>
                        </div>
                        <TranscriptViewer jobId={jobId} targetLanguage={status?.target_language} />
                    </div>
                )}

                {/* Loading state */}
                {!isComplete && !isError && !isWaitingForSrt && !isReviewing && (
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
