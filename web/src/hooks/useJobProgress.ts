'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import { subscribeToJobEvents, getJob, type SSEEvent, type JobStatus } from '@/lib/api';

interface JobProgress {
    status: JobStatus | null;
    step: string;
    stepProgress: number;
    overallProgress: number;
    message: string;
    isComplete: boolean;
    isError: boolean;
    isWaitingForSrt: boolean;
    isReviewing: boolean;
    reviewStep: string;
    error: string | null;
    eta: string;
    restart: () => void;
    // Follow a new run of a finished job (dialogue re-voice): polling only.
    followNewRun: () => void;
}

// After a re-voice is accepted the job may still read done/error for a moment
// before its new run starts: those reads are skipped for this long.
const NEW_RUN_GRACE_MS = 20000;

function formatEta(seconds: number): string {
    if (seconds <= 0 || !isFinite(seconds)) return '';
    if (seconds < 60) return `~${Math.ceil(seconds)}s remaining`;
    if (seconds < 3600) return `~${Math.ceil(seconds / 60)}m remaining`;
    const h = Math.floor(seconds / 3600);
    const m = Math.ceil((seconds % 3600) / 60);
    return `~${h}h ${m}m remaining`;
}

export function useJobProgress(jobId: string | null): JobProgress {
    const [status, setStatus] = useState<JobStatus | null>(null);
    const [step, setStep] = useState('');
    const [stepProgress, setStepProgress] = useState(0);
    const [overallProgress, setOverallProgress] = useState(0);
    const [message, setMessage] = useState('');
    const [isComplete, setIsComplete] = useState(false);
    const [isError, setIsError] = useState(false);
    const [isWaitingForSrt, setIsWaitingForSrt] = useState(false);
    const [isReviewing, setIsReviewing] = useState(false);
    const [reviewStep, setReviewStep] = useState('');
    const [error, setError] = useState<string | null>(null);
    const [eta, setEta] = useState('');
    const unsubRef = useRef<(() => void) | null>(null);
    const pollRef = useRef<NodeJS.Timeout | null>(null);
    const startTimeRef = useRef<number>(Date.now());
    const lastProgressRef = useRef<number>(0);
    // followNewRun(): skip terminal states until this time (0 = off), and time
    // the ETA from the new run's start instead of the job's creation.
    const newRunGraceRef = useRef<number>(0);
    const newRunRef = useRef(false);

    const updateEta = useCallback((progress: number, jobCreatedAt?: number) => {
        if (progress > 0.01 && progress < 1) {
            // Use job's actual start time if available, otherwise page load time
            const startMs = jobCreatedAt ? jobCreatedAt * 1000 : startTimeRef.current;
            const elapsed = (Date.now() - startMs) / 1000;
            if (elapsed > 2 && progress > lastProgressRef.current) {
                const rate = progress / elapsed;
                const remaining = (1 - progress) / rate;
                setEta(formatEta(remaining));
                lastProgressRef.current = progress;
            }
        } else {
            setEta('');
        }
    }, []);

    const startPolling = useCallback((id: string) => {
        if (pollRef.current) clearInterval(pollRef.current);
        pollRef.current = setInterval(async () => {
            try {
                const job = await getJob(id);
                const terminal = job.state === 'done' || job.state === 'error';
                if (newRunGraceRef.current) {
                    if (terminal && Date.now() < newRunGraceRef.current) return;   // new run not started yet
                    newRunGraceRef.current = 0;
                }
                setStatus(job);
                setStep(job.current_step);
                setStepProgress(job.step_progress);
                setOverallProgress(job.overall_progress);
                setMessage(job.message);
                updateEta(job.overall_progress, newRunRef.current ? undefined : job.created_at);
                if (job.state !== 'review_transcription' && job.state !== 'review_translation') {
                    // Continued after a review: the review panel must go away.
                    setIsReviewing(false);
                }
                if (job.state === 'done') {
                    setIsComplete(true);
                    setIsError(false);
                    setIsWaitingForSrt(false);
                    setOverallProgress(1);
                    setEta('');
                    if (pollRef.current) clearInterval(pollRef.current);
                } else if (job.state === 'error') {
                    setIsError(true);
                    setIsComplete(false);
                    setIsWaitingForSrt(false);
                    setError(job.error || 'Unknown error');
                    setEta('');
                    if (pollRef.current) clearInterval(pollRef.current);
                } else if (job.state === 'waiting_for_srt') {
                    setIsWaitingForSrt(true);
                    setIsComplete(false);
                    setIsError(false);
                    setOverallProgress(0.4);
                    setEta('');
                    if (pollRef.current) clearInterval(pollRef.current);
                } else if (job.state === 'review_transcription' || job.state === 'review_translation') {
                    setIsReviewing(true);
                    setReviewStep(job.state === 'review_transcription' ? 'transcription' : 'translation');
                    setIsComplete(false);
                    setIsError(false);
                }
            } catch {
                // Keep polling on transient errors
            }
        }, 2000);
    }, [updateEta]);

    useEffect(() => {
        if (!jobId) return;

        // Reset state
        setStatus(null);
        setStep('');
        setStepProgress(0);
        setOverallProgress(0);
        setMessage('Starting...');
        setIsComplete(false);
        setIsError(false);
        setIsWaitingForSrt(false);
        setIsReviewing(false);
        setError(null);
        setEta('');
        startTimeRef.current = Date.now();
        lastProgressRef.current = 0;
        newRunRef.current = false;
        newRunGraceRef.current = 0;

        // Always start polling for reliable progress updates
        startPolling(jobId);

        // Also try SSE for faster updates
        const unsub = subscribeToJobEvents(
            jobId,
            (event: SSEEvent) => {
                if (event.type === 'complete') {
                    if (event.state === 'done') {
                        setIsComplete(true);
                        setIsError(false);
                        setIsWaitingForSrt(false);
                        setOverallProgress(1);
                        setMessage('Complete');
                        setEta('');
                    } else if (event.state === 'error') {
                        setIsError(true);
                        setIsComplete(false);
                        setIsWaitingForSrt(false);
                        setError(event.error || 'Unknown error');
                        setEta('');
                    } else if (event.state === 'waiting_for_srt') {
                        setIsWaitingForSrt(true);
                        setIsComplete(false);
                        setIsError(false);
                        setOverallProgress(0.4);
                        setMessage('Transcription complete. Download SRT and upload translation.');
                        setEta('');
                    }
                    if (pollRef.current) clearInterval(pollRef.current);
                    getJob(jobId).then(setStatus).catch(() => {});
                    return;
                }
                if (event.step) setStep(event.step);
                if (event.progress !== undefined) setStepProgress(event.progress);
                if (event.overall !== undefined) {
                    setOverallProgress(event.overall);
                    updateEta(event.overall);
                }
                if (event.message) setMessage(event.message);
            },
            () => {
                // SSE failed — polling is already running as backup
            },
        );
        unsubRef.current = unsub;

        return () => {
            if (unsubRef.current) unsubRef.current();
            if (pollRef.current) clearInterval(pollRef.current);
        };
    }, [jobId, startPolling, updateEta]);

    const restart = useCallback(() => {
        if (!jobId) return;
        // Clean up previous SSE/polling before starting new one
        if (unsubRef.current) unsubRef.current();
        if (pollRef.current) clearInterval(pollRef.current);

        setIsComplete(false);
        setIsError(false);
        setIsWaitingForSrt(false);
        setError(null);
        setStep('');
        setStepProgress(0);
        setOverallProgress(0);
        setMessage('Resuming...');
        setEta('');
        startTimeRef.current = Date.now();
        lastProgressRef.current = 0;
        newRunGraceRef.current = 0;

        // Start polling as reliable fallback (same as initial setup)
        startPolling(jobId);

        const unsub = subscribeToJobEvents(
            jobId,
            (event: SSEEvent) => {
                if (event.type === 'complete') {
                    if (event.state === 'done') {
                        setIsComplete(true);
                        setIsError(false);
                        setIsWaitingForSrt(false);
                        setOverallProgress(1);
                        setMessage('Complete');
                    } else if (event.state === 'error') {
                        setIsError(true);
                        setIsComplete(false);
                        setIsWaitingForSrt(false);
                        setError(event.error || 'Unknown error');
                    } else if (event.state === 'waiting_for_srt') {
                        setIsWaitingForSrt(true);
                        setIsComplete(false);
                        setIsError(false);
                        setOverallProgress(0.4);
                        setMessage('Transcription complete. Download SRT and upload translation.');
                    }
                    setEta('');
                    if (pollRef.current) clearInterval(pollRef.current);
                    getJob(jobId).then(setStatus).catch(() => {});
                    return;
                }
                if (event.step) setStep(event.step);
                if (event.progress !== undefined) setStepProgress(event.progress);
                if (event.overall !== undefined) {
                    setOverallProgress(event.overall);
                    updateEta(event.overall);
                }
                if (event.message) setMessage(event.message);
            },
            () => {
                // SSE failed — polling is already running as backup
            },
        );
        unsubRef.current = unsub;
    }, [jobId, startPolling, updateEta]);

    const followNewRun = useCallback(() => {
        if (!jobId) return;
        // The finished run's SSE stream already ended; its events may still
        // hold the old "complete", so the new run is followed by polling.
        if (unsubRef.current) unsubRef.current();
        unsubRef.current = null;
        setIsComplete(false);
        setIsError(false);
        setIsWaitingForSrt(false);
        setIsReviewing(false);
        setError(null);
        setStepProgress(0);
        setOverallProgress(0);
        setMessage('Starting the re-voice...');
        setEta('');
        startTimeRef.current = Date.now();
        lastProgressRef.current = 0;
        newRunRef.current = true;
        newRunGraceRef.current = Date.now() + NEW_RUN_GRACE_MS;
        startPolling(jobId);
    }, [jobId, startPolling]);

    return { status, step, stepProgress, overallProgress, message, isComplete, isError, isWaitingForSrt, isReviewing, reviewStep, error, eta, restart, followNewRun };
}
