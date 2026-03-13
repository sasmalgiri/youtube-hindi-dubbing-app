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
    error: string | null;
    restart: () => void;
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
    const [error, setError] = useState<string | null>(null);
    const unsubRef = useRef<(() => void) | null>(null);
    const pollRef = useRef<NodeJS.Timeout | null>(null);

    const startPolling = useCallback((id: string) => {
        if (pollRef.current) clearInterval(pollRef.current);
        pollRef.current = setInterval(async () => {
            try {
                const job = await getJob(id);
                setStatus(job);
                setStep(job.current_step);
                setStepProgress(job.step_progress);
                setOverallProgress(job.overall_progress);
                setMessage(job.message);
                if (job.state === 'done') {
                    setIsComplete(true);
                    setOverallProgress(1);
                    if (pollRef.current) clearInterval(pollRef.current);
                } else if (job.state === 'error') {
                    setIsError(true);
                    setError(job.error || 'Unknown error');
                    if (pollRef.current) clearInterval(pollRef.current);
                } else if (job.state === 'waiting_for_srt') {
                    setIsWaitingForSrt(true);
                    setOverallProgress(1);
                    if (pollRef.current) clearInterval(pollRef.current);
                }
            } catch {
                // Keep polling on transient errors
            }
        }, 2000);
    }, []);

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
        setError(null);

        // Try SSE first
        const unsub = subscribeToJobEvents(
            jobId,
            (event: SSEEvent) => {
                if (event.type === 'complete') {
                    if (event.state === 'done') {
                        setIsComplete(true);
                        setOverallProgress(1);
                        setMessage('Complete');
                    } else if (event.state === 'error') {
                        setIsError(true);
                        setError(event.error || 'Unknown error');
                    } else if (event.state === 'waiting_for_srt') {
                        setIsWaitingForSrt(true);
                        setOverallProgress(1);
                        setMessage('Transcription complete. Download SRT and upload translation.');
                    }
                    // Fetch final status
                    getJob(jobId).then(setStatus).catch(() => {});
                    return;
                }
                if (event.step) setStep(event.step);
                if (event.progress !== undefined) setStepProgress(event.progress);
                if (event.overall !== undefined) setOverallProgress(event.overall);
                if (event.message) setMessage(event.message);
            },
            () => {
                // SSE failed, fall back to polling
                startPolling(jobId);
            },
        );
        unsubRef.current = unsub;

        return () => {
            unsub();
            if (pollRef.current) clearInterval(pollRef.current);
        };
    }, [jobId, startPolling]);

    const restart = useCallback(() => {
        if (!jobId) return;
        setIsComplete(false);
        setIsError(false);
        setIsWaitingForSrt(false);
        setError(null);
        setStep('');
        setStepProgress(0);
        setOverallProgress(0);
        setMessage('Resuming...');

        const unsub = subscribeToJobEvents(
            jobId,
            (event: SSEEvent) => {
                if (event.type === 'complete') {
                    if (event.state === 'done') {
                        setIsComplete(true);
                        setOverallProgress(1);
                        setMessage('Complete');
                    } else if (event.state === 'error') {
                        setIsError(true);
                        setError(event.error || 'Unknown error');
                    }
                    getJob(jobId).then(setStatus).catch(() => {});
                    return;
                }
                if (event.step) setStep(event.step);
                if (event.progress !== undefined) setStepProgress(event.progress);
                if (event.overall !== undefined) setOverallProgress(event.overall);
                if (event.message) setMessage(event.message);
            },
            () => startPolling(jobId),
        );
        unsubRef.current = unsub;
    }, [jobId, startPolling]);

    return { status, step, stepProgress, overallProgress, message, isComplete, isError, isWaitingForSrt, error, restart };
}
