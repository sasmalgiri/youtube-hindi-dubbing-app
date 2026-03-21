'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import {
    createJob,
    getJob,
    localDownloadAndDub,
    isRemoteBackend,
    subscribeToJobEvents,
    resultVideoUrl,
    type JobCreateRequest,
    type SSEEvent,
} from '@/lib/api';

const MAX_CONCURRENT = 3;

export type BatchItemState = 'pending' | 'creating' | 'running' | 'done' | 'error';

export interface BatchItem {
    url: string;
    jobId: string | null;
    state: BatchItemState;
    progress: number;
    step: string;
    message: string;
    error: string | null;
    videoTitle: string | null;
    downloaded: boolean;
}

export interface BatchSettings {
    source_language?: string;
    target_language?: string;
    asr_model?: string;
    translation_engine?: string;
    tts_rate?: string;
    mix_original?: boolean;
    original_volume?: number;
    use_chatterbox?: boolean;
    use_elevenlabs?: boolean;
    use_google_tts?: boolean;
    use_coqui_xtts?: boolean;
    use_edge_tts?: boolean;
    audio_priority?: boolean;
    audio_bitrate?: string;
    encode_preset?: string;
    prefer_youtube_subs?: boolean;
    multi_speaker?: boolean;
    transcribe_only?: boolean;
}

interface UseBatchManagerReturn {
    items: BatchItem[];
    isRunning: boolean;
    completedCount: number;
    overallProgress: number;
    autoDownload: boolean;
    setAutoDownload: (v: boolean) => void;
    start: (urls: string[], settings: BatchSettings) => void;
}

export function useBatchManager(): UseBatchManagerReturn {
    const [items, setItems] = useState<BatchItem[]>([]);
    const [autoDownload, setAutoDownload] = useState(true);
    const unsubscribesRef = useRef<Map<string, () => void>>(new Map());
    const startedRef = useRef(false);
    const launchingRef = useRef<Set<number>>(new Set());

    // Load from sessionStorage on mount
    useEffect(() => {
        const stored = sessionStorage.getItem('batch_state');
        if (stored) {
            try {
                const parsed = JSON.parse(stored);
                if (parsed.items) setItems(parsed.items);
                if (parsed.autoDownload !== undefined) setAutoDownload(parsed.autoDownload);
            } catch { }
        }
    }, []);

    // Save to sessionStorage on change
    useEffect(() => {
        if (items.length > 0) {
            sessionStorage.setItem('batch_state', JSON.stringify({ items, autoDownload }));
        }
    }, [items, autoDownload]);

    // Cleanup SSE subscriptions
    useEffect(() => {
        return () => {
            unsubscribesRef.current.forEach((unsub) => unsub());
            unsubscribesRef.current.clear();
        };
    }, []);

    const updateItem = useCallback((index: number, updates: Partial<BatchItem>) => {
        setItems((prev) => {
            const next = [...prev];
            next[index] = { ...next[index], ...updates };
            return next;
        });
    }, []);

    const triggerDownload = useCallback((jobId: string, title: string | null) => {
        const a = document.createElement('a');
        a.href = resultVideoUrl(jobId);
        a.download = title ? `${title}_dubbed.mp4` : `dubbed_${jobId}.mp4`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    }, []);

    const subscribeToJob = useCallback((index: number, jobId: string) => {
        // Don't double-subscribe
        if (unsubscribesRef.current.has(jobId)) return;

        const unsub = subscribeToJobEvents(
            jobId,
            (event: SSEEvent) => {
                setItems((prev) => {
                    const item = prev[index];
                    if (!item || item.jobId !== jobId) return prev;

                    const next = [...prev];
                    const updates: Partial<BatchItem> = {};

                    if (event.step) updates.step = event.step;
                    if (event.message) updates.message = event.message;
                    if (event.overall !== undefined) updates.progress = event.overall * 100;

                    if (event.state === 'error' || event.type === 'error') {
                        updates.state = 'error';
                        updates.error = event.error || 'Job failed';
                        unsubscribesRef.current.get(jobId)?.();
                        unsubscribesRef.current.delete(jobId);
                    } else if (event.state === 'waiting_for_srt') {
                        updates.state = 'done'; // Free the queue slot
                        updates.step = 'Awaiting SRT';
                        updates.message = 'Transcription complete. Open job page to upload SRT.';
                        unsubscribesRef.current.get(jobId)?.();
                        unsubscribesRef.current.delete(jobId);
                    } else if (event.type === 'complete' || event.state === 'done') {
                        updates.state = 'done';
                        updates.progress = 100;
                        unsubscribesRef.current.get(jobId)?.();
                        unsubscribesRef.current.delete(jobId);
                        // Fetch video title for proper download filename
                        getJob(jobId).then((job) => {
                            if (job.video_title) {
                                updateItem(index, { videoTitle: job.video_title });
                            }
                        }).catch(() => {});
                    }

                    next[index] = { ...item, ...updates };
                    return next;
                });
            },
            () => {
                // SSE connection error — guard against stale index from prior batch
                setItems((prev) => {
                    const item = prev[index];
                    if (!item || item.jobId !== jobId) return prev;
                    const next = [...prev];
                    next[index] = { ...item, state: 'error', error: 'Connection lost' };
                    return next;
                });
                unsubscribesRef.current.delete(jobId);
            },
        );

        unsubscribesRef.current.set(jobId, unsub);
    }, [updateItem]);

    // Auto-download effect
    useEffect(() => {
        if (!autoDownload) return;
        const toDownload = items.filter((item) => item.state === 'done' && !item.downloaded && item.jobId && item.step !== 'Awaiting SRT');
        if (toDownload.length === 0) return;
        toDownload.forEach((item) => triggerDownload(item.jobId!, item.videoTitle));
        setItems((prev) => {
            const downloadedIds = new Set(toDownload.map((i) => i.jobId));
            return prev.map((item) =>
                downloadedIds.has(item.jobId) ? { ...item, downloaded: true } : item
            );
        });
    }, [items, autoDownload, triggerDownload]);

    // Process queue — start next jobs when slots are available
    useEffect(() => {
        if (!startedRef.current || items.length === 0) return;

        const runningCount = items.filter((i) => i.state === 'creating' || i.state === 'running').length
            + launchingRef.current.size;
        if (runningCount >= MAX_CONCURRENT) return;

        const slotsAvailable = MAX_CONCURRENT - runningCount;
        const pendingIndices = items
            .map((item, idx) => (item.state === 'pending' && !launchingRef.current.has(idx) ? idx : -1))
            .filter((idx) => idx >= 0);

        const toStart = pendingIndices.slice(0, slotsAvailable);
        if (toStart.length === 0) return;

        // Mark indices as launching to prevent double-creation on effect re-fire
        toStart.forEach((idx) => launchingRef.current.add(idx));

        // Create jobs for each pending item
        toStart.forEach(async (index) => {
            const item = items[index];
            updateItem(index, { state: 'creating', message: 'Creating job...' });

            try {
                const storedSettings = sessionStorage.getItem('batch_settings');
                const settings: BatchSettings = storedSettings ? JSON.parse(storedSettings) : {};

                const jobReq: Omit<JobCreateRequest, 'url'> & { url?: string } = {
                    source_language: settings.source_language,
                    target_language: settings.target_language,
                    asr_model: settings.asr_model,
                    translation_engine: settings.translation_engine,
                    tts_rate: settings.tts_rate,
                    mix_original: settings.mix_original,
                    original_volume: settings.original_volume,
                    use_chatterbox: settings.use_chatterbox,
                    use_elevenlabs: settings.use_elevenlabs,
                    use_google_tts: settings.use_google_tts,
                    use_coqui_xtts: settings.use_coqui_xtts,
                    use_edge_tts: settings.use_edge_tts,
                    audio_priority: settings.audio_priority,
                    audio_bitrate: settings.audio_bitrate,
                    encode_preset: settings.encode_preset,
                    prefer_youtube_subs: settings.prefer_youtube_subs,
                    multi_speaker: settings.multi_speaker,
                    transcribe_only: settings.transcribe_only,
                };

                const { id } = isRemoteBackend
                    ? await localDownloadAndDub(item.url, jobReq)
                    : await createJob({ url: item.url, ...jobReq });

                launchingRef.current.delete(index);
                updateItem(index, {
                    jobId: id,
                    state: 'running',
                    message: 'Job started...',
                });

                // Subscribe to SSE events
                subscribeToJob(index, id);
            } catch (e) {
                launchingRef.current.delete(index);
                updateItem(index, {
                    state: 'error',
                    error: e instanceof Error ? e.message : 'Failed to create job',
                });
            }
        });
    }, [items, updateItem, subscribeToJob]);

    const start = useCallback((urls: string[], settings: BatchSettings) => {
        // Clean up prior batch's SSE subscriptions and launching state
        unsubscribesRef.current.forEach((unsub) => unsub());
        unsubscribesRef.current.clear();
        launchingRef.current.clear();

        sessionStorage.setItem('batch_settings', JSON.stringify(settings));

        const newItems: BatchItem[] = urls.map((url) => ({
            url,
            jobId: null,
            state: 'pending' as const,
            progress: 0,
            step: '',
            message: 'Waiting...',
            error: null,
            videoTitle: null,
            downloaded: false,
        }));

        setItems(newItems);
        startedRef.current = true;
    }, []);

    const isRunning = items.some((i) => i.state === 'pending' || i.state === 'creating' || i.state === 'running');
    const completedCount = items.filter((i) => i.state === 'done').length;
    const overallProgress = items.length > 0
        ? Math.round(items.reduce((sum, i) => sum + i.progress, 0) / items.length)
        : 0;

    return {
        items,
        isRunning,
        completedCount,
        overallProgress,
        autoDownload,
        setAutoDownload,
        start,
    };
}
