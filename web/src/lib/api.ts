// Set NEXT_PUBLIC_API_URL to your backend URL when running remotely
// e.g., NEXT_PUBLIC_API_URL=https://abc-123.ngrok-free.app
// For desktop app (static export), defaults to same origin (served by FastAPI)
const API_BASE = process.env.NEXT_PUBLIC_API_URL || '';  // empty = same origin

// ngrok free tier requires this header to skip the interstitial warning page
const EXTRA_HEADERS: Record<string, string> = API_BASE.includes('ngrok')
    ? { 'ngrok-skip-browser-warning': 'true' }
    : {};

// ── Types ───────────────────────────────────────────────────────────────────

export interface Voice {
    ShortName: string;
    Gender: string;
    Locale: string;
    FriendlyName?: string;
}

export interface JobCreateRequest {
    url: string;
    source_language?: string;
    target_language?: string;
    voice?: string;
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
    prefer_youtube_subs?: boolean;
    multi_speaker?: boolean;
    transcribe_only?: boolean;
    audio_priority?: boolean;
    audio_bitrate?: string;
    encode_preset?: string;
    dub_chain?: string[];
}

export interface JobConfig {
    asr_model?: string;
    translation_engine?: string;
    tts_engine?: string;
    audio_priority?: boolean;
    audio_bitrate?: string;
    encode_preset?: string;
}

export interface JobStatus {
    id: string;
    state: 'queued' | 'running' | 'done' | 'error' | 'waiting_for_srt';
    current_step: string;
    step_progress: number;
    overall_progress: number;
    message: string;
    error?: string | null;
    source_url: string;
    video_title: string;
    target_language: string;
    created_at: number;
    config?: JobConfig;
    saved_folder?: string | null;
    saved_video?: string | null;
    description?: string | null;
    qa_score?: number | null;
}

export interface TranscriptSegment {
    start: number;
    end: number;
    text: string;
    text_translated: string;
}

export interface Transcript {
    segments: TranscriptSegment[];
}

export interface SSEEvent {
    step?: string;
    progress?: number;
    overall?: number;
    message?: string;
    type?: string;
    state?: string;
    error?: string;
}

// ── API Functions ───────────────────────────────────────────────────────────

export async function fetchVoices(lang: string = 'hi'): Promise<Voice[]> {
    const res = await fetch(`${API_BASE}/api/voices?lang=${lang}`, {
        headers: { ...EXTRA_HEADERS },
    });
    if (!res.ok) throw new Error('Failed to fetch voices');
    return res.json();
}

export async function createJob(req: JobCreateRequest): Promise<{ id: string }> {
    const res = await fetch(`${API_BASE}/api/jobs`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...EXTRA_HEADERS },
        body: JSON.stringify(req),
    });
    if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Request failed' }));
        throw new Error(err.detail || 'Failed to create job');
    }
    return res.json();
}

export async function createJobUpload(
    file: File,
    settings: Omit<JobCreateRequest, 'url'>,
): Promise<{ id: string }> {
    const form = new FormData();
    form.append('file', file);
    Object.entries(settings).forEach(([key, val]) => {
        if (val !== undefined) form.append(key, String(val));
    });

    const res = await fetch(`${API_BASE}/api/jobs/upload`, {
        method: 'POST',
        headers: { ...EXTRA_HEADERS },
        body: form,
    });
    if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Upload failed' }));
        throw new Error(err.detail || 'Failed to upload and create job');
    }
    return res.json();
}

export async function getJob(id: string): Promise<JobStatus> {
    const res = await fetch(`${API_BASE}/api/jobs/${id}`, {
        headers: { ...EXTRA_HEADERS },
    });
    if (!res.ok) throw new Error('Failed to fetch job');
    return res.json();
}

export async function getJobs(): Promise<JobStatus[]> {
    const res = await fetch(`${API_BASE}/api/jobs`, {
        headers: { ...EXTRA_HEADERS },
    });
    if (!res.ok) throw new Error('Failed to fetch jobs');
    return res.json();
}

export async function getTranscript(id: string): Promise<Transcript> {
    const res = await fetch(`${API_BASE}/api/jobs/${id}/transcript`, {
        headers: { ...EXTRA_HEADERS },
    });
    if (!res.ok) throw new Error('Failed to fetch transcript');
    return res.json();
}

export async function deleteJob(id: string): Promise<void> {
    const res = await fetch(`${API_BASE}/api/jobs/${id}`, {
        method: 'DELETE',
        headers: { ...EXTRA_HEADERS },
    });
    if (!res.ok) throw new Error('Failed to delete job');
}

export function resultVideoUrl(id: string): string {
    return `${API_BASE}/api/jobs/${id}/result`;
}

export function originalVideoUrl(id: string): string {
    return `${API_BASE}/api/jobs/${id}/original`;
}

export function resultSrtUrl(id: string): string {
    return `${API_BASE}/api/jobs/${id}/srt`;
}

export function sourceSrtUrl(id: string): string {
    return `${API_BASE}/api/jobs/${id}/source-srt`;
}

export async function uploadTranslatedSrt(id: string, file: File): Promise<{ id: string; state: string }> {
    const form = new FormData();
    form.append('file', file);
    const res = await fetch(`${API_BASE}/api/jobs/${id}/resume-with-srt`, {
        method: 'POST',
        headers: { ...EXTRA_HEADERS },
        body: form,
    });
    if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Upload failed' }));
        throw new Error(err.detail || 'Failed to upload translated SRT');
    }
    return res.json();
}

// ── Saved Links (persistent) ─────────────────────────────────────────────────

export interface SavedLink {
    id: string;
    url: string;
    title: string;
    added_at: number;
}

export async function getLinks(): Promise<SavedLink[]> {
    const res = await fetch(`${API_BASE}/api/links`, { headers: { ...EXTRA_HEADERS } });
    if (!res.ok) return [];
    return res.json();
}

export async function addLink(url: string, title?: string): Promise<SavedLink[]> {
    const res = await fetch(`${API_BASE}/api/links`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...EXTRA_HEADERS },
        body: JSON.stringify({ url, title }),
    });
    if (!res.ok) return [];
    const data = await res.json();
    return data.links || [];
}

export async function deleteLink(id: string): Promise<SavedLink[]> {
    const res = await fetch(`${API_BASE}/api/links/${id}`, {
        method: 'DELETE',
        headers: { ...EXTRA_HEADERS },
    });
    if (!res.ok) return [];
    const data = await res.json();
    return data.links || [];
}

// ── Local Download + Upload (for remote backend) ────────────────────────────

export const isRemoteBackend = !!API_BASE;

export async function localDownloadAndDub(
    url: string,
    settings: Omit<JobCreateRequest, 'url'>,
): Promise<{ id: string }> {
    const res = await fetch('/api/local-dub', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url, ...settings }),
    });
    if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Local download failed' }));
        throw new Error(err.detail || 'Failed to download and dub');
    }
    return res.json();
}

// ── SSE Helper (with ngrok header support) ──────────────────────────────────

export function subscribeToJobEvents(
    jobId: string,
    onEvent: (event: SSEEvent) => void,
    onError?: (error: Error) => void,
): () => void {
    // EventSource can't send custom headers, so use fetch-based SSE
    // for ngrok compatibility
    const controller = new AbortController();
    let stopped = false;

    (async () => {
        try {
            const res = await fetch(`${API_BASE}/api/jobs/${jobId}/events`, {
                headers: {
                    'Accept': 'text/event-stream',
                    ...EXTRA_HEADERS,
                },
                signal: controller.signal,
            });

            if (!res.ok || !res.body) {
                onError?.(new Error('SSE connection failed'));
                return;
            }

            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (!stopped) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';

                for (const line of lines) {
                    if (line.startsWith('data:')) {
                        const jsonStr = line.slice(5).trim();
                        if (!jsonStr) continue;
                        try {
                            const data: SSEEvent = JSON.parse(jsonStr);
                            onEvent(data);
                            if (data.type === 'complete') {
                                stopped = true;
                                return;
                            }
                        } catch {
                            // ignore parse errors
                        }
                    }
                }
            }
        } catch (err: any) {
            if (!stopped && err.name !== 'AbortError') {
                onError?.(new Error('SSE connection lost'));
            }
        }
    })();

    return () => {
        stopped = true;
        controller.abort();
    };
}
