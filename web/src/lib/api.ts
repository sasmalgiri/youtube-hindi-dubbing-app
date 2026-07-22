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
    use_cosyvoice?: boolean;
    use_chatterbox?: boolean;
    use_indic_parler?: boolean;
    use_sarvam_bulbul?: boolean;
    video_slow_to_match?: boolean;
    use_elevenlabs?: boolean;
    use_google_tts?: boolean;
    use_coqui_xtts?: boolean;
    use_fish_speech?: boolean;
    use_edge_tts?: boolean;
    multi_speaker?: boolean;
    transcribe_only?: boolean;
    audio_priority?: boolean;
    audio_untouchable?: boolean;
    post_tts_level?: string;
    audio_quality_mode?: string;
    enable_sentence_gap?: boolean;
    enable_duration_fit?: boolean;
    audio_bitrate?: string;
    encode_preset?: string;
    download_mode?: string;
    split_duration?: number;
    dub_duration?: number;
    fast_assemble?: boolean;
    dub_chain?: string[];
    enable_manual_review?: boolean;
    use_whisperx?: boolean;
    whisper_gpu_fallback?: boolean;
    whisper_fallback_model?: string;
    simplify_english?: boolean;
    step_by_step?: boolean;
    use_new_pipeline?: boolean;
    enable_tts_verify_retry?: boolean;
    tts_truncation_threshold?: number;
    tts_word_match_verify?: boolean;
    tts_word_match_tolerance?: number;
    tts_word_match_model?: string;
    long_segment_trace?: boolean;
    long_segment_threshold_words?: number;
    tts_no_time_pressure?: boolean;
    tts_dynamic_workers?: boolean;
    tts_dynamic_min?: number;
    tts_dynamic_max?: number;
    tts_dynamic_start?: number;
    tts_rate_mode?: string;
    tts_rate_ceiling?: string;
    tts_rate_target_wpm?: number;
    keep_subject_english?: boolean;
    purge_on_new_url?: boolean;
    pipeline_mode?: string;
    srt_needs_translation?: boolean;
    // ── AV Sync Modules ──
    av_sync_mode?: string;
    max_audio_speedup?: number;
    min_video_speed?: number;
    slot_verify?: string;
    use_global_stretch?: boolean;
    global_stretch_speedup?: number;
    segmenter?: string;
    segmenter_buffer_pct?: number;
    max_sentences_per_cue?: number;
    tts_chunk_words?: number;
    gap_mode?: string;
    // ── SRT Direct mode ──
    sd_srt_content?: string;         // full SRT content (required for srtdub mode)
    sd_max_stretch?: number;         // 1.0 – 10.0
    sd_audio_speed?: number;         // post-TTS atempo (0.5-3.0, default 1.25)
    sd_vx_hflip?: boolean;           // horizontal mirror
    sd_vx_hue?: number;              // hue shift degrees (-30..+30)
    sd_vx_zoom?: number;             // zoom + crop back (1.0 = off, 1.05 = 5% zoom)
}

export interface JobConfig {
    // Core labels
    asr_model?: string;
    translation_engine?: string;
    tts_engine?: string;
    tts_rate?: string;
    audio_bitrate?: string;
    encode_preset?: string;
    post_tts_level?: string;
    audio_quality_mode?: string;
    download_mode?: string;
    split_duration?: number;
    dub_duration?: number;
    pipeline_mode?: string;
    source_language?: string;
    target_language?: string;
    voice?: string;
    original_volume?: number;
    // Boolean toggles
    audio_priority?: boolean;
    audio_untouchable?: boolean;
    video_slow_to_match?: boolean;
    mix_original?: boolean;
    multi_speaker?: boolean;
    fast_assemble?: boolean;
    enable_sentence_gap?: boolean;
    enable_duration_fit?: boolean;
    use_whisperx?: boolean;
    whisper_gpu_fallback?: boolean;
    whisper_fallback_model?: string;
    simplify_english?: boolean;
    enable_manual_review?: boolean;
    transcribe_only?: boolean;
    use_sarvam_bulbul?: boolean;
    enable_tts_verify_retry?: boolean;
    keep_subject_english?: boolean;
    tts_word_match_verify?: boolean;
    long_segment_trace?: boolean;
    tts_no_time_pressure?: boolean;
    tts_dynamic_workers?: boolean;
    purge_on_new_url?: boolean;
    step_by_step?: boolean;
    use_new_pipeline?: boolean;
    srt_needs_translation?: boolean;
    // Numeric thresholds
    tts_truncation_threshold?: number;
    tts_word_match_tolerance?: number;
    tts_word_match_model?: string;
    long_segment_threshold_words?: number;
    tts_dynamic_min?: number;
    tts_dynamic_max?: number;
    tts_dynamic_start?: number;
    tts_rate_mode?: string;
    tts_rate_ceiling?: string;
    tts_rate_target_wpm?: number;
    // ── Preset + AV-sync + segmenter (read by the job details page) ──
    preset_name?: string;
    av_sync_mode?: string;
    max_audio_speedup?: number;
    min_video_speed?: number;
    slot_verify?: string;
    use_global_stretch?: boolean;
    global_stretch_speedup?: number;
    segmenter?: string;
    segmenter_buffer_pct?: number;
    max_sentences_per_cue?: number;
    tts_chunk_words?: number;
    gap_mode?: string;
}

export interface JobStatus {
    id: string;
    state: 'queued' | 'running' | 'done' | 'error' | 'waiting_for_srt' | 'review_transcription' | 'review_translation';
    current_step: string;
    step_progress: number;
    overall_progress: number;
    step_times?: Record<string, number>;
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
    chain_languages?: string[];
    chain_parent_id?: string | null;
    // TTS budget metrics — populated after _pretts_word_budget runs
    total_words?: number;
    total_sentences?: number;
    avg_words_per_sent?: number;
    max_seg_words?: number;
    max_sent_words?: number;
}

export interface TranscriptSegment {
    start: number;
    end: number;
    text: string;
    text_original?: string;  // Original English before simplification
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
        if (val === undefined || val === null) return;
        if (typeof val === 'boolean') {
            form.append(key, val ? 'true' : 'false');
            return;
        }
        // Skip arrays (dub_chain) — not supported by upload endpoint
        if (Array.isArray(val)) return;
        form.append(key, String(val));
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

export async function createJobWithSrt(
    srtFile: File,
    settings: Omit<JobCreateRequest, 'url'>,
    videoSource?: { url?: string; file?: File },
): Promise<{ id: string }> {
    const form = new FormData();
    form.append('srt_file', srtFile);
    if (videoSource?.file) {
        form.append('video_file', videoSource.file);
    }
    if (videoSource?.url) {
        form.append('url', videoSource.url);
    }
    Object.entries(settings).forEach(([key, val]) => {
        if (val === undefined || val === null) return;
        if (typeof val === 'boolean') {
            form.append(key, val ? 'true' : 'false');
            return;
        }
        if (Array.isArray(val)) return;
        form.append(key, String(val));
    });

    const res = await fetch(`${API_BASE}/api/jobs/with-srt`, {
        method: 'POST',
        headers: { ...EXTRA_HEADERS },
        body: form,
    });
    if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Upload failed' }));
        throw new Error(err.detail || 'Failed to create job with SRT');
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

export interface CompareResult {
    engines: Record<string, Array<{ start: number; end: number; text: string; text_translated: string }>>;
    segment_count: number;
    available: Array<{ key: string; label: string }>;
}

export async function compareTranslations(id: string, engines: string[] = [], maxSegments: number = 10): Promise<CompareResult> {
    const res = await fetch(`${API_BASE}/api/jobs/${id}/compare-translations`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...EXTRA_HEADERS },
        body: JSON.stringify({ engines, max_segments: maxSegments }),
    });
    if (!res.ok) {
        const text = await res.text().catch(() => '');
        throw new Error(text || 'Failed to compare translations');
    }
    return res.json();
}

export async function continueJob(id: string): Promise<void> {
    const res = await fetch(`${API_BASE}/api/jobs/${id}/continue`, {
        method: 'POST',
        headers: { ...EXTRA_HEADERS },
    });
    if (!res.ok) throw new Error('Failed to continue job');
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

export interface LinkPreset {
    source_language?: string;
    target_language?: string;
    asr_model?: string;
    translation_engine?: string;
    tts_rate?: string;
    mix_original?: boolean;
    original_volume?: number;
    use_cosyvoice?: boolean;
    use_chatterbox?: boolean;
    use_indic_parler?: boolean;
    use_sarvam_bulbul?: boolean;
    video_slow_to_match?: boolean;
    use_elevenlabs?: boolean;
    use_google_tts?: boolean;
    use_coqui_xtts?: boolean;
    use_fish_speech?: boolean;
    use_edge_tts?: boolean;
    multi_speaker?: boolean;
    transcribe_only?: boolean;
    audio_priority?: boolean;
    audio_untouchable?: boolean;
    post_tts_level?: string;
    audio_quality_mode?: string;
    enable_sentence_gap?: boolean;
    enable_duration_fit?: boolean;
    audio_bitrate?: string;
    encode_preset?: string;
    download_mode?: string;
    split_duration?: number;
    dub_duration?: number;
    fast_assemble?: boolean;
    dub_chain?: string[];
    enable_manual_review?: boolean;
    use_whisperx?: boolean;
    whisper_gpu_fallback?: boolean;
    whisper_fallback_model?: string;
    simplify_english?: boolean;
    step_by_step?: boolean;
    use_new_pipeline?: boolean;
    enable_tts_verify_retry?: boolean;
    tts_truncation_threshold?: number;
    tts_word_match_verify?: boolean;
    tts_word_match_tolerance?: number;
    tts_word_match_model?: string;
    long_segment_trace?: boolean;
    long_segment_threshold_words?: number;
    tts_no_time_pressure?: boolean;
    tts_dynamic_workers?: boolean;
    tts_dynamic_min?: number;
    tts_dynamic_max?: number;
    tts_dynamic_start?: number;
    tts_rate_mode?: string;
    tts_rate_ceiling?: string;
    tts_rate_target_wpm?: number;
    keep_subject_english?: boolean;
    purge_on_new_url?: boolean;
    pipeline_mode?: string;
    srt_needs_translation?: boolean;
    // ── AV Sync + Segmenter ──
    av_sync_mode?: string;
    max_audio_speedup?: number;
    min_video_speed?: number;
    slot_verify?: string;
    use_global_stretch?: boolean;
    global_stretch_speedup?: number;
    segmenter?: string;
    segmenter_buffer_pct?: number;
    max_sentences_per_cue?: number;
    tts_chunk_words?: number;
    gap_mode?: string;
}

export interface SavedLink {
    id: string;
    url: string;
    title: string;
    added_at: number;
    preset?: LinkPreset;
    completed?: boolean;
}

export async function getLinks(): Promise<SavedLink[]> {
    const res = await fetch(`${API_BASE}/api/links`, { headers: { ...EXTRA_HEADERS } });
    if (!res.ok) return [];
    return res.json();
}

export async function addLink(url: string, title?: string, preset?: LinkPreset): Promise<SavedLink[]> {
    const res = await fetch(`${API_BASE}/api/links`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...EXTRA_HEADERS },
        body: JSON.stringify({ url, title, preset }),
    });
    if (!res.ok) return [];
    const data = await res.json();
    return data.links || [];
}

export async function updateLinkPreset(id: string, preset: LinkPreset): Promise<SavedLink[]> {
    const res = await fetch(`${API_BASE}/api/links/${id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json', ...EXTRA_HEADERS },
        body: JSON.stringify({ preset }),
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

// ── Presets (named pipeline configurations) ─────────────────────────────────

export interface Preset {
    name: string;
    slug: string;
    description?: string;
    settings?: Record<string, unknown>;
    builtin?: boolean;
}

export async function getPresets(): Promise<Preset[]> {
    const res = await fetch(`${API_BASE}/api/presets`, { headers: { ...EXTRA_HEADERS } });
    if (!res.ok) return [];
    const data = await res.json();
    return data.presets || [];
}

export async function getBuiltinPresets(): Promise<Preset[]> {
    const res = await fetch(`${API_BASE}/api/presets/builtin`, { headers: { ...EXTRA_HEADERS } });
    if (!res.ok) return [];
    const data = await res.json();
    return data.presets || [];
}

export async function getBuiltinPreset(slug: string): Promise<Preset | null> {
    const res = await fetch(`${API_BASE}/api/presets/builtin/${slug}`, { headers: { ...EXTRA_HEADERS } });
    if (!res.ok) return null;
    return res.json();
}

export async function getPreset(slug: string): Promise<Preset | null> {
    const res = await fetch(`${API_BASE}/api/presets/${slug}`, { headers: { ...EXTRA_HEADERS } });
    if (!res.ok) return null;
    return res.json();
}

export async function savePreset(name: string, settings: Record<string, unknown>): Promise<Preset> {
    const res = await fetch(`${API_BASE}/api/presets`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...EXTRA_HEADERS },
        body: JSON.stringify({ name, settings }),
    });
    if (!res.ok) {
        const err = await res.json().catch(() => ({ error: 'Failed to save preset' }));
        throw new Error(err.error || 'Failed to save preset');
    }
    return res.json();
}

export async function deletePreset(slug: string): Promise<void> {
    const res = await fetch(`${API_BASE}/api/presets/${slug}`, {
        method: 'DELETE',
        headers: { ...EXTRA_HEADERS },
    });
    if (!res.ok) {
        const err = await res.json().catch(() => ({ error: 'Failed to delete preset' }));
        throw new Error(err.error || 'Failed to delete preset');
    }
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

// ── Translation Glossary ─────────────────────────────────────────────────────

export async function getGlossary(): Promise<Record<string, string>> {
    const res = await fetch(`${API_BASE}/api/glossary`, { headers: EXTRA_HEADERS });
    if (!res.ok) return {};
    return res.json();
}

export async function addGlossaryEntry(english: string, hindi: string): Promise<Record<string, string>> {
    const res = await fetch(`${API_BASE}/api/glossary`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...EXTRA_HEADERS },
        body: JSON.stringify({ english, hindi }),
    });
    const data = await res.json();
    return data.glossary || {};
}

export async function deleteGlossaryEntry(word: string): Promise<Record<string, string>> {
    const res = await fetch(`${API_BASE}/api/glossary/${encodeURIComponent(word)}`, {
        method: 'DELETE',
        headers: EXTRA_HEADERS,
    });
    const data = await res.json();
    return data.glossary || {};
}
