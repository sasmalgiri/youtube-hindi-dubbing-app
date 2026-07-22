'use client';

import { useState, useCallback, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import URLInput from '@/components/URLInput';
import LanguageSelector, { LANGUAGES } from '@/components/LanguageSelector';
import SettingsPanel, { type DubbingSettings } from '@/components/SettingsPanel';
import WordTimingPanel from '@/components/WordTimingPanel';
import PresetTabs from '@/components/PresetTabs';
import JobCard from '@/components/JobCard';
import SavedLinks from '@/components/SavedLinks';
import { createJob, createJobUpload, createJobWithSrt, localDownloadAndDub, isRemoteBackend, getJobs, addLink, type JobStatus } from '@/lib/api';


export default function HomePage() {
    const router = useRouter();
    const [sourceLanguage, setSourceLanguage] = useState('auto');
    const [targetLanguage, setTargetLanguage] = useState('hi');
    const [settings, setSettings] = useState<DubbingSettings>({
        voice: 'hi-IN-MadhurNeural',         // Default: Hindi male voice
        // ── Stage 1: Download (aria2c 16x) ──
        asr_model: 'groq-whisper',           // Stage 3: Groq Whisper cloud → fallback local large-v3
        // ── Stage 5: Translation — Google (parallel x20, FASTEST free) ──
        translation_engine: 'google',
        tts_rate: '+0%',
        mix_original: true,   // keep original background music/SFX (Demucs) under the Hindi voice
        video_slow_to_match: true,
        original_volume: 0.10,
        // ── Stage 8: TTS (Managed Quad Parallel) ──
        use_cosyvoice: false,              // OFF: Edge-TTS only for speed
        use_chatterbox: false,
        use_indic_parler: false,
        use_sarvam_bulbul: false,           // OFF: Edge-TTS only for speed
        use_elevenlabs: false,
        use_google_tts: false,
        use_coqui_xtts: false,              // OFF: Edge-TTS only for speed
        use_fish_speech: false,
        use_edge_tts: true,                  // Cloud: fallback overflow
        multi_speaker: false,
        transcribe_only: false,
        // ── Stage 13: Assembly (split-the-diff: audio 1.35x + video 1.50x) ──
        audio_priority: true,
        audio_untouchable: false,
        // ── Stage 9-11: Silence removal + Enhance + Smooth ──
        post_tts_level: 'minimal',           // speechnorm + fade (silence removal always runs)
        audio_quality_mode: 'fast',          // ffmpeg loudnorm (or 'quality' for librosa)
        enable_sentence_gap: false,          // OFF: no artificial gaps, continuous audio
        enable_duration_fit: false,          // OFF: assembly split-the-diff handles timing
        // ── Output quality ──
        audio_bitrate: '192k',
        encode_preset: 'fast',               // NVENC GPU encoding
        download_mode: 'remux',              // 'remux' fast | 'encode' slow but compatible
        split_duration: 30,                  // 30 min chunks: handles long videos (1h+)
        dub_duration: 0,
        fast_assemble: false,
        dub_chain: [],
        enable_manual_review: false,
        use_whisperx: true,
        whisper_gpu_fallback: true,          // local Whisper on GPU if WhisperX not proper
        whisper_fallback_model: 'large-v3',  // large-v3 | medium
        simplify_english: false,             // OFF: translation 35% word cap handles it
        step_by_step: false,
        use_new_pipeline: false,
        enable_tts_verify_retry: false,      // OFF: 70% false-positive rate adds 1-2h on long videos
        tts_truncation_threshold: 0.30,      // 30% catches catastrophic Edge-TTS drops, no false positives
        tts_word_match_verify: true,         // ON: per-segment Whisper word verify + retry on mismatch
        tts_word_match_tolerance: 0.15,      // ±15% wiggle room for word count match
        tts_word_match_model: 'auto',        // 'auto' picks turbo on GPU, tiny on CPU
        tts_word_match_max_segments: 1000,   // auto-disable per-segment word verify above N cues
        long_segment_trace: true,            // ON: write long-segment lifecycle JSON report per job
        long_segment_threshold_words: 15,    // segments with >=15 words are traced
        tts_no_time_pressure: true,          // ON: TTS gets every word, post-processing handles slots
        tts_dynamic_workers: true,           // ON: adapt worker count to Edge-TTS rate limits
        tts_dynamic_min: 10,
        tts_dynamic_max: 120,
        tts_dynamic_start: 30,
        tts_rate_mode: 'auto',       // ON by default: match source video duration
        tts_rate_ceiling: '+50%',    // 1.5x cap per user preference
        tts_rate_target_wpm: 130,    // Hindi natural conversational pace
        keep_subject_english: false,         // OFF: when ON, mask noun subjects so they stay English
        purge_on_new_url: false,             // OFF: opt-in disk cleanup when submitting a different URL
        pipeline_mode: 'classic',
        _input_mode: 'url',
        // ── AV Sync Modules ──
        av_sync_mode: 'original',            // OFF by default — old trim behavior
        max_audio_speedup: 1.30,
        min_video_speed: 0.70,
        slot_verify: 'off',
        // ── Global Stretch ──
        use_global_stretch: false,           // OFF by default
        global_stretch_speedup: 1.25,
        // ── Segmenter ──
        segmenter: 'dp',                     // DP optimal by default
        segmenter_buffer_pct: 0.20,
        max_sentences_per_cue: 2,
        tts_chunk_words: 0,                  // 0=off, 4/8/12=chunk translated text before TTS
        gap_mode: 'micro',                   // "none" (0s) | "micro" (0.2s) | "full" (original gaps)
        sd_srt_content: '',                  // SRT Direct mode: full SRT content (paste or file upload)
        sd_max_stretch: 10.0,                // SRT Direct mode: max video stretch (1.0-10.0×)
        sd_audio_speed: 1.25,                // SRT Direct mode: post-TTS audio speedup (atempo)
        sd_vx_hflip: true,                   // SRT Direct mode: horizontal mirror (Content-ID evasion)
        sd_vx_hue: 5.0,                      // SRT Direct mode: hue shift degrees
        sd_vx_zoom: 1.05,                    // SRT Direct mode: zoom + crop back
        transcript_srt_content: '',          // Upload English SRT to skip transcription entirely
    });
    const [currentUrl, setCurrentUrl] = useState('');
    const [submitting, setSubmitting] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [recentJobs, setRecentJobs] = useState<JobStatus[]>([]);

    const loadJobs = useCallback(() => {
        getJobs().then(setRecentJobs).catch(() => { });
    }, []);

    useEffect(() => {
        loadJobs();
    }, [loadJobs]);

    const targetName = LANGUAGES.find((l) => l.code === targetLanguage)?.name || targetLanguage;

    // Strip mode-specific bloat from request bodies. A 2 MB pasted SRT
    // shouldn't ride along with every classic / hybrid / new / oneflow job.
    const stripModeBloat = useCallback((s: typeof settings) => {
        const cleaned: any = { ...s };
        if (cleaned.pipeline_mode !== 'srtdub') delete cleaned.sd_srt_content;
        return cleaned;
    }, []);

    // Same rule for the saved-link preset — never persist blob fields.
    const stripForPreset = useCallback((s: typeof settings) => {
        const { sd_srt_content, transcript_srt_content, ...rest } = s as any;
        return { source_language: sourceLanguage, target_language: targetLanguage, ...rest };
    }, [sourceLanguage, targetLanguage]);

    const handleSubmit = useCallback(async (url: string) => {
        setSubmitting(true);
        setError(null);

        // Pre-submit guards for mode-specific required inputs
        if (settings.pipeline_mode === 'srtdub' && !((settings as any).sd_srt_content || '').trim() && !((settings as any).transcript_srt_content || '').trim()) {
            setError('Paste or upload a translated SRT or an English transcript SRT before starting SRT Direct.');
            setSubmitting(false);
            return;
        }

        try {
            // Auto-save URL to saved links — WITHOUT blob fields (stripForPreset)
            addLink(url, undefined, stripForPreset(settings)).catch(() => { });

            const cleanedSettings = stripModeBloat(settings);
            const jobSettings = {
                source_language: sourceLanguage,
                target_language: targetLanguage,
                ...cleanedSettings,
            };

            // Remote backend (Colab): download locally first, then upload
            // Local backend: let backend download directly
            const { id } = isRemoteBackend
                ? await localDownloadAndDub(url, jobSettings)
                : await createJob({ url, ...jobSettings });

            router.push(`/jobs/${id}`);
        } catch (e) {
            setError(e instanceof Error ? e.message : 'Failed to start dubbing');
            setSubmitting(false);
        }
    }, [sourceLanguage, targetLanguage, settings, router, stripModeBloat, stripForPreset]);

    const handleFileSubmit = useCallback(async (file: File) => {
        setSubmitting(true);
        setError(null);
        if (settings.pipeline_mode === 'srtdub' && !((settings as any).sd_srt_content || '').trim() && !((settings as any).transcript_srt_content || '').trim()) {
            setError('Paste or upload a translated SRT or an English transcript SRT before starting SRT Direct.');
            setSubmitting(false);
            return;
        }
        try {
            const { id } = await createJobUpload(file, {
                source_language: sourceLanguage,
                target_language: targetLanguage,
                ...stripModeBloat(settings),
            });
            router.push(`/jobs/${id}`);
        } catch (e) {
            setError(e instanceof Error ? e.message : 'Failed to upload and start dubbing');
            setSubmitting(false);
        }
    }, [sourceLanguage, targetLanguage, settings, router, stripModeBloat]);

    const handleBatchSubmit = useCallback((urls: string[]) => {
        // Auto-save all batch URLs with current preset — stripped of blob fields
        const preset = stripForPreset(settings);
        urls.forEach(u => addLink(u, undefined, preset).catch(() => { }));

        sessionStorage.setItem('batch_pending', JSON.stringify({
            urls,
            settings: {
                source_language: sourceLanguage,
                target_language: targetLanguage,
                ...stripModeBloat(settings),
            },
        }));
        router.push('/batch');
    }, [sourceLanguage, targetLanguage, settings, router, stripModeBloat, stripForPreset]);

    const handleSrtSubmit = useCallback(async (srtFile: File, videoSource: { url?: string; file?: File }, needsTranslation?: boolean) => {
        setSubmitting(true);
        setError(null);
        try {
            const { id } = await createJobWithSrt(srtFile, {
                source_language: sourceLanguage,
                target_language: targetLanguage,
                ...settings,
                audio_untouchable: !needsTranslation,  // Only lock audio when already translated
                post_tts_level: needsTranslation ? 'full' : 'none',
                srt_needs_translation: needsTranslation || false,
            }, videoSource);
            router.push(`/jobs/${id}`);
        } catch (e) {
            setError(e instanceof Error ? e.message : 'Failed to start SRT dubbing');
            setSubmitting(false);
        }
    }, [sourceLanguage, targetLanguage, settings, router]);

    return (
        <div className="min-h-screen">
            {/* Hero Section */}
            <div className="border-b border-border bg-gradient-to-b from-primary/[0.03] to-transparent">
                <div className="max-w-3xl mx-auto px-8 py-16">
                    <div className="text-center mb-10">
                        <h1 className="text-4xl font-bold text-text-primary mb-3">
                            Dub YouTube Videos
                            <span className="text-primary"> into {targetName}</span>
                        </h1>
                        <p className="text-text-secondary text-lg">
                            Paste a YouTube URL or upload a video to get it dubbed with Chatterbox AI voice.
                        </p>
                    </div>

                    {/* URL Input */}
                    <URLInput onSubmit={handleSubmit} onFileSubmit={handleFileSubmit} onBatchSubmit={handleBatchSubmit} onSrtSubmit={handleSrtSubmit} onModeChange={(m) => setSettings(s => ({ ...s, _input_mode: m }))} disabled={submitting} url={currentUrl} onUrlChange={setCurrentUrl} getPreset={() => ({ source_language: sourceLanguage, target_language: targetLanguage, ...settings })} dubDuration={settings.dub_duration} onDubDurationChange={(v) => setSettings(s => ({ ...s, dub_duration: v }))} />

                    {/* Saved Links */}
                    <div className="mt-4">
                        <SavedLinks onSelect={setCurrentUrl} onJobStarted={(id) => router.push(`/jobs/${id}`)} />
                    </div>

                    {error && (
                        <div className="mt-4 p-3 rounded-xl bg-error/10 border border-error/20 text-error text-sm">
                            {error}
                        </div>
                    )}
                </div>
            </div>

            {/* Settings Section */}
            <div className="max-w-3xl mx-auto px-8 py-8 space-y-6">
                {/* Language Selector */}
                <div className="glass-card p-5">
                    <LanguageSelector
                        sourceLanguage={sourceLanguage}
                        targetLanguage={targetLanguage}
                        onSourceChange={setSourceLanguage}
                        onTargetChange={setTargetLanguage}
                    />
                </div>

                {/* Preset Tabs — save/load named configurations */}
                <PresetTabs
                    currentSettings={settings}
                    onApply={(loaded) => setSettings(prev => ({ ...prev, ...loaded }))}
                    onLanguageChange={(src, tgt) => { setSourceLanguage(src); setTargetLanguage(tgt); }}
                    sourceLanguage={sourceLanguage}
                    targetLanguage={targetLanguage}
                />

                {/* Pipeline Switcher — 6 modes (hidden in SRT mode — only Classic applies) */}
                {settings._input_mode !== 'srt' && (
                    <div className="flex flex-col gap-2 px-1">
                        <div className="flex flex-wrap gap-1 rounded-xl border border-border p-1">
                            {[
                                { mode: 'classic', label: 'Classic', color: 'bg-primary' },
                                { mode: 'hybrid', label: 'Hybrid', color: 'bg-amber-500' },
                                { mode: 'new', label: 'New (DP)', color: 'bg-green-500' },
                                { mode: 'oneflow', label: 'OneFlow', color: 'bg-red-500' },
                                { mode: 'srtdub', label: 'SRT Direct', color: 'bg-teal-500' },
                            ].map(({ mode, label, color }) => (
                                <button
                                    key={mode}
                                    type="button"
                                    onClick={() => setSettings(s => ({ ...s, pipeline_mode: mode, use_new_pipeline: mode === 'new' }))}
                                    className={`px-3 py-1.5 text-xs font-medium rounded-lg transition-colors whitespace-nowrap ${settings.pipeline_mode === mode || (!(settings.pipeline_mode) && mode === 'classic')
                                        ? `${color} text-white`
                                        : 'bg-white/5 text-text-muted hover:bg-white/10'
                                        }`}
                                >
                                    {label}
                                </button>
                            ))}
                        </div>
                        <span className="text-[10px] text-text-muted px-1">
                            {(({
                                classic: 'Whisper + merge/split + rule engine (proven)',
                                hybrid: 'Old shell + new DP core (best of both)',
                                new: 'Parakeet + WhisperX + DP cues + glossary (experimental)',
                                oneflow: 'Groq Whisper → Google Translate → Edge-TTS → 1.15x (FASTEST)',
                                srtdub: 'Your Hindi SRT → TTS each cue verbatim → 0-gap concat → stretch 1-10× + freeze-pad (audio never trimmed)',
                            }) as Record<string, string>)[settings.pipeline_mode || 'classic']}
                        </span>
                    </div>
                )}

                {/* Voice Clone Preset */}
                <div className="flex items-center gap-3 px-1">
                    <button
                        type="button"
                        onClick={() => {
                            const isVoiceClone = settings.audio_untouchable && settings.use_coqui_xtts && !settings.use_edge_tts;
                            if (isVoiceClone) {
                                // Cancel voice clone — restore defaults
                                setSourceLanguage('auto');
                                setSettings(s => ({
                                    ...s,
                                    use_cosyvoice: false,
                                    use_coqui_xtts: false,
                                    use_edge_tts: true,
                                    use_sarvam_bulbul: false,
                                    use_chatterbox: false,
                                    audio_untouchable: false,
                                    post_tts_level: 'minimal',
                                    audio_priority: true,
                                    video_slow_to_match: true,
                                    tts_rate: '+0%',
                                }));
                            } else {
                                // Activate voice clone
                                setSourceLanguage(targetLanguage === 'hi' ? 'hi' : targetLanguage);
                                setSettings(s => ({
                                    ...s,
                                    use_cosyvoice: false,
                                    use_coqui_xtts: true,
                                    use_edge_tts: false,
                                    use_sarvam_bulbul: false,
                                    use_chatterbox: false,
                                    audio_untouchable: true,
                                    post_tts_level: 'none',
                                    audio_priority: true,
                                    video_slow_to_match: true,
                                    tts_rate: '+0%',
                                    simplify_english: false,
                                    enable_manual_review: false,
                                }));
                            }
                        }}
                        className={`px-4 py-2 rounded-xl text-xs font-medium transition-all border ${settings.audio_untouchable && settings.use_coqui_xtts && !settings.use_edge_tts
                            ? 'bg-violet-500/20 border-violet-500 text-violet-300'
                            : 'bg-white/5 border-white/10 text-text-muted hover:bg-violet-500/10 hover:border-violet-500/30'
                            }`}
                    >
                        Voice Clone
                    </button>
                    {settings.audio_untouchable && settings.use_coqui_xtts && !settings.use_edge_tts && (
                        <span className="text-[10px] text-violet-400">
                            Same-language re-voice: XTTS clones original speaker, zero post-processing
                        </span>
                    )}
                </div>

                {/* Word Timing — dedicated module (backend/dubbing/word_timing.py) */}
                <WordTimingPanel settings={settings} onChange={setSettings} />

                {/* Advanced Settings */}
                <SettingsPanel settings={settings} onChange={setSettings} targetLanguage={targetLanguage} />

                {/* Recent Jobs */}
                {recentJobs.length > 0 && (
                    <div>
                        <h2 className="text-lg font-semibold text-text-primary mb-4">Recent Jobs</h2>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            {recentJobs.slice(0, 6).map((job) => (
                                <JobCard key={job.id} job={job} onDelete={loadJobs} />
                            ))}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
