'use client';

import { useState, useCallback, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import URLInput from '@/components/URLInput';
import LanguageSelector, { LANGUAGES } from '@/components/LanguageSelector';
import SettingsPanel, { type DubbingSettings } from '@/components/SettingsPanel';
import JobCard from '@/components/JobCard';
import SavedLinks from '@/components/SavedLinks';
import { createJob, createJobUpload, createJobWithSrt, localDownloadAndDub, isRemoteBackend, getJobs, addLink, type JobStatus } from '@/lib/api';


export default function HomePage() {
    const router = useRouter();
    const [sourceLanguage, setSourceLanguage] = useState('auto');
    const [targetLanguage, setTargetLanguage] = useState('hi');
    const [settings, setSettings] = useState<DubbingSettings>({
        asr_model: 'large-v3',
        translation_engine: 'auto',
        tts_rate: '+0%',
        mix_original: false,
        original_volume: 0.10,
        use_chatterbox: true,
        use_elevenlabs: false,
        use_google_tts: false,
        use_coqui_xtts: false,
        use_edge_tts: false,
        prefer_youtube_subs: false,
        use_yt_translate: false,
        multi_speaker: false,
        transcribe_only: false,
        audio_priority: true,
        audio_bitrate: '192k',
        encode_preset: 'veryfast',
        dub_chain: [],
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

    const handleSubmit = useCallback(async (url: string) => {
        setSubmitting(true);
        setError(null);
        try {
            // Auto-save URL to saved links with current preset
            addLink(url, undefined, { source_language: sourceLanguage, target_language: targetLanguage, ...settings }).catch(() => {});

            const jobSettings = {
                source_language: sourceLanguage,
                target_language: targetLanguage,
                ...settings,
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
    }, [sourceLanguage, targetLanguage, settings, router]);

    const handleFileSubmit = useCallback(async (file: File) => {
        setSubmitting(true);
        setError(null);
        try {
            const { id } = await createJobUpload(file, {
                source_language: sourceLanguage,
                target_language: targetLanguage,
                ...settings,
            });
            router.push(`/jobs/${id}`);
        } catch (e) {
            setError(e instanceof Error ? e.message : 'Failed to upload and start dubbing');
            setSubmitting(false);
        }
    }, [sourceLanguage, targetLanguage, settings, router]);

    const handleBatchSubmit = useCallback((urls: string[]) => {
        // Auto-save all batch URLs with current preset
        const preset = { source_language: sourceLanguage, target_language: targetLanguage, ...settings };
        urls.forEach(u => addLink(u, undefined, preset).catch(() => {}));

        sessionStorage.setItem('batch_pending', JSON.stringify({
            urls,
            settings: {
                source_language: sourceLanguage,
                target_language: targetLanguage,
                ...settings,
            },
        }));
        router.push('/batch');
    }, [sourceLanguage, targetLanguage, settings, router]);

    const handleSrtSubmit = useCallback(async (srtFile: File, videoSource: { url?: string; file?: File }) => {
        setSubmitting(true);
        setError(null);
        try {
            const { id } = await createJobWithSrt(srtFile, {
                source_language: sourceLanguage,
                target_language: targetLanguage,
                ...settings,
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
                    <URLInput onSubmit={handleSubmit} onFileSubmit={handleFileSubmit} onBatchSubmit={handleBatchSubmit} onSrtSubmit={handleSrtSubmit} disabled={submitting} url={currentUrl} onUrlChange={setCurrentUrl} getPreset={() => ({ source_language: sourceLanguage, target_language: targetLanguage, ...settings })} />

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

                {/* Advanced Settings */}
                <SettingsPanel settings={settings} onChange={setSettings} />

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
