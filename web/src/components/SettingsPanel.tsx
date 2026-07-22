'use client';

import { useState, useEffect, useCallback } from 'react';
import { getGlossary, addGlossaryEntry, deleteGlossaryEntry } from '@/lib/api';

export interface DubbingSettings {
    voice: string;
    asr_model: string;
    translation_engine: string;
    tts_rate: string;
    mix_original: boolean;
    original_volume: number;
    use_cosyvoice: boolean;
    use_chatterbox: boolean;
    use_indic_parler: boolean;
    use_sarvam_bulbul: boolean;
    video_slow_to_match: boolean;
    use_elevenlabs: boolean;
    use_google_tts: boolean;
    use_coqui_xtts: boolean;
    use_fish_speech: boolean;
    use_edge_tts: boolean;
    prefer_youtube_subs: boolean;
    use_yt_translate: boolean;
    multi_speaker: boolean;
    transcribe_only: boolean;
    audio_priority: boolean;
    audio_untouchable: boolean;
    post_tts_level: string;
    audio_quality_mode: string;
    enable_sentence_gap: boolean;
    enable_duration_fit: boolean;
    audio_bitrate: string;
    encode_preset: string;
    download_mode: string;  // "remux" (fast) | "encode" (slow but always works)
    split_duration: number;
    dub_duration: number;
    fast_assemble: boolean;
    dub_chain: string[];
    enable_manual_review: boolean;
    use_whisperx: boolean;
    simplify_english: boolean;
    step_by_step: boolean;
    use_new_pipeline: boolean;
    enable_tts_verify_retry: boolean;
    tts_truncation_threshold: number;  // 0.0 = OFF, 0.30 = default, 0.70 = aggressive
    tts_word_match_verify: boolean;    // post-TTS Whisper word-count verification
    tts_word_match_tolerance: number;  // 0.0–1.0, ±N% wiggle room (default 0.15)
    tts_word_match_model: string;      // "auto" | "tiny" | "turbo"
    tts_word_match_max_segments: number; // auto-disable above N cues (0 = no cap)
    use_wav2lip?: boolean;             // Wav2Lip lip-sync post-process (opt-in, requires GPU + repo + checkpoint)
    long_segment_trace: boolean;       // record long-segment lifecycle to JSON
    long_segment_threshold_words: number; // segments above this many words get traced
    tts_no_time_pressure: boolean;     // skip ALL slot/speed pressure on TTS
    tts_dynamic_workers: boolean;      // adapt worker count to Edge-TTS rate limits
    tts_dynamic_min: number;
    tts_dynamic_max: number;
    tts_dynamic_start: number;
    tts_rate_mode: string;    // "auto" | "manual"
    tts_rate_ceiling: string; // cap for auto mode, e.g., "+50%"
    tts_rate_target_wpm: number; // natural pace baseline, default 130
    keep_subject_english: boolean;     // mask noun subjects, keep them English in output
    purge_on_new_url: boolean;
    pipeline_mode?: string;
    _input_mode?: string;
    preset_name?: string;
    // ── AV Sync Modules ──
    av_sync_mode: string;            // "original" | "capped" | "audio_first"
    max_audio_speedup: number;       // cap for capped mode (default 1.30)
    min_video_speed: number;         // floor before flagging (default 0.70)
    slot_verify: string;             // "off" | "dry_run" | "auto_fix"
    // ── Global Stretch ──
    use_global_stretch: boolean;     // uniform video slowdown
    global_stretch_speedup: number;  // TTS speedup for global stretch (default 1.25)
    // ── Segmenter ──
    segmenter: string;               // "dp" | "sentence"
    segmenter_buffer_pct: number;    // Hindi expansion buffer (default 0.20)
    max_sentences_per_cue: number;   // max sentences per segment (default 2)
    // ── YouTube Transcript Mode ──
    yt_transcript_mode: string;      // "yt_timeline" | "whisper_timeline"
    yt_segment_mode: string;         // "sentence" | "wordcount"
    yt_text_correction: boolean;     // correct Whisper text using YouTube subs
    yt_replace_mode: string;         // "full" | "diff"
    tts_chunk_words: number;         // 0=off, 4/8/12=chunk size for TTS
    gap_mode: string;                // "none" | "micro" | "full"
    // ── WordChunk mode ──
    wc_chunk_size?: number;          // 4 | 8 | 12
    wc_max_stretch?: number;         // 1.0 – 20.0
    wc_transcript?: string;          // optional pasted transcript override
    // ── SRT Direct mode ──
    sd_srt_content?: string;         // full SRT content (required for srtdub mode)
    sd_max_stretch?: number;         // 1.0 – 20.0
    sd_audio_speed?: number;         // post-TTS atempo (0.5-3.0, default 1.25)
    sd_vx_hflip?: boolean;           // horizontal mirror
    sd_vx_hue?: number;              // hue shift degrees (-30..+30)
    sd_vx_zoom?: number;             // zoom + crop back (1.0 = off, 1.05 = 5% zoom)
    // ── Transcript Upload (skip transcription) ──
    transcript_srt_content?: string;  // English SRT to skip transcription entirely
}

interface SettingsPanelProps {
    settings: DubbingSettings;
    onChange: (settings: DubbingSettings) => void;
    targetLanguage?: string;
}

function GlossaryEditor() {
    const [glossary, setGlossary] = useState<Record<string, string>>({});
    const [newEn, setNewEn] = useState('');
    const [newHi, setNewHi] = useState('');
    const [open, setOpen] = useState(false);

    const load = useCallback(() => {
        getGlossary().then(setGlossary).catch(() => { });
    }, []);

    useEffect(() => { load(); }, [load]);

    const handleAdd = async () => {
        const en = newEn.trim().toLowerCase();
        const hi = newHi.trim();
        if (!en || !hi) return;
        const updated = await addGlossaryEntry(en, hi);
        setGlossary(updated);
        setNewEn('');
        setNewHi('');
    };

    const handleDelete = async (word: string) => {
        const updated = await deleteGlossaryEntry(word);
        setGlossary(updated);
    };

    const entries = Object.entries(glossary);

    return (
        <div className="border border-border rounded-lg p-3">
            <button type="button" onClick={() => setOpen(!open)}
                className="flex items-center justify-between w-full text-left">
                <div>
                    <p className="text-xs font-medium text-text-secondary">
                        Translation Glossary ({entries.length} words)
                    </p>
                    <p className="text-[10px] text-text-muted">
                        Words to keep in English or transliterate in Hindi output
                    </p>
                </div>
                <span className="text-text-muted text-xs">{open ? '\u25B2' : '\u25BC'}</span>
            </button>

            {open && (
                <div className="mt-3 space-y-2">
                    {/* Add new entry */}
                    <div className="flex gap-2">
                        <input
                            type="text" placeholder="English (e.g. noble)"
                            value={newEn} onChange={e => setNewEn(e.target.value)}
                            onKeyDown={e => e.key === 'Enter' && handleAdd()}
                            className="flex-1 px-2 py-1 rounded text-xs bg-white/5 border border-border text-text-primary placeholder:text-text-muted"
                        />
                        <input
                            type="text" placeholder="Hindi output (e.g. noble)"
                            value={newHi} onChange={e => setNewHi(e.target.value)}
                            onKeyDown={e => e.key === 'Enter' && handleAdd()}
                            className="flex-1 px-2 py-1 rounded text-xs bg-white/5 border border-border text-text-primary placeholder:text-text-muted"
                        />
                        <button type="button" onClick={handleAdd}
                            className="px-3 py-1 rounded text-xs bg-primary text-white hover:bg-primary/80">
                            Add
                        </button>
                    </div>

                    {/* Existing entries */}
                    {entries.length > 0 && (
                        <div className="max-h-40 overflow-y-auto space-y-1">
                            {entries.map(([en, hi]) => (
                                <div key={en} className="flex items-center justify-between px-2 py-1 rounded bg-white/5 text-xs">
                                    <span className="text-text-primary">{en}</span>
                                    <span className="text-text-muted mx-2">&rarr;</span>
                                    <span className="text-green-400 flex-1">{hi}</span>
                                    <button type="button" onClick={() => handleDelete(en)}
                                        className="ml-2 text-red-400 hover:text-red-300 text-[10px]">
                                        x
                                    </button>
                                </div>
                            ))}
                        </div>
                    )}

                    {entries.length === 0 && (
                        <p className="text-[10px] text-text-muted text-center py-2">
                            No entries yet. Add English words that should stay as-is in Hindi.
                        </p>
                    )}
                </div>
            )}
        </div>
    );
}

export default function SettingsPanel({ settings, onChange, targetLanguage = 'hi' }: SettingsPanelProps) {
    const [open, setOpen] = useState(false);

    const update = (partial: Partial<DubbingSettings>) => {
        onChange({ ...settings, ...partial });
    };

    // When the user switches INTO the new pipeline while a non-new-compatible
    // ASR model is selected (currently only "groq-whisper" — it isn't wired to
    // the DP runner yet), auto-migrate to Parakeet so no button looks orphaned.
    useEffect(() => {
        if (settings.pipeline_mode === 'new' && settings.asr_model === 'groq-whisper') {
            onChange({ ...settings, asr_model: 'parakeet' });
        }
        // Only re-run when the pipeline mode or selected model changes.
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [settings.pipeline_mode, settings.asr_model]);

    // SRT Direct and WordChunk have REQUIRED inputs inside the advanced panel
    // (the SRT textarea / transcript). Auto-expand so users don't think the
    // tab did nothing.
    useEffect(() => {
        if (settings.pipeline_mode === 'srtdub' || settings.pipeline_mode === 'wordchunk') {
            setOpen(true);
        }
    }, [settings.pipeline_mode]);

    return (
        <div className="glass-card overflow-hidden">
            <button
                onClick={() => setOpen(!open)}
                className="w-full flex items-center justify-between px-5 py-3.5 hover:bg-white/[0.02] transition-colors"
            >
                <div className="flex items-center gap-2">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-text-muted">
                        <path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z" />
                        <circle cx="12" cy="12" r="3" />
                    </svg>
                    <span className="text-sm font-medium text-text-secondary">Advanced Settings</span>
                </div>
                <svg
                    width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
                    className={`text-text-muted transition-transform duration-200 ${open ? 'rotate-180' : ''}`}
                >
                    <path d="m6 9 6 6 6-6" />
                </svg>
            </button>

            {open && (() => {
                // ── Pipeline mode ──
                const mode = (settings as any).pipeline_mode || 'classic';
                const isClassic = mode === 'classic';
                const isHybrid = mode === 'hybrid';
                const isNew = mode === 'new';
                const isOneFlow = mode === 'oneflow';
                const isWordChunk = mode === 'wordchunk';
                const isSrtDub = mode === 'srtdub';
                const isSrtMode = settings._input_mode === 'srt';

                // ── Dependency flags ──
                const isVoiceClone = settings.audio_untouchable && settings.use_coqui_xtts && !settings.use_edge_tts;
                const ytTranslateOn = settings.use_yt_translate;
                const ytSubsOn = settings.prefer_youtube_subs;
                const transcribeOnly = settings.transcribe_only;
                // Whisper disabled when YouTube provides subs, New pipeline, OneFlow, WordChunk, SrtDub, or SRT mode
                const whisperDisabled = ytTranslateOn || ytSubsOn || isOneFlow || isWordChunk || isSrtDub || isSrtMode;
                const whisperxDisabled = whisperDisabled;
                // Translation disabled when YT gives Hindi directly, or in TTS-only modes
                const translationDisabled = ytTranslateOn || isOneFlow || isWordChunk || isSrtDub || isSrtMode;
                // Simplify disabled when no English source or in TTS-only modes
                const simplifyDisabled = ytTranslateOn || isOneFlow || isWordChunk || isSrtDub || isSrtMode;

                return (<div className="px-5 pb-5 space-y-5 animate-slide-up border-t border-border pt-4">
                    {/* ── Quick Start Tip — always visible at top of advanced ── */}
                    <div className="rounded-lg p-3 bg-blue-500/5 border border-blue-500/20">
                        <p className="text-xs font-medium text-blue-400 mb-1">Quick Start — defaults are good for most videos</p>
                        <ul className="text-[11px] text-text-muted space-y-0.5 list-disc list-inside">
                            <li><b className="text-text-secondary">Hindi dubbing</b>: leave everything default → paste URL → submit</li>
                            <li><b className="text-text-secondary">Long video (1h+)</b>: turn on <i>Use YouTube Transcript</i> → skips slow Whisper step</li>
                            <li><b className="text-text-secondary">Best Hindi voice</b>: enable <i>CosyVoice 2</i> or <i>Sarvam Bulbul v3</i></li>
                            <li><b className="text-text-secondary">Already have a translated SRT</b>: switch input mode to <i>SRT</i> → upload it</li>
                        </ul>
                    </div>

                    {/* ── Pipeline Mode Banner ── */}
                    <div className={`rounded-lg p-3 text-xs ${isSrtDub ? 'bg-teal-500/10 border border-teal-500/30 text-teal-400' :
                        isWordChunk ? 'bg-purple-500/10 border border-purple-500/30 text-purple-400' :
                            isOneFlow ? 'bg-red-500/10 border border-red-500/30 text-red-400' :
                                isNew ? 'bg-green-500/10 border border-green-500/30 text-green-400' :
                                    isHybrid ? 'bg-amber-500/10 border border-amber-500/30 text-amber-400' :
                                        'bg-primary/10 border border-primary/30 text-primary-light'
                        }`}>
                        <p className="font-medium mb-1">
                            {isSrtDub ? 'SRT Direct (Your SRT → TTS → Stretch)' :
                                isWordChunk ? 'WordChunk (YouTube + Super-Stretch)' :
                                    isOneFlow ? 'OneFlow (Fastest)' :
                                        isNew ? 'New Pipeline (Experimental)' :
                                            isHybrid ? 'Hybrid Pipeline (Recommended)' :
                                                'Classic Pipeline (Proven)'}
                        </p>
                        <p className="text-[10px] opacity-80">
                            {isSrtDub ? 'You provide a perfect translated SRT. We TTS each cue verbatim → concatenate back-to-back with zero gap → stretch video 1-10× to match audio. If audio still overflows after max stretch, we freeze-pad the last frame. Audio is NEVER trimmed.' :
                                isWordChunk ? 'YouTube English subs → split by sentence → Google Translate each → Edge-TTS each → concatenate → video stretched 1-5× to match audio. No Whisper.' :
                                    isOneFlow ? 'Groq Whisper → Google Translate (100 workers) → Edge-TTS (150 workers) → fixed 1.15x → video adapts. No LLM, no cue rebuild, just speed.' :
                                        isNew ? 'Parakeet ASR + WhisperX timing + DP cue builder + glossary lock. All new modular code.' :
                                            isHybrid ? 'Whisper ASR (all options) + DP cue builder + glossary + Hindi fitting + QC gates. Best quality + proven infrastructure.' :
                                                'Full monolith pipeline. All engines, all options, battle-tested. Maximum flexibility.'}
                        </p>
                    </div>

                    {/* ── SRT Direct-specific controls ── */}
                    {isSrtDub && (
                        <div className="rounded-lg p-4 bg-teal-500/5 border border-teal-500/20 space-y-4">
                            <div>
                                <label className="text-xs font-medium text-teal-300 block mb-2">
                                    Translated SRT content
                                </label>
                                <div className="flex items-center gap-2 mb-2">
                                    <input
                                        type="file"
                                        accept=".srt,text/plain"
                                        aria-label="Upload SRT file"
                                        title="Upload SRT file"
                                        onChange={async (e) => {
                                            const file = e.target.files?.[0];
                                            if (!file) return;
                                            // Try UTF-8 first; if the decoded text has replacement characters
                                            // (U+FFFD) the file is likely in a different encoding (Windows-1252
                                            // / UTF-16). Fall back to those in order.
                                            const buf = await file.arrayBuffer();
                                            const tryDecode = (label: string): string | null => {
                                                try {
                                                    const dec = new TextDecoder(label, { fatal: false });
                                                    return dec.decode(buf);
                                                } catch { return null; }
                                            };
                                            let text = tryDecode("utf-8") ?? "";
                                            if (text.includes("\uFFFD")) {
                                                for (const enc of ["utf-16le", "utf-16be", "windows-1252"]) {
                                                    const alt = tryDecode(enc);
                                                    if (alt && !alt.includes("\uFFFD")) { text = alt; break; }
                                                }
                                            }
                                            onChange({ ...settings, sd_srt_content: text } as any);
                                            // Reset the file input so re-uploading the same file triggers onChange
                                            e.target.value = "";
                                        }}
                                        className="text-[11px] text-text-muted file:mr-2 file:px-3 file:py-1 file:rounded file:border-0 file:bg-teal-500/20 file:text-teal-300 file:text-xs hover:file:bg-teal-500/30"
                                    />
                                    {((settings as any).sd_srt_content ?? "").trim() && (
                                        <button
                                            type="button"
                                            onClick={() => onChange({ ...settings, sd_srt_content: '' } as any)}
                                            className="text-[11px] text-red-400 hover:text-red-300 px-2 py-1"
                                        >
                                            Clear
                                        </button>
                                    )}
                                </div>
                                <textarea
                                    rows={8}
                                    placeholder={"Paste or upload your translated SRT here. Example:\n\n1\n00:00:01,000 --> 00:00:04,000\nमेरे पिता राजा थे और मेरी माँ रानी थीं।\n\n2\n00:00:04,000 --> 00:00:07,500\nवे दोनों कुलीन सफेद लोमड़ियाँ थीं।"}
                                    aria-label="Translated SRT content"
                                    value={(settings as any).sd_srt_content ?? ""}
                                    onChange={e => onChange({ ...settings, sd_srt_content: e.target.value } as any)}
                                    className="w-full bg-white/5 border border-border rounded px-3 py-2 text-xs text-text-primary placeholder:text-text-muted/60 focus:outline-none focus:border-teal-500 font-mono"
                                />
                                <p className="text-[10px] text-text-muted mt-1">
                                    {(() => {
                                        const text = ((settings as any).sd_srt_content ?? "").trim();
                                        if (!text) return "Upload a .srt file or paste SRT content. Required for this mode.";
                                        const cueCount = (text.match(/^\d+\s*$/gm) || []).length;
                                        return `Loaded ${text.length} chars, ~${cueCount} cues detected`;
                                    })()}
                                </p>
                            </div>

                            {/* ── OR: English Transcript SRT (auto-translate before TTS) ── */}
                            <div className="border-t border-teal-500/20 pt-4">
                                <p className="text-xs font-medium text-teal-300 mb-1">
                                    OR: Upload English Transcript (auto-translate)
                                </p>
                                <p className="text-[10px] text-text-muted mb-2">
                                    Have an English SRT but no translation? Upload it here — we&apos;ll translate to the target language first, then TTS + assemble.
                                </p>
                                <div className="flex items-center gap-2 mb-2">
                                    <input
                                        type="file"
                                        accept=".srt,text/plain"
                                        aria-label="Upload English transcript SRT file"
                                        title="Upload English transcript SRT file"
                                        onChange={async (e) => {
                                            const file = e.target.files?.[0];
                                            if (!file) return;
                                            const buf = await file.arrayBuffer();
                                            const tryDecode = (label: string): string | null => {
                                                try {
                                                    const dec = new TextDecoder(label, { fatal: false });
                                                    return dec.decode(buf);
                                                } catch { return null; }
                                            };
                                            let text = tryDecode("utf-8") ?? "";
                                            if (text.includes("\uFFFD")) {
                                                for (const enc of ["utf-16le", "utf-16be", "windows-1252"]) {
                                                    const alt = tryDecode(enc);
                                                    if (alt && !alt.includes("\uFFFD")) { text = alt; break; }
                                                }
                                            }
                                            onChange({ ...settings, transcript_srt_content: text.trim() } as any);
                                            e.target.value = "";
                                        }}
                                        className="text-[11px] text-text-muted file:mr-2 file:px-3 file:py-1 file:rounded file:border-0 file:bg-teal-500/20 file:text-teal-300 file:text-xs hover:file:bg-teal-500/30"
                                    />
                                    {((settings as any).transcript_srt_content ?? "").trim() && (
                                        <button
                                            type="button"
                                            onClick={() => onChange({ ...settings, transcript_srt_content: '' } as any)}
                                            className="text-[11px] text-red-400 hover:text-red-300 px-2 py-1"
                                        >
                                            Clear
                                        </button>
                                    )}
                                </div>
                                <textarea
                                    rows={4}
                                    placeholder={"Paste English SRT here (will be translated automatically):\n\n1\n00:00:01,000 --> 00:00:04,000\nHello, welcome to the video."}
                                    aria-label="English transcript SRT content"
                                    value={(settings as any).transcript_srt_content ?? ""}
                                    onChange={e => onChange({ ...settings, transcript_srt_content: e.target.value } as any)}
                                    className="w-full bg-white/5 border border-border rounded px-3 py-2 text-xs text-text-primary placeholder:text-text-muted/60 focus:outline-none focus:border-teal-500 font-mono resize-y"
                                />
                                {(settings as any).transcript_srt_content && (
                                    <p className="text-[10px] text-emerald-400 mt-1">
                                        ✓ English transcript loaded ({((settings as any).transcript_srt_content ?? '').length} chars) — will be translated before TTS
                                    </p>
                                )}
                            </div>

                            <div>
                                <label className="text-xs font-medium text-teal-300 block mb-2">
                                    Max video stretch: {((settings as any).sd_max_stretch ?? 20.0).toFixed(1)}×
                                </label>
                                <input
                                    type="range"
                                    min="1"
                                    max="20"
                                    step="0.5"
                                    aria-label="Maximum video stretch multiplier"
                                    title="Maximum video stretch multiplier"
                                    value={(settings as any).sd_max_stretch ?? 20.0}
                                    onChange={e => onChange({ ...settings, sd_max_stretch: parseFloat(e.target.value) } as any)}
                                    className="w-full accent-teal-500"
                                />
                                <p className="text-[10px] text-text-muted mt-1">
                                    If audio still overflows after {((settings as any).sd_max_stretch ?? 20.0).toFixed(1)}× stretch, the last video frame freezes to fill the remaining audio. Audio is NEVER trimmed.
                                </p>
                            </div>

                            {/* Post-TTS audio speedup (atempo) */}
                            <div>
                                <label className="text-xs font-medium text-teal-300 block mb-2">
                                    Post-TTS audio speed: {((settings as any).sd_audio_speed ?? 1.25).toFixed(2)}×
                                </label>
                                <input
                                    type="range"
                                    min="0.5"
                                    max="4.0"
                                    step="0.05"
                                    aria-label="Post-TTS audio speedup"
                                    title="Post-TTS audio speedup"
                                    value={(settings as any).sd_audio_speed ?? 1.25}
                                    onChange={e => onChange({ ...settings, sd_audio_speed: parseFloat(e.target.value) } as any)}
                                    className="w-full accent-teal-500"
                                />
                                <div className="flex gap-2 mt-1">
                                    {[
                                        { v: 1.0, l: '1.00× (off)' },
                                        { v: 1.1, l: '1.10×' },
                                        { v: 1.25, l: '1.25× default' },
                                        { v: 1.5, l: '1.50×' },
                                        { v: 1.75, l: '1.75×' },
                                    ].map(p => (
                                        <button
                                            key={p.v}
                                            type="button"
                                            onClick={() => onChange({ ...settings, sd_audio_speed: p.v } as any)}
                                            className={`px-2 py-0.5 text-[10px] rounded border transition-colors ${Math.abs(((settings as any).sd_audio_speed ?? 1.25) - p.v) < 0.01
                                                ? 'bg-teal-500 text-white border-teal-500'
                                                : 'bg-white/5 text-text-muted border-border hover:bg-white/10'
                                                }`}
                                        >
                                            {p.l}
                                        </button>
                                    ))}
                                </div>
                                <p className="text-[10px] text-text-muted mt-1">
                                    ffmpeg <span className="font-mono">atempo</span> applied after concat — no pitch shift.
                                    Faster audio → less video stretch needed.
                                </p>
                            </div>

                            {/* Visual transforms — Content-ID evasion */}
                            <div className="rounded-lg p-3 bg-teal-500/5 border border-teal-500/20 space-y-3">
                                <div className="flex items-center justify-between">
                                    <div>
                                        <p className="text-xs font-medium text-teal-300">Visual transforms</p>
                                        <p className="text-[10px] text-text-muted">
                                            Break YouTube Content ID visual fingerprint (does NOT create rights — you still need a license / fair-use basis).
                                        </p>
                                    </div>
                                </div>

                                {/* Horizontal flip */}
                                <div className="flex items-center justify-between">
                                    <div>
                                        <p className="text-xs text-text-primary">Horizontal flip (mirror)</p>
                                        <p className="text-[10px] text-text-muted">Very effective. Reverses all on-screen text/logos.</p>
                                    </div>
                                    <button
                                        type="button"
                                        title="Toggle horizontal flip"
                                        onClick={() => onChange({ ...settings, sd_vx_hflip: !((settings as any).sd_vx_hflip ?? true) } as any)}
                                        className={`w-11 h-6 rounded-full transition-colors relative ${((settings as any).sd_vx_hflip ?? true) ? 'bg-teal-500' : 'bg-white/10'
                                            }`}
                                    >
                                        <div className={`w-4 h-4 rounded-full bg-white absolute top-1 transition-transform ${((settings as any).sd_vx_hflip ?? true) ? 'translate-x-6' : 'translate-x-1'
                                            }`} />
                                    </button>
                                </div>

                                {/* Hue shift */}
                                <div>
                                    <label className="text-xs text-text-primary block mb-1">
                                        Hue shift: {((settings as any).sd_vx_hue ?? 5).toFixed(0)}°
                                    </label>
                                    <input
                                        type="range"
                                        min="-30"
                                        max="30"
                                        step="1"
                                        aria-label="Hue shift degrees"
                                        title="Hue shift degrees"
                                        value={(settings as any).sd_vx_hue ?? 5}
                                        onChange={e => onChange({ ...settings, sd_vx_hue: parseFloat(e.target.value) } as any)}
                                        className="w-full accent-teal-500"
                                    />
                                    <p className="text-[10px] text-text-muted">0 = off. ±5-10° = subtle tint, viewer barely notices.</p>
                                </div>

                                {/* Zoom + crop back */}
                                <div>
                                    <label className="text-xs text-text-primary block mb-1">
                                        Zoom + crop: {((((settings as any).sd_vx_zoom ?? 1.05) - 1) * 100).toFixed(1)}%
                                    </label>
                                    <input
                                        type="range"
                                        min="1.0"
                                        max="1.15"
                                        step="0.01"
                                        aria-label="Zoom and crop percentage"
                                        title="Zoom and crop percentage"
                                        value={(settings as any).sd_vx_zoom ?? 1.05}
                                        onChange={e => onChange({ ...settings, sd_vx_zoom: parseFloat(e.target.value) } as any)}
                                        className="w-full accent-teal-500"
                                    />
                                    <p className="text-[10px] text-text-muted">0% = off. 5% = zooms in slightly then crops back; invisible loss but shifts fingerprint.</p>
                                </div>

                                <p className="text-[10px] text-amber-400 italic">
                                    Note: any visual transform forces a video re-encode (even when audio ≈ video length). Cost: one ffmpeg libx264 encode pass.
                                </p>
                            </div>

                            {/* B25: voice picker (Edge-TTS engine block is hidden in srtdub, so expose a slim picker here) */}
                            <div>
                                <label className="text-xs font-medium text-teal-300 block mb-2">
                                    Edge-TTS voice
                                </label>
                                <div className="grid grid-cols-2 gap-2">
                                    {([
                                        { value: 'hi-IN-SwaraNeural', label: 'Hindi · Swara (F)' },
                                        { value: 'hi-IN-MadhurNeural', label: 'Hindi · Madhur (M)' },
                                        { value: 'bn-IN-TanishaaNeural', label: 'Bengali · Tanishaa (F)' },
                                        { value: 'bn-IN-BashkarNeural', label: 'Bengali · Bashkar (M)' },
                                        { value: 'ta-IN-PallaviNeural', label: 'Tamil · Pallavi (F)' },
                                        { value: 'ta-IN-ValluvarNeural', label: 'Tamil · Valluvar (M)' },
                                        { value: 'te-IN-ShrutiNeural', label: 'Telugu · Shruti (F)' },
                                        { value: 'mr-IN-AarohiNeural', label: 'Marathi · Aarohi (F)' },
                                        { value: 'en-US-JennyNeural', label: 'English · Jenny (F)' },
                                        { value: 'en-US-GuyNeural', label: 'English · Guy (M)' },
                                    ]).map(v => (
                                        <button
                                            key={v.value}
                                            type="button"
                                            onClick={() => update({ voice: v.value })}
                                            className={`px-3 py-2 text-[11px] rounded border transition-colors text-left ${settings.voice === v.value
                                                ? 'bg-teal-500 text-white border-teal-500'
                                                : 'bg-white/5 text-text-muted border-border hover:bg-white/10'
                                                }`}
                                        >
                                            {v.label}
                                        </button>
                                    ))}
                                </div>
                                <p className="text-[10px] text-text-muted mt-1">
                                    Pick the voice that matches your SRT's language. Current: <span className="font-mono">{settings.voice}</span>
                                </p>
                            </div>
                        </div>
                    )}

                    {/* ── WordChunk-specific controls ── */}
                    {isWordChunk && (
                        <div className="rounded-lg p-4 bg-purple-500/5 border border-purple-500/20 space-y-4">
                            <div>
                                <label className="text-xs font-medium text-purple-300 block mb-2">
                                    Chunk size (words per TTS clip)
                                </label>
                                <div className="flex gap-2">
                                    {[4, 8, 12].map(n => (
                                        <button
                                            key={n}
                                            type="button"
                                            onClick={() => onChange({ ...settings, wc_chunk_size: n } as any)}
                                            className={`px-4 py-2 text-xs rounded border transition-colors ${((settings as any).wc_chunk_size ?? 8) === n
                                                ? 'bg-purple-500 text-white border-purple-500'
                                                : 'bg-white/5 text-text-muted border-border hover:bg-white/10'
                                                }`}
                                        >
                                            {n} words
                                        </button>
                                    ))}
                                </div>
                                <p className="text-[10px] text-text-muted mt-1">
                                    Smaller = tighter sync; larger = smoother prosody.
                                </p>
                            </div>
                            <div>
                                <label className="text-xs font-medium text-purple-300 block mb-2">
                                    Max video stretch: {((settings as any).wc_max_stretch ?? 20.0).toFixed(1)}×
                                </label>
                                <input
                                    type="range"
                                    min="1"
                                    max="20"
                                    step="0.5"
                                    aria-label="Maximum video stretch multiplier"
                                    title="Maximum video stretch multiplier"
                                    value={(settings as any).wc_max_stretch ?? 20.0}
                                    onChange={e => onChange({ ...settings, wc_max_stretch: parseFloat(e.target.value) } as any)}
                                    className="w-full accent-purple-500"
                                />
                                <p className="text-[10px] text-text-muted mt-1">
                                    Cap for slowing video to match audio. Video never speeds up.
                                </p>
                            </div>

                            <div>
                                <label className="text-xs font-medium text-purple-300 block mb-2">
                                    Transcript override (optional)
                                </label>
                                <textarea
                                    rows={6}
                                    placeholder="Paste the English transcript here to skip YouTube subs fetch. Leave blank to auto-download subs. Copy from YouTube's transcript panel or any other source — timestamps and speaker labels are stripped automatically."
                                    aria-label="Pasted English transcript override"
                                    value={(settings as any).wc_transcript ?? ""}
                                    onChange={e => onChange({ ...settings, wc_transcript: e.target.value } as any)}
                                    className="w-full bg-white/5 border border-border rounded px-3 py-2 text-xs text-text-primary placeholder:text-text-muted/60 focus:outline-none focus:border-purple-500 font-mono"
                                />
                                <p className="text-[10px] text-text-muted mt-1">
                                    {((settings as any).wc_transcript ?? "").trim()
                                        ? `Using pasted transcript (${((settings as any).wc_transcript ?? "").trim().length} chars) — YouTube subs fetch SKIPPED`
                                        : "Leave empty to auto-fetch YouTube English subs via yt-dlp."}
                                </p>
                            </div>
                        </div>
                    )}

                    {/* SRT Mode: lock transcription + translation (SRT already has translated text) */}
                    {isSrtMode && (
                        <div className="rounded-lg bg-purple-500/5 border border-purple-500/20 p-3">
                            <p className="text-xs font-medium text-purple-400">SRT Dub Mode</p>
                            <p className="text-[10px] text-text-muted mt-1">
                                Transcription + Translation are skipped — your SRT file provides the text.
                                Only TTS, Audio, and Assembly settings apply.
                            </p>
                        </div>
                    )}

                    {/* OneFlow: lock ALL settings — everything is pre-configured */}
                    {isOneFlow && (
                        <div className="rounded-lg bg-red-500/5 border border-red-500/20 p-3 space-y-2">
                            <p className="text-xs font-medium text-red-400">OneFlow — All Settings Fixed</p>
                            <div className="grid grid-cols-2 gap-2 text-[10px] text-text-muted">
                                <div>ASR: Groq Whisper (cloud)</div>
                                <div>Translate: Google (100 workers)</div>
                                <div>TTS: Edge-TTS (150 workers)</div>
                                <div>Speed: fixed 1.15x uniform</div>
                                <div>QC: 1 check + 1 retry per stage</div>
                                <div>Video: adapts to audio (freeze/slow)</div>
                            </div>
                            <p className="text-[10px] opacity-60">All settings below are ignored in OneFlow mode.</p>
                        </div>
                    )}

                    {/* ── SRT Direct: lock notice — only Voice + Rate + Bitrate apply ── */}
                    {isSrtDub && (
                        <div className="rounded-lg bg-teal-500/5 border border-teal-500/20 p-3 space-y-2">
                            <p className="text-xs font-medium text-teal-400">SRT Direct — Simplified Settings</p>
                            <div className="grid grid-cols-2 gap-2 text-[10px] text-text-muted">
                                <div>ASR: skipped (your SRT)</div>
                                <div>Translate: skipped (your SRT)</div>
                                <div>TTS Engine: Edge-TTS only</div>
                                <div>Concat: zero gap, verbatim</div>
                                <div>Stretch: 1×–10× (your slider above)</div>
                                <div>Audio: never trimmed</div>
                            </div>
                            <p className="text-[10px] opacity-60">
                                Only <b>Voice</b>, <b>Speech Rate</b>, and <b>Audio Bitrate</b> below take effect.
                                All other settings are ignored in SRT Direct mode.
                            </p>
                        </div>
                    )}

                    {/* ── Transcription + Translation: HIDDEN in SRT mode and SRT Direct ── */}
                    {!isSrtMode && !isSrtDub && (<>
                        {/* ── Transcription Section ── */}
                        {/* Classic + Hybrid: show Whisper options. New: ASR engine picker. */}
                        <div>
                            <p className="text-sm font-medium text-text-primary mb-1">
                                Step 1 — Transcription{isNew ? ' (ASR Engine)' : ''}
                            </p>
                            <p className="text-[10px] text-text-muted mb-1">
                                Converts spoken audio into text with timestamps. This is the slowest step on long videos.
                            </p>
                            <p className="text-[10px] text-text-muted mb-3">
                                {isNew ? (
                                    <>
                                        <b className="text-text-secondary">New pipeline:</b> pick <i>Parakeet</i> for NVIDIA&apos;s SOTA text quality (reconciled with Whisper timing), or any <i>Whisper</i> size for a Whisper-only run. Every engine runs in an isolated child process, so a native crash can&apos;t take down the server.
                                    </>
                                ) : (
                                    <>
                                        <b className="text-text-secondary">Recommended:</b> use <i>Groq</i> (cloud, fastest) or <i>Use YouTube Transcript</i> below if the video has captions.
                                        Local models only matter if Groq is rate-limited.
                                    </>
                                )}
                                {whisperDisabled && !isNew && <span className="text-yellow-400 ml-1"> — Whisper currently skipped, using YouTube subs</span>}
                            </p>
                            <div className="space-y-3">
                                <div className={(whisperDisabled && !isNew) ? 'opacity-40 pointer-events-none' : ''}>
                                    <p className="text-xs text-text-muted mb-1.5">
                                        {isNew ? 'ASR Engine' : 'Whisper Model'}
                                    </p>
                                    <div className="grid grid-cols-5 gap-1.5">
                                        {[
                                            { value: 'base', label: 'Base', desc: 'Fastest' },
                                            { value: 'small', label: 'Small', desc: 'Fast' },
                                            { value: 'medium', label: 'Medium', desc: 'Balanced' },
                                            { value: 'large-v3-turbo', label: 'Turbo', desc: 'Fast+accurate' },
                                            { value: 'large-v3', label: 'Large-v3', desc: 'Best' },
                                            { value: 'parakeet', label: 'Parakeet', desc: 'NVIDIA SOTA' },
                                            { value: 'groq-whisper', label: 'Groq', desc: 'Cloud, fastest' },
                                        ]
                                            // Groq is not wired to the new DP pipeline — hide it there.
                                            .filter((m) => !(isNew && m.value === 'groq-whisper'))
                                            .map((m) => (
                                                <button
                                                    key={m.value}
                                                    onClick={() => update({ asr_model: m.value })}
                                                    className={`
                                                px-3 py-2 rounded-lg text-xs text-center transition-all border
                                                ${settings.asr_model === m.value
                                                            ? 'bg-primary/20 border-primary text-primary-light'
                                                            : 'bg-white/5 border-white/10 text-text-muted hover:bg-white/10'}
                                            `}
                                                >
                                                    <div className="font-medium">{m.label}</div>
                                                    <div className="text-[10px] opacity-70 mt-0.5">{m.desc}</div>
                                                </button>
                                            ))}
                                    </div>
                                </div>

                                {/* YouTube Subtitles — disabled when YT Auto-Translate is on */}
                                <div className={`flex items-center justify-between ${(ytTranslateOn || isOneFlow || isSrtMode) ? 'opacity-40 pointer-events-none' : ''}`}>
                                    <div>
                                        <p className="text-sm text-text-primary">Use YouTube Transcript ⚡</p>
                                        <p className="text-xs text-text-muted">
                                            Skip Whisper completely — download YouTube&apos;s caption file (~5-10s). Best for lectures, TED talks, tutorials with clean audio. Falls back to Whisper if no captions exist.
                                            {ytTranslateOn && <span className="text-yellow-400 ml-1">— disabled: YT Translate is on (cascade includes this automatically)</span>}
                                        </p>
                                    </div>
                                    <button
                                        type="button" title="Toggle YouTube Subtitles"
                                        onClick={() => update({
                                            prefer_youtube_subs: !settings.prefer_youtube_subs,
                                            ...(!settings.prefer_youtube_subs ? { use_yt_translate: false } : {}),
                                        })}
                                        className={`w-11 h-6 rounded-full transition-colors relative ${settings.prefer_youtube_subs ? 'bg-primary' : 'bg-white/10'}`}
                                    >
                                        <div className={`w-4 h-4 rounded-full bg-white absolute top-1 transition-transform ${settings.prefer_youtube_subs ? 'translate-x-6' : 'translate-x-1'}`} />
                                    </button>
                                </div>

                                {/* YouTube Auto-Translate — the CASCADE toggle */}
                                <div className={`flex items-center justify-between ${(ytSubsOn || transcribeOnly || isOneFlow || isSrtMode) ? 'opacity-40 pointer-events-none' : ''}`}>
                                    <div>
                                        <p className="text-sm text-text-primary">YT Auto-Translate ⚡⚡ <span className="text-[10px] text-green-400">CASCADE</span></p>
                                        <p className="text-xs text-text-muted">
                                            <b>Recommended.</b> Full cascade: tries YouTube Hindi first (best quality, fastest).
                                            If Hindi unavailable/429, falls back to YouTube English subs + Google Translate.
                                            If no subs at all, falls back to Whisper. All with retries + Premium cookies.
                                            {ytSubsOn && <span className="text-yellow-400 ml-1">— disabled: YT Transcript is on</span>}
                                            {transcribeOnly && !ytSubsOn && <span className="text-yellow-400 ml-1">— disabled: Transcribe Only is on</span>}
                                        </p>
                                    </div>
                                    <button
                                        type="button" title="Toggle YouTube Translate"
                                        onClick={() => update({
                                            use_yt_translate: !settings.use_yt_translate,
                                            ...(!settings.use_yt_translate ? { prefer_youtube_subs: false, transcribe_only: false } : {}),
                                        })}
                                        className={`w-11 h-6 rounded-full transition-colors relative ${settings.use_yt_translate ? 'bg-primary' : 'bg-white/10'}`}
                                    >
                                        <div className={`w-4 h-4 rounded-full bg-white absolute top-1 transition-transform ${settings.use_yt_translate ? 'translate-x-6' : 'translate-x-1'}`} />
                                    </button>
                                </div>

                                {/* YT Text Correction — works WITH Whisper, not instead of it */}
                                <div className={`flex items-center justify-between ${(ytTranslateOn || ytSubsOn || isOneFlow || isSrtMode) ? 'opacity-40 pointer-events-none' : ''}`}>
                                    <div>
                                        <p className="text-sm text-text-primary">YT Text Correction</p>
                                        <p className="text-xs text-text-muted">
                                            Whisper runs normally (keeps precise timestamps). YouTube subs fix text errors — proper nouns, punctuation, hallucinations. Best of both.
                                            {(ytTranslateOn || ytSubsOn) && <span className="text-yellow-400 ml-1">— disabled: YT subs mode already replaces Whisper</span>}
                                        </p>
                                    </div>
                                    <button
                                        type="button" title="Toggle YT Text Correction"
                                        onClick={() => update({ yt_text_correction: !settings.yt_text_correction })}
                                        className={`w-11 h-6 rounded-full transition-colors relative ${settings.yt_text_correction ? 'bg-primary' : 'bg-white/10'}`}
                                    >
                                        <div className={`w-4 h-4 rounded-full bg-white absolute top-1 transition-transform ${settings.yt_text_correction ? 'translate-x-6' : 'translate-x-1'}`} />
                                    </button>
                                </div>

                                {/* YT Replace Mode — only visible when YT Text Correction is ON */}
                                {settings.yt_text_correction && !ytTranslateOn && !ytSubsOn && !isOneFlow && !isSrtMode && (
                                    <div className="flex items-center justify-between ml-4">
                                        <div>
                                            <p className="text-xs font-medium text-text-secondary">Replace Mode</p>
                                            <p className="text-xs text-text-muted">
                                                {settings.yt_replace_mode === 'full'
                                                    ? 'Full — replace all Whisper text with YouTube text. Clean names but loses Whisper punctuation.'
                                                    : 'Diff — word-level comparison, only swap words that differ (names, nouns, misspellings). Keeps Whisper punctuation.'}
                                            </p>
                                        </div>
                                        <div className="flex rounded-lg overflow-hidden border border-border">
                                            <button
                                                type="button"
                                                onClick={() => update({ yt_replace_mode: 'diff' })}
                                                className={`px-3 py-1 text-[10px] font-medium transition-colors ${settings.yt_replace_mode === 'diff'
                                                    ? 'bg-primary text-white'
                                                    : 'bg-white/5 text-text-muted hover:bg-white/10'
                                                    }`}
                                            >
                                                Diff
                                            </button>
                                            <button
                                                type="button"
                                                onClick={() => update({ yt_replace_mode: 'full' })}
                                                className={`px-3 py-1 text-[10px] font-medium transition-colors ${settings.yt_replace_mode === 'full'
                                                    ? 'bg-primary text-white'
                                                    : 'bg-white/5 text-text-muted hover:bg-white/10'
                                                    }`}
                                            >
                                                Full
                                            </button>
                                        </div>
                                    </div>
                                )}

                                {/* Translation Glossary — words to keep/transliterate */}
                                <GlossaryEditor />

                                {/* YouTube Transcript Mode — only visible when YT subs are enabled */}
                                {(ytTranslateOn || ytSubsOn) && (
                                    <div className="flex items-center justify-between">
                                        <div>
                                            <p className="text-sm text-text-primary">YT Transcript Mode</p>
                                            <p className="text-xs text-text-muted">
                                                How YouTube subs are structured before feeding to TTS pipeline.
                                            </p>
                                        </div>
                                        <div className="flex rounded-lg overflow-hidden border border-border">
                                            {([
                                                { mode: 'yt_timeline', label: 'YT Timeline', desc: 'Fast (no Whisper)' },
                                                { mode: 'whisper_timeline', label: 'Whisper Timeline', desc: 'Precise (slower)' },
                                            ] as const).map(({ mode, label }) => (
                                                <button
                                                    key={mode}
                                                    type="button"
                                                    onClick={() => update({ yt_transcript_mode: mode })}
                                                    className={`px-3 py-1.5 text-xs font-medium transition-colors ${settings.yt_transcript_mode === mode
                                                        ? 'bg-primary text-white'
                                                        : 'bg-white/5 text-text-muted hover:bg-white/10'
                                                        }`}
                                                >
                                                    {label}
                                                </button>
                                            ))}
                                        </div>
                                    </div>
                                )}
                                {(ytTranslateOn || ytSubsOn) && (
                                    <div className="text-[10px] text-text-muted -mt-1 ml-1">
                                        {settings.yt_transcript_mode === 'whisper_timeline'
                                            ? 'YouTube text + Whisper precise timestamps. Runs Whisper for timing only, uses YouTube text (better quality). Slower but exact slot durations.'
                                            : 'YouTube text + YouTube timelines. Fast, no Whisper needed.'}
                                    </div>
                                )}

                                {/* Segment Split Mode — only visible when YT subs are enabled */}
                                {(ytTranslateOn || ytSubsOn) && (
                                    <div className="flex items-center justify-between">
                                        <div>
                                            <p className="text-sm text-text-primary">Segment Split Mode</p>
                                            <p className="text-xs text-text-muted">
                                                How text is split into segments for TTS.
                                            </p>
                                        </div>
                                        <div className="flex rounded-lg overflow-hidden border border-border">
                                            <button
                                                type="button"
                                                onClick={() => update({ yt_segment_mode: 'sentence' })}
                                                className={`px-3 py-1.5 text-xs font-medium transition-colors ${settings.yt_segment_mode === 'sentence'
                                                    ? 'bg-primary text-white'
                                                    : 'bg-white/5 text-text-muted hover:bg-white/10'
                                                    }`}
                                            >
                                                Sentence
                                            </button>
                                            <button
                                                type="button"
                                                onClick={() => update({ yt_segment_mode: 'wordcount' })}
                                                className={`px-3 py-1.5 text-xs font-medium transition-colors ${settings.yt_segment_mode === 'wordcount'
                                                    ? 'bg-primary text-white'
                                                    : 'bg-white/5 text-text-muted hover:bg-white/10'
                                                    }`}
                                            >
                                                Word Count
                                            </button>
                                        </div>
                                    </div>
                                )}
                                {(ytTranslateOn || ytSubsOn) && (
                                    <div className="text-[10px] text-text-muted -mt-1 ml-1">
                                        {settings.yt_segment_mode === 'wordcount'
                                            ? 'Even word-count split (~20 words/segment). Uniform speed, works without punctuation. Gaps removed anyway.'
                                            : '2 sentences per segment (atomic, never split). Needs punctuation in captions. Auto-falls back to word count if no punctuation detected.'}
                                    </div>
                                )}

                                {/* Chain Dub — disabled when YT Translate or New pipeline */}
                                <div className={`flex items-center justify-between ${(ytTranslateOn || isNew || isOneFlow || isSrtMode) ? 'opacity-40 pointer-events-none' : ''}`}>
                                    <div>
                                        <p className="text-sm text-text-primary">Chain Dub (English → Hindi)</p>
                                        <p className="text-xs text-text-muted">Dub to English first using subs, then English to Hindi (best for non-English videos)</p>
                                    </div>
                                    <button
                                        type="button" title="Toggle Chain Dub"
                                        onClick={() => update({ dub_chain: settings.dub_chain.length > 0 ? [] : ['en', 'hi'] })}
                                        className={`w-11 h-6 rounded-full transition-colors relative ${settings.dub_chain.length > 0 ? 'bg-primary' : 'bg-white/10'}`}
                                    >
                                        <div className={`w-4 h-4 rounded-full bg-white absolute top-1 transition-transform ${settings.dub_chain.length > 0 ? 'translate-x-6' : 'translate-x-1'}`} />
                                    </button>
                                </div>

                                {/* WhisperX — disabled when Whisper skipped or New pipeline (built-in) */}
                                <div className={`flex items-center justify-between ${(whisperxDisabled || isNew || isOneFlow || isSrtMode) ? 'opacity-40 pointer-events-none' : ''}`}>
                                    <div>
                                        <p className="text-sm text-text-primary">WhisperX Alignment</p>
                                        <p className="text-xs text-text-muted">
                                            Force word-level timestamp alignment (requires whisperx)
                                            {whisperDisabled && <span className="text-yellow-400 ml-1">— needs Whisper</span>}
                                        </p>
                                    </div>
                                    <button
                                        type="button" title="Toggle WhisperX Alignment" onClick={() => update({ use_whisperx: !settings.use_whisperx })}
                                        className={`w-11 h-6 rounded-full transition-colors relative ${settings.use_whisperx ? 'bg-primary' : 'bg-white/10'}`}
                                    >
                                        <div className={`w-4 h-4 rounded-full bg-white absolute top-1 transition-transform ${settings.use_whisperx ? 'translate-x-6' : 'translate-x-1'}`} />
                                    </button>
                                </div>

                                {/* Wav2Lip lip-sync — post-assembly, opt-in. Needs GPU + backend/wav2lip/ + checkpoint */}
                                <div className="flex items-center justify-between">
                                    <div>
                                        <p className="text-sm text-text-primary">Wav2Lip Lip Sync</p>
                                        <p className="text-xs text-text-muted">
                                            Re-sync speaker&apos;s lips to dubbed audio (post-process). Needs GPU + manual setup: clone <span className="font-mono">Rudrabha/Wav2Lip</span> into <span className="font-mono">backend/wav2lip/</span> and place <span className="font-mono">wav2lip_gan.pth</span> in <span className="font-mono">checkpoints/</span>. Skipped gracefully if missing.
                                        </p>
                                    </div>
                                    <button
                                        type="button" title="Toggle Wav2Lip lip sync" onClick={() => update({ use_wav2lip: !settings.use_wav2lip } as any)}
                                        className={`w-11 h-6 rounded-full transition-colors relative ${settings.use_wav2lip ? 'bg-primary' : 'bg-white/10'}`}
                                    >
                                        <div className={`w-4 h-4 rounded-full bg-white absolute top-1 transition-transform ${settings.use_wav2lip ? 'translate-x-6' : 'translate-x-1'}`} />
                                    </button>
                                </div>

                                {/* Simplify English — only for Classic mode. Hybrid/New use DP cue builder */}
                                <div className={`flex items-center justify-between ${simplifyDisabled || !isClassic ? 'opacity-40 pointer-events-none' : ''}`}>
                                    <div>
                                        <p className="text-sm text-text-primary">Simplify English</p>
                                        <p className="text-xs text-text-muted">
                                            Rewrite complex English into simple sentences — much better Hindi
                                            {ytTranslateOn && <span className="text-yellow-400 ml-1">— not needed with YT Translate</span>}
                                        </p>
                                    </div>
                                    <button
                                        type="button" title="Toggle Simplify English" onClick={() => update({ simplify_english: !settings.simplify_english })}
                                        className={`w-11 h-6 rounded-full transition-colors relative ${settings.simplify_english ? 'bg-primary' : 'bg-white/10'}`}
                                    >
                                        <div className={`w-4 h-4 rounded-full bg-white absolute top-1 transition-transform ${settings.simplify_english ? 'translate-x-6' : 'translate-x-1'}`} />
                                    </button>
                                </div>
                            </div>
                        </div>

                        {/* ── Upload Transcript SRT — skip transcription entirely ── */}
                        <div className="rounded-lg p-4 bg-emerald-500/5 border border-emerald-500/20 space-y-3">
                            <div>
                                <p className="text-sm font-medium text-emerald-300">
                                    Upload Transcript (Skip Transcription)
                                </p>
                                <p className="text-xs text-text-muted mt-1">
                                    Have your own English SRT transcript? Upload it here to skip Whisper/YouTube transcription entirely.
                                    The pipeline will go straight to translation → TTS → assembly.
                                </p>
                            </div>
                            <div className="flex items-center gap-2">
                                <input
                                    type="file"
                                    accept=".srt,text/plain"
                                    aria-label="Upload transcript SRT file"
                                    title="Upload transcript SRT file"
                                    onChange={async (e) => {
                                        const file = e.target.files?.[0];
                                        if (!file) return;
                                        const buf = await file.arrayBuffer();
                                        const tryDecode = (label: string): string | null => {
                                            try {
                                                const dec = new TextDecoder(label, { fatal: false });
                                                return dec.decode(buf);
                                            } catch { return null; }
                                        };
                                        let text = tryDecode("utf-8") ?? "";
                                        if (text.includes("\uFFFD")) {
                                            for (const enc of ["utf-16le", "utf-16be", "windows-1252"]) {
                                                const alt = tryDecode(enc);
                                                if (alt && !alt.includes("\uFFFD")) { text = alt; break; }
                                            }
                                        }
                                        update({ transcript_srt_content: text.trim() } as any);
                                    }}
                                    className="text-xs text-text-muted file:mr-2 file:py-1 file:px-3 file:rounded file:border-0 file:bg-emerald-600 file:text-white file:text-xs file:cursor-pointer"
                                />
                                {(settings as any).transcript_srt_content && (
                                    <button
                                        type="button"
                                        onClick={() => update({ transcript_srt_content: '' } as any)}
                                        className="text-xs text-red-400 hover:text-red-300 px-2 py-1 rounded bg-red-500/10"
                                    >
                                        Clear
                                    </button>
                                )}
                            </div>
                            <textarea
                                value={(settings as any).transcript_srt_content || ''}
                                onChange={(e) => update({ transcript_srt_content: e.target.value } as any)}
                                placeholder={"1\n00:00:01,000 --> 00:00:04,000\nHello, welcome to the video.\n\n2\n00:00:04,500 --> 00:00:08,000\nToday we will learn about..."}
                                rows={4}
                                className="w-full rounded-md bg-white/5 border border-white/10 px-3 py-2 text-xs text-text-primary placeholder:text-text-muted/50 focus:outline-none focus:ring-1 focus:ring-emerald-500/50 font-mono resize-y"
                            />
                            {(settings as any).transcript_srt_content && (
                                <p className="text-[10px] text-emerald-400">
                                    ✓ Transcript loaded ({((settings as any).transcript_srt_content ?? '').length} chars) — transcription step will be SKIPPED
                                </p>
                            )}
                        </div>

                        {/* ── Translation Section ── */}
                        <div className={translationDisabled ? 'opacity-40 pointer-events-none' : ''}>
                            <p className="text-sm font-medium text-text-primary mb-1">Step 2 — Translation</p>
                            <p className="text-[10px] text-text-muted mb-1">
                                {(isHybrid || isNew)
                                    ? 'DP cue boundaries + glossary lock + Hindi fitting + QC gates applied automatically.'
                                    : 'Translates transcribed text into your target language.'}
                            </p>
                            <p className="text-[10px] text-text-muted mb-3">
                                <b className="text-text-secondary">Recommended:</b> <i>Google</i> = fastest free (used by default). <i>Gemma 4</i> = best Hindi quality but needs GEMINI_API_KEY. <i>Cerebras</i> = ultra-fast LLM, also good.
                                {ytTranslateOn && <span className="text-yellow-400 ml-1"> Skipped — YouTube already translated.</span>}
                            </p>
                            <div className="grid grid-cols-6 gap-2 mb-3">
                                {[
                                    { value: 'auto', label: 'Auto', desc: 'Best available' },
                                    { value: 'gemma4', label: 'Gemma 4', desc: 'Best Hindi (free, 31B)' },
                                    { value: 'turbo', label: 'Turbo', desc: 'Groq+SambaNova parallel' },
                                    { value: 'groq', label: 'Groq', desc: 'Llama 3.3 70B (free)' },
                                    { value: 'sambanova', label: 'SambaNova', desc: 'Llama 3.3 70B (free)' },
                                    { value: 'gemini', label: 'Gemini', desc: 'Google AI (free)' },
                                ].map((m) => (
                                    <button
                                        key={m.value}
                                        onClick={() => update({ translation_engine: m.value })}
                                        className={`
                                        px-3 py-2 rounded-lg text-xs text-center transition-all border
                                        ${settings.translation_engine === m.value
                                                ? 'bg-primary/20 border-primary text-primary-light'
                                                : 'bg-white/5 border-white/10 text-text-muted hover:bg-white/10'}
                                    `}
                                    >
                                        <div className="font-medium">{m.label}</div>
                                        <div className="text-[10px] opacity-70 mt-0.5">{m.desc}</div>
                                    </button>
                                ))}
                            </div>
                            <div className="grid grid-cols-4 gap-2">
                                {[
                                    { value: 'chain_dub', label: 'Chain Dub', desc: 'IndicTrans2+ → Turbo refine' },
                                    { value: 'nllb_polish', label: 'IndicTrans2+', desc: 'IndicTrans2 → LLM → Rules' },
                                    { value: 'google_polish', label: 'Google+Polish', desc: 'Fast Google → LLM polish' },
                                    { value: 'nllb', label: 'IndicTrans2', desc: 'Local meaning model' },
                                    { value: 'ollama', label: 'Ollama', desc: 'Local LLM (GPU)' },
                                    { value: 'hinglish', label: 'Hinglish AI', desc: 'Custom Hindi model' },
                                    { value: 'google', label: 'Google', desc: 'Free, FASTEST (parallel x20)' },
                                    { value: 'cerebras', label: 'Cerebras', desc: 'Llama 3.3 70B, fastest LLM' },
                                    { value: 'seamless', label: 'SeamlessM4T', desc: 'Meta end-to-end (GPU)' },
                                ].map((m) => (
                                    <button
                                        key={m.value}
                                        onClick={() => update({ translation_engine: m.value })}
                                        className={`
                                        px-3 py-2 rounded-lg text-xs text-center transition-all border
                                        ${settings.translation_engine === m.value
                                                ? 'bg-primary/20 border-primary text-primary-light'
                                                : 'bg-white/5 border-white/10 text-text-muted hover:bg-white/10'}
                                    `}
                                    >
                                        <div className="font-medium">{m.label}</div>
                                        <div className="text-[10px] opacity-70 mt-0.5">{m.desc}</div>
                                    </button>
                                ))}
                            </div>
                        </div>

                        {/* ── New Pipeline Features (hybrid + new only) ── */}
                        {(isHybrid || isNew) && (
                            <div className="rounded-lg bg-green-500/5 border border-green-500/20 p-3 space-y-2">
                                <p className="text-xs font-medium text-green-400">New Pipeline Features (Active)</p>
                                <div className="grid grid-cols-2 gap-2 text-[10px] text-text-muted">
                                    <div className="flex items-center gap-1.5">
                                        <span className="w-1.5 h-1.5 rounded-full bg-green-400" />
                                        DP Cue Builder
                                    </div>
                                    <div className="flex items-center gap-1.5">
                                        <span className="w-1.5 h-1.5 rounded-full bg-green-400" />
                                        Glossary Lock
                                    </div>
                                    <div className="flex items-center gap-1.5">
                                        <span className="w-1.5 h-1.5 rounded-full bg-green-400" />
                                        Hindi Fitting
                                    </div>
                                    <div className="flex items-center gap-1.5">
                                        <span className="w-1.5 h-1.5 rounded-full bg-green-400" />
                                        Pre-TTS QC Gate
                                    </div>
                                    <div className="flex items-center gap-1.5">
                                        <span className="w-1.5 h-1.5 rounded-full bg-green-400" />
                                        English QC
                                    </div>
                                    <div className="flex items-center gap-1.5">
                                        <span className="w-1.5 h-1.5 rounded-full bg-green-400" />
                                        {isNew ? 'Parakeet + WhisperX' : 'Whisper + DP Cues'}
                                    </div>
                                </div>
                            </div>
                        )}

                        {/* Transcribe Only — disabled when YT Auto-Translate or New pipeline */}
                        <div className={`flex items-center justify-between ${(ytTranslateOn || isNew || isOneFlow || isSrtMode) ? 'opacity-40 pointer-events-none' : ''}`}>
                            <div>
                                <p className="text-sm text-text-primary">Transcribe Only</p>
                                <p className="text-xs text-text-muted">
                                    Get SRT to translate yourself (e.g. with Claude), then upload back
                                    {ytTranslateOn && <span className="text-yellow-400 ml-1">— off: YT Translate active</span>}
                                </p>
                            </div>
                            <button
                                type="button" title="Toggle Transcribe Only"
                                onClick={() => update({
                                    transcribe_only: !settings.transcribe_only,
                                    ...(!settings.transcribe_only ? { use_yt_translate: false } : {}),
                                })}
                                className={`w-11 h-6 rounded-full transition-colors relative ${settings.transcribe_only ? 'bg-primary' : 'bg-white/10'}`}
                            >
                                <div className={`w-4 h-4 rounded-full bg-white absolute top-1 transition-transform ${settings.transcribe_only ? 'translate-x-6' : 'translate-x-1'}`} />
                            </button>
                        </div>

                        {/* Multi-Speaker Voices — disabled when YT Translate or New pipeline */}
                        <div className={`flex items-center justify-between ${(ytTranslateOn || isNew || isOneFlow || isSrtMode) ? 'opacity-40 pointer-events-none' : ''}`}>
                            <div>
                                <p className="text-sm text-text-primary">Multi-Speaker Voices</p>
                                <p className="text-xs text-text-muted">Detect speakers & assign distinct voices (needs HF_TOKEN, adds ~30s)</p>
                            </div>
                            <button
                                type="button" title="Toggle Multi-speaker" onClick={() => update({ multi_speaker: !settings.multi_speaker })}
                                className={`
                                w-11 h-6 rounded-full transition-colors relative
                                ${settings.multi_speaker ? 'bg-primary' : 'bg-white/10'}
                            `}
                            >
                                <div className={`
                                w-4 h-4 rounded-full bg-white absolute top-1 transition-transform
                                ${settings.multi_speaker ? 'translate-x-6' : 'translate-x-1'}
                            `} />
                            </button>
                        </div>

                    </>)}
                    {/* ── End of Transcription + Translation (hidden in SRT mode) ── */}

                    {/* TTS Engines + Rate Mode — HIDDEN in SRT Direct (Edge-TTS only, no auto-rate) */}
                    {!isSrtDub && (<>
                        {/* TTS Engines */}
                        <div className={isVoiceClone ? 'opacity-40 pointer-events-none' : ''}>
                            {isVoiceClone && (
                                <div className="rounded-lg p-2 mb-2 bg-violet-500/10 border border-violet-500/30 pointer-events-auto opacity-100">
                                    <p className="text-[10px] text-violet-400">Voice Clone active — TTS locked to XTTS. Click Voice Clone button to deactivate.</p>
                                </div>
                            )}
                            <p className="text-sm font-medium text-text-primary mb-1">Step 3 — TTS Voice Engine</p>

                            {/* Edge-TTS Voice — dynamic per language */}
                            {(() => {
                                const voiceMap: Record<string, { male: { value: string; label: string }; female: { value: string; label: string } }> = {
                                    hi: { male: { value: 'hi-IN-MadhurNeural', label: 'Madhur' }, female: { value: 'hi-IN-SwaraNeural', label: 'Swara' } },
                                    bn: { male: { value: 'bn-IN-BashkarNeural', label: 'Bashkar' }, female: { value: 'bn-IN-TanishaaNeural', label: 'Tanishaa' } },
                                    ta: { male: { value: 'ta-IN-ValluvarNeural', label: 'Valluvar' }, female: { value: 'ta-IN-PallaviNeural', label: 'Pallavi' } },
                                    te: { male: { value: 'te-IN-MohanNeural', label: 'Mohan' }, female: { value: 'te-IN-ShrutiNeural', label: 'Shruti' } },
                                    mr: { male: { value: 'mr-IN-ManoharNeural', label: 'Manohar' }, female: { value: 'mr-IN-AarohiNeural', label: 'Aarohi' } },
                                    gu: { male: { value: 'gu-IN-NiranjanNeural', label: 'Niranjan' }, female: { value: 'gu-IN-DhwaniNeural', label: 'Dhwani' } },
                                    kn: { male: { value: 'kn-IN-GaganNeural', label: 'Gagan' }, female: { value: 'kn-IN-SapnaNeural', label: 'Sapna' } },
                                    ml: { male: { value: 'ml-IN-MidhunNeural', label: 'Midhun' }, female: { value: 'ml-IN-SobhanaNeural', label: 'Sobhana' } },
                                    pa: { male: { value: 'pa-IN-GurpreetNeural', label: 'Gurpreet' }, female: { value: 'pa-IN-OjasNeural', label: 'Ojas' } },
                                    ur: { male: { value: 'ur-PK-AsadNeural', label: 'Asad' }, female: { value: 'ur-PK-UzmaNeural', label: 'Uzma' } },
                                    en: { male: { value: 'en-US-GuyNeural', label: 'Guy' }, female: { value: 'en-US-JennyNeural', label: 'Jenny' } },
                                    es: { male: { value: 'es-ES-AlvaroNeural', label: 'Alvaro' }, female: { value: 'es-ES-ElviraNeural', label: 'Elvira' } },
                                    fr: { male: { value: 'fr-FR-HenriNeural', label: 'Henri' }, female: { value: 'fr-FR-DeniseNeural', label: 'Denise' } },
                                    de: { male: { value: 'de-DE-ConradNeural', label: 'Conrad' }, female: { value: 'de-DE-KatjaNeural', label: 'Katja' } },
                                    ja: { male: { value: 'ja-JP-KeitaNeural', label: 'Keita' }, female: { value: 'ja-JP-NanamiNeural', label: 'Nanami' } },
                                    ko: { male: { value: 'ko-KR-InJoonNeural', label: 'InJoon' }, female: { value: 'ko-KR-SunHiNeural', label: 'SunHi' } },
                                    zh: { male: { value: 'zh-CN-YunxiNeural', label: 'Yunxi' }, female: { value: 'zh-CN-XiaoxiaoNeural', label: 'Xiaoxiao' } },
                                    pt: { male: { value: 'pt-BR-AntonioNeural', label: 'Antonio' }, female: { value: 'pt-BR-FranciscaNeural', label: 'Francisca' } },
                                    ru: { male: { value: 'ru-RU-DmitryNeural', label: 'Dmitry' }, female: { value: 'ru-RU-SvetlanaNeural', label: 'Svetlana' } },
                                    ar: { male: { value: 'ar-SA-HamedNeural', label: 'Hamed' }, female: { value: 'ar-SA-ZariyahNeural', label: 'Zariyah' } },
                                    it: { male: { value: 'it-IT-DiegoNeural', label: 'Diego' }, female: { value: 'it-IT-ElsaNeural', label: 'Elsa' } },
                                    tr: { male: { value: 'tr-TR-AhmetNeural', label: 'Ahmet' }, female: { value: 'tr-TR-EmelNeural', label: 'Emel' } },
                                };
                                const lang = targetLanguage || 'hi';
                                const voices = voiceMap[lang] || voiceMap['hi'];
                                const options = [
                                    { ...voices.male, desc: 'Male', color: 'blue' as const },
                                    { ...voices.female, desc: 'Female', color: 'pink' as const },
                                ];
                                return (
                                    <div className="mb-3">
                                        <p className="text-xs text-text-secondary mb-1">Edge-TTS Voice</p>
                                        <div className="grid grid-cols-2 gap-2">
                                            {options.map((v) => (
                                                <button key={v.value} type="button"
                                                    onClick={() => update({ voice: v.value })}
                                                    className={`flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-medium transition-all border ${settings.voice === v.value
                                                        ? v.color === 'blue'
                                                            ? 'bg-blue-500/20 text-blue-400 border-blue-500/30'
                                                            : 'bg-pink-500/20 text-pink-400 border-pink-500/30'
                                                        : 'bg-white/5 text-text-muted border-white/5 hover:border-white/20'
                                                        }`}
                                                >
                                                    <span className={`w-6 h-6 rounded-full flex items-center justify-center text-[10px] ${v.color === 'blue' ? 'bg-blue-500/20 text-blue-400' : 'bg-pink-500/20 text-pink-400'
                                                        }`}>
                                                        {v.color === 'blue' ? 'M' : 'F'}
                                                    </span>
                                                    <div>
                                                        <div>{v.label}</div>
                                                        <div className="text-[10px] opacity-60">{v.desc}</div>
                                                    </div>
                                                </button>
                                            ))}
                                        </div>
                                    </div>
                                );
                            })()}
                            <p className="text-[10px] text-text-muted mb-1">
                                Generates the spoken Hindi (or target-language) audio. Pick ONE main engine — enabling multiple runs them in parallel for speed.
                            </p>
                            <p className="text-[10px] text-text-muted mb-3">
                                <b className="text-text-secondary">For Hindi:</b> CosyVoice 2 (best, GPU) → Sarvam Bulbul (cloud, premium) → Indic Parler → Edge-TTS (free fallback).
                                <b className="text-text-secondary"> For English:</b> auto-chains Chatterbox → XTTS → Edge.
                                <b className="text-text-secondary"> Free + no setup:</b> Edge-TTS only.
                            </p>
                            <div className="space-y-3">
                                {/* CosyVoice 2 */}
                                <div className="flex items-center justify-between">
                                    <div>
                                        <p className="text-sm text-text-primary">CosyVoice 2 ⭐ <span className="text-[10px] text-green-400">BEST FOR HINDI</span></p>
                                        <p className="text-xs text-text-muted">Free · GPU required · clones the original speaker&apos;s voice in Hindi · near-ElevenLabs quality · slow on CPU</p>
                                    </div>
                                    <button
                                        type="button"
                                        title="Toggle CosyVoice 2"
                                        onClick={() => update({ use_cosyvoice: !settings.use_cosyvoice })}
                                        className={`w-11 h-6 rounded-full transition-colors relative ${settings.use_cosyvoice ? 'bg-primary' : 'bg-white/10'}`}
                                    >
                                        <div className={`w-4 h-4 rounded-full bg-white absolute top-1 transition-transform ${settings.use_cosyvoice ? 'translate-x-6' : 'translate-x-1'}`} />
                                    </button>
                                </div>
                                {/* Chatterbox */}
                                {/* Indic Parler-TTS */}
                                <div className="flex items-center justify-between">
                                    <div>
                                        <p className="text-sm text-text-primary">Indic Parler-TTS <span className="text-[10px] text-blue-400">HINDI-NATIVE</span></p>
                                        <p className="text-xs text-text-muted">Free · GPU · purpose-built for Hindi &amp; Indic languages by AI4Bharat · good fallback if CosyVoice fails</p>
                                    </div>
                                    <button
                                        type="button" title="Toggle Indic Parler-TTS" onClick={() => update({ use_indic_parler: !settings.use_indic_parler })}
                                        className={`w-11 h-6 rounded-full transition-colors relative ${settings.use_indic_parler ? 'bg-primary' : 'bg-white/10'}`}
                                    >
                                        <div className={`w-4 h-4 rounded-full bg-white absolute top-1 transition-transform ${settings.use_indic_parler ? 'translate-x-6' : 'translate-x-1'}`} />
                                    </button>
                                </div>
                                {/* Sarvam Bulbul v3 */}
                                <div className="flex items-center justify-between">
                                    <div>
                                        <p className="text-sm text-text-primary">Sarvam Bulbul v3 <span className="text-[10px] text-purple-400">PREMIUM CLOUD</span></p>
                                        <p className="text-xs text-text-muted">Cloud API · arguably the best-sounding Hindi voice available · needs SARVAM_API_KEY in .env · paid after free quota</p>
                                    </div>
                                    <button
                                        type="button" title="Toggle Sarvam Bulbul" onClick={() => update({ use_sarvam_bulbul: !settings.use_sarvam_bulbul })}
                                        className={`w-11 h-6 rounded-full transition-colors relative ${settings.use_sarvam_bulbul ? 'bg-primary' : 'bg-white/10'}`}
                                    >
                                        <div className={`w-4 h-4 rounded-full bg-white absolute top-1 transition-transform ${settings.use_sarvam_bulbul ? 'translate-x-6' : 'translate-x-1'}`} />
                                    </button>
                                </div>
                                {/* Chatterbox */}
                                <div className="flex items-center justify-between">
                                    <div>
                                        <p className="text-sm text-text-primary">Chatterbox AI <span className="text-[10px] text-blue-400">ENGLISH ONLY</span></p>
                                        <p className="text-xs text-text-muted">Free · GPU · most human-like English voice · use only for English-target dubs</p>
                                    </div>
                                    <button
                                        type="button" title="Toggle Chatterbox" onClick={() => update({ use_chatterbox: !settings.use_chatterbox })}
                                        className={`
                                        w-11 h-6 rounded-full transition-colors relative
                                        ${settings.use_chatterbox ? 'bg-primary' : 'bg-white/10'}
                                    `}
                                    >
                                        <div className={`
                                        w-4 h-4 rounded-full bg-white absolute top-1 transition-transform
                                        ${settings.use_chatterbox ? 'translate-x-6' : 'translate-x-1'}
                                    `} />
                                    </button>
                                </div>

                                {/* ElevenLabs */}
                                <div className="flex items-center justify-between">
                                    <div>
                                        <p className="text-sm text-text-primary">ElevenLabs</p>
                                        <p className="text-xs text-text-muted">
                                            Paid API, needs ELEVENLABS_API_KEY in .env
                                            {settings.use_elevenlabs && <span className="text-yellow-400 ml-1">— make sure API key is set or job will fail!</span>}
                                        </p>
                                    </div>
                                    <button
                                        type="button" title="Toggle ElevenLabs" onClick={() => update({ use_elevenlabs: !settings.use_elevenlabs })}
                                        className={`
                                        w-11 h-6 rounded-full transition-colors relative
                                        ${settings.use_elevenlabs ? 'bg-primary' : 'bg-white/10'}
                                    `}
                                    >
                                        <div className={`
                                        w-4 h-4 rounded-full bg-white absolute top-1 transition-transform
                                        ${settings.use_elevenlabs ? 'translate-x-6' : 'translate-x-1'}
                                    `} />
                                    </button>
                                </div>

                                {/* Coqui XTTS v2 */}
                                <div className="flex items-center justify-between">
                                    <div>
                                        <p className="text-sm text-text-primary">Coqui XTTS v2 <span className="text-[10px] text-green-400">VOICE CLONE</span></p>
                                        <p className="text-xs text-text-muted">Free · GPU · clones the original speaker&apos;s voice · best for same-language re-voicing (use the &quot;Voice Clone&quot; preset above)</p>
                                    </div>
                                    <button
                                        type="button" title="Toggle Coqui XTTS v2" onClick={() => update({ use_coqui_xtts: !settings.use_coqui_xtts })}
                                        className={`
                                        w-11 h-6 rounded-full transition-colors relative
                                        ${settings.use_coqui_xtts ? 'bg-primary' : 'bg-white/10'}
                                    `}
                                    >
                                        <div className={`
                                        w-4 h-4 rounded-full bg-white absolute top-1 transition-transform
                                        ${settings.use_coqui_xtts ? 'translate-x-6' : 'translate-x-1'}
                                    `} />
                                    </button>
                                </div>

                                {/* Google Cloud TTS */}
                                <div className="flex items-center justify-between">
                                    <div>
                                        <p className="text-sm text-text-primary">Google Cloud TTS</p>
                                        <p className="text-xs text-text-muted">Free 1M chars/mo, WaveNet/Neural2 voices, needs GCP credentials</p>
                                    </div>
                                    <button
                                        type="button" title="Toggle Google TTS" onClick={() => update({ use_google_tts: !settings.use_google_tts })}
                                        className={`
                                        w-11 h-6 rounded-full transition-colors relative
                                        ${settings.use_google_tts ? 'bg-primary' : 'bg-white/10'}
                                    `}
                                    >
                                        <div className={`
                                        w-4 h-4 rounded-full bg-white absolute top-1 transition-transform
                                        ${settings.use_google_tts ? 'translate-x-6' : 'translate-x-1'}
                                    `} />
                                    </button>
                                </div>

                                {/* Edge-TTS */}
                                <div className="flex items-center justify-between">
                                    <div>
                                        <p className="text-sm text-text-primary">Edge-TTS <span className="text-[10px] text-amber-400">FASTEST · NO SETUP</span></p>
                                        <p className="text-xs text-text-muted">Free · cloud · no GPU · supports 70+ languages · decent quality · use this when you want speed and zero installation</p>
                                    </div>
                                    <button
                                        type="button" title="Toggle Edge TTS" onClick={() => update({ use_edge_tts: !settings.use_edge_tts })}
                                        className={`
                                        w-11 h-6 rounded-full transition-colors relative
                                        ${settings.use_edge_tts ? 'bg-primary' : 'bg-white/10'}
                                    `}
                                    >
                                        <div className={`
                                        w-4 h-4 rounded-full bg-white absolute top-1 transition-transform
                                        ${settings.use_edge_tts ? 'translate-x-6' : 'translate-x-1'}
                                    `} />
                                    </button>
                                </div>
                            </div>
                            {settings.use_cosyvoice && settings.use_coqui_xtts && settings.use_edge_tts ? (
                                <p className="text-[10px] text-primary mt-2 font-medium">
                                    Managed Parallel: CosyVoice + XTTS + Sarvam (quality reserved) / Edge (overflow only)
                                </p>
                            ) : settings.use_coqui_xtts && settings.use_edge_tts ? (
                                <p className="text-[10px] text-primary mt-2 font-medium">
                                    Hybrid Mode: Coqui XTTS + Edge-TTS will run in parallel (~2x faster)
                                </p>
                            ) : (
                                <p className="text-[10px] text-text-muted mt-2">First enabled engine from top to bottom will be used. Enable CosyVoice + Coqui + Edge for triple parallel mode.</p>
                            )}
                        </div>

                        {/* TTS Rate Mode — Auto (match source duration) or Manual */}
                        <div>
                            <label className="label mb-2 block">
                                Speech Rate Mode
                                {settings.tts_rate_mode === 'auto' && (
                                    <span className="text-[10px] text-green-400 ml-2">AUTO · RECOMMENDED</span>
                                )}
                            </label>
                            <p className="text-xs text-text-muted mb-2">
                                <b>Auto</b> (default): the pipeline computes the optimal speech rate from
                                the source video duration and translated word count so the dubbed output
                                matches the original runtime. Capped at <b>+50% (1.5x)</b> — beyond that,
                                the remaining overflow is absorbed by video stretching in assembly.
                                {' '}<b>Manual</b>: uses the Speech Rate slider below as-is (the old behavior).
                            </p>
                            <div className="grid grid-cols-2 gap-2 mb-3">
                                <button
                                    type="button"
                                    onClick={() => update({ tts_rate_mode: 'auto' })}
                                    className={`
                                    px-2 py-2 rounded-lg text-xs text-center transition-all border
                                    ${settings.tts_rate_mode === 'auto'
                                            ? 'bg-primary/20 border-primary text-primary-light'
                                            : 'bg-white/5 border-white/10 text-text-muted hover:bg-white/10'}
                                `}
                                >
                                    <div className="font-medium">Auto</div>
                                    <div className="text-[10px] opacity-70 mt-0.5">Match source duration</div>
                                </button>
                                <button
                                    type="button"
                                    onClick={() => update({ tts_rate_mode: 'manual' })}
                                    className={`
                                    px-2 py-2 rounded-lg text-xs text-center transition-all border
                                    ${settings.tts_rate_mode === 'manual'
                                            ? 'bg-primary/20 border-primary text-primary-light'
                                            : 'bg-white/5 border-white/10 text-text-muted hover:bg-white/10'}
                                `}
                                >
                                    <div className="font-medium">Manual</div>
                                    <div className="text-[10px] opacity-70 mt-0.5">Use slider below</div>
                                </button>
                            </div>
                        </div>
                    </>)}
                    {/* ── End of TTS Engines + Rate Mode (hidden in SRT Direct) ── */}

                    {/* TTS Speech Rate — Edge-TTS native prosody rate (visible in all modes) */}
                    <div className={(!isSrtDub && settings.tts_rate_mode === 'auto') ? 'opacity-50 pointer-events-none' : ''}>
                        <label className="label mb-2 block">
                            Speech Rate: <span className="text-primary-light font-mono">{settings.tts_rate}</span>
                            <span className="text-[10px] text-text-muted ml-2 font-mono">
                                ({(1 + (parseInt(settings.tts_rate) || 0) / 100).toFixed(2)}x)
                            </span>
                            {settings.tts_rate_mode === 'auto' && (
                                <span className="text-[10px] text-amber-400 ml-2">
                                    disabled (Auto mode computes this)
                                </span>
                            )}
                        </label>
                        <p className="text-xs text-text-muted mb-2">
                            Edge-TTS native prosody rate at synthesis time. Sounds <b>natural</b> at
                            higher speeds (no chipmunk effect, no time-stretch artifacts). Only used
                            when mode is <b>Manual</b>.
                        </p>
                        <input
                            type="range"
                            min={-50}
                            max={75}
                            step={5}
                            value={parseInt(settings.tts_rate) || 0}
                            onChange={(e) => {
                                const v = parseInt(e.target.value);
                                update({ tts_rate: `${v >= 0 ? '+' : ''}${v}%` });
                            }}
                            disabled={settings.tts_rate_mode === 'auto'}
                            className="w-full accent-primary"
                            aria-label="TTS speech rate"
                        />
                        <div className="flex justify-between text-[10px] text-text-muted mt-1">
                            <span>-50%</span>
                            <span>0%</span>
                            <span>+25%</span>
                            <span>+50%</span>
                            <span>+75%</span>
                        </div>
                        <div className="flex justify-between text-[9px] text-text-muted">
                            <span>0.5x</span>
                            <span>1.0x</span>
                            <span>1.25x</span>
                            <span>1.5x</span>
                            <span>1.75x</span>
                        </div>
                    </div>

                    {/* Keep Background Music — Demucs bed ducked under the Hindi voice */}
                    {!isSrtDub && (
                        <div className={`flex items-center justify-between mb-3 ${(isOneFlow || isSrtMode) ? 'opacity-40 pointer-events-none' : ''}`}>
                            <div className="pr-3">
                                <p className="text-sm text-text-primary">Keep Background Music</p>
                                <p className="text-xs text-text-muted">
                                    Demucs isolates the original music/SFX and mixes it (auto-ducked) under the Hindi voice.
                                    The bed is pitch/EQ-shifted to reduce Content-ID matching — this lowers, but does not
                                    guarantee, avoiding a music copyright claim. Adds a GPU separation step.
                                </p>
                            </div>
                            <button
                                type="button" title="Toggle Keep Background Music" onClick={() => update({ mix_original: !settings.mix_original })}
                                className={`w-11 h-6 rounded-full transition-colors relative shrink-0 ${settings.mix_original ? 'bg-primary' : 'bg-white/10'}`}
                            >
                                <div className={`w-4 h-4 rounded-full bg-white absolute top-1 transition-transform ${settings.mix_original ? 'translate-x-6' : 'translate-x-1'}`} />
                            </button>
                        </div>
                    )}

                    {/* ── Audio & Performance Section — HIDDEN in SRT Direct (it has its own assembly) ── */}
                    {!isSrtDub && (
                        <div>
                            <p className="text-sm font-medium text-text-primary mb-1">Step 4 — Audio &amp; Video Assembly</p>
                            {isVoiceClone && (
                                <div className="rounded-lg p-2 mb-2 bg-violet-500/10 border border-violet-500/30">
                                    <p className="text-[10px] text-violet-400">Voice Clone active — audio is untouchable, post-processing disabled. AV sync modules blocked.</p>
                                </div>
                            )}
                            <p className="text-[10px] text-text-muted mb-1">
                                Controls how the dubbed audio gets stitched onto the video. Hindi sentences are usually <b className="text-text-secondary">longer</b> than English, so something has to give: either speed up audio, slow down video, or both.
                            </p>
                            <p className="text-[10px] text-text-muted mb-3">
                                <b className="text-text-secondary">Default (split-the-diff):</b> Audio Priority ON + Video Slow to Match ON → mild audio speedup + mild video slowdown → most natural result.
                                Don&apos;t change these unless you know what you&apos;re doing.
                            </p>
                            <div className="space-y-3">
                                {/* Audio Priority */}
                                <div className="flex items-center justify-between">
                                    <div>
                                        <p className="text-sm text-text-primary">Audio Priority <span className="text-[10px] text-green-400">RECOMMENDED ON</span></p>
                                        <p className="text-xs text-text-muted">TTS speaks at natural pace; video stretches/freezes to match. Best listening experience. Leave ON unless lip-sync matters more than voice quality.</p>
                                    </div>
                                    <button
                                        type="button" title="Toggle Audio Priority" onClick={() => update({ audio_priority: !settings.audio_priority })}
                                        className={`
                                        w-11 h-6 rounded-full transition-colors relative
                                        ${settings.audio_priority ? 'bg-primary' : 'bg-white/10'}
                                    `}
                                    >
                                        <div className={`
                                        w-4 h-4 rounded-full bg-white absolute top-1 transition-transform
                                        ${settings.audio_priority ? 'translate-x-6' : 'translate-x-1'}
                                    `} />
                                    </button>
                                </div>

                                {/* Audio Untouchable */}
                                <div className={`flex items-center justify-between ${isVoiceClone ? 'opacity-40 pointer-events-none' : ''}`}>
                                    <div>
                                        <p className="text-sm text-text-primary">Audio Untouchable <span className="text-[10px] text-amber-400">ADVANCED</span></p>
                                        <p className="text-xs text-text-muted">Locks the raw TTS output — no trim, no loudness normalize, no speed adjustment. Turn ON only when you&apos;re providing pre-perfect audio (e.g. SRT mode with high-quality TTS).</p>
                                    </div>
                                    <button
                                        type="button" title="Toggle Audio Untouchable" onClick={() => update({ audio_untouchable: !settings.audio_untouchable })}
                                        className={`
                                        w-11 h-6 rounded-full transition-colors relative
                                        ${settings.audio_untouchable ? 'bg-primary' : 'bg-white/10'}
                                    `}
                                    >
                                        <div className={`
                                        w-4 h-4 rounded-full bg-white absolute top-1 transition-transform
                                        ${settings.audio_untouchable ? 'translate-x-6' : 'translate-x-1'}
                                    `} />
                                    </button>
                                </div>

                                {/* Video Slow to Match */}
                                <div className="flex items-center justify-between">
                                    <div>
                                        <p className="text-sm text-text-primary">Video Slow to Match <span className="text-[10px] text-green-400">RECOMMENDED ON</span></p>
                                        <p className="text-xs text-text-muted">When Hindi audio runs longer than the original, slow the video down evenly so everything stays in sync. The lip movements are slower but everything fits.</p>
                                    </div>
                                    <button
                                        type="button" title="Toggle Video Slow to Match" onClick={() => update({ video_slow_to_match: !settings.video_slow_to_match })}
                                        className={`
                                        w-11 h-6 rounded-full transition-colors relative
                                        ${settings.video_slow_to_match ? 'bg-primary' : 'bg-white/10'}
                                    `}
                                    >
                                        <div className={`
                                        w-4 h-4 rounded-full bg-white absolute top-1 transition-transform
                                        ${settings.video_slow_to_match ? 'translate-x-6' : 'translate-x-1'}
                                    `} />
                                    </button>
                                </div>

                                {/* Post-TTS Processing Level */}
                                <div className={isVoiceClone ? 'opacity-40 pointer-events-none' : ''}>
                                    <p className="text-xs text-text-secondary mb-0.5">Post-TTS Audio Cleanup</p>
                                    <p className="text-[10px] text-text-muted mb-1.5">How aggressively to clean up the generated speech. <b>Minimal</b> is the default — safe for most cases. <b>Full</b> when TTS sounds uneven. <b>None</b> when audio is already perfect (SRT mode).</p>
                                    <div className="grid grid-cols-3 gap-2">
                                        {[
                                            { value: 'full', label: 'Full', desc: 'Trim + Norm + Compress' },
                                            { value: 'minimal', label: 'Minimal ⭐', desc: 'Fade + Loudness only' },
                                            { value: 'none', label: 'None', desc: 'Zero processing' },
                                        ].map((m) => (
                                            <button key={m.value} type="button"
                                                onClick={() => update({ post_tts_level: m.value })}
                                                className={`
                                                px-2 py-1.5 rounded-lg text-xs font-medium transition-all text-center
                                                ${settings.post_tts_level === m.value
                                                        ? 'bg-primary/20 text-primary border border-primary/30'
                                                        : 'bg-white/5 text-text-muted border border-white/5 hover:border-white/20'}
                                            `}
                                            >
                                                <div>{m.label}</div>
                                                <div className="text-[10px] opacity-60">{m.desc}</div>
                                            </button>
                                        ))}
                                    </div>
                                </div>

                                {/* Audio Quality Mode */}
                                <div className={isVoiceClone ? 'opacity-40 pointer-events-none' : ''}>
                                    <p className="text-xs text-text-secondary mb-0.5">Audio Processing Engine</p>
                                    <p className="text-[10px] text-text-muted mb-1.5"><b>Fast</b> uses FFmpeg (recommended). <b>Quality</b> uses Librosa for slower but cleaner denoising. Most users should leave this on Fast.</p>
                                    <div className="grid grid-cols-2 gap-2">
                                        {[
                                            { value: 'fast', label: 'Fast ⭐', desc: 'FFmpeg loudnorm + EQ' },
                                            { value: 'quality', label: 'Quality', desc: 'Librosa denoise + trim' },
                                        ].map((m) => (
                                            <button key={m.value} type="button"
                                                onClick={() => update({ audio_quality_mode: m.value })}
                                                className={`
                                                px-2 py-1.5 rounded-lg text-xs font-medium transition-all text-center
                                                ${settings.audio_quality_mode === m.value
                                                        ? 'bg-primary/20 text-primary border border-primary/30'
                                                        : 'bg-white/5 text-text-muted border border-white/5 hover:border-white/20'}
                                            `}
                                            >
                                                <div>{m.label}</div>
                                                <div className="text-[10px] opacity-60">{m.desc}</div>
                                            </button>
                                        ))}
                                    </div>
                                </div>

                                {/* ═══════════════════════════════════════════════ */}
                                {/* AV SYNC MODULES — 3 approaches to match audio & video */}
                                {/* ═══════════════════════════════════════════════ */}
                                <div className={`border-t border-border pt-4 mt-2 ${isVoiceClone ? 'opacity-40 pointer-events-none' : ''}`}>
                                    <p className="text-sm font-medium text-text-primary mb-1">AV Sync Modules</p>
                                    <p className="text-[10px] text-text-muted mb-3">
                                        Three ways to handle Hindi audio being longer than the original video. Pick <b className="text-text-secondary">one</b> — they are alternatives, not layers.
                                    </p>

                                    {/* ── Module 1: Per-Segment Slot Recompute ── */}
                                    <div className="rounded-lg border border-white/10 p-3 mb-3">
                                        <div className="flex items-center justify-between mb-2">
                                            <div>
                                                <p className="text-xs font-medium text-text-primary">Module 1 — Per-Segment Slot Recompute</p>
                                                <p className="text-[10px] text-text-muted">Speed audio up to cap, then expand each segment&apos;s slot by word-weight. Each video segment gets its own speed.</p>
                                            </div>
                                        </div>
                                        <div className="space-y-2">
                                            <div>
                                                <p className="text-[10px] text-text-muted mb-1">AV Sync Mode</p>
                                                <div className="grid grid-cols-3 gap-2">
                                                    {[
                                                        { value: 'original', label: 'Off', desc: 'Trim audio (old)' },
                                                        { value: 'capped', label: 'Capped', desc: 'Speedup + expand' },
                                                        { value: 'audio_first', label: 'Audio First', desc: 'No speedup' },
                                                    ].map((m) => (
                                                        <button key={m.value} type="button"
                                                            onClick={() => update({ av_sync_mode: m.value, use_global_stretch: false })}
                                                            className={`px-2 py-1.5 rounded-lg text-xs font-medium transition-all text-center ${settings.av_sync_mode === m.value
                                                                ? 'bg-primary/20 text-primary border border-primary/30'
                                                                : 'bg-white/5 text-text-muted border border-white/5 hover:border-white/20'
                                                                }`}
                                                        >
                                                            <div>{m.label}</div>
                                                            <div className="text-[10px] opacity-60">{m.desc}</div>
                                                        </button>
                                                    ))}
                                                </div>
                                            </div>
                                            {settings.av_sync_mode !== 'original' && (
                                                <div className="grid grid-cols-2 gap-3">
                                                    <div>
                                                        <p className="text-[10px] text-text-muted mb-0.5">Max Audio Speedup</p>
                                                        <input type="number" step="0.05" min="1.0" max="2.0"
                                                            value={settings.max_audio_speedup}
                                                            onChange={(e) => update({ max_audio_speedup: parseFloat(e.target.value) || 1.3 })}
                                                            className="w-full px-2 py-1 rounded-lg bg-white/5 border border-white/10 text-xs text-text-primary"
                                                        />
                                                    </div>
                                                    <div>
                                                        <p className="text-[10px] text-text-muted mb-0.5">Min Video Speed</p>
                                                        <input type="number" step="0.05" min="0.1" max="1.0"
                                                            value={settings.min_video_speed}
                                                            onChange={(e) => update({ min_video_speed: parseFloat(e.target.value) || 0.7 })}
                                                            className="w-full px-2 py-1 rounded-lg bg-white/5 border border-white/10 text-xs text-text-primary"
                                                        />
                                                    </div>
                                                </div>
                                            )}
                                            {settings.av_sync_mode !== 'original' && (
                                                <div>
                                                    <p className="text-[10px] text-text-muted mb-1">Slot Verify</p>
                                                    <div className="grid grid-cols-3 gap-2">
                                                        {[
                                                            { value: 'off', label: 'Off' },
                                                            { value: 'dry_run', label: 'Dry Run' },
                                                            { value: 'auto_fix', label: 'Auto Fix' },
                                                        ].map((m) => (
                                                            <button key={m.value} type="button"
                                                                onClick={() => update({ slot_verify: m.value })}
                                                                className={`px-2 py-1 rounded-lg text-[10px] font-medium transition-all ${settings.slot_verify === m.value
                                                                    ? 'bg-primary/20 text-primary border border-primary/30'
                                                                    : 'bg-white/5 text-text-muted border border-white/5 hover:border-white/20'
                                                                    }`}
                                                            >{m.label}</button>
                                                        ))}
                                                    </div>
                                                </div>
                                            )}
                                        </div>
                                    </div>

                                    {/* ── Module 2: Global Stretch ── */}
                                    <div className="rounded-lg border border-white/10 p-3 mb-3">
                                        <div className="flex items-center justify-between">
                                            <div>
                                                <p className="text-xs font-medium text-text-primary">Module 2 — Global Stretch</p>
                                                <p className="text-[10px] text-text-muted">One uniform video speed for the whole video. Simple: total TTS audio at speed X vs original video length.</p>
                                            </div>
                                            <button
                                                type="button" title="Toggle Global Stretch"
                                                onClick={() => update({ use_global_stretch: !settings.use_global_stretch, av_sync_mode: !settings.use_global_stretch ? 'original' : settings.av_sync_mode })}
                                                className={`w-11 h-6 rounded-full transition-colors relative ${settings.use_global_stretch ? 'bg-primary' : 'bg-white/10'}`}
                                            >
                                                <div className={`w-4 h-4 rounded-full bg-white absolute top-1 transition-transform ${settings.use_global_stretch ? 'translate-x-6' : 'translate-x-1'}`} />
                                            </button>
                                        </div>
                                        {settings.use_global_stretch && (
                                            <div className="mt-2">
                                                <p className="text-[10px] text-text-muted mb-0.5">TTS Audio Speedup</p>
                                                <input type="number" step="0.05" min="1.0" max="2.0"
                                                    value={settings.global_stretch_speedup}
                                                    onChange={(e) => update({ global_stretch_speedup: parseFloat(e.target.value) || 1.25 })}
                                                    className="w-24 px-2 py-1 rounded-lg bg-white/5 border border-white/10 text-xs text-text-primary"
                                                />
                                            </div>
                                        )}
                                    </div>

                                    {/* ── Module 3: Sentence Segmenter ── */}
                                    <div className="rounded-lg border border-white/10 p-3">
                                        <div className="flex items-center justify-between mb-2">
                                            <div>
                                                <p className="text-xs font-medium text-text-primary">Module 3 — Segmenter</p>
                                                <p className="text-[10px] text-text-muted">How to split Whisper words into segments. <b>DP</b> = globally optimal (default). <b>Sentence</b> = pack full sentences with 20% Hindi buffer.</p>
                                            </div>
                                        </div>
                                        <div className="space-y-2">
                                            <div className="grid grid-cols-2 gap-2">
                                                {[
                                                    { value: 'dp', label: 'DP Optimal', desc: 'Score every split point' },
                                                    { value: 'sentence', label: 'Sentence Pack', desc: 'Full sentences + buffer' },
                                                ].map((m) => (
                                                    <button key={m.value} type="button"
                                                        onClick={() => update({ segmenter: m.value })}
                                                        className={`px-2 py-1.5 rounded-lg text-xs font-medium transition-all text-center ${settings.segmenter === m.value
                                                            ? 'bg-primary/20 text-primary border border-primary/30'
                                                            : 'bg-white/5 text-text-muted border border-white/5 hover:border-white/20'
                                                            }`}
                                                    >
                                                        <div>{m.label}</div>
                                                        <div className="text-[10px] opacity-60">{m.desc}</div>
                                                    </button>
                                                ))}
                                            </div>
                                            {settings.segmenter === 'sentence' && (
                                                <div className="grid grid-cols-2 gap-3">
                                                    <div>
                                                        <p className="text-[10px] text-text-muted mb-0.5">Hindi Buffer %</p>
                                                        <input type="number" step="0.05" min="0" max="0.50"
                                                            value={settings.segmenter_buffer_pct}
                                                            onChange={(e) => update({ segmenter_buffer_pct: parseFloat(e.target.value) || 0.2 })}
                                                            className="w-full px-2 py-1 rounded-lg bg-white/5 border border-white/10 text-xs text-text-primary"
                                                        />
                                                    </div>
                                                    <div>
                                                        <p className="text-[10px] text-text-muted mb-0.5">Max Sentences/Cue</p>
                                                        <input type="number" step="1" min="1" max="5"
                                                            value={settings.max_sentences_per_cue}
                                                            onChange={(e) => update({ max_sentences_per_cue: parseInt(e.target.value) || 2 })}
                                                            className="w-full px-2 py-1 rounded-lg bg-white/5 border border-white/10 text-xs text-text-primary"
                                                        />
                                                    </div>
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                </div>

                                {/* Sentence Gap (1s silence between segments) */}
                                <div className="flex items-center justify-between">
                                    <div>
                                        <p className="text-xs font-medium text-text-secondary">Sentence Gap (1s pause)</p>
                                        <p className="text-xs text-text-muted">Adds a 1-second pause between sentences. Off by default — speech flows continuously like natural speech. Turn ON only if your TTS sentences feel rushed together.</p>
                                    </div>
                                    <button
                                        type="button" title="Toggle Sentence Gap" onClick={() => update({ enable_sentence_gap: !settings.enable_sentence_gap })}
                                        className={`
                                        w-11 h-6 rounded-full transition-colors relative
                                        ${settings.enable_sentence_gap ? 'bg-primary' : 'bg-white/10'}
                                    `}
                                    >
                                        <div className={`
                                        w-4 h-4 rounded-full bg-white absolute top-1 transition-transform
                                        ${settings.enable_sentence_gap ? 'translate-x-6' : 'translate-x-1'}
                                    `} />
                                    </button>
                                </div>

                                {/* Gap Mode — controls silence between segments in assembly */}
                                <div className="flex items-center justify-between">
                                    <div>
                                        <p className="text-xs font-medium text-text-secondary">Gap Mode</p>
                                        <p className="text-xs text-text-muted">
                                            {settings.gap_mode === 'none' ? 'No gaps — pure back-to-back segments (shortest output).'
                                                : settings.gap_mode === 'full' ? 'Full gaps — keep original silence durations between speech.'
                                                    : 'Micro gaps — 0.2s breathing room between segments (proven default).'}
                                        </p>
                                    </div>
                                    <div className="flex rounded-lg overflow-hidden border border-border">
                                        {([
                                            { mode: 'none', label: 'None' },
                                            { mode: 'micro', label: 'Micro' },
                                            { mode: 'full', label: 'Full' },
                                        ] as const).map(({ mode, label }) => (
                                            <button
                                                key={mode}
                                                type="button"
                                                onClick={() => update({ gap_mode: mode })}
                                                className={`px-3 py-1 text-[10px] font-medium transition-colors ${settings.gap_mode === mode
                                                    ? 'bg-primary text-white'
                                                    : 'bg-white/5 text-text-muted hover:bg-white/10'
                                                    }`}
                                            >
                                                {label}
                                            </button>
                                        ))}
                                    </div>
                                </div>

                                {/* TTS Chunk Words — split translated text into small chunks */}
                                <div className="flex items-center justify-between">
                                    <div>
                                        <p className="text-xs font-medium text-text-secondary">TTS Chunk Size</p>
                                        <p className="text-xs text-text-muted">
                                            {settings.tts_chunk_words === 0 ? 'Off — TTS gets full segments (best natural sound).'
                                                : `${settings.tts_chunk_words} words per TTS call — prevents truncation but less natural prosody.`}
                                        </p>
                                    </div>
                                    <div className="flex rounded-lg overflow-hidden border border-border">
                                        {([
                                            { val: 0, label: 'Off' },
                                            { val: 4, label: '4w' },
                                            { val: 8, label: '8w' },
                                            { val: 12, label: '12w' },
                                        ] as const).map(({ val, label }) => (
                                            <button
                                                key={val}
                                                type="button"
                                                onClick={() => update({ tts_chunk_words: val })}
                                                className={`px-3 py-1 text-[10px] font-medium transition-colors ${settings.tts_chunk_words === val
                                                    ? 'bg-primary text-white'
                                                    : 'bg-white/5 text-text-muted hover:bg-white/10'
                                                    }`}
                                            >
                                                {label}
                                            </button>
                                        ))}
                                    </div>
                                </div>

                                {/* Duration Fitting (speed-up/slow-down) */}
                                <div className="flex items-center justify-between">
                                    <div>
                                        <p className="text-xs font-medium text-text-secondary">Force TTS into Original Timing</p>
                                        <p className="text-xs text-text-muted">Aggressively speeds up / slows down each Hindi sentence to fit the original English duration. OFF by default — assembly handles this more gracefully (split-the-diff). Turn ON only if you absolutely cannot stretch the video.</p>
                                    </div>
                                    <button
                                        type="button" title="Toggle Duration Fitting" onClick={() => update({ enable_duration_fit: !settings.enable_duration_fit })}
                                        className={`
                                        w-11 h-6 rounded-full transition-colors relative
                                        ${settings.enable_duration_fit ? 'bg-primary' : 'bg-white/10'}
                                    `}
                                    >
                                        <div className={`
                                        w-4 h-4 rounded-full bg-white absolute top-1 transition-transform
                                        ${settings.enable_duration_fit ? 'translate-x-6' : 'translate-x-1'}
                                    `} />
                                    </button>
                                </div>

                                {/* Audio Bitrate */}
                                <div>
                                    <p className="text-xs text-text-secondary mb-0.5">Audio Bitrate</p>
                                    <p className="text-[10px] text-text-muted mb-1.5">Higher = better sound quality, larger file. <b>192k</b> is the sweet spot for speech.</p>
                                    <div className="grid grid-cols-4 gap-2">
                                        {[
                                            { value: '128k', label: '128k', desc: 'Small file' },
                                            { value: '192k', label: '192k', desc: 'Standard' },
                                            { value: '256k', label: '256k', desc: 'High' },
                                            { value: '320k', label: '320k', desc: 'Best' },
                                        ].map((m) => (
                                            <button
                                                type="button"
                                                key={m.value}
                                                onClick={() => update({ audio_bitrate: m.value })}
                                                className={`
                                                px-2 py-2 rounded-lg text-xs text-center transition-all border
                                                ${settings.audio_bitrate === m.value
                                                        ? 'bg-primary/20 border-primary text-primary-light'
                                                        : 'bg-white/5 border-white/10 text-text-muted hover:bg-white/10'}
                                            `}
                                            >
                                                <div className="font-medium">{m.label}</div>
                                                <div className="text-[10px] opacity-70 mt-0.5">{m.desc}</div>
                                            </button>
                                        ))}
                                    </div>
                                </div>

                                {/* Download Mode — remux (fast) vs encode (compatible) */}
                                <div>
                                    <p className="text-xs text-text-secondary mb-0.5">Download Mode</p>
                                    <p className="text-[10px] text-text-muted mb-1.5"><b>Remux</b> = instant container swap (fast, current default). <b>Encode</b> = old behaviour, ffmpeg re-encodes during merge — slower but always works if a video&apos;s codec combo breaks remux.</p>
                                    <div className="grid grid-cols-2 gap-2">
                                        {[
                                            { value: 'remux', label: 'Remux', desc: 'Fast (default)' },
                                            { value: 'encode', label: 'Encode', desc: 'Slow but compatible' },
                                        ].map((m) => (
                                            <button
                                                type="button"
                                                key={m.value}
                                                onClick={() => update({ download_mode: m.value })}
                                                className={`
                                                px-2 py-2 rounded-lg text-xs text-center transition-all border
                                                ${settings.download_mode === m.value
                                                        ? 'bg-primary/20 border-primary text-primary-light'
                                                        : 'bg-white/5 border-white/10 text-text-muted hover:bg-white/10'}
                                            `}
                                            >
                                                <div className="font-medium">{m.label}</div>
                                                <div className="text-[10px] opacity-70 mt-0.5">{m.desc}</div>
                                            </button>
                                        ))}
                                    </div>
                                </div>

                                {/* Encode Speed */}
                                <div>
                                    <p className="text-xs text-text-secondary mb-0.5">Video Encode Speed</p>
                                    <p className="text-[10px] text-text-muted mb-1.5">Faster preset = quicker job, larger file. Slower preset = better compression, smaller file. NVENC GPU encoding is used regardless. <b>Fast</b> is the default sweet spot.</p>
                                    <div className="grid grid-cols-4 gap-2">
                                        {[
                                            { value: 'ultrafast', label: 'Ultra Fast', desc: 'Fastest' },
                                            { value: 'veryfast', label: 'Very Fast', desc: 'Default' },
                                            { value: 'fast', label: 'Fast', desc: 'Better' },
                                            { value: 'medium', label: 'Medium', desc: 'Best video' },
                                        ].map((m) => (
                                            <button
                                                type="button"
                                                key={m.value}
                                                onClick={() => update({ encode_preset: m.value })}
                                                className={`
                                                px-2 py-2 rounded-lg text-xs text-center transition-all border
                                                ${settings.encode_preset === m.value
                                                        ? 'bg-primary/20 border-primary text-primary-light'
                                                        : 'bg-white/5 border-white/10 text-text-muted hover:bg-white/10'}
                                            `}
                                            >
                                                <div className="font-medium">{m.label}</div>
                                                <div className="text-[10px] opacity-70 mt-0.5">{m.desc}</div>
                                            </button>
                                        ))}
                                    </div>
                                </div>

                                {/* Assembly mode info — always per-segment NVENC now */}
                                <div className="rounded-lg p-2.5 bg-green-500/10 border border-green-500/20">
                                    <p className="text-xs text-green-400 font-medium">Assembly: Per-segment NVENC (4x parallel)</p>
                                    <p className="text-[10px] text-green-400/70">Audio priority — video adapts per sentence. GPU-accelerated encoding.</p>
                                </div>

                                {/* Manual Review Queue */}
                                <div className="flex items-center justify-between">
                                    <div>
                                        <p className="text-sm text-text-primary">Manual Review Queue</p>
                                        <p className="text-xs text-text-muted">Save segments that failed QC after all retries to manual_review_queue.json for inspection</p>
                                    </div>
                                    <button
                                        type="button" title="Toggle Manual Review Queue" onClick={() => update({ enable_manual_review: !settings.enable_manual_review })}
                                        className={`
                                        w-11 h-6 rounded-full transition-colors relative
                                        ${settings.enable_manual_review ? 'bg-primary' : 'bg-white/10'}
                                    `}
                                    >
                                        <div className={`
                                        w-4 h-4 rounded-full bg-white absolute top-1 transition-transform
                                        ${settings.enable_manual_review ? 'translate-x-6' : 'translate-x-1'}
                                    `} />
                                    </button>
                                </div>

                                {/* TTS Verify Retry */}
                                <div className="flex items-center justify-between">
                                    <div>
                                        <p className="text-sm text-text-primary">TTS Verify Retry <span className="text-[10px] text-amber-400">SLOW · OFF</span></p>
                                        <p className="text-xs text-text-muted">
                                            Re-checks every TTS segment for duration / silence and regenerates failures.
                                            Hindi WPM variance causes ~70% false positives → adds 1-2 hours on long videos.
                                            Turn ON only for short videos where you want extra quality safety.
                                        </p>
                                    </div>
                                    <button
                                        type="button" title="Toggle TTS Verify Retry" onClick={() => update({ enable_tts_verify_retry: !settings.enable_tts_verify_retry })}
                                        className={`
                                        w-11 h-6 rounded-full transition-colors relative
                                        ${settings.enable_tts_verify_retry ? 'bg-primary' : 'bg-white/10'}
                                    `}
                                    >
                                        <div className={`
                                        w-4 h-4 rounded-full bg-white absolute top-1 transition-transform
                                        ${settings.enable_tts_verify_retry ? 'translate-x-6' : 'translate-x-1'}
                                    `} />
                                    </button>
                                </div>

                                {/* TTS Truncation Guard (inline, fast, no false positives at default) */}
                                <div>
                                    <div className="flex items-center justify-between mb-1">
                                        <p className="text-sm text-text-primary">
                                            TTS Truncation Guard{' '}
                                            <span className="text-[10px] text-green-400">FAST · INLINE</span>
                                        </p>
                                        <span className="text-xs text-primary-light font-mono">
                                            {settings.tts_truncation_threshold === 0
                                                ? 'OFF'
                                                : `${Math.round(settings.tts_truncation_threshold * 100)}%`}
                                        </span>
                                    </div>
                                    <p className="text-xs text-text-muted mb-2">
                                        Catches Edge-TTS WebSocket drops that silently save partial audio (e.g. only the
                                        first half of a Hindi sentence). Probes each MP3 right after save and retries if
                                        duration is below threshold &times; (word_count / 250&nbsp;WPM).
                                        {' '}<b>30%</b> = catastrophic only (default, no false positives).
                                        {' '}<b>50-70%</b> = stricter, may false-positive on naturally fast speech.
                                        {' '}<b>0%</b> = OFF (legacy behavior).
                                    </p>
                                    <input
                                        type="range"
                                        min="0"
                                        max="0.9"
                                        step="0.05"
                                        value={settings.tts_truncation_threshold}
                                        onChange={(e) => update({ tts_truncation_threshold: parseFloat(e.target.value) })}
                                        className="w-full accent-primary"
                                        aria-label="TTS truncation threshold"
                                    />
                                    <div className="flex justify-between text-[10px] text-text-muted mt-1">
                                        <span>OFF</span>
                                        <span>30% default</span>
                                        <span>60%</span>
                                        <span>90% paranoid</span>
                                    </div>
                                </div>

                                {/* Post-TTS Whisper word-match verification */}
                                <div>
                                    <div className="flex items-center justify-between mb-1">
                                        <div className="flex-1 pr-3">
                                            <p className="text-sm text-text-primary">
                                                Post-TTS Word-Match Verify{' '}
                                                <span className="text-[10px] text-amber-400">SLOW · EXACT</span>
                                            </p>
                                            <p className="text-xs text-text-muted">
                                                After every segment is synthesized, run Whisper-tiny on the audio
                                                to count the actual spoken words and compare to the translated word
                                                count. Re-runs Edge-TTS for any segment outside the tolerance window
                                                (up to 3 retries). The most accurate guard available, but adds
                                                <b> ~150–300 ms per segment</b> on CPU — a 5000-segment job pays
                                                ~15–25 minutes extra.
                                            </p>
                                        </div>
                                        <button
                                            type="button"
                                            title="Toggle Post-TTS Word-Match Verify"
                                            onClick={() => update({ tts_word_match_verify: !settings.tts_word_match_verify })}
                                            className={`
                                            w-11 h-6 rounded-full transition-colors relative shrink-0
                                            ${settings.tts_word_match_verify ? 'bg-primary' : 'bg-white/10'}
                                        `}
                                        >
                                            <div className={`
                                            w-4 h-4 rounded-full bg-white absolute top-1 transition-transform
                                            ${settings.tts_word_match_verify ? 'translate-x-6' : 'translate-x-1'}
                                        `} />
                                        </button>
                                    </div>
                                    {settings.tts_word_match_verify && (
                                        <div className="mt-2 ml-1">
                                            <div className="flex items-center justify-between mb-1">
                                                <span className="text-xs text-text-secondary">
                                                    Tolerance window
                                                </span>
                                                <span className="text-xs text-primary-light font-mono">
                                                    ±{Math.round(settings.tts_word_match_tolerance * 100)}%
                                                </span>
                                            </div>
                                            <input
                                                type="range"
                                                min="0"
                                                max="0.5"
                                                step="0.05"
                                                value={settings.tts_word_match_tolerance}
                                                onChange={(e) => update({ tts_word_match_tolerance: parseFloat(e.target.value) })}
                                                className="w-full accent-primary"
                                                aria-label="TTS word-match tolerance"
                                            />
                                            <div className="flex justify-between text-[10px] text-text-muted mt-1">
                                                <span>Strict 0%</span>
                                                <span>15% default</span>
                                                <span>30%</span>
                                                <span>50% loose</span>
                                            </div>
                                            <p className="text-[10px] text-text-muted mt-1">
                                                Even Whisper has ~10% natural variance on Hindi (contractions,
                                                quiet syllables, number forms). 0% will trigger false retries
                                                on segments that are actually fine.
                                            </p>

                                            {/* Whisper model picker — only visible when verify is ON */}
                                            <div className="mt-3">
                                                <p className="text-xs text-text-secondary mb-1">
                                                    Whisper model
                                                </p>
                                                <div className="grid grid-cols-3 gap-2">
                                                    {[
                                                        { value: 'auto', label: 'Auto', desc: 'Turbo on GPU, Tiny on CPU' },
                                                        { value: 'tiny', label: 'Tiny', desc: 'Fast · 85% acc' },
                                                        { value: 'turbo', label: 'Turbo', desc: 'Best · GPU only' },
                                                    ].map((m) => (
                                                        <button
                                                            type="button"
                                                            key={m.value}
                                                            onClick={() => update({ tts_word_match_model: m.value })}
                                                            className={`
                                                            px-2 py-2 rounded-lg text-xs text-center transition-all border
                                                            ${settings.tts_word_match_model === m.value
                                                                    ? 'bg-primary/20 border-primary text-primary-light'
                                                                    : 'bg-white/5 border-white/10 text-text-muted hover:bg-white/10'}
                                                        `}
                                                        >
                                                            <div className="font-medium">{m.label}</div>
                                                            <div className="text-[10px] opacity-70 mt-0.5">{m.desc}</div>
                                                        </button>
                                                    ))}
                                                </div>
                                                <p className="text-[10px] text-text-muted mt-1">
                                                    <b>Auto</b> picks the best model for your hardware. <b>Turbo</b> (large-v3-turbo)
                                                    is 96-97% accurate but ~5-10x slower than Tiny on CPU — only pick it if
                                                    you have a GPU. The verifier auto-falls-back to Tiny on GPU OOM.
                                                </p>
                                            </div>
                                        </div>
                                    )}
                                </div>

                                {/* No Time Pressure on TTS — master switch */}
                                <div className="flex items-center justify-between">
                                    <div className="flex-1 pr-3">
                                        <p className="text-sm text-text-primary">
                                            TTS: No Time Pressure{' '}
                                            <span className="text-[10px] text-green-400">DEFAULT ON · RECOMMENDED</span>
                                        </p>
                                        <p className="text-xs text-text-muted">
                                            Master switch: when ON, TTS produces full natural-pace audio for
                                            every word with <b>no slot constraint</b>. ALL of the downstream
                                            time-pressure points are bypassed: the 1.15× pre-speedup, the
                                            speed-fit clamp (which was cutting words on long segments), and
                                            the QC duration mismatch failure. Slot timing is handled by
                                            assembly via the audio-priority path (video adapts to audio).
                                            Forces <code>tts_rate=&quot;+0%&quot;</code>, <code>audio_priority=true</code>,
                                            <code>enable_duration_fit=false</code>.
                                        </p>
                                    </div>
                                    <button
                                        type="button"
                                        title="Toggle TTS No Time Pressure"
                                        onClick={() => update({ tts_no_time_pressure: !settings.tts_no_time_pressure })}
                                        className={`
                                        w-11 h-6 rounded-full transition-colors relative shrink-0
                                        ${settings.tts_no_time_pressure ? 'bg-primary' : 'bg-white/10'}
                                    `}
                                    >
                                        <div className={`
                                        w-4 h-4 rounded-full bg-white absolute top-1 transition-transform
                                        ${settings.tts_no_time_pressure ? 'translate-x-6' : 'translate-x-1'}
                                    `} />
                                    </button>
                                </div>

                                {/* Dynamic TTS Workers */}
                                <div>
                                    <div className="flex items-center justify-between mb-1">
                                        <div className="flex-1 pr-3">
                                            <p className="text-sm text-text-primary">
                                                TTS: Dynamic Workers{' '}
                                                <span className="text-[10px] text-green-400">ADAPTIVE</span>
                                            </p>
                                            <p className="text-xs text-text-muted">
                                                Concurrency adapts to Edge-TTS rate limits between batches.
                                                Starts at <b>{settings.tts_dynamic_start}</b>, halves on &gt;10% failure rate
                                                (down to {settings.tts_dynamic_min}), grows by 25% on clean batches
                                                (up to {settings.tts_dynamic_max}). When OFF, fixed at 120 (the old behavior).
                                            </p>
                                        </div>
                                        <button
                                            type="button"
                                            title="Toggle TTS Dynamic Workers"
                                            onClick={() => update({ tts_dynamic_workers: !settings.tts_dynamic_workers })}
                                            className={`
                                            w-11 h-6 rounded-full transition-colors relative shrink-0
                                            ${settings.tts_dynamic_workers ? 'bg-primary' : 'bg-white/10'}
                                        `}
                                        >
                                            <div className={`
                                            w-4 h-4 rounded-full bg-white absolute top-1 transition-transform
                                            ${settings.tts_dynamic_workers ? 'translate-x-6' : 'translate-x-1'}
                                        `} />
                                        </button>
                                    </div>
                                    {settings.tts_dynamic_workers && (
                                        <div className="mt-2 ml-1 grid grid-cols-3 gap-2">
                                            <div>
                                                <p className="text-[10px] text-text-muted mb-1">Start</p>
                                                <input
                                                    type="number"
                                                    min="5" max="120" step="5"
                                                    value={settings.tts_dynamic_start}
                                                    onChange={(e) => update({ tts_dynamic_start: parseInt(e.target.value, 10) || 30 })}
                                                    title="Initial worker count"
                                                    className="w-full px-2 py-1 rounded bg-white/5 border border-white/10 text-xs text-text-primary font-mono"
                                                />
                                            </div>
                                            <div>
                                                <p className="text-[10px] text-text-muted mb-1">Min</p>
                                                <input
                                                    type="number"
                                                    min="1" max="50" step="1"
                                                    value={settings.tts_dynamic_min}
                                                    onChange={(e) => update({ tts_dynamic_min: parseInt(e.target.value, 10) || 10 })}
                                                    title="Minimum worker count"
                                                    className="w-full px-2 py-1 rounded bg-white/5 border border-white/10 text-xs text-text-primary font-mono"
                                                />
                                            </div>
                                            <div>
                                                <p className="text-[10px] text-text-muted mb-1">Max</p>
                                                <input
                                                    type="number"
                                                    min="20" max="240" step="10"
                                                    value={settings.tts_dynamic_max}
                                                    onChange={(e) => update({ tts_dynamic_max: parseInt(e.target.value, 10) || 120 })}
                                                    title="Maximum worker count"
                                                    className="w-full px-2 py-1 rounded bg-white/5 border border-white/10 text-xs text-text-primary font-mono"
                                                />
                                            </div>
                                        </div>
                                    )}
                                </div>

                                {/* Long-Segment Trace Watchdog */}
                                <div>
                                    <div className="flex items-center justify-between mb-1">
                                        <div className="flex-1 pr-3">
                                            <p className="text-sm text-text-primary">
                                                Long-Segment Trace Watchdog{' '}
                                                <span className="text-[10px] text-blue-400">DIAGNOSTIC · CHEAP</span>
                                            </p>
                                            <p className="text-xs text-text-muted">
                                                Records the full lifecycle of every long segment (text fingerprint,
                                                slot duration, post-save file size, post-WAV duration, truncation
                                                guard verdict, word verifier verdict, speed-fit clamping, assembly cut)
                                                to a JSON report at <code>backend/logs/long_segment_trace_&lt;job_id&gt;.json</code>.
                                                When a long segment&apos;s audio comes out wrong, you can grep the
                                                report to see <b>exactly which stage</b> dropped or modified it.
                                                Cost is microseconds per event.
                                            </p>
                                        </div>
                                        <button
                                            type="button"
                                            title="Toggle Long-Segment Trace Watchdog"
                                            onClick={() => update({ long_segment_trace: !settings.long_segment_trace })}
                                            className={`
                                            w-11 h-6 rounded-full transition-colors relative shrink-0
                                            ${settings.long_segment_trace ? 'bg-primary' : 'bg-white/10'}
                                        `}
                                        >
                                            <div className={`
                                            w-4 h-4 rounded-full bg-white absolute top-1 transition-transform
                                            ${settings.long_segment_trace ? 'translate-x-6' : 'translate-x-1'}
                                        `} />
                                        </button>
                                    </div>
                                    {settings.long_segment_trace && (
                                        <div className="mt-2 ml-1">
                                            <div className="flex items-center justify-between mb-1">
                                                <span className="text-xs text-text-secondary">
                                                    Word threshold (segments at or above are traced)
                                                </span>
                                                <span className="text-xs text-primary-light font-mono">
                                                    {settings.long_segment_threshold_words} words
                                                </span>
                                            </div>
                                            <input
                                                type="range"
                                                min="5"
                                                max="50"
                                                step="1"
                                                value={settings.long_segment_threshold_words}
                                                onChange={(e) => update({ long_segment_threshold_words: parseInt(e.target.value, 10) })}
                                                className="w-full accent-primary"
                                                aria-label="Long segment trace word threshold"
                                            />
                                            <div className="flex justify-between text-[10px] text-text-muted mt-1">
                                                <span>5 (verbose)</span>
                                                <span>15 (default)</span>
                                                <span>30</span>
                                                <span>50 (minimal)</span>
                                            </div>
                                            <p className="text-[10px] text-text-muted mt-1">
                                                Lower threshold = more segments traced = larger JSON report.
                                                <b> 15</b> is the right default for diagnosing TTS truncation
                                                issues on Hindi dubs. Drop to <b>5</b> if you want every
                                                non-trivial segment recorded.
                                            </p>
                                        </div>
                                    )}
                                </div>

                                {/* Keep Noun Subjects in English */}
                                <div className="flex items-center justify-between">
                                    <div className="flex-1 pr-3">
                                        <p className="text-sm text-text-primary">
                                            Keep Noun Subjects in English{' '}
                                            <span className="text-[10px] text-blue-400">SPACY REQUIRED</span>
                                        </p>
                                        <p className="text-xs text-text-muted">
                                            For every sentence, the main NOUN subject (names, common nouns,
                                            full noun phrases like &quot;The young warrior&quot;) is masked
                                            before translation and restored verbatim afterward. Pronoun
                                            subjects (he, she, it, they) are still translated to Hindi
                                            normally — keeping pronouns English sounds jarring.
                                            Edge-TTS Madhur will pronounce English nouns with a Hindi accent.
                                            Requires <code>spacy + en_core_web_sm</code> in the backend.
                                        </p>
                                    </div>
                                    <button
                                        type="button"
                                        title="Toggle Keep Noun Subjects in English"
                                        onClick={() => update({ keep_subject_english: !settings.keep_subject_english })}
                                        className={`
                                        w-11 h-6 rounded-full transition-colors relative shrink-0
                                        ${settings.keep_subject_english ? 'bg-primary' : 'bg-white/10'}
                                    `}
                                    >
                                        <div className={`
                                        w-4 h-4 rounded-full bg-white absolute top-1 transition-transform
                                        ${settings.keep_subject_english ? 'translate-x-6' : 'translate-x-1'}
                                    `} />
                                    </button>
                                </div>

                                {/* Purge On New URL */}
                                <div className="flex items-center justify-between">
                                    <div>
                                        <p className="text-sm text-text-primary">Auto-Clean Disk on New URL <span className="text-[10px] text-blue-400">SAVES SPACE</span></p>
                                        <p className="text-xs text-text-muted">
                                            When you submit a different YouTube URL, automatically delete the previous video&apos;s
                                            work files (raw audio, source.mp4, intermediate wavs — usually 2-4 GB per video).
                                            Won&apos;t touch your final dubbed output in the saved folder.
                                        </p>
                                    </div>
                                    <button
                                        type="button" title="Toggle Auto-Clean on New URL" onClick={() => update({ purge_on_new_url: !settings.purge_on_new_url })}
                                        className={`
                                        w-11 h-6 rounded-full transition-colors relative
                                        ${settings.purge_on_new_url ? 'bg-primary' : 'bg-white/10'}
                                    `}
                                    >
                                        <div className={`
                                        w-4 h-4 rounded-full bg-white absolute top-1 transition-transform
                                        ${settings.purge_on_new_url ? 'translate-x-6' : 'translate-x-1'}
                                    `} />
                                    </button>
                                </div>

                                {/* Step-by-Step Review — disabled when YT Translate or New pipeline */}
                                <div className={`flex items-center justify-between ${(ytTranslateOn || isNew || isOneFlow || isSrtMode) ? 'opacity-40 pointer-events-none' : ''}`}>
                                    <div>
                                        <p className="text-sm text-text-primary">Step-by-Step Review</p>
                                        <p className="text-xs text-text-muted">
                                            Pause after transcription & translation to review output before continuing
                                            {ytTranslateOn && <span className="text-yellow-400 ml-1">— off: YT Translate skips these steps</span>}
                                        </p>
                                    </div>
                                    <button
                                        type="button" title="Toggle Step-by-Step Review" onClick={() => update({ step_by_step: !settings.step_by_step })}
                                        className={`
                                        w-11 h-6 rounded-full transition-colors relative
                                        ${settings.step_by_step ? 'bg-primary' : 'bg-white/10'}
                                    `}
                                    >
                                        <div className={`
                                        w-4 h-4 rounded-full bg-white absolute top-1 transition-transform
                                        ${settings.step_by_step ? 'translate-x-6' : 'translate-x-1'}
                                    `} />
                                    </button>
                                </div>
                            </div>

                            {/* Dub Duration Limit */}
                            <div>
                                <p className="text-xs text-text-secondary mb-0.5">Dub Duration Limit</p>
                                <p className="text-[10px] text-text-muted mb-1.5">Stop dubbing after N minutes. Use this to preview a long video before committing to the full run, or when you only need a portion.</p>
                                <div className="flex items-center gap-2">
                                    <button
                                        type="button"
                                        onClick={() => update({ dub_duration: 0 })}
                                        className={`px-3 py-2 rounded-lg text-xs text-center transition-all border ${settings.dub_duration === 0
                                            ? 'bg-primary/20 border-primary text-primary-light'
                                            : 'bg-white/5 border-white/10 text-text-muted hover:bg-white/10'}`}
                                    >
                                        <div className="font-medium">Full</div>
                                    </button>
                                    <div className="flex items-center gap-1.5 flex-1">
                                        <input
                                            type="number"
                                            min={1}
                                            max={9999}
                                            placeholder="e.g. 30"
                                            value={settings.dub_duration || ''}
                                            onChange={(e) => {
                                                const v = parseInt(e.target.value, 10);
                                                update({ dub_duration: isNaN(v) ? 0 : Math.max(0, Math.min(9999, v)) });
                                            }}
                                            className={`w-24 px-2 py-2 rounded-lg text-xs text-center border bg-white/5 outline-none transition-all
                                                ${settings.dub_duration > 0
                                                    ? 'border-primary text-primary-light'
                                                    : 'border-white/10 text-text-muted'}`}
                                        />
                                        <span className="text-xs text-text-muted">min</span>
                                    </div>
                                    {[10, 30, 60, 120].map((m) => (
                                        <button
                                            type="button"
                                            key={m}
                                            onClick={() => update({ dub_duration: m })}
                                            className={`px-2.5 py-2 rounded-lg text-xs text-center transition-all border
                                                ${settings.dub_duration === m
                                                    ? 'bg-primary/20 border-primary text-primary-light'
                                                    : 'bg-white/5 border-white/10 text-text-muted hover:bg-white/10'}`}
                                        >
                                            {m >= 60 ? `${m / 60}h` : `${m}m`}
                                        </button>
                                    ))}
                                </div>
                            </div>

                            {/* Split Long Videos */}
                            <div>
                                <p className="text-xs text-text-secondary mb-0.5">Split Long Videos</p>
                                <p className="text-[10px] text-text-muted mb-1.5">Breaks the video into N-minute parts and dubs each separately, then joins them. Strongly recommended for videos over 1 hour to avoid memory issues. <b>Default: 30 min.</b></p>
                                <div className="grid grid-cols-3 gap-2">
                                    {[
                                        { value: 0, label: 'Off', desc: 'No splitting' },
                                        { value: 30, label: '30 min', desc: 'Split every 30m' },
                                        { value: 40, label: '40 min', desc: 'Split every 40m' },
                                        { value: 60, label: '1 hour', desc: 'Split every 1h' },
                                        { value: 120, label: '2 hours', desc: 'Split every 2h' },
                                        { value: 180, label: '3 hours', desc: 'Split every 3h' },
                                    ].map((m) => (
                                        <button
                                            type="button"
                                            key={m.value}
                                            onClick={() => update({ split_duration: m.value })}
                                            className={`
                                            px-3 py-2 rounded-lg text-xs text-center transition-all border
                                            ${settings.split_duration === m.value
                                                    ? 'bg-primary/20 border-primary text-primary-light'
                                                    : 'bg-white/5 border-white/10 text-text-muted hover:bg-white/10'}
                                        `}
                                        >
                                            <div className="font-medium">{m.label}</div>
                                            <div className="text-[10px] opacity-70 mt-0.5">{m.desc}</div>
                                        </button>
                                    ))}
                                </div>
                            </div>
                        </div>
                    )}
                    {/* ── End of Audio & Performance Section (hidden in SRT Direct) ── */}
                </div>);
            })()}
        </div>
    );
}
