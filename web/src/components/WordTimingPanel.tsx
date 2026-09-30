'use client';

import type { DubbingSettings } from '@/components/SettingsPanel';

interface WordTimingPanelProps {
    settings: DubbingSettings;
    onChange: (settings: DubbingSettings) => void;
    /** Force-disable (e.g. embedding it somewhere Whisper never runs). */
    disabled?: boolean;
}

/**
 * Word Timing — a dedicated, self-contained control module for the tiered
 * word-level timing strategy in backend/dubbing/word_timing.py:
 *
 *   1. WhisperX forced alignment            (most precise)
 *   2. Local faster-whisper on the GPU      (auto-fallback when WhisperX "not proper")
 *   3. Segment-level timing                 (last resort)
 *
 * Drop it anywhere in the UI:
 *   <WordTimingPanel settings={settings} onChange={setSettings} />
 */
export default function WordTimingPanel({ settings, onChange, disabled = false }: WordTimingPanelProps) {
    const update = (patch: Partial<DubbingSettings>) => onChange({ ...settings, ...patch });

    const wx = settings.use_whisperx;
    const gpuFallback = settings.whisper_gpu_fallback;
    const model = settings.whisper_fallback_model || 'large-v3';

    // Modes that never transcribe → word timing doesn't apply.
    const mode = settings.pipeline_mode || 'classic';
    const whisperSkipped =
        disabled ||
        ['srtdub', 'oneflow'].includes(mode) ||
        settings._input_mode === 'srt';

    const Toggle = ({ on, onClick, label }: { on: boolean; onClick: () => void; label: string }) => (
        <button
            type="button" title={label} onClick={onClick}
            className={`w-11 h-6 rounded-full transition-colors relative shrink-0 ${on ? 'bg-primary' : 'bg-white/10'}`}
        >
            <div className={`w-4 h-4 rounded-full bg-white absolute top-1 transition-transform ${on ? 'translate-x-6' : 'translate-x-1'}`} />
        </button>
    );

    return (
        <div className={`rounded-xl border border-border bg-white/[0.02] p-4 ${whisperSkipped ? 'opacity-40 pointer-events-none' : ''}`}>
            {/* Header */}
            <div className="flex items-center gap-2 mb-1">
                <span className="text-base">🎯</span>
                <p className="text-sm font-semibold text-text-primary">Word Timing</p>
                <span className={`ml-auto text-[10px] px-2 py-0.5 rounded-full ${wx ? 'bg-primary/20 text-primary' : 'bg-white/10 text-text-muted'}`}>
                    {whisperSkipped ? 'N/A this mode' : wx ? 'WhisperX ON' : 'Segment-level'}
                </span>
            </div>
            <p className="text-[11px] text-text-muted mb-3">
                How tightly each word is timed to the video. Tighter timing = better lip/subtitle sync.
                {whisperSkipped && <span className="text-yellow-400"> This mode skips transcription.</span>}
            </p>

            {/* Tier ladder */}
            <div className="text-[11px] text-text-muted mb-3 space-y-1">
                <div className="flex items-center gap-2">
                    <span className={`w-1.5 h-1.5 rounded-full ${wx ? 'bg-primary' : 'bg-white/20'}`} />
                    <span className={wx ? 'text-text-secondary' : ''}>1 · WhisperX alignment <span className="text-text-muted">— most precise</span></span>
                </div>
                <div className="flex items-center gap-2">
                    <span className={`w-1.5 h-1.5 rounded-full ${wx && gpuFallback ? 'bg-primary' : 'bg-white/20'}`} />
                    <span className={wx && gpuFallback ? 'text-text-secondary' : ''}>2 · Local Whisper on GPU <span className="text-text-muted">— auto-fallback</span></span>
                </div>
                <div className="flex items-center gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-white/20" />
                    <span>3 · Segment-level <span className="text-text-muted">— last resort</span></span>
                </div>
            </div>

            {/* WhisperX toggle */}
            <div className="flex items-center justify-between py-2 border-t border-border">
                <div className="pr-3">
                    <p className="text-sm text-text-primary">WhisperX Alignment</p>
                    <p className="text-[11px] text-text-muted">Word-level forced alignment (wav2vec2) after transcription.</p>
                </div>
                <Toggle on={wx} label="Toggle WhisperX" onClick={() => update({ use_whisperx: !wx })} />
            </div>

            {/* GPU fallback toggle */}
            <div className={`flex items-center justify-between py-2 border-t border-border ${!wx ? 'opacity-40 pointer-events-none' : ''}`}>
                <div className="pr-3">
                    <p className="text-sm text-text-primary">GPU fallback (local Whisper)</p>
                    <p className="text-[11px] text-text-muted">If WhisperX fails or looks wrong, re-transcribe on the GPU for word timings.</p>
                </div>
                <Toggle on={gpuFallback} label="Toggle GPU fallback" onClick={() => update({ whisper_gpu_fallback: !gpuFallback })} />
            </div>

            {/* Fallback model */}
            <div className={`flex items-center justify-between py-2 border-t border-border ${(!wx || !gpuFallback) ? 'opacity-40 pointer-events-none' : ''}`}>
                <div className="pr-3">
                    <p className="text-sm text-text-primary">Fallback model</p>
                    <p className="text-[11px] text-text-muted">large-v3 = best accuracy · medium = ~2× faster, lighter.</p>
                </div>
                <div className="flex rounded-lg overflow-hidden border border-border shrink-0">
                    {['large-v3', 'medium'].map(m => (
                        <button
                            key={m} type="button"
                            onClick={() => update({ whisper_fallback_model: m })}
                            className={`px-3 py-1 text-xs transition-colors ${model === m ? 'bg-primary text-white' : 'bg-white/5 text-text-muted hover:text-text-secondary'}`}
                        >
                            {m}
                        </button>
                    ))}
                </div>
            </div>
        </div>
    );
}
