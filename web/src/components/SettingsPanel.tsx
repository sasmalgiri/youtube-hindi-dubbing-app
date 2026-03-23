'use client';

import { useState } from 'react';

export interface DubbingSettings {
    asr_model: string;
    translation_engine: string;
    tts_rate: string;
    mix_original: boolean;
    original_volume: number;
    use_chatterbox: boolean;
    use_elevenlabs: boolean;
    use_google_tts: boolean;
    use_coqui_xtts: boolean;
    use_edge_tts: boolean;
    prefer_youtube_subs: boolean;
    multi_speaker: boolean;
    transcribe_only: boolean;
    audio_priority: boolean;
    audio_bitrate: string;
    encode_preset: string;
    dub_chain: string[];
}

interface SettingsPanelProps {
    settings: DubbingSettings;
    onChange: (settings: DubbingSettings) => void;
}

export default function SettingsPanel({ settings, onChange }: SettingsPanelProps) {
    const [open, setOpen] = useState(false);

    const update = (partial: Partial<DubbingSettings>) => {
        onChange({ ...settings, ...partial });
    };

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

            {open && (
                <div className="px-5 pb-5 space-y-5 animate-slide-up border-t border-border pt-4">
                    {/* ── Transcription Section ── */}
                    <div>
                        <p className="text-sm font-medium text-text-primary mb-3">Transcription (Whisper)</p>
                        <div className="space-y-3">
                            <div>
                                <p className="text-xs text-text-muted mb-1.5">Whisper Model</p>
                                <div className="grid grid-cols-3 gap-2">
                                    {[
                                        { value: 'base', label: 'Base', desc: 'Fast, basic' },
                                        { value: 'medium', label: 'Medium', desc: 'Balanced' },
                                        { value: 'large-v3', label: 'Large-v3', desc: 'Best quality' },
                                    ].map((m) => (
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

                            {/* YouTube Subtitles */}
                            <div className="flex items-center justify-between">
                                <div>
                                    <p className="text-sm text-text-primary">Use YouTube Subtitles</p>
                                    <p className="text-xs text-text-muted">Skip Whisper, use existing subs (faster)</p>
                                </div>
                                <button
                                    onClick={() => update({ prefer_youtube_subs: !settings.prefer_youtube_subs })}
                                    className={`
                                        w-11 h-6 rounded-full transition-colors relative
                                        ${settings.prefer_youtube_subs ? 'bg-primary' : 'bg-white/10'}
                                    `}
                                >
                                    <div className={`
                                        w-4 h-4 rounded-full bg-white absolute top-1 transition-transform
                                        ${settings.prefer_youtube_subs ? 'translate-x-6' : 'translate-x-1'}
                                    `} />
                                </button>
                            </div>

                            {/* Chain Dub */}
                            <div className="flex items-center justify-between">
                                <div>
                                    <p className="text-sm text-text-primary">Chain Dub (English → Hindi)</p>
                                    <p className="text-xs text-text-muted">Dub to English first using subs, then English to Hindi (best for non-English videos)</p>
                                </div>
                                <button
                                    onClick={() => update({
                                        dub_chain: settings.dub_chain.length > 0 ? [] : ['en', 'hi'],
                                    })}
                                    className={`
                                        w-11 h-6 rounded-full transition-colors relative
                                        ${settings.dub_chain.length > 0 ? 'bg-primary' : 'bg-white/10'}
                                    `}
                                >
                                    <div className={`
                                        w-4 h-4 rounded-full bg-white absolute top-1 transition-transform
                                        ${settings.dub_chain.length > 0 ? 'translate-x-6' : 'translate-x-1'}
                                    `} />
                                </button>
                            </div>
                        </div>
                    </div>

                    {/* ── Translation Section ── */}
                    <div>
                        <p className="text-sm font-medium text-text-primary mb-3">Translation Engine</p>
                        <div className="grid grid-cols-5 gap-2 mb-3">
                            {[
                                { value: 'auto', label: 'Auto', desc: 'Best available' },
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
                        <div className="grid grid-cols-3 gap-2">
                            {[
                                { value: 'ollama', label: 'Ollama', desc: 'Local LLM (GPU)' },
                                { value: 'hinglish', label: 'Hinglish AI', desc: 'Custom Hindi model' },
                                { value: 'google', label: 'Google', desc: 'Free, basic' },
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

                    {/* Transcribe Only (manual translation) */}
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-sm text-text-primary">Transcribe Only</p>
                            <p className="text-xs text-text-muted">Get SRT to translate yourself (e.g. with Claude), then upload back</p>
                        </div>
                        <button
                            onClick={() => update({ transcribe_only: !settings.transcribe_only })}
                            className={`
                                w-11 h-6 rounded-full transition-colors relative
                                ${settings.transcribe_only ? 'bg-primary' : 'bg-white/10'}
                            `}
                        >
                            <div className={`
                                w-4 h-4 rounded-full bg-white absolute top-1 transition-transform
                                ${settings.transcribe_only ? 'translate-x-6' : 'translate-x-1'}
                            `} />
                        </button>
                    </div>

                    {/* Multi-Speaker Voices */}
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-sm text-text-primary">Multi-Speaker Voices</p>
                            <p className="text-xs text-text-muted">Detect speakers & assign distinct voices (needs HF_TOKEN, adds ~30s)</p>
                        </div>
                        <button
                            onClick={() => update({ multi_speaker: !settings.multi_speaker })}
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

                    {/* TTS Engines */}
                    <div>
                        <p className="text-sm font-medium text-text-primary mb-3">TTS Engines</p>
                        <div className="space-y-3">
                            {/* Chatterbox */}
                            <div className="flex items-center justify-between">
                                <div>
                                    <p className="text-sm text-text-primary">Chatterbox AI</p>
                                    <p className="text-xs text-text-muted">Free, GPU required, most human-like</p>
                                </div>
                                <button
                                    onClick={() => update({ use_chatterbox: !settings.use_chatterbox })}
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
                                    <p className="text-xs text-text-muted">Paid API, needs ELEVENLABS_API_KEY in .env</p>
                                </div>
                                <button
                                    onClick={() => update({ use_elevenlabs: !settings.use_elevenlabs })}
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
                                    <p className="text-sm text-text-primary">Coqui XTTS v2</p>
                                    <p className="text-xs text-text-muted">Free, GPU required, voice cloning from original speaker</p>
                                </div>
                                <button
                                    onClick={() => update({ use_coqui_xtts: !settings.use_coqui_xtts })}
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
                                    onClick={() => update({ use_google_tts: !settings.use_google_tts })}
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
                                    <p className="text-sm text-text-primary">Edge-TTS</p>
                                    <p className="text-xs text-text-muted">Free, no GPU needed, decent quality</p>
                                </div>
                                <button
                                    onClick={() => update({ use_edge_tts: !settings.use_edge_tts })}
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
                        {settings.use_coqui_xtts && settings.use_edge_tts ? (
                            <p className="text-[10px] text-primary mt-2 font-medium">
                                Hybrid Mode: Coqui XTTS + Edge-TTS will run in parallel (~2x faster)
                            </p>
                        ) : (
                            <p className="text-[10px] text-text-muted mt-2">First enabled engine from top to bottom will be used. Enable both Coqui + Edge for hybrid parallel mode.</p>
                        )}
                    </div>

                    {/* TTS Speech Rate */}
                    <div>
                        <label className="label mb-2 block">
                            Speech Rate: <span className="text-primary-light">{settings.tts_rate}</span>
                        </label>
                        <input
                            type="range"
                            min={-50}
                            max={50}
                            value={parseInt(settings.tts_rate) || 0}
                            onChange={(e) => {
                                const v = parseInt(e.target.value);
                                update({ tts_rate: `${v >= 0 ? '+' : ''}${v}%` });
                            }}
                            className="w-full accent-primary"
                        />
                        <div className="flex justify-between text-[10px] text-text-muted">
                            <span>Slower</span>
                            <span>Normal</span>
                            <span>Faster</span>
                        </div>
                    </div>

                    {/* Mix Background Music */}
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-sm text-text-primary">Mix Background Music</p>
                            <p className="text-xs text-text-muted">Keep original background music (vocals removed) behind dubbed voice</p>
                        </div>
                        <button
                            onClick={() => update({ mix_original: !settings.mix_original })}
                            className={`
                                w-11 h-6 rounded-full transition-colors relative
                                ${settings.mix_original ? 'bg-primary' : 'bg-white/10'}
                            `}
                        >
                            <div className={`
                                w-4 h-4 rounded-full bg-white absolute top-1 transition-transform
                                ${settings.mix_original ? 'translate-x-6' : 'translate-x-1'}
                            `} />
                        </button>
                    </div>

                    {/* Music Volume */}
                    {settings.mix_original && (
                        <div className="animate-slide-up">
                            <label className="label mb-2 block">
                                Music Volume: <span className="text-primary-light">{Math.round(settings.original_volume * 100)}%</span>
                            </label>
                            <input
                                type="range"
                                min={0}
                                max={50}
                                value={settings.original_volume * 100}
                                onChange={(e) => update({ original_volume: parseInt(e.target.value) / 100 })}
                                className="w-full accent-primary"
                                title="Original volume"
                            />
                        </div>
                    )}

                    {/* ── Audio & Performance Section ── */}
                    <div>
                        <p className="text-sm font-medium text-text-primary mb-3">Audio & Performance</p>
                        <div className="space-y-3">
                            {/* Audio Priority */}
                            <div className="flex items-center justify-between">
                                <div>
                                    <p className="text-sm text-text-primary">Audio Priority</p>
                                    <p className="text-xs text-text-muted">TTS speaks naturally, video adjusts to match (best for listening)</p>
                                </div>
                                <button
                                    type="button"
                                    title="Toggle audio priority"
                                    onClick={() => update({ audio_priority: !settings.audio_priority })}
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

                            {/* Audio Bitrate */}
                            <div>
                                <p className="text-xs text-text-muted mb-1.5">Audio Quality</p>
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

                            {/* Encode Speed */}
                            <div>
                                <p className="text-xs text-text-muted mb-1.5">Video Encode Speed</p>
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
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
