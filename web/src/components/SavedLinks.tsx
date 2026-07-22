'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { getLinks, deleteLink, updateLinkPreset, createJob, type SavedLink, type LinkPreset } from '@/lib/api';
import { extractYouTubeId, getThumbnailUrl } from '@/lib/utils';

interface SavedLinksProps {
    onSelect: (url: string) => void;
    onJobStarted?: (jobId: string) => void;
}

const PRESET_LABELS: { key: keyof LinkPreset; label: string; type: 'select' | 'toggle'; options?: { value: string; label: string }[] }[] = [
    {
        key: 'target_language', label: 'Language', type: 'select',
        options: [
            { value: 'hi', label: 'Hindi' }, { value: 'bn', label: 'Bengali' },
            { value: 'ta', label: 'Tamil' }, { value: 'te', label: 'Telugu' },
            { value: 'mr', label: 'Marathi' }, { value: 'ja', label: 'Japanese' },
            { value: 'ko', label: 'Korean' }, { value: 'es', label: 'Spanish' },
            { value: 'fr', label: 'French' }, { value: 'de', label: 'German' },
            { value: 'en', label: 'English' },
        ],
    },
    {
        key: 'translation_engine', label: 'Translation', type: 'select',
        options: [
            { value: 'auto', label: 'Auto' }, { value: 'turbo', label: 'Turbo' },
            { value: 'groq', label: 'Groq' }, { value: 'sambanova', label: 'SambaNova' },
            { value: 'gemini', label: 'Gemini' }, { value: 'google', label: 'Google' },
            { value: 'ollama', label: 'Ollama' }, { value: 'hinglish', label: 'Hinglish' },
        ],
    },
    {
        key: 'asr_model', label: 'Whisper', type: 'select',
        options: [
            { value: 'base', label: 'Base' }, { value: 'small', label: 'Small' },
            { value: 'medium', label: 'Medium' }, { value: 'large-v3-turbo', label: 'Turbo' },
            { value: 'large-v3', label: 'Large-v3' },
        ],
    },
    {
        key: 'encode_preset', label: 'Encode', type: 'select',
        options: [
            { value: 'ultrafast', label: 'Ultra Fast' }, { value: 'veryfast', label: 'Very Fast' },
            { value: 'fast', label: 'Fast' }, { value: 'medium', label: 'Medium' },
        ],
    },
    { key: 'use_cosyvoice', label: 'CosyVoice 2', type: 'toggle' },
    { key: 'use_chatterbox', label: 'Chatterbox', type: 'toggle' },
    { key: 'use_edge_tts', label: 'Edge TTS', type: 'toggle' },
    { key: 'use_coqui_xtts', label: 'Coqui XTTS', type: 'toggle' },
    { key: 'mix_original', label: 'Mix BG Music', type: 'toggle' },
    { key: 'audio_priority', label: 'Audio Priority', type: 'toggle' },
    { key: 'enable_manual_review', label: 'Manual Review', type: 'toggle' },
    { key: 'use_whisperx', label: 'WhisperX Align', type: 'toggle' },
];

function PresetSummary({ preset }: { preset?: LinkPreset }) {
    if (!preset || Object.keys(preset).length === 0) return <span className="text-text-muted">Default</span>;
    const parts: string[] = [];
    if (preset.target_language) parts.push(preset.target_language.toUpperCase());
    if (preset.translation_engine && preset.translation_engine !== 'auto') parts.push(preset.translation_engine);
    if (preset.use_cosyvoice) parts.push('CosyVoice');
    if (preset.use_chatterbox) parts.push('CBX');
    if (preset.use_edge_tts) parts.push('Edge');
    if (preset.use_coqui_xtts) parts.push('XTTS');
    if (preset.mix_original) parts.push('BG');
    if (preset.use_whisperx) parts.push('WhisperX');
    return <span className="text-primary-light">{parts.join(' · ') || 'Default'}</span>;
}

function PresetEditor({ preset, onSave }: { preset: LinkPreset; onSave: (p: LinkPreset) => void }) {
    const [local, setLocal] = useState<LinkPreset>({ ...preset });
    const timerRef = useRef<ReturnType<typeof setTimeout>>();

    const update = (key: keyof LinkPreset, value: unknown) => {
        setLocal(prev => {
            const next = { ...prev, [key]: value };
            // Debounce saves to avoid rapid-fire API calls
            clearTimeout(timerRef.current);
            timerRef.current = setTimeout(() => onSave(next), 500);
            return next;
        });
    };

    return (
        <div className="px-3 pb-3 pt-1 space-y-2 animate-slide-up">
            <div className="grid grid-cols-2 gap-2">
                {PRESET_LABELS.map((field) => (
                    <div key={field.key} className="flex items-center justify-between gap-2">
                        <span className="text-[11px] text-text-muted whitespace-nowrap">{field.label}</span>
                        {field.type === 'select' && field.options ? (
                            <select
                                title={field.label}
                                value={(local[field.key] as string) || field.options[0].value}
                                onChange={(e) => update(field.key, e.target.value)}
                                className="bg-white/5 border border-white/10 rounded px-1.5 py-0.5 text-[11px] text-text-primary outline-none focus:border-primary min-w-[80px]"
                            >
                                {field.options.map((o) => (
                                    <option key={o.value} value={o.value} className="bg-gray-900 text-white">{o.label}</option>
                                ))}
                            </select>
                        ) : (
                            <button
                                title={`Toggle ${field.label}`}
                                onClick={() => update(field.key, !local[field.key])}
                                className={`w-8 h-4 rounded-full transition-colors relative flex-shrink-0 ${local[field.key] ? 'bg-primary' : 'bg-white/10'}`}
                            >
                                <div className={`w-3 h-3 rounded-full bg-white absolute top-0.5 transition-transform ${local[field.key] ? 'translate-x-4' : 'translate-x-0.5'}`} />
                            </button>
                        )}
                    </div>
                ))}
            </div>
        </div>
    );
}

export default function SavedLinks({ onSelect, onJobStarted }: SavedLinksProps) {
    const [links, setLinks] = useState<SavedLink[]>([]);
    const [open, setOpen] = useState(false);
    const [editingId, setEditingId] = useState<string | null>(null);
    const [queuingId, setQueuingId] = useState<string | null>(null);

    const loadLinks = useCallback(() => {
        getLinks().then(setLinks).catch(() => {});
    }, []);

    useEffect(() => { loadLinks(); }, [loadLinks]);

    // Refresh links periodically to pick up auto-saved ones
    useEffect(() => {
        const interval = setInterval(loadLinks, 5000);
        return () => clearInterval(interval);
    }, [loadLinks]);

    const handleDelete = useCallback(async (id: string) => {
        const updated = await deleteLink(id);
        setLinks(updated);
    }, []);

    const handlePresetSave = useCallback(async (id: string, preset: LinkPreset) => {
        const updated = await updateLinkPreset(id, preset);
        if (updated.length > 0) setLinks(updated);
    }, []);

    const handleQueue = useCallback(async (link: SavedLink) => {
        setQueuingId(link.id);
        try {
            const preset = link.preset || {};
            const { id } = await createJob({ url: link.url, ...preset });
            onJobStarted?.(id);
        } catch (e) {
            console.error('Failed to queue:', e);
        } finally {
            setQueuingId(null);
        }
    }, [onJobStarted]);

    return (
        <div className="glass-card overflow-hidden">
            {/* Header */}
            <button
                onClick={() => setOpen(!open)}
                className="w-full flex items-center justify-between p-4 hover:bg-white/[0.02] transition-colors"
            >
                <div className="flex items-center gap-2">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-primary">
                        <path d="m19 21-7-4-7 4V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2v16z" />
                    </svg>
                    <span className="text-sm font-medium text-text-primary">
                        Saved Links
                    </span>
                    {links.length > 0 && (
                        <span className="text-xs bg-primary/20 text-primary px-1.5 py-0.5 rounded-full">
                            {links.length}
                        </span>
                    )}
                </div>
                <svg
                    width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
                    className={`text-text-muted transition-transform ${open ? 'rotate-180' : ''}`}
                >
                    <path d="m6 9 6 6 6-6" />
                </svg>
            </button>

            {open && (
                <div className="border-t border-border">
                    {/* Links list */}
                    {links.length === 0 ? (
                        <div className="p-4 text-center text-sm text-text-muted">
                            No saved links yet. Paste a URL above and save it here.
                        </div>
                    ) : (
                        <div className="max-h-[500px] overflow-y-auto">
                            {links.map((link) => {
                                const videoId = extractYouTubeId(link.url);
                                const isEditing = editingId === link.id;
                                return (
                                    <div
                                        key={link.id}
                                        className="border-b border-border/30 last:border-0"
                                    >
                                        <div className={`flex items-center gap-3 p-3 hover:bg-white/[0.03] transition-colors group ${link.completed ? 'bg-green-500/[0.05]' : ''}`}>
                                            {/* Thumbnail */}
                                            {videoId && (
                                                <div className="relative flex-shrink-0">
                                                    <img
                                                        src={getThumbnailUrl(videoId)}
                                                        alt=""
                                                        className={`w-20 h-12 object-cover rounded-md ${link.completed ? 'opacity-70' : ''}`}
                                                    />
                                                    {link.completed && (
                                                        <div className="absolute inset-0 flex items-center justify-center">
                                                            <div className="bg-green-500 rounded-full p-0.5">
                                                                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                                                                    <polyline points="20 6 9 17 4 12" />
                                                                </svg>
                                                            </div>
                                                        </div>
                                                    )}
                                                </div>
                                            )}

                                            {/* URL + title + preset summary */}
                                            <div
                                                className="flex-1 min-w-0 cursor-pointer"
                                                onClick={() => onSelect(link.url)}
                                            >
                                                <div className="flex items-center gap-1.5">
                                                    {link.title && (
                                                        <p className="text-sm text-text-primary truncate">{link.title}</p>
                                                    )}
                                                    {link.completed && (
                                                        <span className="text-[10px] bg-green-500/20 text-green-400 px-1.5 py-0.5 rounded-full flex-shrink-0">Done</span>
                                                    )}
                                                </div>
                                                <p className="text-xs text-text-muted truncate">{link.url}</p>
                                                <p className="text-[10px] mt-0.5">
                                                    <PresetSummary preset={link.preset} />
                                                </p>
                                            </div>

                                            {/* Actions */}
                                            <div className="flex items-center gap-1 flex-shrink-0">
                                                {/* Settings button */}
                                                <button
                                                    onClick={() => setEditingId(isEditing ? null : link.id)}
                                                    className={`p-1.5 rounded-md transition-colors ${isEditing ? 'bg-primary/20 text-primary' : 'hover:bg-primary/20 text-text-muted hover:text-primary'}`}
                                                    title="Edit preset"
                                                >
                                                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                                        <path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z" />
                                                        <circle cx="12" cy="12" r="3" />
                                                    </svg>
                                                </button>
                                                {/* Queue dubbing button */}
                                                <button
                                                    onClick={() => handleQueue(link)}
                                                    disabled={queuingId === link.id}
                                                    className={`p-1.5 rounded-md transition-colors ${queuingId === link.id ? 'bg-primary/30 text-primary animate-pulse' : 'hover:bg-primary/20 text-text-muted hover:text-primary'}`}
                                                    title="Start dubbing with preset"
                                                >
                                                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                                        <polygon points="5 3 19 12 5 21 5 3" />
                                                    </svg>
                                                </button>
                                                {/* Delete button */}
                                                <button
                                                    onClick={() => handleDelete(link.id)}
                                                    className="p-1.5 rounded-md hover:bg-error/20 text-text-muted hover:text-error transition-colors opacity-0 group-hover:opacity-100"
                                                    title="Remove"
                                                >
                                                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                                        <path d="M18 6 6 18M6 6l12 12" />
                                                    </svg>
                                                </button>
                                            </div>
                                        </div>

                                        {/* Inline Preset Editor */}
                                        {isEditing && (
                                            <PresetEditor
                                                preset={link.preset || {}}
                                                onSave={(p) => handlePresetSave(link.id, p)}
                                            />
                                        )}
                                    </div>
                                );
                            })}
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}
