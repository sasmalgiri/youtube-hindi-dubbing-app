'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { getPresets, getPreset, savePreset, deletePreset, getBuiltinPresets, getBuiltinPreset, type Preset } from '@/lib/api';
import type { DubbingSettings } from './SettingsPanel';

interface PresetTabsProps {
    currentSettings: DubbingSettings;
    onApply: (settings: DubbingSettings) => void;
    onLanguageChange?: (source: string, target: string) => void;
    sourceLanguage: string;
    targetLanguage: string;
}

const MAX_PRESETS = 8;

export default function PresetTabs({ currentSettings, onApply, onLanguageChange, sourceLanguage, targetLanguage }: PresetTabsProps) {
    const [presets, setPresets] = useState<Preset[]>([]);
    const [builtinPresets, setBuiltinPresets] = useState<Preset[]>([]);
    const [activeSlug, setActiveSlug] = useState<string | null>(null);
    const [saving, setSaving] = useState(false);
    const [showNameInput, setShowNameInput] = useState(false);
    const [newName, setNewName] = useState('');
    const [error, setError] = useState<string | null>(null);
    const selectGenRef = useRef(0);

    const loadPresets = useCallback(async () => {
        const [user, builtin] = await Promise.all([getPresets(), getBuiltinPresets()]);
        setPresets(user);
        setBuiltinPresets(builtin);
    }, []);

    useEffect(() => { loadPresets(); }, [loadPresets]);

    const handleSelectPreset = async (slug: string, builtin = false) => {
        // Click active preset again to deselect
        if (activeSlug === slug) {
            setActiveSlug(null);
            onApply({ preset_name: '' } as any);
            return;
        }
        const gen = ++selectGenRef.current;
        const data = builtin ? await getBuiltinPreset(slug) : await getPreset(slug);
        if (gen !== selectGenRef.current) return; // stale response, discard
        if (data?.settings) {
            // Separate language keys from settings to avoid leaking into DubbingSettings
            const { source_language, target_language, ...rest } = data.settings as Record<string, unknown>;
            setActiveSlug(slug);
            onApply({ ...rest, preset_name: data.name } as DubbingSettings);
            if (onLanguageChange && source_language && target_language) {
                onLanguageChange(source_language as string, target_language as string);
            }
        }
    };

    const handleSave = async () => {
        const name = newName.trim();
        if (!name) return;
        setSaving(true);
        setError(null);
        try {
            // Presets capture modes + knobs, not per-job content. Strip blob
            // fields (pasted SRT, pasted transcript) so saved presets stay small
            // and don't leak one job's content into other jobs.
            const { sd_srt_content, ...presetSafeSettings } = currentSettings as any;
            const fullSettings = {
                source_language: sourceLanguage,
                target_language: targetLanguage,
                ...presetSafeSettings,
            };
            const result = await savePreset(name, fullSettings as Record<string, unknown>);
            setActiveSlug(result.slug);
            setShowNameInput(false);
            setNewName('');
            await loadPresets();
        } catch (e) {
            setError(e instanceof Error ? e.message : 'Failed to save');
        }
        setSaving(false);
    };

    const handleDelete = async (slug: string, e: React.MouseEvent) => {
        e.stopPropagation();
        try {
            await deletePreset(slug);
            if (activeSlug === slug) setActiveSlug(null);
            await loadPresets();
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to delete preset');
        }
    };

    const handleOverwrite = async (slug: string, name: string, e: React.MouseEvent) => {
        e.stopPropagation();
        setSaving(true);
        try {
            const fullSettings = {
                source_language: sourceLanguage,
                target_language: targetLanguage,
                ...currentSettings,
            };
            await savePreset(name, fullSettings as Record<string, unknown>);
            setActiveSlug(slug);
            await loadPresets();
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to update');
        }
        setSaving(false);
    };

    return (
        <div className="glass-card overflow-hidden">
            <div className="px-5 py-3.5">
                <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-text-muted">
                            <path d="M4 20h16a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.93a2 2 0 0 1-1.66-.9l-.82-1.2A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13c0 1.1.9 2 2 2Z" />
                        </svg>
                        <span className="text-sm font-medium text-text-secondary">Presets</span>
                        <span className="text-[10px] text-text-muted">({presets.length}/{MAX_PRESETS})</span>
                    </div>
                </div>

                {/* Built-in quality presets (read-only) */}
                {builtinPresets.length > 0 && (
                    <div className="mb-3">
                        <p className="text-[10px] uppercase tracking-wide text-text-muted mb-1.5">Quality presets</p>
                        <div className="flex items-center gap-2 flex-wrap">
                            {builtinPresets.map((p) => (
                                <button
                                    key={p.slug}
                                    type="button"
                                    onClick={() => handleSelectPreset(p.slug, true)}
                                    title={p.description || p.name}
                                    className={`px-4 py-2 rounded-xl text-xs font-medium transition-all border ${
                                        activeSlug === p.slug
                                            ? 'bg-violet-500/20 border-violet-500 text-violet-200'
                                            : 'bg-violet-500/5 border-violet-500/20 text-violet-300 hover:bg-violet-500/10 hover:border-violet-500/40'
                                    }`}
                                >
                                    {p.name}
                                </button>
                            ))}
                        </div>
                    </div>
                )}

                {/* User-saved preset label */}
                {builtinPresets.length > 0 && (
                    <p className="text-[10px] uppercase tracking-wide text-text-muted mb-1.5">Your presets</p>
                )}

                {/* Preset tabs */}
                <div className="flex items-center gap-2 flex-wrap">
                    {presets.map((p) => (
                        <div key={p.slug} className="group relative">
                            <button
                                type="button"
                                onClick={() => handleSelectPreset(p.slug)}
                                className={`px-4 py-2 rounded-xl text-xs font-medium transition-all border ${
                                    activeSlug === p.slug
                                        ? 'bg-primary/20 border-primary text-primary-light'
                                        : 'bg-white/5 border-white/10 text-text-muted hover:bg-white/10 hover:border-white/20'
                                }`}
                            >
                                {p.name}
                            </button>

                            {/* Action buttons on hover */}
                            <div className="absolute -top-2 -right-2 hidden group-hover:flex gap-0.5">
                                {/* Overwrite */}
                                <button
                                    type="button"
                                    onClick={(e) => handleOverwrite(p.slug, p.name, e)}
                                    title="Update with current settings"
                                    className="w-5 h-5 rounded-full bg-amber-500/80 text-white flex items-center justify-center text-[10px] hover:bg-amber-500 transition-colors"
                                >
                                    <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                                        <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z" />
                                        <polyline points="17,21 17,13 7,13 7,21" />
                                    </svg>
                                </button>
                                {/* Delete */}
                                <button
                                    type="button"
                                    onClick={(e) => handleDelete(p.slug, e)}
                                    title="Delete preset"
                                    className="w-5 h-5 rounded-full bg-red-500/80 text-white flex items-center justify-center text-[10px] hover:bg-red-500 transition-colors"
                                >
                                    <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                                        <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
                                    </svg>
                                </button>
                            </div>
                        </div>
                    ))}

                    {/* Add new preset */}
                    {showNameInput ? (
                        <div className="flex items-center gap-2">
                            <input
                                type="text"
                                value={newName}
                                onChange={(e) => setNewName(e.target.value)}
                                onKeyDown={(e) => { if (e.key === 'Enter') handleSave(); if (e.key === 'Escape') { setShowNameInput(false); setNewName(''); } }}
                                placeholder="Preset name..."
                                autoFocus
                                maxLength={30}
                                className="px-3 py-1.5 rounded-lg bg-white/5 border border-white/20 text-xs text-text-primary placeholder:text-text-muted focus:outline-none focus:border-primary w-36"
                            />
                            <button
                                type="button"
                                onClick={handleSave}
                                disabled={saving || !newName.trim()}
                                className="px-3 py-1.5 rounded-lg text-xs font-medium bg-primary text-white hover:bg-primary/80 disabled:opacity-40 transition-colors"
                            >
                                {saving ? '...' : 'Save'}
                            </button>
                            <button
                                type="button"
                                onClick={() => { setShowNameInput(false); setNewName(''); setError(null); }}
                                className="px-2 py-1.5 rounded-lg text-xs text-text-muted hover:text-text-primary transition-colors"
                            >
                                Cancel
                            </button>
                        </div>
                    ) : (
                        presets.length < MAX_PRESETS && (
                            <button
                                type="button"
                                onClick={() => setShowNameInput(true)}
                                className="px-3 py-2 rounded-xl text-xs font-medium transition-all border border-dashed border-white/20 text-text-muted hover:border-primary/50 hover:text-primary-light hover:bg-primary/5"
                            >
                                + Save Current
                            </button>
                        )
                    )}
                </div>

                {error && (
                    <p className="text-[11px] text-red-400 mt-2">{error}</p>
                )}

                {presets.length === 0 && !showNameInput && (
                    <p className="text-[11px] text-text-muted mt-2">
                        No presets yet. Configure your settings, then click &quot;+ Save Current&quot; to create a preset.
                    </p>
                )}
            </div>
        </div>
    );
}
