'use client';

/**
 * Hindi Dialogue — module matrix panel.
 *
 * Preset picker + every stage's options with cost/availability badges.
 * Every change is previewed through /api/dialogue/resolve, which shows what
 * this PC will actually run: options that cannot run are switched off (with
 * the exact fix) and the first workable fallback is switched on.
 */
import { useEffect, useMemo, useRef, useState } from 'react';
import {
    fetchDialogueModules, fetchDialoguePresets, resolveDialogue,
    type DialogueMatrix, type DialogueOverrides, type DialoguePreset, type DialogueResolution,
} from '@/lib/api';

interface Props {
    preset: string;
    overrides: DialogueOverrides;
    onChange: (preset: string, overrides: DialogueOverrides) => void;
    sourceKind: 'url' | 'file';
}

const COST_BADGE: Record<string, string> = {
    free: 'bg-green-500/15 text-green-400',
    free_tier: 'bg-sky-500/15 text-sky-400',
    paid: 'bg-amber-500/15 text-amber-400',
};
const COST_LABEL: Record<string, string> = { free: 'free', free_tier: 'free tier', paid: 'paid' };

export default function DialogueModulesPanel({ preset, overrides, onChange, sourceKind }: Props) {
    const [matrix, setMatrix] = useState<DialogueMatrix | null>(null);
    const [presets, setPresets] = useState<DialoguePreset[]>([]);
    const [resolution, setResolution] = useState<DialogueResolution | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [open, setOpen] = useState(false);
    const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

    useEffect(() => {
        let alive = true;
        Promise.all([fetchDialogueModules(sourceKind), fetchDialoguePresets()])
            .then(([m, p]) => { if (alive) { setMatrix(m); setPresets(p); } })
            .catch((e) => alive && setError(e instanceof Error ? e.message : String(e)));
        return () => { alive = false; };
    }, [sourceKind]);

    const activePreset = preset || matrix?.default_preset || 'free-online';
    const presetDef = presets.find((p) => p.id === activePreset);

    // Debounced preview of what this PC will actually run.
    useEffect(() => {
        if (timer.current) clearTimeout(timer.current);
        timer.current = setTimeout(() => {
            resolveDialogue(activePreset, overrides, sourceKind)
                .then(setResolution)
                .catch((e) => setError(e instanceof Error ? e.message : String(e)));
        }, 300);
        return () => { if (timer.current) clearTimeout(timer.current); };
    }, [activePreset, overrides, sourceKind]);

    const params = (overrides.params as Record<string, unknown>) || {};
    const effectiveParams: Record<string, unknown> = { ...(presetDef?.params || {}), ...params };

    const selected = (stageId: string): string[] =>
        (overrides[stageId] as string[] | undefined) ?? presetDef?.selections?.[stageId] ?? [];

    const setStage = (stageId: string, value: string[]) =>
        onChange(activePreset, { ...overrides, [stageId]: value });
    const setParam = (key: string, value: unknown) =>
        onChange(activePreset, { ...overrides, params: { ...params, [key]: value } });

    const toggle = (stageId: string, multi: boolean, choiceId: string) => {
        const cur = selected(stageId);
        if (!multi) return setStage(stageId, [choiceId]);
        setStage(stageId, cur.includes(choiceId) ? cur.filter((c) => c !== choiceId) : [...cur, choiceId]);
    };

    const changedCount = useMemo(() => Object.keys(overrides).length, [overrides]);

    if (error && !matrix) {
        return <div className="glass-card p-3 text-xs text-red-400">Dialogue modules unavailable: {error}</div>;
    }
    if (!matrix) return <div className="glass-card p-3 text-xs text-text-muted">Loading dialogue modules…</div>;

    return (
        <div className="glass-card p-3 space-y-3">
            {/* Presets */}
            <div>
                <div className="text-xs font-semibold text-text-primary mb-2">Hindi Dialogue preset</div>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                    {presets.map((p) => (
                        <button key={p.id} type="button"
                            onClick={() => onChange(p.id, {})}
                            className={`text-left rounded-lg border px-3 py-2 transition-colors ${p.id === activePreset
                                ? 'border-pink-500 bg-pink-500/10' : 'border-border bg-white/5 hover:bg-white/10'}`}>
                            <div className="text-xs font-semibold text-text-primary">{p.name}</div>
                            <div className="text-[11px] text-text-muted leading-snug">{p.description}</div>
                        </button>
                    ))}
                </div>
            </div>

            {/* Effective result on this PC */}
            {resolution && (
                <div className={`rounded-lg border px-3 py-2 text-[11px] space-y-1 ${resolution.ok ? 'border-border' : 'border-red-500/60'}`}>
                    <div className="font-semibold text-text-primary">
                        {resolution.ok ? 'Will run on this PC' : 'Cannot run yet'}
                        {matrix.environment.gpu === false && <span className="ml-2 text-amber-400">(no NVIDIA GPU detected)</span>}
                    </div>
                    <div className="flex flex-wrap gap-x-3 gap-y-0.5 text-text-muted">
                        {matrix.stages.map((st) => (
                            <span key={st.id}>{st.label}: <span className="text-text-primary">
                                {(resolution.selections[st.id] || []).join(' → ') || 'skipped'}</span></span>
                        ))}
                    </div>
                    {resolution.changes.map((c, i) => (
                        <div key={i} className={c.action === 'deactivated' ? 'text-amber-400' : 'text-sky-400'}>
                            {c.action === 'deactivated' ? '○' : '●'} {c.stage}: {c.action} <b>{c.choice}</b> — {c.reason}
                            {c.fix && <div className="pl-4 text-text-muted">fix: {c.fix}</div>}
                        </div>
                    ))}
                    {resolution.warnings.map((w, i) => <div key={i} className="text-text-muted">! {w}</div>)}
                    {resolution.blocking.map((b, i) => <div key={i} className="text-red-400">✕ {b}</div>)}
                </div>
            )}

            {/* Customise */}
            <button type="button" onClick={() => setOpen(!open)}
                className="text-xs text-text-muted hover:text-text-primary">
                {open ? '▾' : '▸'} Customise modules{changedCount ? ` (${changedCount} changed)` : ''}
            </button>
            {open && (
                <div className="space-y-3">
                    {matrix.stages.map((st) => {
                        const cur = selected(st.id);
                        return (
                            <div key={st.id}>
                                <div className="text-xs font-semibold text-text-primary">
                                    {st.label}{st.multi && <span className="ml-1 font-normal text-text-muted">(order = priority; later ones are fallbacks)</span>}
                                </div>
                                <div className="text-[11px] text-text-muted mb-1">{st.description}</div>
                                <div className="flex flex-wrap gap-1.5">
                                    {st.choices.map((c) => {
                                        const idx = cur.indexOf(c.id);
                                        const on = idx >= 0;
                                        return (
                                            <button key={c.id} type="button" onClick={() => toggle(st.id, st.multi, c.id)}
                                                title={[c.description, ...c.missing.map((m) => `missing ${m.name}: ${m.fix}`), ...c.notes].join('\n')}
                                                className={`rounded-lg border px-2 py-1 text-[11px] transition-colors ${on
                                                    ? 'border-pink-500 bg-pink-500/15 text-text-primary'
                                                    : 'border-border bg-white/5 text-text-muted hover:bg-white/10'} ${!c.available ? 'opacity-50' : ''}`}>
                                                {st.multi && on && <span className="mr-1 font-bold">{idx + 1}.</span>}
                                                {c.label}
                                                <span className={`ml-1 rounded px-1 ${COST_BADGE[c.cost]}`}>{COST_LABEL[c.cost]}</span>
                                                {!c.available && <span className="ml-1 text-amber-400">not installed</span>}
                                            </button>
                                        );
                                    })}
                                </div>
                            </div>
                        );
                    })}

                    {/* Parameters */}
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs">
                        <label className="flex items-center gap-2">
                            <input type="checkbox" checked={Boolean(effectiveParams.allow_paid)}
                                onChange={(e) => setParam('allow_paid', e.target.checked)} />
                            Allow paid services
                        </label>
                        <label className="flex items-center gap-2">
                            <input type="checkbox" checked={Boolean(effectiveParams.local_only)}
                                onChange={(e) => setParam('local_only', e.target.checked)} />
                            Local AI only (no cloud AI)
                        </label>
                        <label className="flex items-center gap-2">
                            Speakers
                            <input type="number" min={0} max={20} className="w-16 rounded bg-white/5 border border-border px-1"
                                value={Number(effectiveParams.num_speakers ?? 0)}
                                onChange={(e) => setParam('num_speakers', Number(e.target.value))} />
                            <span className="text-text-muted">0 = detect</span>
                        </label>
                        <label className="flex items-center gap-2">
                            Max line speed-up
                            <input type="number" step={0.05} min={1} max={1.3} className="w-16 rounded bg-white/5 border border-border px-1"
                                value={Number(effectiveParams.max_stretch ?? 1.15)}
                                onChange={(e) => setParam('max_stretch', Number(e.target.value))} />
                        </label>
                        <label className="flex items-center gap-2 sm:col-span-2">
                            Ollama model
                            {matrix.environment.ollama_models.length > 0 ? (
                                <select className="rounded bg-white/5 border border-border px-1"
                                    value={String(effectiveParams.ollama_model ?? '')}
                                    onChange={(e) => setParam('ollama_model', e.target.value)}>
                                    <option value="">(OLLAMA_MODEL from backend/.env)</option>
                                    {matrix.environment.ollama_models.map((m) => <option key={m} value={m}>{m}</option>)}
                                </select>
                            ) : (
                                <span className="text-text-muted">Ollama not running (install from ollama.com, then `ollama pull gemma3:12b`)</span>
                            )}
                        </label>
                    </div>
                    {changedCount > 0 && (
                        <button type="button" onClick={() => onChange(activePreset, {})}
                            className="text-xs text-text-muted hover:text-text-primary underline">
                            Reset to preset
                        </button>
                    )}
                </div>
            )}
        </div>
    );
}
