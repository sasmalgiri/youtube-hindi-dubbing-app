'use client';

import { cn } from '@/lib/utils';

const STEPS = [
    { key: 'download', label: 'Download', icon: 'download' },
    { key: 'extract', label: 'Extract', icon: 'audio' },
    { key: 'transcribe', label: 'Transcribe', icon: 'text' },
    { key: 'translate', label: 'Translate', icon: 'translate' },
    { key: 'synthesize', label: 'Synthesize', icon: 'voice' },
    { key: 'assemble', label: 'Assemble', icon: 'video' },
];

interface ProgressPipelineProps {
    currentStep: string;
    stepProgress: number;
    overallProgress: number;
    message: string;
    isComplete: boolean;
    isError: boolean;
    eta?: string;
    stepTimes?: Record<string, number>;
    skippedSteps?: Record<string, string>;
}

function formatStepTime(seconds: number): string {
    if (!seconds || seconds < 0) return '';
    if (seconds < 60) return `${Math.round(seconds)}s`;
    const m = Math.floor(seconds / 60);
    const s = Math.round(seconds % 60);
    return s > 0 ? `${m}m ${s}s` : `${m}m`;
}

function StepIcon({ icon, size = 16 }: { icon: string; size?: number }) {
    const props = { width: size, height: size, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', strokeWidth: 2, strokeLinecap: 'round' as const, strokeLinejoin: 'round' as const };

    switch (icon) {
        case 'download':
            return <svg {...props}><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="7 10 12 15 17 10" /><line x1="12" x2="12" y1="15" y2="3" /></svg>;
        case 'audio':
            return <svg {...props}><path d="M2 10v3" /><path d="M6 6v11" /><path d="M10 3v18" /><path d="M14 8v7" /><path d="M18 5v13" /><path d="M22 10v3" /></svg>;
        case 'text':
            return <svg {...props}><path d="M17 6.1H3" /><path d="M21 12.1H3" /><path d="M15.1 18H3" /></svg>;
        case 'translate':
            return <svg {...props}><path d="m5 8 6 6" /><path d="m4 14 6-6 2-3" /><path d="M2 5h12" /><path d="M7 2h1" /><path d="m22 22-5-10-5 10" /><path d="M14 18h6" /></svg>;
        case 'voice':
            return <svg {...props}><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z" /><path d="M19 10v2a7 7 0 0 1-14 0v-2" /><line x1="12" x2="12" y1="19" y2="22" /></svg>;
        case 'video':
            return <svg {...props}><path d="m16 13 5.223 3.482a.5.5 0 0 0 .777-.416V7.87a.5.5 0 0 0-.752-.432L16 10.5" /><rect x="2" y="6" width="14" height="12" rx="2" /></svg>;
        default:
            return null;
    }
}

export default function ProgressPipeline({ currentStep, stepProgress, overallProgress, message, isComplete, isError, eta, stepTimes, skippedSteps }: ProgressPipelineProps) {
    const currentIndex = STEPS.findIndex(s => s.key === currentStep);
    const currentStepLabel = STEPS[currentIndex]?.label || currentStep;

    return (
        <div className="glass-card p-6">
            {/* Overall progress bar */}
            <div className="mb-4">
                <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium text-text-primary">
                        {isComplete ? 'Dubbing Complete!' : isError ? 'Error Occurred' : 'Overall Progress'}
                    </span>
                    <div className="flex items-center gap-3">
                        {eta && !isComplete && !isError && (
                            <span className="text-xs text-text-muted">{eta}</span>
                        )}
                        <span className="text-sm text-text-muted">
                            {Math.round(overallProgress * 100)}%
                        </span>
                    </div>
                </div>
                <div className="h-2 bg-white/5 rounded-full overflow-hidden">
                    <div
                        className={cn(
                            'h-full rounded-full transition-all duration-700 ease-out',
                            isError ? 'bg-error' : isComplete ? 'bg-success' : 'bg-gradient-to-r from-primary to-accent',
                        )}
                        style={{ width: `${overallProgress * 100}%` }}
                    />
                </div>
            </div>

            {/* Current step progress bar — each step has its own 0-100% */}
            {!isComplete && !isError && currentIndex >= 0 && (
                <div className="mb-6">
                    <div className="flex items-center justify-between mb-2">
                        <span className="text-xs font-medium text-primary-light">
                            Current Step: {currentStepLabel}
                        </span>
                        <span className="text-xs text-primary-light font-mono">
                            {Math.round(stepProgress * 100)}%
                        </span>
                    </div>
                    <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
                        <div
                            className="h-full bg-primary rounded-full transition-all duration-500 ease-out"
                            style={{ width: `${stepProgress * 100}%` }}
                        />
                    </div>
                </div>
            )}

            {/* Steps */}
            <div className="flex items-center justify-between">
                {STEPS.map((step, i) => {
                    const skipReason = skippedSteps?.[step.key];
                    const isDone = !skipReason && (isComplete || i < currentIndex);
                    const isActive = !isComplete && !isError && i === currentIndex;
                    const isPending = !isComplete && i > currentIndex;
                    const hasError = isError && i === currentIndex;

                    return (
                        <div key={step.key} className="flex items-center flex-1" title={skipReason ? `Skipped: ${skipReason}` : undefined} data-skipped={skipReason ? 'true' : undefined}>
                            {/* Step circle */}
                            <div className="flex flex-col items-center">
                                <div
                                    className={cn(
                                        'w-10 h-10 rounded-full flex items-center justify-center border-2 transition-all duration-300',
                                        isDone && 'bg-success/20 border-success text-success',
                                        isActive && 'bg-primary/20 border-primary text-primary-light animate-pulse-glow',
                                        isPending && 'bg-white/[0.02] border-border border-dashed text-text-muted',
                                        hasError && 'bg-error/20 border-error text-error',
                                    )}
                                >
                                    {isDone ? (
                                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                                            <path d="M20 6 9 17l-5-5" />
                                        </svg>
                                    ) : hasError ? (
                                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                                            <path d="M18 6 6 18" /><path d="m6 6 12 12" />
                                        </svg>
                                    ) : (
                                        <StepIcon icon={step.icon} />
                                    )}
                                </div>
                                <span className={cn(
                                    'text-[10px] mt-2 font-medium',
                                    isDone && 'text-success',
                                    isActive && 'text-primary-light',
                                    isPending && 'text-text-muted',
                                    hasError && 'text-error',
                                )}>
                                    {step.label}
                                </span>

                                {/* Per-step numeric % — done steps always 100%, active step shows live %, pending steps show 0% */}
                                <span className={cn(
                                    'text-[9px] mt-0.5 font-mono',
                                    isDone && 'text-success/80',
                                    isActive && 'text-primary-light',
                                    isPending && 'text-text-muted/40',
                                    hasError && 'text-error/80',
                                )}>
                                    {skipReason ? 'Skipped' : isDone ? '100%' : isActive ? `${Math.round(stepProgress * 100)}%` : '0%'}
                                </span>

                                {/* Step time */}
                                {stepTimes?.[step.key] != null && (isDone || isActive) && (
                                    <span className={cn(
                                        'text-[9px] mt-0.5',
                                        isDone ? 'text-success/70' : 'text-primary-light/70',
                                    )}>
                                        {formatStepTime(stepTimes[step.key])}
                                    </span>
                                )}

                                {/* Step progress bar */}
                                {isActive && (
                                    <div className="w-10 h-0.5 bg-white/5 rounded-full mt-1 overflow-hidden">
                                        <div
                                            className="h-full bg-primary rounded-full transition-all duration-500"
                                            style={{ width: `${stepProgress * 100}%` }}
                                        />
                                    </div>
                                )}
                            </div>

                            {/* Connector line */}
                            {i < STEPS.length - 1 && (
                                <div className={cn(
                                    'flex-1 h-0.5 mx-2 rounded-full transition-colors duration-300',
                                    i < currentIndex || isComplete ? 'bg-success/50' : 'bg-border',
                                )} />
                            )}
                        </div>
                    );
                })}
            </div>

            {skippedSteps && Object.keys(skippedSteps).length > 0 && (
                <ul className="mt-4 space-y-1 text-xs text-text-muted" data-testid="skipped-steps">
                    {STEPS.filter((s) => skippedSteps[s.key]).map((s) => (
                        <li key={s.key}><span className="font-medium text-text-secondary">{s.label} skipped:</span> {skippedSteps[s.key]}</li>
                    ))}
                </ul>
            )}

            {/* Current message */}
            {message && (
                <div className={cn(
                    'mt-4 text-center text-sm',
                    isError ? 'text-error' : 'text-text-secondary',
                )}>
                    {message}
                </div>
            )}
        </div>
    );
}
