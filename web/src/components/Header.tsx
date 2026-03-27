'use client';

export default function Header() {
    return (
        <header className="h-16 border-b border-border bg-card/50 backdrop-blur-sm flex items-center px-6 sticky top-0 z-50">
            <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-lg bg-primary flex items-center justify-center">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z" />
                        <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
                        <line x1="12" x2="12" y1="19" y2="22" />
                    </svg>
                </div>
                <h1 className="text-lg font-semibold text-text-primary">
                    Hindi<span className="text-primary">Dub</span>
                </h1>
            </div>
            <div className="ml-auto flex items-center gap-4">
                <span className="text-xs text-text-muted px-3 py-1 rounded-full bg-white/5 border border-border">
                    Hindi Dubbing
                </span>
            </div>
        </header>
    );
}
