'use client';

const LANGUAGES = [
    { code: 'hi', name: 'Hindi', flag: '\u0939\u093F' },
    { code: 'en', name: 'English', flag: 'En' },
    { code: 'es', name: 'Spanish', flag: 'Es' },
    { code: 'fr', name: 'French', flag: 'Fr' },
    { code: 'de', name: 'German', flag: 'De' },
    { code: 'ja', name: 'Japanese', flag: '\u65E5' },
    { code: 'ko', name: 'Korean', flag: '\uD55C' },
    { code: 'zh', name: 'Chinese', flag: '\u4E2D' },
    { code: 'pt', name: 'Portuguese', flag: 'Pt' },
    { code: 'ru', name: 'Russian', flag: '\u0420\u0443' },
    { code: 'ar', name: 'Arabic', flag: '\u0639\u0631' },
    { code: 'it', name: 'Italian', flag: 'It' },
    { code: 'tr', name: 'Turkish', flag: 'Tr' },
    { code: 'bn', name: 'Bengali', flag: '\u09AC\u09BE' },
    { code: 'ta', name: 'Tamil', flag: '\u0BA4' },
    { code: 'te', name: 'Telugu', flag: '\u0C24\u0C46' },
    { code: 'mr', name: 'Marathi', flag: '\u092E' },
    { code: 'gu', name: 'Gujarati', flag: '\u0A97\u0AC1' },
    { code: 'kn', name: 'Kannada', flag: '\u0C95' },
    { code: 'ml', name: 'Malayalam', flag: '\u0D2E' },
    { code: 'pa', name: 'Punjabi', flag: '\u0A2A' },
    { code: 'ur', name: 'Urdu', flag: '\u0627\u0631' },
];

interface LanguageSelectorProps {
    sourceLanguage: string;
    targetLanguage: string;
    onSourceChange: (lang: string) => void;
    onTargetChange: (lang: string) => void;
}

export default function LanguageSelector({
    sourceLanguage,
    targetLanguage,
    onSourceChange,
    onTargetChange,
}: LanguageSelectorProps) {
    // The dub is locked to an English source: there is no 'Auto Detect' any
    // more (the backend treats 'auto' as English), so an old saved 'auto' is
    // shown, and swapped, as English instead of falling through to Hindi.
    const source = sourceLanguage === 'auto' ? 'en' : sourceLanguage;

    const handleSwap = () => {
        onSourceChange(targetLanguage);
        onTargetChange(source);
    };

    return (
        <div className="flex items-center gap-3">
            {/* Source language */}
            <div className="flex-1">
                <label className="label mb-1.5 block text-xs">From</label>
                <select
                    value={source}
                    onChange={(e) => onSourceChange(e.target.value)}
                    className="w-full px-3 py-2.5 rounded-xl border border-border bg-white/[0.03] text-text-primary text-sm focus:outline-none focus:ring-1 focus:ring-primary/40 appearance-none cursor-pointer"
                >
                    {LANGUAGES.map((l) => (
                        <option key={l.code} value={l.code}>
                            {l.flag} {l.name}
                        </option>
                    ))}
                </select>
            </div>

            {/* Swap button */}
            <button
                onClick={handleSwap}
                className="mt-5 p-2 rounded-lg border transition-all border-border text-text-secondary hover:bg-primary/10 hover:text-primary hover:border-primary/30"
                title="Swap languages"
            >
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M7 16V4m0 0L3 8m4-4l4 4" />
                    <path d="M17 8v12m0 0l4-4m-4 4l-4-4" />
                </svg>
            </button>

            {/* Target language */}
            <div className="flex-1">
                <label className="label mb-1.5 block text-xs">To</label>
                <select
                    value={targetLanguage}
                    onChange={(e) => onTargetChange(e.target.value)}
                    className="w-full px-3 py-2.5 rounded-xl border border-border bg-white/[0.03] text-text-primary text-sm focus:outline-none focus:ring-1 focus:ring-primary/40 appearance-none cursor-pointer"
                >
                    {LANGUAGES.map((l) => (
                        <option key={l.code} value={l.code}>
                            {l.flag} {l.name}
                        </option>
                    ))}
                </select>
            </div>
        </div>
    );
}

export { LANGUAGES };
