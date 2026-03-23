'use client';

import { useState, useCallback, useRef, useEffect } from 'react';
import { extractYouTubeId, isValidYouTubeUrl, getThumbnailUrl } from '@/lib/utils';
import { addLink } from '@/lib/api';

type InputMode = 'url' | 'upload' | 'batch';

interface URLInputProps {
    onSubmit: (url: string) => void;
    onFileSubmit: (file: File) => void;
    onBatchSubmit?: (urls: string[]) => void;
    disabled?: boolean;
    url?: string;
    onUrlChange?: (url: string) => void;
}

export default function URLInput({ onSubmit, onFileSubmit, onBatchSubmit, disabled, url: controlledUrl, onUrlChange }: URLInputProps) {
    const [mode, setMode] = useState<InputMode>('url');
    const [internalUrl, setInternalUrl] = useState('');

    const url = controlledUrl ?? internalUrl;
    const setUrl = (val: string) => {
        setInternalUrl(val);
        onUrlChange?.(val);
    };
    const [file, setFile] = useState<File | null>(null);
    const [dragOver, setDragOver] = useState(false);
    const [batchText, setBatchText] = useState('');
    const fileInputRef = useRef<HTMLInputElement>(null);

    const videoId = extractYouTubeId(url);
    const isValid = isValidYouTubeUrl(url);

    // Auto-save valid URL to saved links
    const savedRef = useRef<Set<string>>(new Set());
    useEffect(() => {
        if (isValid && url && !savedRef.current.has(url)) {
            savedRef.current.add(url);
            addLink(url).catch(() => {});
        }
    }, [url, isValid]);

    // Batch URL parsing
    const parsedLines = batchText.split('\n').map(l => l.trim()).filter(Boolean);
    const validUrls = parsedLines.filter(isValidYouTubeUrl);
    const invalidCount = parsedLines.length - validUrls.length;

    // Auto-save valid batch URLs
    useEffect(() => {
        validUrls.forEach(u => {
            if (!savedRef.current.has(u)) {
                savedRef.current.add(u);
                addLink(u).catch(() => {});
            }
        });
    }, [batchText]); // eslint-disable-line react-hooks/exhaustive-deps

    const handleUrlSubmit = useCallback(() => {
        if (isValid && !disabled) {
            onSubmit(url.trim());
        }
    }, [url, isValid, disabled, onSubmit]);

    const handleFileSubmit = useCallback(() => {
        if (file && !disabled) {
            onFileSubmit(file);
        }
    }, [file, disabled, onFileSubmit]);

    const handlePaste = useCallback((e: React.ClipboardEvent) => {
        const pasted = e.clipboardData.getData('text').trim();
        if (isValidYouTubeUrl(pasted)) {
            e.preventDefault();
            setUrl(pasted);
        }
    }, []);

    const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
        const f = e.target.files?.[0];
        if (f) setFile(f);
    }, []);

    const handleDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setDragOver(false);
        const f = e.dataTransfer.files?.[0];
        if (f && f.type.startsWith('video/')) {
            setFile(f);
        }
    }, []);

    const formatFileSize = (bytes: number) => {
        if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
        if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
        return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
    };

    return (
        <div className="space-y-4">
            {/* Tab Switcher */}
            <div className="flex gap-1 p-1 rounded-xl bg-card/50 border border-border">
                <button
                    onClick={() => setMode('url')}
                    className={`flex-1 py-2.5 px-4 rounded-lg text-sm font-medium transition-all flex items-center justify-center gap-2 ${
                        mode === 'url'
                            ? 'bg-primary text-white shadow-sm'
                            : 'text-text-secondary hover:text-text-primary'
                    }`}
                >
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M2.5 17a24.12 24.12 0 0 1 0-10 2 2 0 0 1 1.4-1.4 49.56 49.56 0 0 1 16.2 0A2 2 0 0 1 21.5 7a24.12 24.12 0 0 1 0 10 2 2 0 0 1-1.4 1.4 49.55 49.55 0 0 1-16.2 0A2 2 0 0 1 2.5 17" />
                        <path d="m10 15 5-3-5-3z" />
                    </svg>
                    YouTube URL
                </button>
                <button
                    onClick={() => setMode('upload')}
                    className={`flex-1 py-2.5 px-4 rounded-lg text-sm font-medium transition-all flex items-center justify-center gap-2 ${
                        mode === 'upload'
                            ? 'bg-primary text-white shadow-sm'
                            : 'text-text-secondary hover:text-text-primary'
                    }`}
                >
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                        <polyline points="17 8 12 3 7 8" />
                        <line x1="12" y1="3" x2="12" y2="15" />
                    </svg>
                    Upload Video
                </button>
                <button
                    onClick={() => setMode('batch')}
                    className={`flex-1 py-2.5 px-4 rounded-lg text-sm font-medium transition-all flex items-center justify-center gap-2 ${
                        mode === 'batch'
                            ? 'bg-primary text-white shadow-sm'
                            : 'text-text-secondary hover:text-text-primary'
                    }`}
                >
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M16 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V8Z" />
                        <path d="M15 3v4a2 2 0 0 0 2 2h4" />
                        <path d="M10 13l-2 2 2 2" /><path d="M14 17l2-2-2-2" />
                    </svg>
                    Batch URLs
                </button>
            </div>

            {/* YouTube URL Mode */}
            {mode === 'url' && (
                <>
                    <div className="relative">
                        <div className="absolute left-4 top-1/2 -translate-y-1/2 text-text-muted">
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <path d="M2.5 17a24.12 24.12 0 0 1 0-10 2 2 0 0 1 1.4-1.4 49.56 49.56 0 0 1 16.2 0A2 2 0 0 1 21.5 7a24.12 24.12 0 0 1 0 10 2 2 0 0 1-1.4 1.4 49.55 49.55 0 0 1-16.2 0A2 2 0 0 1 2.5 17" />
                                <path d="m10 15 5-3-5-3z" />
                            </svg>
                        </div>
                        <input
                            type="text"
                            value={url}
                            onChange={(e) => setUrl(e.target.value)}
                            onPaste={handlePaste}
                            onKeyDown={(e) => e.key === 'Enter' && handleUrlSubmit()}
                            placeholder="Paste YouTube URL here..."
                            className="input-field pl-12 pr-4 py-4 text-base"
                            disabled={disabled}
                        />
                        {url && !isValid && (
                            <div className="absolute right-4 top-1/2 -translate-y-1/2">
                                <span className="text-xs text-error">Invalid URL</span>
                            </div>
                        )}
                        {isValid && (
                            <div className="absolute right-4 top-1/2 -translate-y-1/2">
                                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#22c55e" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                                    <path d="M20 6 9 17l-5-5" />
                                </svg>
                            </div>
                        )}
                    </div>

                    {/* Thumbnail Preview */}
                    {videoId && (
                        <div className="animate-slide-up glass-card p-3 flex items-center gap-4">
                            <img
                                src={getThumbnailUrl(videoId)}
                                alt="Video thumbnail"
                                className="w-32 h-20 object-cover rounded-lg"
                            />
                            <div className="flex-1 min-w-0">
                                <p className="text-sm text-text-secondary mb-1">Ready to dub</p>
                                <p className="text-xs text-text-muted truncate">{url}</p>
                            </div>
                        </div>
                    )}

                    <button
                        onClick={handleUrlSubmit}
                        disabled={!isValid || disabled}
                        className="btn-primary w-full py-4 text-base font-semibold flex items-center justify-center gap-2"
                    >
                        {disabled ? (
                            <>
                                <svg className="animate-spin" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                    <path d="M21 12a9 9 0 1 1-6.219-8.56" />
                                </svg>
                                Processing...
                            </>
                        ) : (
                            <>
                                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                    <path d="m5 8 6 4-6 4V8Z" />
                                    <path d="m13 8 6 4-6 4V8Z" />
                                </svg>
                                Start Dubbing
                            </>
                        )}
                    </button>
                </>
            )}

            {/* Batch Mode */}
            {mode === 'batch' && (
                <>
                    <textarea
                        value={batchText}
                        onChange={(e) => setBatchText(e.target.value)}
                        placeholder={"Paste YouTube URLs here (one per line)...\n\nhttps://youtube.com/watch?v=abc123\nhttps://youtu.be/def456\nhttps://youtube.com/watch?v=ghi789"}
                        className="input-field w-full h-40 p-4 text-sm resize-none"
                        disabled={disabled}
                    />

                    {/* URL count info */}
                    {parsedLines.length > 0 && (
                        <div className="flex items-center gap-3 text-sm">
                            <span className="text-green-400 font-medium">
                                {validUrls.length} valid URL{validUrls.length !== 1 ? 's' : ''}
                            </span>
                            {invalidCount > 0 && (
                                <span className="text-error">
                                    {invalidCount} invalid
                                </span>
                            )}
                        </div>
                    )}

                    {/* Thumbnail previews (first 3) */}
                    {validUrls.length > 0 && (
                        <div className="flex gap-2 overflow-hidden">
                            {validUrls.slice(0, 3).map((u) => {
                                const vid = extractYouTubeId(u);
                                return vid ? (
                                    <div key={vid} className="animate-slide-up glass-card p-2 flex items-center gap-3 flex-1 min-w-0">
                                        <img
                                            src={getThumbnailUrl(vid)}
                                            alt="Thumbnail"
                                            className="w-20 h-12 object-cover rounded-md"
                                        />
                                        <p className="text-xs text-text-muted truncate flex-1">{u}</p>
                                    </div>
                                ) : null;
                            })}
                            {validUrls.length > 3 && (
                                <div className="glass-card p-2 flex items-center justify-center min-w-[80px]">
                                    <span className="text-sm text-text-secondary">+{validUrls.length - 3} more</span>
                                </div>
                            )}
                        </div>
                    )}

                    <button
                        onClick={() => onBatchSubmit?.(validUrls)}
                        disabled={validUrls.length === 0 || disabled}
                        className="btn-primary w-full py-4 text-base font-semibold flex items-center justify-center gap-2"
                    >
                        {disabled ? (
                            <>
                                <svg className="animate-spin" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                    <path d="M21 12a9 9 0 1 1-6.219-8.56" />
                                </svg>
                                Processing...
                            </>
                        ) : (
                            <>
                                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                    <path d="m5 8 6 4-6 4V8Z" />
                                    <path d="m13 8 6 4-6 4V8Z" />
                                </svg>
                                Start Batch Dubbing ({validUrls.length} video{validUrls.length !== 1 ? 's' : ''})
                            </>
                        )}
                    </button>
                </>
            )}

            {/* Upload Mode */}
            {mode === 'upload' && (
                <>
                    <input
                        ref={fileInputRef}
                        type="file"
                        accept="video/*"
                        onChange={handleFileChange}
                        className="hidden"
                    />

                    {/* Drop Zone */}
                    <div
                        onClick={() => fileInputRef.current?.click()}
                        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                        onDragLeave={() => setDragOver(false)}
                        onDrop={handleDrop}
                        className={`cursor-pointer border-2 border-dashed rounded-xl p-8 text-center transition-all ${
                            dragOver
                                ? 'border-primary bg-primary/10'
                                : file
                                    ? 'border-green-500/50 bg-green-500/5'
                                    : 'border-border hover:border-primary/50 hover:bg-primary/5'
                        }`}
                    >
                        {file ? (
                            <div className="space-y-2">
                                <svg className="mx-auto text-green-400" width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                                    <path d="m15 2 5 5v11a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9Z" />
                                    <path d="m14 2v4a2 2 0 0 0 2 2h4" />
                                    <path d="m9 15 2 2 4-4" />
                                </svg>
                                <p className="text-sm text-text-primary font-medium">{file.name}</p>
                                <p className="text-xs text-text-muted">{formatFileSize(file.size)}</p>
                                <p className="text-xs text-text-secondary">Click to choose a different file</p>
                            </div>
                        ) : (
                            <div className="space-y-2">
                                <svg className="mx-auto text-text-muted" width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                                    <polyline points="17 8 12 3 7 8" />
                                    <line x1="12" y1="3" x2="12" y2="15" />
                                </svg>
                                <p className="text-sm text-text-primary font-medium">
                                    Drop a video file here or click to browse
                                </p>
                                <p className="text-xs text-text-muted">
                                    MP4, MKV, WebM, AVI — any video format
                                </p>
                            </div>
                        )}
                    </div>

                    <button
                        onClick={handleFileSubmit}
                        disabled={!file || disabled}
                        className="btn-primary w-full py-4 text-base font-semibold flex items-center justify-center gap-2"
                    >
                        {disabled ? (
                            <>
                                <svg className="animate-spin" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                    <path d="M21 12a9 9 0 1 1-6.219-8.56" />
                                </svg>
                                Uploading...
                            </>
                        ) : (
                            <>
                                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                                    <polyline points="17 8 12 3 7 8" />
                                    <line x1="12" y1="3" x2="12" y2="15" />
                                </svg>
                                Upload &amp; Start Dubbing
                            </>
                        )}
                    </button>
                </>
            )}
        </div>
    );
}
