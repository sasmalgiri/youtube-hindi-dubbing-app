import { NextRequest, NextResponse } from 'next/server';
import { execFile } from 'child_process';
import { promisify } from 'util';
import { readFile, mkdtemp, rm } from 'fs/promises';
import { tmpdir } from 'os';
import path from 'path';

const execFileAsync = promisify(execFile);

const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export async function POST(req: NextRequest) {
    // Module on/off matrix (S02): this route downloads on the server and buffers the
    // whole file, so it exists only while surface.local_dub is on (never in hosted).
    try {
        const f = await fetch(`${BACKEND_URL}/api/features`, { cache: 'no-store' });
        const m = f.ok ? await f.json() : null;
        const mod = m?.modules?.find((x: { id: string }) => x.id === 'surface.local_dub');
        if (!m || m.profile === 'hosted' || !mod?.on) {
            return NextResponse.json({ detail: 'Not Found' }, { status: 404 });
        }
    } catch {
        return NextResponse.json({ detail: 'Not Found' }, { status: 404 });
    }
    let tempDir = '';
    let videoPath = '';

    try {
        const body = await req.json();
        const { url, ...settings } = body;

        if (!url) {
            return NextResponse.json({ detail: 'URL is required' }, { status: 400 });
        }

        // Create temp directory for download
        tempDir = await mkdtemp(path.join(tmpdir(), 'hindidub-'));
        videoPath = path.join(tempDir, 'video.mp4');

        // Download video locally using yt-dlp
        const ytdlpPath = 'yt-dlp';
        const args = [
            '-f', 'best[height<=720]/best',
            '--no-playlist',
            '-o', videoPath,
            '--merge-output-format', 'mp4',
            url,
        ];

        try {
            await execFileAsync(ytdlpPath, args, {
                timeout: 300000, // 5 min timeout
                maxBuffer: 10 * 1024 * 1024,
            });
        } catch (dlErr: any) {
            const msg = dlErr.stderr || dlErr.message || 'Download failed';
            return NextResponse.json(
                { detail: `Download failed: ${msg.slice(0, 500)}` },
                { status: 500 },
            );
        }

        // Read the downloaded file
        const fileBuffer = await readFile(videoPath);

        // Upload to backend
        const form = new FormData();
        form.append('file', new Blob([fileBuffer], { type: 'video/mp4' }), 'video.mp4');

        // Append all settings
        for (const [key, val] of Object.entries(settings)) {
            if (val !== undefined && val !== null) {
                form.append(key, String(val));
            }
        }

        const uploadRes = await fetch(`${BACKEND_URL}/api/jobs/upload`, {
            method: 'POST',
            body: form,
        });

        if (!uploadRes.ok) {
            const err = await uploadRes.json().catch(() => ({ detail: 'Upload to backend failed' }));
            return NextResponse.json(
                { detail: err.detail || 'Upload to backend failed' },
                { status: 500 },
            );
        }

        const result = await uploadRes.json();
        return NextResponse.json(result);

    } catch (err: any) {
        return NextResponse.json(
            { detail: err.message || 'Internal error' },
            { status: 500 },
        );
    } finally {
        // Cleanup temp file
        try {
            if (tempDir) await rm(tempDir, { recursive: true, force: true }).catch(() => {});
        } catch { /* ignore cleanup errors */ }
    }
}
