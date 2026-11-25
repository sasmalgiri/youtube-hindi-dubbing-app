/** @type {import('next').NextConfig} */
// desktop.py always runs the backend on 8000 (it stops an older VoiceDub
// backend there first, and refuses to start next to anything else) and still
// passes BACKEND_PORT explicitly, so this proxy and the backend can never
// disagree about the port.
const BACKEND_PORT = process.env.BACKEND_PORT || '8000';

import path from 'node:path';
import { fileURLToPath } from 'node:url';

const nextConfig = {
    reactStrictMode: true,
    experimental: {
        // The /api proxy cuts request bodies at 10 MB by default (Next 15). Uploads over
        // 50 MB go in 50 MB pieces (lib/api.ts uploadChunked), so 64 MB covers every request.
        middlewareClientMaxBodySize: '64mb',
    },
    // a stray lockfile in the user folder must not become the tracing root
    outputFileTracingRoot: path.dirname(fileURLToPath(import.meta.url)),
    async rewrites() {
        return [
            {
                source: '/api/:path*',
                destination: `http://localhost:${BACKEND_PORT}/api/:path*`,
            },
        ];
    },
};

export default nextConfig;
