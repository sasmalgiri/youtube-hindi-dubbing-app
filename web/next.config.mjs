/** @type {import('next').NextConfig} */
// desktop.py always runs the backend on 8000 (it stops an older VoiceDub
// backend there first, and refuses to start next to anything else) and still
// passes BACKEND_PORT explicitly, so this proxy and the backend can never
// disagree about the port.
const BACKEND_PORT = process.env.BACKEND_PORT || '8000';

const nextConfig = {
    reactStrictMode: true,
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
