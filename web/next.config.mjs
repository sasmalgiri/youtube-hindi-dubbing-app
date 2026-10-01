/** @type {import('next').NextConfig} */
// desktop.py starts the backend on the first free port from 8000 and passes it
// as BACKEND_PORT; proxying to a hard-coded 8000 sent the UI to the wrong (or
// no) backend whenever 8000 was busy.
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
