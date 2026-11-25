import './globals.css';
import type { Metadata } from 'next';
import Sidebar from '@/components/Sidebar';

export const metadata: Metadata = {
    title: 'VoiceDub',   // desktop.py FRONTEND_TITLE identifies our server by this; keep them equal
    description: 'Dub your videos into other languages with AI translation and voices.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
    return (
        <html lang="en">
            <body className="bg-background text-text-primary">
                <div className="flex min-h-screen">
                    <Sidebar />
                    <main className="flex-1 min-w-0 pt-14 md:pt-0 md:ml-64">
                        {children}
                    </main>
                </div>
            </body>
        </html>
    );
}
