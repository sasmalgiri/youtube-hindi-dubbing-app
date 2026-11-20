'use client';

import { useEffect, useState } from 'react';
import { getAuthConfig, getMe, type AuthConfig, type AuthUser } from '@/lib/api';

// One shared copy for the page; refreshAuth() after sign-in / sign-out.
let cache: { user: AuthUser | null; config: AuthConfig | null; loaded: boolean } = { user: null, config: null, loaded: false };
const listeners = new Set<(c: typeof cache) => void>();
let inflight: Promise<void> | null = null;

export function refreshAuth(): Promise<void> {
    inflight = Promise.all([getMe().catch(() => ({ user: null })), getAuthConfig().catch(() => null)])
        .then(([me, config]) => {
            cache = { user: me.user, config, loaded: true };
            listeners.forEach((l) => l(cache));
        })
        .finally(() => { inflight = null; });
    return inflight;
}

export function useAuth() {
    const [state, setState] = useState(cache);
    useEffect(() => {
        listeners.add(setState);
        if (!cache.loaded && !inflight) refreshAuth();
        return () => { listeners.delete(setState); };
    }, []);
    return state;
}
