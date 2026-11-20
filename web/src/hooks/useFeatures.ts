'use client';

import { useEffect, useState } from 'react';
import { getFeatures, type FeatureMatrix } from '@/lib/api';

// One shared copy for every component on the page, refreshed by refreshFeatures().
let cache: FeatureMatrix | null = null;
let inflight: Promise<FeatureMatrix> | null = null;
const listeners = new Set<(m: FeatureMatrix | null) => void>();

export function refreshFeatures(): Promise<FeatureMatrix> {
    inflight = getFeatures()
        .then((m) => { cache = m; listeners.forEach((l) => l(m)); return m; })
        .finally(() => { inflight = null; });
    return inflight;
}

/**
 * isOn(id): false only when the backend says the module is switched off.
 * Unknown ids and the loading state count as on, so an option is never hidden
 * by accident; the backend still refuses a switched-off module (HTTP 400/403).
 */
export function useFeatures() {
    const [matrix, setMatrix] = useState<FeatureMatrix | null>(cache);
    useEffect(() => {
        listeners.add(setMatrix);
        if (!cache && !inflight) refreshFeatures().catch(() => { /* backend offline: show everything */ });
        return () => { listeners.delete(setMatrix); };
    }, []);
    const byId = new Map((matrix?.modules || []).map((m) => [m.id, m]));
    const isOn = (id: string) => byId.get(id)?.on !== false;
    return { matrix, isOn, profile: matrix?.profile || 'local' };
}
