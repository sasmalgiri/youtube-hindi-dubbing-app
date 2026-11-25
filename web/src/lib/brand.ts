/** Product name shown in the UI. Override with NEXT_PUBLIC_BRAND (e.g. once a final name is chosen). */
export const BRAND = process.env.NEXT_PUBLIC_BRAND || 'VoiceDub';
export const BRAND_SPLIT = BRAND.endsWith('Dub') ? [BRAND.slice(0, -3), 'Dub'] : [BRAND, ''];
