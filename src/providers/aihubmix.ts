// src/providers/aihubmix.ts — AIHubMix provider definition
import type { ProviderDef } from './types.js';

export const AIHUBMIX: ProviderDef = {
    id: 'aihubmix',
    name: 'AIHubMix',
    baseUrl: 'https://aihubmix.com/v1/chat/completions',
    apiKeyEnvVars: ['AIHUBMIX_API_KEY'],
    rateLimits: {
        rpm: 3,
        rpd: 100,
    },
};
