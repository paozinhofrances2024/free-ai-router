// src/selection/task-router.ts — Route models based on task type
import type { ModelDef, ProviderDef } from '../providers/types.js';

/** Supported task types for intelligent model routing */
export type TaskType = 'coding' | 'reasoning' | 'creative' | 'fast' | 'vision' | 'general';

/** Task profile: what makes a model good for a task */
interface TaskProfile {
    /** Preferred model families (in order) */
    preferredModels: string[];
    /** Minimum tier required */
    minTier: string;
    /** Required capabilities */
    requiredCaps: {
        tools?: boolean;
        vision?: boolean;
        streaming?: boolean;
    };
    /** Preferred context window (K tokens) */
    minContextK: number;
    /** Prefer speed over quality? */
    preferSpeed: boolean;
}

/** Task profiles defining what each task type needs */
const TASK_PROFILES: Record<TaskType, TaskProfile> = {
    coding: {
        preferredModels: [
            'qwen3-coder', 'coding-minimax', 'kimi-for-coding', 'codestral',
            'deepseek-v3', 'deepseek-r1', 'glm-4', 'glm-5',
            'devstral', 'qwen3', 'granite',
        ],
        minTier: 'A',
        requiredCaps: { tools: true },
        minContextK: 32,
        preferSpeed: false,
    },
    reasoning: {
        preferredModels: [
            'gemini-3-pro', 'gemini-3.1-pro', 'gemini-2.5-pro',
            'deepseek-r1', 'qwq', 'kimi-k2-thinking',
            'minimax-m2.7', 'minimax-m2.5', 'glm-5', 'glm-4.7',
        ],
        minTier: 'S',
        requiredCaps: {},
        minContextK: 128,
        preferSpeed: false,
    },
    creative: {
        preferredModels: [
            'gpt-4.1', 'gpt-4o', 'gemini-2.5-flash', 'gemini-3-flash',
            'claude', 'minimax', 'glm',
        ],
        minTier: 'A',
        requiredCaps: {},
        minContextK: 64,
        preferSpeed: false,
    },
    fast: {
        preferredModels: [
            'gemini-2.5-flash', 'gemini-2.0-flash', 'gpt-4.1-nano',
            'llama-3.1-8b', 'gemma-3-27b', 'step-3.5-flash',
        ],
        minTier: 'B',
        requiredCaps: {},
        minContextK: 8,
        preferSpeed: true,
    },
    vision: {
        preferredModels: [
            'gemini-3-pro', 'gemini-2.5-flash', 'gpt-4o',
            'llama-4-scout', 'llama-4-maverick',
        ],
        minTier: 'A',
        requiredCaps: { vision: true },
        minContextK: 64,
        preferSpeed: false,
    },
    general: {
        preferredModels: [],
        minTier: 'B',
        requiredCaps: {},
        minContextK: 16,
        preferSpeed: false,
    },
};

/** Scored model candidate */
interface ScoredCandidate {
    model: ModelDef;
    provider: ProviderDef;
    score: number;
    reason: string;
}

/**
 * Score and rank models for a specific task type.
 * Returns candidates sorted by task relevance (best first).
 */
export function scoreForTask(
    candidates: Array<{ model: ModelDef; provider: ProviderDef; health: any }>,
    task: TaskType,
): ScoredCandidate[] {
    const profile = TASK_PROFILES[task] || TASK_PROFILES.general;
    const TIER_RANK: Record<string, number> = {
        'S+': 80, 'S': 70, 'A+': 60, 'A': 50, 'A-': 45, 'B+': 35, 'B': 25, 'C': 10,
    };

    const scored: ScoredCandidate[] = [];

    for (const { model, provider, health } of candidates) {
        if (!health.available) continue;

        // Filter by minimum tier
        const modelTierRank = TIER_RANK[model.tier] ?? 0;
        const minTierRank = TIER_RANK[profile.minTier] ?? 0;
        if (modelTierRank < minTierRank) continue;

        // Filter by required capabilities
        if (profile.requiredCaps.tools && !model.supportsTools) continue;
        if (profile.requiredCaps.vision && !model.supportsVision) continue;

        // Filter by context window
        if (model.contextK < profile.minContextK) continue;

        let score = 0;
        const reasons: string[] = [];

        // 1. Model family match (0-50 points)
        const modelIdLower = model.modelId.toLowerCase();
        let familyMatch = 0;
        for (const preferred of profile.preferredModels) {
            if (modelIdLower.includes(preferred.toLowerCase())) {
                familyMatch = 50;
                reasons.push(`task match: ${preferred}`);
                break;
            }
        }
        score += familyMatch;

        // 2. Tier score (0-30 points)
        const tierScore = Math.min(30, modelTierRank / 3);
        score += tierScore;

        // 3. Health score (0-20 points)
        const healthScore = Math.min(20, (health.score || 50) / 5);
        score += healthScore;

        // 4. Speed bonus (prefer lower latency if preferSpeed)
        if (profile.preferSpeed && health.latency) {
            const avgLatency = health.latency.avg || 5000;
            if (avgLatency < 1000) score += 15;
            else if (avgLatency < 3000) score += 8;
        }

        scored.push({
            model,
            provider,
            score,
            reason: reasons.join(', ') || `tier ${model.tier}`,
        });
    }

    // Sort by score descending
    scored.sort((a, b) => b.score - a.score);
    return scored;
}

/**
 * Get the task profile for a given task type.
 */
export function getTaskProfile(task: TaskType): TaskProfile {
    return TASK_PROFILES[task] || TASK_PROFILES.general;
}

/**
 * Detect task type from a prompt heuristically.
 */
export function detectTask(prompt: string): TaskType {
    const lower = prompt.toLowerCase();

    // Coding indicators
    if (/\b(code|function|class|method|api|debug|fix bug|implement|refactor|typescript|python|javascript|rust|sql|regex|algorithm)\b/.test(lower)) {
        return 'coding';
    }

    // Reasoning indicators
    if (/\b(analyze|reason|logic|proof|math|calculate|explain why|compare|evaluate|deduce|solve)\b/.test(lower)) {
        return 'reasoning';
    }

    // Vision indicators
    if (/\b(image|photo|screenshot|picture|diagram|chart|visual|see|look at)\b/.test(lower)) {
        return 'vision';
    }

    // Fast/short indicators
    if (prompt.length < 50 && /\b(what is|who is|when|where|yes or no|true or false|\d+\s*[\+\-\*\/]\s*\d+)\b/.test(lower)) {
        return 'fast';
    }

    // Creative indicators
    if (/\b(write|story|poem|creative|imagine|draft|compose|brainstorm|idea)\b/.test(lower)) {
        return 'creative';
    }

    return 'general';
}
