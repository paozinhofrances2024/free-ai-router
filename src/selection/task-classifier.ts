// src/selection/task-classifier.ts — LLM-based task classification with heuristic fallback
import { detectTask as heuristicDetect, type TaskType } from './task-router.js';

const CLASSIFIER_PROMPT = 'Classify this prompt. Reply ONLY with one of: coding reasoning creative fast general\nPrompt: ';

/** Normalize with heuristic fallback for bad LLM responses */
function classifyWithHeuristicFallback(prompt: string, raw: string): TaskType {
    const fromLLM = normalize(raw);
    if (fromLLM !== 'general') return fromLLM; // LLM gave a valid category
    // LLM gave garbage — use heuristic
    return detectTask(prompt);
}

/** Cache for classification results (LRU, 100 entries) */
const classCache = new Map<string, { task: TaskType; ts: number }>();
const CACHE_MAX = 100;
const CACHE_TTL = 5 * 60 * 1000; // 5 min

/** Validate and normalize a classification response */
function normalize(raw: string): TaskType {
    const clean = raw.trim().toLowerCase().replace(/[^a-z]/g, '');
    const valid: TaskType[] = ['coding', 'reasoning', 'creative', 'fast', 'vision', 'general'];
    const match = valid.find(v => clean.includes(v));
    return match || 'general';
}

/**
 * Classify a prompt using a remote LLM API (fast/small model).
 * Falls back to heuristic if API fails.
 */
export async function classifyWithLLM(
    messages: Array<{ role: string; content: string }>,
    apiUrl: string,
    apiKey: string,
    model: string,
): Promise<{ task: TaskType; method: 'llm' | 'heuristic'; confidence: number }> {
    // Build context from conversation history
    const contextParts = messages.slice(-10).map(m => `${m.role}: ${m.content}`) // last 10 messages
    const contextStr = contextParts.join('\n').slice(0, 800)
    const lastMsg = messages[messages.length - 1]?.content || ''

    // Check cache based on last message
    const cacheKey = lastMsg.slice(0, 200);
    const cached = classCache.get(cacheKey);
    if (cached && Date.now() - cached.ts < CACHE_TTL) {
        return { task: cached.task, method: 'llm', confidence: 0.95 };
    }

    try {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 3000); // 3s max

        const resp = await fetch(apiUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${apiKey}`,
            },
            body: JSON.stringify({
                model,
                messages: [{ role: 'user', content: CLASSIFIER_PROMPT + contextStr }],
                max_tokens: 10,
                temperature: 0,
            }),
            signal: controller.signal,
        });

        clearTimeout(timeout);

        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

        const data = await resp.json() as any;
        const content = data?.choices?.[0]?.message?.content || '';

        if (!content.trim()) throw new Error('Empty response');

        const task = normalize(content);

        // Cache result
        classCache.set(cacheKey, { task, ts: Date.now() });
        if (classCache.size > CACHE_MAX) {
            const oldest = [...classCache.entries()].sort((a, b) => a[1].ts - b[1].ts)[0];
            if (oldest) classCache.delete(oldest[0]);
        }

        return { task, method: 'llm', confidence: 0.85 };
    } catch {
        // Fallback to heuristic
        return { task: heuristicDetect(prompt), method: 'heuristic', confidence: 0.5 };
    }
}

/**
 * Quick heuristic-only classification (no API call).
 */
export function classifyHeuristic(prompt: string): { task: TaskType; method: 'heuristic'; confidence: number } {
    return { task: heuristicDetect(prompt), method: 'heuristic', confidence: 0.6 };
}

/**
 * Try multiple classifiers in sequence: free small models first, Groq fallback.
 * Returns first successful result.
 */
export async function classifyWithCascade(
    messages: Array<{ role: string; content: string }>,
    keys: Record<string, string>,
): Promise<{ task: TaskType; method: string; confidence: number }> {
    // Chain: Google Gemma 1B → Google Gemma 4B → OpenRouter → Groq (fallback)
    const chain: Array<{ preset: typeof CLASSIFIER_PRESETS[keyof typeof CLASSIFIER_PRESETS]; key: string; name: string }> = []

    if (keys.GOOGLE_API_KEY) {
        chain.push({ preset: CLASSIFIER_PRESETS.googleai_gemma1b, key: keys.GOOGLE_API_KEY, name: 'gemma-1b' })
        chain.push({ preset: CLASSIFIER_PRESETS.googleai_gemma4b, key: keys.GOOGLE_API_KEY, name: 'gemma-4b' })
    }
    if (keys.OPENROUTER_API_KEY) {
        chain.push({ preset: CLASSIFIER_PRESETS.openrouter, key: keys.OPENROUTER_API_KEY, name: 'openrouter-gemma' })
    }
    if (keys.GROQ_API_KEY) {
        chain.push({ preset: CLASSIFIER_PRESETS.groq, key: keys.GROQ_API_KEY, name: 'groq-llama8b' })
    }

    for (const { preset, key, name } of chain) {
        const result = await classifyWithLLM(messages, preset.apiUrl, key, preset.model)
        if (result.method === 'llm' && result.confidence >= 0.7) {
            return { ...result, method: name }
        }
        // If LLM returned garbage, try next in chain
    }

    // All failed — heuristic
    const lastMsg = messages[messages.length - 1]?.content || ''
    return { task: heuristicDetect(lastMsg), method: 'heuristic', confidence: 0.5 }
}

/**
 * Default classifier config for free tier providers.
 */
export const CLASSIFIER_PRESETS = {
    /** Primary classifiers (free, small models) */
    googleai_gemma1b: {
        apiUrl: 'https://generativelanguage.googleapis.com/v1beta/openai/chat/completions',
        envVar: 'GOOGLE_API_KEY',
        model: 'gemma-3-1b-it', // 14.4k req/day free
    },
    googleai_gemma4b: {
        apiUrl: 'https://generativelanguage.googleapis.com/v1beta/openai/chat/completions',
        envVar: 'GOOGLE_API_KEY',
        model: 'gemma-3-4b-it', // 14.4k req/day free
    },
    openrouter: {
        apiUrl: 'https://openrouter.ai/api/v1/chat/completions',
        envVar: 'OPENROUTER_API_KEY',
        model: 'google/gemma-3-27b-it:free', // free on OpenRouter
    },
    /** Fallback (paid, reliable, never stops) */
    groq: {
        apiUrl: 'https://api.groq.com/openai/v1/chat/completions',
        envVar: 'GROQ_API_KEY',
        model: 'llama-3.1-8b-instant', // billing enabled, ~26ms
    },
};
