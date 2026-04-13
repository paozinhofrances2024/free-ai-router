// src/selection/task-classifier.ts — LLM-based task classification with heuristic fallback
import { detectTask as heuristicDetect, type TaskType } from './task-router.js';

const CLASSIFIER_PROMPT = `Classify this user prompt into exactly one category. Reply with ONLY the category word, nothing else.

Categories:
- coding (programming, debugging, writing code)
- reasoning (analysis, math, logic, explanation)
- creative (writing, stories, poems, brainstorming)
- fast (simple questions, quick lookups, yes/no)
- vision (image analysis, visual content)
- general (anything else)

Prompt: `;

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
    prompt: string,
    apiUrl: string,
    apiKey: string,
    model: string,
): Promise<{ task: TaskType; method: 'llm' | 'heuristic'; confidence: number }> {
    // Check cache
    const cacheKey = prompt.slice(0, 200);
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
                messages: [{ role: 'user', content: CLASSIFIER_PROMPT + prompt.slice(0, 500) }],
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
 * Default classifier config for free tier providers.
 */
export const CLASSIFIER_PRESETS = {
    groq: {
        apiUrl: 'https://api.groq.com/openai/v1/chat/completions',
        envVar: 'GROQ_API_KEY',
        model: 'gemma-3-1b-it', // Free, fast (~50ms)
    },
    cerebras: {
        apiUrl: 'https://api.cerebras.ai/v1/chat/completions',
        envVar: 'CEREBRAS_API_KEY',
        model: 'llama3.1-8b',
    },
    googleai: {
        apiUrl: 'https://generativelanguage.googleapis.com/v1beta/openai/chat/completions',
        envVar: 'GOOGLE_API_KEY',
        model: 'gemma-3-1b-it', // Free, 14k req/day
    },
};
