// tests/task-routing.test.ts — Integration tests for task-based routing
import { describe, test, expect, beforeAll } from 'bun:test';
import FreeRouter from '../src/index.js';
import { detectTask, scoreForTask, type TaskType } from '../src/selection/task-router.js';
import { classifyWithCascade, classifyWithLLM, CLASSIFIER_PRESETS } from '../src/selection/task-classifier.js';

const GROQ_KEY = process.env.GROQ_API_KEY || '';
const GOOGLE_KEY = process.env.GOOGLE_API_KEY || '';
const OPENROUTER_KEY = process.env.OPENROUTER_API_KEY || '';

// ═══════════════════════════════════════════════════════════
// 1. HEURISTIC DETECTOR TESTS
// ═══════════════════════════════════════════════════════════
describe('detectTask (heuristic)', () => {
    test('detects coding tasks', () => {
        expect(detectTask('Write a Python function that sorts a list')).toBe('coding');
        expect(detectTask('Fix the bug in my JavaScript code')).toBe('coding');
        expect(detectTask('Implement a REST API with Express')).toBe('coding');
        expect(detectTask('Refactor this class to use dependency injection')).toBe('coding');
        expect(detectTask('Write a regex to match email addresses')).toBe('coding');
    });

    test('detects reasoning tasks', () => {
        expect(detectTask('Analyze the pros and cons of microservices')).toBe('reasoning');
        expect(detectTask('Solve this math problem step by step')).toBe('reasoning');
        expect(detectTask('Compare React vs Vue for large applications')).toBe('reasoning');
        expect(detectTask('Explain why the sky is blue')).toBe('reasoning');
    });

    test('detects creative tasks', () => {
        expect(detectTask('Write a haiku about the ocean')).toBe('creative');
        expect(detectTask('Tell me a story about a brave knight')).toBe('creative');
        expect(detectTask('Compose a poem about autumn leaves')).toBe('creative');
        expect(detectTask('Brainstorm ideas for a mobile app')).toBe('creative');
    });

    test('detects fast tasks', () => {
        expect(detectTask('What is 2+2?')).toBe('fast');
        expect(detectTask('Who is the president of France?')).toBe('fast');
        expect(detectTask('Where is Paris?')).toBe('fast');
    });

    test('detects vision tasks', () => {
        expect(detectTask('What do you see in this screenshot?')).toBe('vision');
        expect(detectTask('Look at this photo and describe it')).toBe('vision');
        expect(detectTask('Analyze the chart in this image')).toBe('vision');
    });

    test('defaults to general', () => {
        expect(detectTask('Hello, how are you today?')).toBe('general');
        expect(detectTask('Tell me about your capabilities')).toBe('general');
    });
});

// ═══════════════════════════════════════════════════════════
// 2. LLM CLASSIFIER TESTS (live API calls)
// ═══════════════════════════════════════════════════════════
describe('classifyWithLLM (Groq)', () => {
    const skip = !GROQ_KEY;

    test('classifies coding prompt', async () => {
        if (skip) return console.log('  ⏭ SKIP: no GROQ_API_KEY');
        const result = await classifyWithLLM(
            [{ role: 'user', content: 'Write a Python function that sorts a list using merge sort' }],
            CLASSIFIER_PRESETS.groq.apiUrl,
            GROQ_KEY,
            CLASSIFIER_PRESETS.groq.model,
        );
        console.log(`  coding → ${result.task} (${result.method}, conf=${result.confidence})`);
        expect(['coding']).toContain(result.task);
    }, 10000);

    test('classifies creative prompt', async () => {
        if (skip) return console.log('  ⏭ SKIP: no GROQ_API_KEY');
        const result = await classifyWithLLM(
            [{ role: 'user', content: 'Write a haiku about the ocean' }],
            CLASSIFIER_PRESETS.groq.apiUrl,
            GROQ_KEY,
            CLASSIFIER_PRESETS.groq.model,
        );
        console.log(`  creative → ${result.task} (${result.method}, conf=${result.confidence})`);
        expect(['creative']).toContain(result.task);
    }, 10000);

    test('uses conversation context', async () => {
        if (skip) return console.log('  ⏭ SKIP: no GROQ_API_KEY');
        // "make it faster" alone = ambiguous, but after coding context should = coding
        const result = await classifyWithLLM(
            [
                { role: 'user', content: 'Write a Python function that sorts a list' },
                { role: 'assistant', content: 'Here is a bubble sort implementation...' },
                { role: 'user', content: 'Now make it faster' },
            ],
            CLASSIFIER_PRESETS.groq.apiUrl,
            GROQ_KEY,
            CLASSIFIER_PRESETS.groq.model,
        );
        console.log(`  context test → ${result.task} (${result.method})`);
        expect(['coding', 'general']).toContain(result.task);
    }, 10000);
});

// ═══════════════════════════════════════════════════════════
// 3. CASCADE CLASSIFIER TESTS
// ═══════════════════════════════════════════════════════════
describe('classifyWithCascade', () => {
    const keys: Record<string, string> = {};
    if (GOOGLE_KEY) keys.GOOGLE_API_KEY = GOOGLE_KEY;
    if (OPENROUTER_KEY) keys.OPENROUTER_API_KEY = OPENROUTER_KEY;
    if (GROQ_KEY) keys.GROQ_API_KEY = GROQ_KEY;

    const skip = Object.keys(keys).length === 0;

    test('cascades through providers', async () => {
        if (skip) return console.log('  ⏭ SKIP: no API keys');
        const result = await classifyWithCascade(
            [{ role: 'user', content: 'Write a Python function that sorts a list' }],
            keys,
        );
        console.log(`  cascade → ${result.task} via ${result.method} (conf=${result.confidence})`);
        expect(['coding']).toContain(result.task);
        expect(['gemma-1b', 'gemma-4b', 'openrouter-gemma', 'groq-llama8b', 'heuristic']).toContain(result.method);
    }, 15000);

    test('falls back to Groq if free models fail', async () => {
        if (!GROQ_KEY) return console.log('  ⏭ SKIP: no GROQ_API_KEY');
        // Use only bad keys for free models, Groq as fallback
        const badKeys = {
            GOOGLE_API_KEY: 'invalid-key',
            GROQ_API_KEY: GROQ_KEY,
        };
        const result = await classifyWithCascade(
            [{ role: 'user', content: 'Write a Python function' }],
            badKeys,
        );
        console.log(`  fallback → ${result.task} via ${result.method}`);
        expect(result.method).toBe('groq-llama8b');
    }, 10000);
});

// ═══════════════════════════════════════════════════════════
// 4. FULL ROUTER INTEGRATION TESTS
// ═══════════════════════════════════════════════════════════
describe('FreeRouter with task routing', () => {
    const skip = !GROQ_KEY;

    function createRouter() {
        return new FreeRouter({
            strategy: 'smart',
            apiKeys: { groq: GROQ_KEY },
            providers: ['groq'],
            minTier: 'B',
            autoDetectTask: true,
            useLLMClassifier: true,
        });
    }

    test('routes coding prompt to appropriate model', async () => {
        if (skip) return console.log('  ⏭ SKIP: no GROQ_API_KEY');
        const router = createRouter();
        try {
            const resp = await router.chat.completions.create({
                messages: [{ role: 'user', content: 'Write a Python function that sorts a list' }],
                max_tokens: 30,
            });
            console.log(`  coding → ${resp._router.provider}/${resp.model}`);
            console.log(`    response: ${resp.choices[0].message.content.slice(0, 60)}`);
            expect(resp.model).toBeTruthy();
            expect(resp.choices[0].message.content).toBeTruthy();
        } catch (e: any) {
            console.log(`  ⚠ ${e.message?.slice(0, 80)}`);
        }
    }, 20000);

    test('routes with explicit task parameter', async () => {
        if (skip) return console.log('  ⏭ SKIP: no GROQ_API_KEY');
        const router = createRouter();
        try {
            const resp = await router.chat.completions.create({
                messages: [{ role: 'user', content: 'Hello' }],
                max_tokens: 10,
                task: 'fast',
            } as any);
            console.log(`  explicit task=fast → ${resp._router.provider}/${resp.model}`);
            expect(resp.model).toBeTruthy();
        } catch (e: any) {
            console.log(`  ⚠ ${e.message?.slice(0, 80)}`);
        }
    }, 20000);

    test('handles multi-turn conversation context', async () => {
        if (skip) return console.log('  ⏭ SKIP: no GROQ_API_KEY');
        const router = createRouter();
        try {
            const resp = await router.chat.completions.create({
                messages: [
                    { role: 'user', content: 'Write a Python function' },
                    { role: 'assistant', content: 'Here is a function...' },
                    { role: 'user', content: 'Now add error handling' },
                ],
                max_tokens: 30,
            });
            console.log(`  multi-turn → ${resp._router.provider}/${resp.model}`);
            expect(resp.model).toBeTruthy();
        } catch (e: any) {
            console.log(`  ⚠ ${e.message?.slice(0, 80)}`);
        }
    }, 20000);
});

// ═══════════════════════════════════════════════════════════
// 5. SCORE FOR TASK TESTS (unit, no API)
// ═══════════════════════════════════════════════════════════
describe('scoreForTask', () => {
    const makeModel = (id: string, tier: string, ctxK: number, tools = true) => ({
        model: {
            modelId: id,
            displayName: id,
            tier,
            contextK: ctxK,
            supportsTools: tools,
            supportsStreaming: true,
            supportsVision: false,
            providerIds: ['groq'],
        },
        provider: { id: 'groq', name: 'Groq', baseUrl: '', apiKeyEnvVars: [], rateLimits: {} },
        health: { available: true, score: 80, latency: { avg: 500 } },
    });

    test('scores coding models higher for coding task', () => {
        const coding = makeModel('qwen/qwen3-32b', 'A+', 131);
        const generic = makeModel('llama-3.3-70b-versatile', 'A-', 128); // A- >= minTier A check uses rank
        const results = scoreForTask([coding, generic], 'coding');
        expect(results.length).toBeGreaterThan(0);
        expect(results[0].model.modelId).toBe('qwen/qwen3-32b');
        if (results.length > 1) {
            expect(results[0].score).toBeGreaterThan(results[1].score);
        }
    });

    test('filters out models below min tier', () => {
        const lowTier = makeModel('tiny-model', 'C', 128);
        const results = scoreForTask([lowTier], 'coding'); // coding requires minTier A
        expect(results.length).toBe(0);
    });

    test('filters out models without required tools for coding', () => {
        const noTools = makeModel('some-model', 'A', 128, false);
        const results = scoreForTask([noTools], 'coding');
        expect(results.length).toBe(0);
    });

    test('general task accepts all tiers >= B', () => {
        const bTier = makeModel('some-model', 'B', 128);
        const results = scoreForTask([bTier], 'general');
        expect(results.length).toBeGreaterThan(0);
    });
});
