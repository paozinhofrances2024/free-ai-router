// src/persistence/state-store.ts — File-based state persistence for router
import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'node:fs';
import { dirname } from 'node:path';
import { createLogger, type Logger } from '../utils/logger.js';

/** Serializable router state */
export interface RouterState {
    /** When this state was last saved */
    savedAt: number;
    /** Provider-level stats */
    providers: Record<string, ProviderPersistState>;
    /** Global request counters */
    global: {
        totalRequests: number;
        successfulRequests: number;
        failedRequests: number;
    };
}

export interface ProviderPersistState {
    /** Quota info from last known response */
    quota?: {
        remaining: number;
        limit: number;
        quotaPercent: number;
        recordedAt: number;
    };
    /** Latency stats */
    latency?: {
        avg: number;
        p95: number;
        sampleCount: number;
    };
    /** Circuit breaker */
    circuit?: {
        failures: number;
        state: string;
        lastFailureTime: number;
    };
    /** Request counters */
    requests: number;
    successes: number;
    failures: number;
}

const DEFAULT_STATE: RouterState = {
    savedAt: 0,
    providers: {},
    global: { totalRequests: 0, successfulRequests: 0, failedRequests: 0 },
};

/**
 * File-based state store with debounced writes.
 * Persists router state to disk so it survives restarts.
 */
export class StateStore {
    private filePath: string;
    private logger: Logger;
    private state: RouterState;
    private dirty = false;
    private writeTimer: ReturnType<typeof setTimeout> | null = null;
    private debounceMs: number;

    constructor(filePath: string, debounceMs = 5000, debug = false) {
        this.filePath = filePath;
        this.debounceMs = debounceMs;
        this.logger = createLogger(debug);
        this.state = this.load();
    }

    /** Load state from disk (or return default) */
    private load(): RouterState {
        try {
            if (existsSync(this.filePath)) {
                const raw = readFileSync(this.filePath, 'utf-8');
                const parsed = JSON.parse(raw);
                // Validate basic structure
                if (parsed && typeof parsed.providers === 'object') {
                    return parsed as RouterState;
                }
            }
        } catch (err: any) {
            this.logger.warn(`StateStore: failed to load ${this.filePath}: ${err.message}`);
        }
        return { ...DEFAULT_STATE, providers: {} };
    }

    /** Flush to disk immediately */
    private flush(): void {
        try {
            const dir = dirname(this.filePath);
            if (!existsSync(dir)) {
                mkdirSync(dir, { recursive: true });
            }
            this.state.savedAt = Date.now();
            writeFileSync(this.filePath, JSON.stringify(this.state, null, 2), 'utf-8');
            this.dirty = false;
        } catch (err: any) {
            this.logger.warn(`StateStore: failed to write ${this.filePath}: ${err.message}`);
        }
    }

    /** Schedule a debounced write */
    private scheduleWrite(): void {
        this.dirty = true;
        if (this.writeTimer) clearTimeout(this.writeTimer);
        this.writeTimer = setTimeout(() => {
            if (this.dirty) this.flush();
            this.writeTimer = null;
        }, this.debounceMs);
    }

    /** Force immediate save */
    save(): void {
        if (this.writeTimer) {
            clearTimeout(this.writeTimer);
            this.writeTimer = null;
        }
        this.flush();
    }

    // ─── Read Methods ───────────────────────────────────────

    /** Get the full persisted state */
    getState(): RouterState {
        return this.state;
    }

    /** Get state for a specific provider */
    getProvider(providerId: string): ProviderPersistState | undefined {
        return this.state.providers[providerId];
    }

    /** Get global counters */
    getGlobal(): { totalRequests: number; successfulRequests: number; failedRequests: number } {
        return this.state.global;
    }

    // ─── Write Methods ──────────────────────────────────────

    /** Record a successful request */
    recordSuccess(providerId: string, latencyMs: number): void {
        this.state.global.totalRequests++;
        this.state.global.successfulRequests++;

        const prov = this.getOrCreateProvider(providerId);
        prov.requests++;
        prov.successes++;

        // Update latency rolling stats
        if (prov.latency) {
            const prevAvg = prov.latency.avg;
            const prevCount = prov.latency.sampleCount;
            const newCount = prevCount + 1;
            prov.latency.avg = Math.round((prevAvg * prevCount + latencyMs) / newCount);
            prov.latency.sampleCount = newCount;
            // Simple p95 approximation: if current > p95, drift up slowly
            if (latencyMs > prov.latency.p95) {
                prov.latency.p95 = Math.round(prov.latency.p95 * 0.9 + latencyMs * 0.1);
            }
        } else {
            prov.latency = { avg: latencyMs, p95: latencyMs, sampleCount: 1 };
        }

        // Reset circuit on success
        if (prov.circuit) {
            prov.circuit.failures = 0;
            prov.circuit.state = 'CLOSED';
        }

        this.scheduleWrite();
    }

    /** Record a failed request */
    recordFailure(providerId: string, _error?: string): void {
        this.state.global.totalRequests++;
        this.state.global.failedRequests++;

        const prov = this.getOrCreateProvider(providerId);
        prov.requests++;
        prov.failures++;

        // Update circuit breaker
        if (!prov.circuit) {
            prov.circuit = { failures: 0, state: 'CLOSED', lastFailureTime: 0 };
        }
        prov.circuit.failures++;
        prov.circuit.lastFailureTime = Date.now();
        if (prov.circuit.failures >= 5) {
            prov.circuit.state = 'OPEN';
        }

        this.scheduleWrite();
    }

    /** Record quota info from response headers */
    recordQuota(providerId: string, remaining: number, limit: number): void {
        const prov = this.getOrCreateProvider(providerId);
        prov.quota = {
            remaining,
            limit,
            quotaPercent: limit > 0 ? Math.round((remaining / limit) * 100) : 100,
            recordedAt: Date.now(),
        };
        this.scheduleWrite();
    }

    /** Record circuit breaker state */
    recordCircuit(providerId: string, state: string, failures: number): void {
        const prov = this.getOrCreateProvider(providerId);
        prov.circuit = {
            failures,
            state,
            lastFailureTime: Date.now(),
        };
        this.scheduleWrite();
    }

    /** Update provider state from current in-memory trackers */
    updateProvider(providerId: string, data: Partial<ProviderPersistState>): void {
        const prov = this.getOrCreateProvider(providerId);
        if (data.quota) prov.quota = data.quota;
        if (data.latency) prov.latency = data.latency;
        if (data.circuit) prov.circuit = data.circuit;
        if (data.requests !== undefined) prov.requests = data.requests;
        if (data.successes !== undefined) prov.successes = data.successes;
        if (data.failures !== undefined) prov.failures = data.failures;
        this.scheduleWrite();
    }

    /** Get or create provider state */
    private getOrCreateProvider(providerId: string): ProviderPersistState {
        if (!this.state.providers[providerId]) {
            this.state.providers[providerId] = {
                requests: 0,
                successes: 0,
                failures: 0,
            };
        }
        return this.state.providers[providerId];
    }

    /** Clear all persisted state */
    clear(): void {
        this.state = { ...DEFAULT_STATE, providers: {} };
        this.scheduleWrite();
    }

    /** Cleanup: flush pending writes */
    destroy(): void {
        this.save();
    }
}
