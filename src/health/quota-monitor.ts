// src/health/quota-monitor.ts — Proactive quota reset detection and monitoring
import type { ProviderDef } from '../providers/types.js';
import { createLogger, type Logger } from '../utils/logger.js';

/** Provider-specific quota reset schedule */
interface QuotaSchedule {
    /** Provider ID */
    providerId: string;
    /** When does daily quota reset? (hour in UTC, e.g. 7 = 00:00 PST) */
    resetHourUTC: number;
    /** How often to probe (ms) when provider is rate-limited */
    probeIntervalMs: number;
    /** Last known rate-limit timestamp */
    lastRateLimitedAt: number;
    /** Is currently rate-limited? */
    isRateLimited: boolean;
    /** Models that were rate-limited */
    blockedModels: Set<string>;
    /** Last successful probe */
    lastProbeAt: number;
}

/** Known provider reset schedules */
const PROVIDER_SCHEDULES: Record<string, Partial<QuotaSchedule>> = {
    googleai: { resetHourUTC: 7, probeIntervalMs: 5 * 60 * 1000 },        // Resets ~midnight PST
    openrouter: { resetHourUTC: 0, probeIntervalMs: 10 * 60 * 1000 },     // Resets midnight UTC
    aihubmix: { resetHourUTC: 0, probeIntervalMs: 15 * 60 * 1000 },       // Resets midnight UTC
    groq: { resetHourUTC: 0, probeIntervalMs: 5 * 60 * 1000 },            // Resets midnight UTC
    cerebras: { resetHourUTC: 0, probeIntervalMs: 5 * 60 * 1000 },        // Resets midnight UTC
    huggingface: { resetHourUTC: 0, probeIntervalMs: 30 * 60 * 1000 },    // Monthly reset
};

export interface QuotaMonitorOptions {
    /** How often to check for quota resets (ms). Default: 5 min */
    checkIntervalMs?: number;
    /** Debug logging */
    debug?: boolean;
}

/**
 * Monitors quota state per provider and detects when quotas reset.
 * Works with the StateStore to persist quota observations.
 */
export class QuotaMonitor {
    private schedules: Map<string, QuotaSchedule> = new Map();
    private logger: Logger;
    private checkIntervalMs: number;
    private timer: ReturnType<typeof setInterval> | null = null;
    private onProviderRecovered: ((providerId: string) => void) | null = null;

    constructor(options?: QuotaMonitorOptions) {
        this.logger = createLogger(options?.debug ?? false);
        this.checkIntervalMs = options?.checkIntervalMs ?? 5 * 60 * 1000;

        // Initialize schedules for all known providers
        for (const [providerId, schedule] of Object.entries(PROVIDER_SCHEDULES)) {
            this.schedules.set(providerId, {
                providerId,
                resetHourUTC: schedule.resetHourUTC ?? 0,
                probeIntervalMs: schedule.probeIntervalMs ?? 10 * 60 * 1000,
                lastRateLimitedAt: 0,
                isRateLimited: false,
                blockedModels: new Set(),
                lastProbeAt: 0,
            });
        }
    }

    /** Set callback for when a provider recovers from rate limit */
    onRecovery(callback: (providerId: string) => void): void {
        this.onProviderRecovered = callback;
    }

    /** Record that a provider hit a rate limit */
    recordRateLimit(providerId: string, modelId?: string): void {
        const sched = this.getOrCreate(providerId);
        sched.isRateLimited = true;
        sched.lastRateLimitedAt = Date.now();
        if (modelId) sched.blockedModels.add(modelId);
        this.logger.log(`QuotaMonitor: ${providerId} rate-limited${modelId ? ` (model: ${modelId})` : ''}`);
    }

    /** Record that a provider had a successful request */
    recordSuccess(providerId: string, modelId?: string): void {
        const sched = this.getOrCreate(providerId);
        if (sched.isRateLimited) {
            sched.isRateLimited = false;
            this.logger.log(`QuotaMonitor: ${providerId} recovered!`);
            this.onProviderRecovered?.(providerId);
        }
        if (modelId) sched.blockedModels.delete(modelId);
        sched.lastProbeAt = Date.now();
    }

    /** Check if a provider is currently rate-limited */
    isProviderLimited(providerId: string): boolean {
        const sched = this.schedules.get(providerId);
        if (!sched || !sched.isRateLimited) return false;

        // Check if quota should have reset by now
        const now = new Date();
        const resetHour = sched.resetHourUTC;
        const lastRateLimit = new Date(sched.lastRateLimitedAt);

        // If we crossed the reset hour since last rate limit, assume recovered
        if (now.getUTCHours() >= resetHour) {
            const lastRLHour = lastRateLimit.getUTCHours();
            const lastRLDay = lastRateLimit.getUTCDate();
            const nowDay = now.getUTCDate();

            if (nowDay > lastRLDay || (nowDay === lastRLDay && lastRLHour < resetHour && now.getUTCHours() >= resetHour)) {
                sched.isRateLimited = false;
                sched.blockedModels.clear();
                this.logger.log(`QuotaMonitor: ${providerId} quota reset detected (passed reset hour ${resetHourUTC}h UTC)`);
                this.onProviderRecovered?.(providerId);
                return false;
            }
        }

        return true;
    }

    /** Check if a specific model is blocked for a provider */
    isModelBlocked(providerId: string, modelId: string): boolean {
        const sched = this.schedules.get(providerId);
        if (!sched) return false;
        if (!sched.isRateLimited) return false;
        return sched.blockedModels.has(modelId);
    }

    /** Get all currently blocked providers */
    getBlockedProviders(): string[] {
        const blocked: string[] = [];
        for (const [id, sched] of this.schedules) {
            if (sched.isRateLimited) blocked.push(id);
        }
        return blocked;
    }

    /** Get status of all providers */
    getStatus(): Record<string, {
        isRateLimited: boolean;
        blockedModels: string[];
        lastRateLimitedAt: number;
        resetHourUTC: number;
        timeUntilResetMs: number;
    }> {
        const result: Record<string, any> = {};
        const now = Date.now();

        for (const [id, sched] of this.schedules) {
            // Calculate time until next reset
            const nextReset = new Date();
            nextReset.setUTCHours(sched.resetHourUTC, 0, 0, 0);
            if (nextReset.getTime() <= now) {
                nextReset.setUTCDate(nextReset.getUTCDate() + 1);
            }

            result[id] = {
                isRateLimited: sched.isRateLimited,
                blockedModels: [...sched.blockedModels],
                lastRateLimitedAt: sched.lastRateLimitedAt,
                resetHourUTC: sched.resetHourUTC,
                timeUntilResetMs: nextReset.getTime() - now,
            };
        }
        return result;
    }

    /** Start periodic monitoring */
    start(): void {
        if (this.timer) return;
        this.timer = setInterval(() => this.check(), this.checkIntervalMs);
        this.logger.log('QuotaMonitor: started periodic monitoring');
    }

    /** Stop monitoring */
    stop(): void {
        if (this.timer) {
            clearInterval(this.timer);
            this.timer = null;
        }
    }

    /** Check all providers for potential quota resets */
    private check(): void {
        for (const [id, sched] of this.schedules) {
            if (sched.isRateLimited) {
                this.isProviderLimited(id); // Will auto-detect reset
            }
        }
    }

    private getOrCreate(providerId: string): QuotaSchedule {
        if (!this.schedules.has(providerId)) {
            const known = PROVIDER_SCHEDULES[providerId];
            this.schedules.set(providerId, {
                providerId,
                resetHourUTC: known?.resetHourUTC ?? 0,
                probeIntervalMs: known?.probeIntervalMs ?? 10 * 60 * 1000,
                lastRateLimitedAt: 0,
                isRateLimited: false,
                blockedModels: new Set(),
                lastProbeAt: 0,
            });
        }
        return this.schedules.get(providerId)!;
    }
}
