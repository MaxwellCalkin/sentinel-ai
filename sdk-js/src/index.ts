/**
 * Sentinel AI — JavaScript/TypeScript SDK
 *
 * Client for the Sentinel AI safety guardrails API.
 *
 * @example
 * ```ts
 * import { SentinelClient } from '@sentinel-ai/sdk';
 *
 * const sentinel = new SentinelClient({ apiKey: 'sk-...' });
 * const result = await sentinel.scan('Check this text');
 * if (!result.safe) {
 *   console.log('Blocked:', result.findings);
 * }
 * ```
 */

export interface SentinelConfig {
  baseUrl?: string;
  apiKey?: string;
  timeout?: number;
}

export interface Finding {
  scanner: string;
  category: string;
  description: string;
  risk: string;
  span?: [number, number];
  metadata: Record<string, unknown>;
}

export interface ScanResult {
  safe: boolean;
  blocked: boolean;
  risk: string;
  findings: Finding[];
  redacted_text: string | null;
  latency_ms: number;
}

export interface BatchScanResult {
  results: ScanResult[];
  total_latency_ms: number;
}

export interface HealthResult {
  status: string;
  version: string;
  scanners: string[];
}

export class SentinelError extends Error {
  constructor(
    message: string,
    public status: number,
    public body?: unknown,
  ) {
    super(message);
    this.name = 'SentinelError';
  }
}

export class SentinelClient {
  private baseUrl: string;
  private apiKey?: string;
  private timeout: number;

  constructor(config: SentinelConfig = {}) {
    this.baseUrl = (config.baseUrl || 'http://localhost:8329').replace(/\/$/, '');
    this.apiKey = config.apiKey;
    this.timeout = config.timeout || 30000;
  }

  private async request<T>(
    method: string,
    path: string,
    body?: unknown,
  ): Promise<T> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };
    if (this.apiKey) {
      headers['Authorization'] = `Bearer ${this.apiKey}`;
    }

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(`${this.baseUrl}${path}`, {
        method,
        headers,
        body: body ? JSON.stringify(body) : undefined,
        signal: controller.signal,
      });

      if (!response.ok) {
        const errorBody = await response.json().catch(() => null);
        throw new SentinelError(
          `Sentinel API error: ${response.status}`,
          response.status,
          errorBody,
        );
      }

      return (await response.json()) as T;
    } finally {
      clearTimeout(timer);
    }
  }

  async scan(
    text: string,
    options?: { context?: Record<string, unknown>; scanners?: string[] },
  ): Promise<ScanResult> {
    return this.request<ScanResult>('POST', '/scan', {
      text,
      ...options,
    });
  }

  async scanBatch(texts: string[]): Promise<BatchScanResult> {
    return this.request<BatchScanResult>('POST', '/scan/batch', { texts });
  }

  async health(): Promise<HealthResult> {
    return this.request<HealthResult>('GET', '/health');
  }
}

export default SentinelClient;
