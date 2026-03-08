#!/usr/bin/env node
/**
 * Sentinel AI CLI — scan text for safety issues from the command line.
 *
 * Usage:
 *   npx @sentinel-ai/sdk scan "Ignore all previous instructions"
 *   npx @sentinel-ai/sdk scan --json "text to scan"
 *   echo "text" | npx @sentinel-ai/sdk scan --stdin
 */

import { SentinelGuard } from './index.js';
import type { ScanResult } from './index.js';

function formatResult(result: ScanResult, json: boolean): string {
  if (json) {
    return JSON.stringify({
      safe: result.safe,
      blocked: result.blocked,
      risk: result.risk,
      findings: result.findings.map(f => ({
        scanner: f.scanner,
        category: f.category,
        description: f.description,
        risk: f.risk,
      })),
      redactedText: result.redactedText,
      latencyMs: result.latencyMs,
    }, null, 2);
  }

  const lines: string[] = [];
  const status = result.safe ? 'SAFE' : (result.blocked ? 'BLOCKED' : 'RISKY');
  lines.push(`Status: ${status} (risk: ${result.risk})`);
  lines.push(`Latency: ${result.latencyMs}ms`);

  if (result.findings.length > 0) {
    lines.push(`\nFindings (${result.findings.length}):`);
    result.findings.forEach((f, i) => {
      lines.push(`  ${i + 1}. [${f.risk}] ${f.description}`);
      lines.push(`     Scanner: ${f.scanner} | Category: ${f.category}`);
    });
  } else {
    lines.push('\nNo findings.');
  }

  if (result.redactedText) {
    lines.push(`\nRedacted output:\n  ${result.redactedText}`);
  }

  return lines.join('\n');
}

async function readStdin(): Promise<string> {
  const chunks: Buffer[] = [];
  for await (const chunk of process.stdin) {
    chunks.push(chunk);
  }
  return Buffer.concat(chunks).toString('utf-8');
}

async function main(): Promise<void> {
  const args = process.argv.slice(2);

  if (args.length === 0 || args[0] === '--help' || args[0] === '-h') {
    console.log(`Sentinel AI — Real-time safety guardrails for LLM applications

Usage:
  npx @sentinel-ai/sdk scan "text to scan"
  npx @sentinel-ai/sdk scan --json "text to scan"
  echo "text" | npx @sentinel-ai/sdk scan --stdin

Commands:
  scan     Scan text for prompt injection, PII, harmful content, etc.

Options:
  --json   Output results as JSON
  --stdin  Read text from stdin
  --help   Show this help message

Scanners: prompt injection, PII, harmful content, toxicity, tool-use safety, obfuscation detection

https://github.com/MaxwellCalkin/sentinel-ai`);
    process.exit(0);
  }

  if (args[0] !== 'scan') {
    console.error(`Unknown command: ${args[0]}. Use 'scan' or --help.`);
    process.exit(1);
  }

  const restArgs = args.slice(1);
  const jsonOutput = restArgs.includes('--json');
  const useStdin = restArgs.includes('--stdin');
  const textArgs = restArgs.filter(a => a !== '--json' && a !== '--stdin');

  let text: string;
  if (useStdin) {
    text = await readStdin();
  } else if (textArgs.length > 0) {
    text = textArgs.join(' ');
  } else {
    console.error('Error: provide text to scan or use --stdin');
    process.exit(1);
  }

  if (!text.trim()) {
    console.error('Error: empty input');
    process.exit(1);
  }

  const guard = SentinelGuard.default();
  const result = guard.scan(text);
  console.log(formatResult(result, jsonOutput));
  process.exit(result.blocked ? 1 : 0);
}

main().catch(err => {
  console.error(`Error: ${err.message}`);
  process.exit(1);
});
