/**
 * Sentinel AI — Claude Code PostToolUse Hook (TypeScript)
 *
 * Scans every Write/Edit tool output for:
 * - OWASP vulnerabilities (SQL injection, XSS, command injection, etc.)
 * - Supply chain attacks in package manifests
 * - Hardcoded secrets and API keys
 *
 * Setup — add to .claude/settings.json:
 * {
 *   "hooks": {
 *     "PostToolUse": [
 *       {
 *         "matcher": "Write|Edit",
 *         "hooks": [{
 *           "type": "command",
 *           "command": "npx tsx examples/claude_code_hook.ts"
 *         }]
 *       }
 *     ]
 *   }
 * }
 *
 * Run: reads tool output from stdin, exits 2 to block on high/critical findings
 */

import { CodeScanner, DependencyScanner } from '../sdk-js/src/index';

const MANIFEST_FILES = new Set([
  'requirements.txt', 'package.json', 'pyproject.toml',
  'Pipfile', 'setup.py', 'setup.cfg',
]);

async function main(): Promise<number> {
  // Read stdin
  let raw = '';
  try {
    const chunks: Buffer[] = [];
    for await (const chunk of process.stdin) {
      chunks.push(chunk);
    }
    raw = Buffer.concat(chunks).toString('utf-8');
  } catch {
    return 0;
  }

  if (!raw.trim()) return 0;

  let event: Record<string, unknown>;
  try {
    event = JSON.parse(raw);
  } catch {
    return 0;
  }

  const toolName = (event.tool_name as string) ?? '';
  const toolInput = (event.tool_input as Record<string, unknown>) ?? {};

  if (!['Write', 'Edit', 'write', 'edit'].includes(toolName)) {
    return 0;
  }

  const filePath = (toolInput.file_path ?? toolInput.path ?? '') as string;
  const content = (toolInput.content ?? toolInput.new_string ?? '') as string;

  if (!content) return 0;

  const allFindings: Array<{ risk: string; description: string }> = [];

  // Code vulnerability scan
  const codeScanner = new CodeScanner();
  allFindings.push(...codeScanner.scan(content, filePath));

  // Dependency scan (if manifest file)
  const fileName = filePath.split('/').pop() ?? filePath.split('\\').pop() ?? '';
  if (MANIFEST_FILES.has(fileName)) {
    const depScanner = new DependencyScanner();
    allFindings.push(...depScanner.scan(content, fileName));
  }

  if (allFindings.length === 0) return 0;

  const criticalHigh = allFindings.filter(f =>
    f.risk === 'CRITICAL' || f.risk === 'HIGH'
  );
  const medium = allFindings.filter(f => f.risk === 'MEDIUM');

  // Warn on medium findings
  for (const f of medium) {
    console.log(`  WARNING: ${f.description}`);
  }

  // Block on high/critical
  if (criticalHigh.length > 0) {
    console.log(`\nSentinel AI: blocked ${toolName} — ${criticalHigh.length} security issue(s) found:\n`);
    for (const f of criticalHigh) {
      console.log(`  [${f.risk}] ${f.description}`);
    }
    console.log(`\n  File: ${filePath}`);
    console.log('  Fix the issues above before writing this code.');
    return 2;
  }

  return 0;
}

main().then(code => process.exit(code));
