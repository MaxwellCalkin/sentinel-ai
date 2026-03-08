# PyPI Publishing Setup

The package is named `sentinel-guardrails` on PyPI (since `sentinel-ai` was already taken).

## One-time setup: Configure Trusted Publishing

1. Go to https://pypi.org/manage/account/publishing/
2. Click "Add a new pending publisher"
3. Fill in:
   - **PyPI Project Name:** `sentinel-guardrails`
   - **Owner:** `MaxwellCalkin`
   - **Repository name:** `sentinel-ai`
   - **Workflow name:** `publish.yml`
   - **Environment name:** `pypi`
4. Click "Add"

Also add for TestPyPI at https://test.pypi.org/manage/account/publishing/:
   - Same settings but **Environment name:** `testpypi`

## Publishing

Once configured, publishing happens automatically:
- **On release:** Create a GitHub release → package auto-publishes to PyPI
- **Manual:** Go to Actions → "Publish to PyPI" → Run workflow

## Install

```bash
pip install sentinel-guardrails
```

The `sentinel` CLI command and all `from sentinel import ...` imports work the same.
