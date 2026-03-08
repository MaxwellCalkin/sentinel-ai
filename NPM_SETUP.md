# Publishing @sentinel-ai/sdk to npm

## One-time setup

1. Create an npm account at https://www.npmjs.com/signup (if you don't have one)

2. Create an npm access token:
   - Go to https://www.npmjs.com/settings/~/tokens
   - Click "Generate New Token" > "Classic Token"
   - Select "Automation" type
   - Copy the token

3. Add the token to GitHub:
   - Go to https://github.com/MaxwellCalkin/sentinel-ai/settings/secrets/actions
   - Click "New repository secret"
   - Name: `NPM_TOKEN`
   - Value: paste the npm token

4. Create the npm org (optional, for scoped packages):
   - Go to https://www.npmjs.com/org/create
   - Create org: `sentinel-ai`
   - Or publish without scope by changing `name` in `sdk-js/package.json` to `sentinel-guardrails-js`

## Publishing

### Automatic (on GitHub release)
The npm publish workflow runs automatically when you create a GitHub release. The JS SDK version in `sdk-js/package.json` should match the release tag.

### Manual
```bash
cd sdk-js
npm login
npm publish --access public
```

## What gets published

The package includes:
- `dist/index.js` — compiled JavaScript (CommonJS)
- `dist/index.d.ts` — TypeScript type declarations
- `README.md` — package documentation
- `LICENSE` — Apache 2.0

Source code (`src/`) and tests are NOT included in the npm package.
