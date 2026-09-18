# SyncSentry webapp

Browser UI for SyncSentry. Upload a video (or a `.zip` bundling a video +
captions), optionally toggle the SyncNet detector, and see detected/
residual offset bars, a confidence gauge, and download links for the
corrected files -- all backed by the real `analyzer` FastAPI service
(`/v1/fix`), no mocked data.

## Run it

```bash
# 1. the analyzer API (from ../analyzer, with its .venv active)
uvicorn syncsentry.api:app --port 8000

# 2. this webapp
npm install
npm run dev        # http://localhost:5173
```

To point at a non-localhost API, set `VITE_API_BASE_URL` (e.g. in
`.env.local`) before `npm run build` / `npm run dev`.

## Structure

```
src/
  api/          Typed fetch client for the analyzer API (client.ts, types.ts)
  components/   FileDrop, UploadForm, OffsetBar, ConfidenceGauge, IssueCard,
                ResultsPanel, ProcessingView, Header/Footer
  pages/        Home (about/how-it-works), Analyze (upload -> processing -> results)
  hooks/        useApiHealth (polls /healthz)
```

No chart library -- the offset bar and confidence gauge are small,
purpose-built SVG/CSS components, since the one thing this tool needs to
communicate ("which way, how far, do we trust it") is more specific than
what a generic chart affords.
