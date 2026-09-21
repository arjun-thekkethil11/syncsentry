# Setup

## Requirements

- Python 3.11+
- Node.js 18+
- FFmpeg (`brew install ffmpeg` on macOS, `apt install ffmpeg` on Debian/Ubuntu)

## Backend

```bash
cd analyzer
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-core.txt -r requirements-api.txt -r requirements-ml.txt
pip install -e .

uvicorn syncsentry.api:app --port 8000
```

The API is now at `http://localhost:8000`.

### SyncNet (recommended, needed for the accurate detector)

```bash
bash scripts/fetch_syncnet.sh
```

Downloads the pretrained SyncNet and S3FD weights into `third_party/` (gitignored). Without this step, only the faster, less accurate coarse detector is available.

### Optional models

```bash
bash scripts/fetch_light_asd.sh      # active speaker detection
bash scripts/fetch_mtdvocalist.sh    # short-clip sync scoring
```

## Frontend

```bash
cd webapp
npm install
npm run dev
```

Open `http://localhost:5173`. Override the backend URL with `VITE_API_BASE_URL` in `webapp/.env.local`.

## Command line

```bash
syncsentry fix --video asset.mp4 --out-dir out/
syncsentry fix --video asset.mp4 --captions asset.vtt --out-dir out/ --use-syncnet
```

## Tests

```bash
cd analyzer && pytest tests/ -v
cd webapp && npm run test
```

Some analyzer tests need real content or the SyncNet weights and skip automatically if those are not present. To include them:

```bash
bash analyzer/scripts/fetch_real_content.sh
bash analyzer/scripts/fetch_syncnet.sh
```

## Benchmark

```bash
cd analyzer
syncsentry benchmark --out-dir ../benchmark/results
python3 scripts/blind_benchmark/gen_clips.py
python3 scripts/blind_benchmark/run_benchmark.py my_run
```
