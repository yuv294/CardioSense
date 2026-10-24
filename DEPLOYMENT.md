# Deploying CardioSense

The Flask backend serves both the API and the built React app, so one service runs
everything.

## Option 1: Docker (any host)

```bash
cd scripts && python run_pipeline.py --skip-nested   # only if artifacts/ is missing
cd .. && docker build -t cardiosense .
docker run -p 8000:8000 cardiosense                  # http://localhost:8000
```

## Option 2: Render (free tier)

1. Push the repository to GitHub, including `artifacts/` and `reports/` (a few MB).
2. On render.com choose **New → Blueprint** and pick the repository. `render.yaml`
   builds the Dockerfile and checks `/api/health`.
3. The free instance sleeps after ~15 minutes idle; the first request then takes ~30–60 s.

## Option 3: split hosting (e.g. Netlify + Render)

Build the frontend with the backend's URL baked in:

```bash
cd frontend && VITE_API_BASE=https://your-api.onrender.com npm run build
```

Upload `frontend/dist` to any static host (`netlify.toml` already handles SPA routing),
and set `CORS_ORIGIN=https://your-site.netlify.app` on the backend.
