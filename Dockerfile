# One image runs the whole app: Flask serves the API and the built React frontend.
#   docker build -t cardiosense .
#   docker run -p 8000:8000 cardiosense      ->  http://localhost:8000
# The trained artifacts/ and reports/ must exist (run scripts/run_pipeline.py first).

FROM node:22-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt
COPY scripts/ scripts/
COPY backend/ backend/
COPY artifacts/ artifacts/
COPY reports/ reports/
COPY --from=web /web/dist frontend/dist
WORKDIR /app/backend
ENV PORT=8000
EXPOSE 8000
CMD gunicorn -w 2 -b 0.0.0.0:${PORT} app:app
