# Deploying CardioSense — free tier, step by step

This guide assumes you've never deployed anything. It uses **two free services**:

- **Render** — runs the Python backend (the API).
- **Netlify** — hosts the React frontend (the website).

Both have a free tier with **no credit card** needed to start. Total cost: ₹0.

> One honest caveat about free hosting: Render's free backend **goes to sleep
> after about 15 minutes with no visitors**. The next visit wakes it, which takes
> ~30–50 seconds. After that it's fast again. Fine for a student project or demo;
> just don't panic at the first slow load.

---

## Before you start

1. Put this whole project on **GitHub** (both Render and Netlify deploy from there).
   - Make a free account at github.com.
   - Create a new repository called `CardioSense`.
   - Upload the project (drag-and-drop in the browser works, or use git).
2. Make sure `artifacts/` is committed — it holds the trained model the API
   loads. (It's small; no TensorFlow.) If you retrain, re-commit it.

---

## Part A — the backend on Render

1. Go to **render.com** and sign up with your GitHub account.
2. Click **New → Web Service**, then pick your `cardiosense` repo.
3. Fill in the settings:
   - **Name:** `cardiosense-api` (this becomes part of your URL)
   - **Root Directory:** `backend`
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
   - **Instance Type:** **Free**
4. Click **Create Web Service**. Wait for the log to say "Live" (a few minutes).
5. Copy your backend URL — it looks like
   `https://cardiosense-api.onrender.com`. Test it by opening
   `https://cardiosense-api.onrender.com/api/health` in your browser; you should
   see `{"status":"ok"}`.

That's the backend done.

---

## Part B — the frontend on Netlify

1. Go to **netlify.com** and sign up with GitHub.
2. Click **Add new site → Import an existing project**, pick `cardiosense`.
3. Fill in the build settings:
   - **Base directory:** `frontend`
   - **Build command:** `npm run build`
   - **Publish directory:** `frontend/dist`
4. Before deploying, add **one environment variable** so the website knows where
   the backend lives:
   - Key: `VITE_API_BASE`
   - Value: your Render URL from Part A, e.g. `https://cardiosense-api.onrender.com`
5. Click **Deploy**. When it finishes, Netlify gives you a URL like
   `https://cardiosense.netlify.app` — that's your live app. 🎉

---

## If something breaks

| Symptom | Likely cause | Fix |
|---|---|---|
| Website loads but every assessment errors | `VITE_API_BASE` wrong or missing | Re-check the env var in Netlify, redeploy |
| First request takes ~40s | Render free tier woke from sleep | Normal; subsequent requests are fast |
| Build fails on Netlify | Node version | In Netlify env vars add `NODE_VERSION` = `20` |
| API 500 on predict | `artifacts/` not committed | Commit the `artifacts/` folder and redeploy |
| CORS error in browser console | backend not sending headers | The app already sets CORS headers; confirm the API URL has no trailing slash |

---

## Updating the site later

Both services **auto-deploy when you push to GitHub**. Change code, push, and in
a minute or two the live site updates itself. No manual step.

---

## A note on keeping the backend awake (optional)

If you want to avoid the 15-minute sleep for a demo, a free uptime pinger (e.g.
UptimeRobot) can hit `/api/health` every 10 minutes. Don't leave it running
forever — it's only polite to use it around a demo or submission.
