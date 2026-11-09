# Deploying the fraud detection system

There are two ways to run the whole system:

1. **On your own computer** with Docker: one command.
2. **As a public demo** on the internet: the API and its database on Render, the dashboard on Vercel. Both have free plans and both sign in with GitHub.

Nothing here needs a credit card.

---

## 1. On your own computer

You need Docker Desktop. From the project folder:

```bash
docker compose up --build
```

The first build takes a while (it installs Python, PyTorch and the dashboard). When it is ready:

| What | Address |
|---|---|
| Dashboard | http://localhost:5173 |
| API | http://localhost:8000 |
| API documentation | http://localhost:8000/docs |
| Service status | http://localhost:8000/status |

This starts three containers:

- **db**: PostgreSQL 16. Data is kept in a Docker volume, so it survives a restart.
- **api**: the FastAPI service with two worker processes. Before the workers start, it applies the database migrations and runs the one-time setup (tables, demo users, model versions, demo history).
- **frontend**: the dashboard.

It runs in demo mode, so the login page shows the demo accounts (`analyst` / `analyst123` and `admin` / `admin123`).

To stop it: `Ctrl+C`, then `docker compose down`. To also delete the database: `docker compose down -v`.

---

## 2. Public demo

The plan:

```
visitor's browser ──► dashboard on Vercel (static files)
        │
        └──────────► API on Render (Docker) ──► PostgreSQL on Render
```

The dashboard calls the API directly, so the API has to know the dashboard's address (CORS), and the dashboard has to know the API's address. That is why the steps go back and forth once.

### Step 1: push the code to GitHub

Render and Vercel build from the GitHub repository, so the branch you want to deploy must be pushed first.

```bash
git push origin <your-branch>
```

### Step 2: create the API and database on Render

1. Go to https://dashboard.render.com and sign in with GitHub.
2. Click **New** → **Blueprint**.
3. Choose the repository `Cost-Aware-Fraud-Detection-with-Delay-Aware-Decisioning` and the branch.
4. Render reads `render.yaml` and shows two things to create: the `fraud-api` web service and the `fraud-db` database. It asks for one value, `FRAUD_CORS_ORIGINS`. For now type `http://localhost:5173` (you will change it in step 4).
5. Click **Apply**. The first build takes about 10 minutes.
6. When it says **Live**, copy the API's address, for example `https://fraud-api-xxxx.onrender.com`.
7. Check it works: open `https://fraud-api-xxxx.onrender.com/status`. You should see `"status": "ok"` and `"backend": "postgresql"`.

### Step 3: put the dashboard on Vercel

1. Go to https://vercel.com and sign in with GitHub.
2. Click **Add New** → **Project** and import the same repository.
3. Set **Root Directory** to `dashboard`. Vercel recognises Vite by itself.
4. Under **Environment Variables** add:

   | Name | Value |
   |---|---|
   | `VITE_API_URL` | the API address from step 2, e.g. `https://fraud-api-xxxx.onrender.com` |

5. Click **Deploy**. Copy the dashboard's address, for example `https://fraud-dashboard.vercel.app`.

If you prefer the command line, the same thing is:

```bash
cd dashboard
npx vercel link
npx vercel env add VITE_API_URL production      # paste the API address when asked
npx vercel --prod
```

### Step 4: tell the API where the dashboard is

1. On Render, open `fraud-api` → **Environment**.
2. Change `FRAUD_CORS_ORIGINS` to the dashboard address from step 3, e.g. `https://fraud-dashboard.vercel.app` (no slash at the end).
3. Save. Render restarts the API.

### Step 5: check it

- Open the dashboard address. The login page should show the demo accounts.
- Sign in as `analyst` and open **Live feed**. Transactions should start streaming.
- Open `/status` on the API again and check the model version is the one you expect.

---

## Things to know about the free plans

These were true when this was written (November 2026). Check Render's pricing page if something behaves differently.

- **The API goes to sleep** after about 15 minutes without visitors. The next visit wakes it, which takes about a minute. Before a presentation, open the dashboard a few minutes early.
- **Memory is limited to 512 MB.** The deployed model (V3) has no autoencoder, so the API never loads PyTorch while scoring. If you roll back to V2, which carries the autoencoder, PyTorch is loaded and takes a large share of the memory; if Render's logs then show the service running out of memory, that is the cause.
- **The free PostgreSQL database is deleted after 30 days.** That is enough for the presentation. Redeploying creates a new one; the setup step fills it again on start.

---

## Settings

The API reads its settings from environment variables. The most important ones:

| Variable | What it does | Demo value |
|---|---|---|
| `FRAUD_ENVIRONMENT` | `production` makes the API refuse to start with unsafe settings | `production` |
| `FRAUD_DEMO_MODE` | Shows the demo accounts on the login page; turns off model rollback, so visitors cannot change the model for everyone; threshold proposals are only logged | `true` |
| `FRAUD_DEMO_RESET_HOURS` | In demo mode, how often visitors' work (cases, decisions, replayed transactions, labels, audit log) is cleared. `0` turns the reset off | `24` (default) |
| `FRAUD_JWT_SECRET` | Signs the login tokens. Must be random and at least 32 characters in production | generated by Render |
| `FRAUD_DATABASE_URL` | Database address. `postgres://` addresses from hosting platforms are accepted as they are | from Render |
| `FRAUD_CORS_ORIGINS` | Which websites may call the API (comma-separated) | the Vercel address |
| `FRAUD_ADMIN_PASSWORD`, `FRAUD_ANALYST_PASSWORD` | Passwords for the two accounts. In production they must be changed unless demo mode is on | demo defaults |
| `FRAUD_RATE_LIMIT_PER_MINUTE` | Requests per minute per visitor (0 turns it off) | `300` |
| `FRAUD_LOGIN_ATTEMPTS_PER_MINUTE` | Login attempts per minute per visitor | `20` (default) |
| `WEB_CONCURRENCY` | Number of API worker processes | `1` on the free plan |
| `PORT` | Port to listen on. Hosting platforms set it themselves | set by Render |

The dashboard has one setting, `VITE_API_URL`, used when it is built. Leave it empty locally; the development server forwards `/api` to the API.

---

## Security basics in place

- In production the API **will not start** with the development token secret, with the demo passwords (unless demo mode is on), or with `*` as the allowed website.
- **Passwords** are stored as salted PBKDF2 hashes (200,000 rounds). Login tokens expire after 8 hours.
- **Rate limits**: a general limit per visitor, and a much smaller one for login attempts against password guessing. Each worker process counts separately.
- **Request size**: bodies over 1 MB are refused.
- **Security headers** on every response (`X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`, `Permissions-Policy`, and `Strict-Transport-Security` in production).
- **Roles**: analysts and admins see and can do different things. Every decision is written to the audit log.
- `/status` shows the model version, database health and monitoring status, but never addresses, secrets or transactions. In demo mode it also lists the demo accounts, which are public anyway, and when the demo data is next reset.
- **Demo mode** (`fraud/service/demo.py`): switching the model is refused, threshold proposals only go to the audit log, and once a day visitors' work is cleared and the recipient history is seeded again. Users, model versions and drift reports are kept. On PostgreSQL a database lock makes sure only one worker does the reset.

## Several workers

With `WEB_CONCURRENCY` above 1 there are several API processes. Three things make that safe:

1. The one-time setup runs once, before the workers start (`python -m fraud.service.setup`), so they do not all create the same users and history rows at the same moment.
2. Each worker keeps its own copy of the model. After a rollback, the new active version is recorded in the database and every worker switches to it within 10 seconds.
3. Several workers need PostgreSQL. SQLite only allows one writer at a time, so with SQLite keep one worker.

## What has and has not been checked

- The automated tests (`pytest`) cover the status endpoint, demo mode, the production settings check, the security headers, the size limit, the login limit, running the setup twice, and workers following a rollback.
- The container's start sequence was run on this laptop without Docker, on an empty SQLite database, with the production settings Render uses (`FRAUD_ENVIRONMENT=production`, demo mode, a random token secret, a CORS address). The migrations ran, the setup step prepared the database, and the API served V3: `/status`, sign-in, scoring, the Model lab and Explainability data all worked, with the production security headers. The API process stayed well under Render's 512 MB, and PyTorch was not loaded.
- The API has been run on PostgreSQL 17 with 1, 2 and 4 workers, with the same start-up steps as the container (setup once, then the workers), by `scripts/benchmark_api.py`. Every request in those runs succeeded. The timings are in the README.
- `docker compose up` itself has not been run. Docker Desktop on this laptop fails to start ("Docker Desktop is unable to start"), and drive C: has little free space for the images. To try it: free some space on C: or move Docker's disk image to D: (Docker Desktop → Settings → Resources → Advanced → Disk image location), then run `docker compose up --build`.
- The CI workflow (`.github/workflows/ci.yml`) builds and starts the full `docker compose` stack on PostgreSQL and scores a transaction through it on every push. It has not run on GitHub yet; it will on the first push, and that is the first full test of the Docker and PostgreSQL path.
- The public deployment above has not been done yet. It needs your Render and Vercel accounts.
