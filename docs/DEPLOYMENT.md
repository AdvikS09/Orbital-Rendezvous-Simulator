# Deployment guide

How to put the pilot platform on a public URL that participants can use without
installing anything.

Two supported routes: **Streamlit Community Cloud** (free, easiest, recommended
for a pilot) and **Docker** (any host).

---

## 0. Before you deploy

You need three things.

### A database

The platform falls back to a local SQLite file, which is fine on your laptop and
**wrong for a hosted pilot**: Streamlit Community Cloud and most container hosts
give the app an ephemeral filesystem that is wiped on every restart and every
redeploy. You would lose all pilot data, silently, and only notice when the
dashboard showed zeros.

Use managed Postgres. Free tiers that work: [Neon](https://neon.tech),
[Supabase](https://supabase.com), [Railway](https://railway.app). You need the
connection string, in this shape:

```
postgresql+psycopg://USER:PASSWORD@HOST:5432/DBNAME?sslmode=require
```

Note the `+psycopg` driver prefix. If your provider gives you a plain
`postgres://...` URL, rewrite the scheme.

### A participant salt

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

Generate it once, **before** the pilot starts, and never change it. Changing it
makes every returning participant look like a new one and destroys your
return-rate figures.

### Administrator credentials

```bash
python scripts/hash_password.py
```

Enter a password of at least 12 characters. It prints a
`PILOT_ADMIN_PASSWORD_HASH=` line — that is what goes in your secrets, so the
plaintext never exists in the deployment environment.

---

## 1. Streamlit Community Cloud (recommended)

Free, gives you a public `https://` URL, redeploys on push. Limits: one process,
about 1 GB of RAM, and the app sleeps after inactivity (first visitor after a
sleep waits ~30 seconds).

1. **Push the repository to GitHub.** Public or private both work.

2. **Create the app** at [share.streamlit.io](https://share.streamlit.io):
   * Repository: your fork
   * Branch: `pilot-platform` (or `main` once merged)
   * **Main file path: `pilot_app.py`** — not `app.py`, which is the original
     research MVP

3. **Set the secrets.** In the app's *Settings → Secrets*, paste TOML. Streamlit
   exposes these as environment variables, which is what the platform reads:

   ```toml
   PILOT_ENV = "production"
   DATABASE_URL = "postgresql+psycopg://USER:PASSWORD@HOST:5432/DBNAME?sslmode=require"
   PILOT_PARTICIPANT_SALT = "your-64-hex-character-salt"
   PILOT_ADMIN_USERNAME = "coordinator"
   PILOT_ADMIN_PASSWORD_HASH = "pbkdf2_sha256$240000$...$..."
   PILOT_APP_VERSION = "1.0.0"
   ```

   `PILOT_ENV = "production"` matters. Without it, every record is flagged as
   demo data and excluded from your reports.

4. **Add the Postgres driver.** `psycopg` is commented out in
   `requirements.txt` so local development needs no compiler. Uncomment it:

   ```
   psycopg[binary]==3.2.3
   ```

   Commit and push. Streamlit rebuilds.

5. **Verify.** Open the app, then `https://YOUR-APP.streamlit.app/?view=admin`,
   sign in, and open the **System** tab. It must say *"Database reachable and
   migrated"* and show **no configuration warnings**. Warnings there are the
   fastest way to catch a misconfigured deployment.

6. **Register your organisations** (optional but recommended — see
   [PILOT_COORDINATOR.md](PILOT_COORDINATOR.md)).

---

## 2. Docker

```bash
docker build -t orbital-pilot .

docker run -d -p 8501:8501 \
  -e PILOT_ENV=production \
  -e DATABASE_URL='postgresql+psycopg://USER:PASSWORD@HOST:5432/DBNAME' \
  -e PILOT_PARTICIPANT_SALT='your-64-hex-salt' \
  -e PILOT_ADMIN_USERNAME='coordinator' \
  -e PILOT_ADMIN_PASSWORD_HASH='pbkdf2_sha256$240000$...$...' \
  --name orbital-pilot orbital-pilot
```

Or `--env-file .env` if you prefer a file (it is git-ignored).

The image runs as a non-root user, includes `psycopg`, and **runs the numerical
regression tests during the build** — an image whose engine outputs have moved
cannot be produced. It has a `HEALTHCHECK` against Streamlit's own health
endpoint.

Put a TLS-terminating reverse proxy in front of it. Streamlit uses WebSockets,
so the proxy must be configured to upgrade connections. Minimal nginx:

```nginx
location / {
    proxy_pass http://127.0.0.1:8501;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_set_header Host $host;
    proxy_read_timeout 86400;
}
```

---

## 3. Database setup

The app migrates automatically on startup, so usually nothing is needed. To do
it as an explicit release step, or to check a connection string before going
live:

```bash
python scripts/init_db.py --check    # report only
python scripts/init_db.py            # apply migrations
```

`--check` reports the schema version and any missing tables without changing
anything. Both print a password-free description of the database.

---

## 4. Post-deployment checklist

| Check | How |
| --- | --- |
| App loads | Open the public URL; the landing page renders |
| Database connected | Admin → System says "reachable and migrated" |
| No configuration warnings | Admin → System shows none |
| Records are real, not demo | Admin → System shows `Environment: production` |
| Admin locked down | Sign out, try a wrong password, confirm refusal |
| A run is counted | Complete one simulation; the dashboard shows 1 completed run |
| Return visits work | Sign in with the same code tomorrow; return rate rises |
| No secrets in the repo | `git grep -nE 'PILOT_ADMIN_PASSWORD=|postgres.*:.*@'` |

Then work through [`MANUAL_REVIEW_CHECKLIST.md`](../MANUAL_REVIEW_CHECKLIST.md).

---

## 5. Capacity, honestly

The engine is pure-Python loops and Streamlit runs one process.

* A typical interactive run is 0.07–0.5 s.
* The heaviest permitted run is capped at roughly 20 s by the predicted-cost
  check.
* While a simulation runs, that worker is busy. Other users wait.

For a classroom of ~30 working through guided activities — mostly reading, with
short runs — a single Community Cloud instance is adequate. For a large
simultaneous cohort all running the heaviest comparison scenario, it will not
be. Mitigations, in order of preference:

1. Lower `PILOT_MAX_SIMULATION_STEPS` (e.g. 20 000) and
   `PILOT_MAX_ESTIMATED_SECONDS` (e.g. 10).
2. Lower `PILOT_MAX_RUNS_PER_MINUTE`.
3. Move to a container host and run several replicas behind a load balancer,
   sharing the Postgres database. **Note:** the rate limiter is in-process, so
   with N replicas the effective limit is N times the configured one.

This has not been load-tested. See `KNOWN_GAPS.md`.

---

## 6. Updating a live pilot

1. Run the full suite locally: `python -m pytest -q`.
2. Confirm `tests/test_regression_baseline.py` passes — that is the guard that
   the physics has not moved.
3. Bump `PILOT_APP_VERSION` so records are attributable to the new build.
4. Push. Streamlit redeploys; for Docker, rebuild and restart.
5. Migrations apply automatically on startup and are idempotent.

Never edit an applied migration — append a new numbered one in `pilot/db.py`.

---

## 7. Backups

Managed Postgres providers offer automated backups; turn them on. The pilot
dataset is small, so a periodic logical dump is also cheap:

```bash
pg_dump "$DATABASE_URL" > pilot_backup_$(date +%F).sql
```

Store backups somewhere consistent with your retention policy — a backup that
outlives `PILOT_RETENTION_DAYS` quietly breaks the promise made on the consent
page.
