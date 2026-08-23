# Troubleshooting

Symptoms, causes, fixes. Roughly ordered by how often they come up.

**First stop for anything deployment-related:** the administrator area, System
tab (`/?view=admin`). It reports database health, the schema version and every
configuration warning, and it never displays a secret.

---

## 1. The dashboard shows zeros but people definitely used it

**Almost always:** `PILOT_ENV` is not `production`.

Any value other than `production` — including unset — flags every record as demo
data, and every report excludes demo data by default. This is deliberate: it is
what stops your own testing from contaminating pilot figures.

**Check:** Admin → System → "Environment". **Fix:** set `PILOT_ENV=production`
and restart.

Records already written as demo stay flagged. To see them, tick *Include
development / demo records* in the sidebar — but they are not pilot data and the
page says so while it is on.

**Other causes:**

* The date filter excludes the activity. Try *All time*.
* An organisation filter is set to one with no data.
* Nobody **completed** a simulation. Signing in is not enough — "unique user"
  requires at least one completed run.

---

## 2. All the pilot data disappeared after a restart

**Cause:** SQLite on an ephemeral filesystem. Streamlit Community Cloud and most
container hosts wipe the app's disk on every restart and redeploy.

**Check:** Admin → System. If the database line starts `sqlite (`, this is it.
There is also a standing warning in production.

**Fix:** set `DATABASE_URL` to managed Postgres and redeploy. Data already lost
cannot be recovered. See [DEPLOYMENT.md](DEPLOYMENT.md) §0.

---

## 3. Returning participants are not recognised; return rate stuck at zero

**Cause:** `PILOT_PARTICIPANT_SALT` is unset, so a random salt is generated per
process. The same person typing the same code after a restart hashes to a
different value and looks new.

**Check:** Admin → System shows a warning in production.

**Fix:** generate one, set it, never change it:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

**Also check:** "returning" requires **two separate dates**. Two sessions the
same afternoon is not returning, by definition.

Changing the salt later has the same effect as never setting one. If you have
already changed it, earlier participants are unrecoverable — the old hashes
cannot be recomputed.

---

## 4. Cannot sign in to the administrator area

| Message | Cause | Fix |
| --- | --- | --- |
| "No administrator credentials are configured" | Neither `PILOT_ADMIN_PASSWORD_HASH` nor `PILOT_ADMIN_PASSWORD` is set | Set them and restart. There is no default password by design. |
| "Incorrect username or password. N attempt(s) remaining" | Wrong credentials, or the hash was truncated when pasted | Regenerate with `python scripts/hash_password.py` and paste the whole `pbkdf2_sha256$...` string |
| "Too many failed attempts" | Five failures locked the form | Wait five minutes, or restart the app to clear the in-process counter |

The hash contains `$` characters. Some shells and secret managers interpret
those — quote it, and check the value that actually arrived is complete.

---

## 5. "Database problem" on the System tab

Read the error class shown; the message never contains the password.

| Error | Likely cause |
| --- | --- |
| `ModuleNotFoundError: psycopg` | The Postgres driver is not installed. Uncomment `psycopg[binary]` in `requirements.txt` and redeploy. |
| `OperationalError: could not translate host name` | Wrong host in `DATABASE_URL`. |
| `OperationalError: password authentication failed` | Wrong credentials. |
| `OperationalError: SSL required` | Append `?sslmode=require`. |
| `ArgumentError: Could not parse SQLAlchemy URL` | Scheme wrong. Use `postgresql+psycopg://`, not `postgres://`. |

Test before deploying:

```bash
DATABASE_URL='postgresql+psycopg://...' python scripts/init_db.py --check
```

---

## 6. A participant sees a red error when they run something

That is the input validator, and it is working. It catches inputs that would
otherwise produce a **silently wrong answer** — a negative altitude gives an
all-`NaN` trajectory, an over-long horizon gets quietly truncated.

The message says which value is out of range and what the range is.

| Message | Meaning |
| --- | --- |
| "Altitude must be at least 150 km" | Below that a spacecraft is re-entering and the atmosphere model is meaningless |
| "must be a number, not NaN" | A field was left in a bad state; retype it |
| "Duration must be at most 50 orbits" | Long studies belong on a local install, not a shared server |
| "This run needs N integration steps, above the limit" | Increase the timestep or shorten the duration |
| "estimated to take about N s, above the limit" | Same |
| "starts exactly on the target with zero relative velocity" | There is nothing to simulate; offset an axis |

To allow heavier runs, raise `PILOT_MAX_SIMULATION_STEPS` and
`PILOT_MAX_ESTIMATED_SECONDS` — but see §8 first.

---

## 7. "Too many simulations in under a minute"

The rate limiter. Defaults: 20 runs/minute and 300 per session.

Usually a stuck button or a genuinely enthusiastic user. Adjust with
`PILOT_MAX_RUNS_PER_MINUTE` / `PILOT_MAX_RUNS_PER_SESSION`.

**Note:** the limiter is in-process. With multiple app replicas each keeps its
own counters, so the effective limit is multiplied by the replica count.

---

## 8. The app is slow, or several users are blocking each other

The engine is pure-Python loops and Streamlit runs one process. While a
simulation runs, that worker is busy and other users wait.

Measured costs: a typical run is 0.07–0.5 s; the CW-versus-reference comparison
costs about 0.4 ms per step.

**Fixes, in order:**

1. Lower `PILOT_MAX_SIMULATION_STEPS` (e.g. 20 000) and
   `PILOT_MAX_ESTIMATED_SECONDS` (e.g. 10).
2. Lower `PILOT_MAX_RUNS_PER_MINUTE`.
3. Move off Community Cloud to a container host with several replicas sharing
   one Postgres database.

On Community Cloud the app also sleeps after inactivity; the first visitor
afterwards waits ~30 s. That is the free tier, not a fault.

---

## 9. A plot is empty or looks wrong

* **Check the warnings first.** An under-resolved timestep or an out-of-range
  separation explains most odd-looking results.
* **A flat line is often correct.** In the two-body scenario the altitude plot
  *should* be flat — that is the point.
* **Look at the axis range.** A "flat" line may be varying by 10⁻⁹ km.
* If the message says the results could not be drawn but the simulation
  completed, the numbers are still downloadable. Please report it.

---

## 10. Something failed and the count went up

`simulation_failed` means the engine raised an exception. The participant sees a
plain-language message, never a traceback.

**Find it:** Admin → System → Recent events. The `error_class` column names the
exception type. The message is deliberately not stored, because it could echo
participant input.

Reproduce locally with the same scenario and parameters, where the full
traceback is available.

**Not the same as** `simulation_rejected`, which means validation or rate
limiting blocked it before the engine ran. Rejections are not failures, and are
counted separately.

---

## 11. Tests fail after a change

**`tests/test_regression_baseline.py` fails.** Stop. This means a numerical
result the research depends on has moved. Do **not** regenerate the baseline to
get a green suite — find out why. The failure message shows the value, the
baseline, and the tolerance.

Legitimate causes: a genuine engine change (then the baseline is regenerated
deliberately and the change documented), or a different numpy/BLAS build
producing last-bit differences (tolerance is `rtol=1e-9`, far looser than that
should need).

**`tests/test_integrator_convergence.py` fails.** The RK4 order test measures
observed order between 3.6 and 4.4. At very fine steps, round-off can exceed
truncation error and drag the measured order down — a limit of floating-point
arithmetic, not of the method.

**Platform tests fail with a database error.** Each test uses a temporary SQLite
file. On Windows a previous run may still hold one; the fixture ignores
unlink failures, so a stale temp file is harmless.

---

## 12. Locally: `ModuleNotFoundError: No module named 'src'` or `'pilot'`

Run from the project root, or with the root on `PYTHONPATH`:

```bash
cd "Orbital Stability Platform"
python -m pytest -q
```

`pilot_app.py` inserts its own directory on `sys.path`, so `streamlit run
pilot_app.py` works from anywhere.

---

## 13. Something looks like a privacy problem

Check first — several things look alarming and are not:

* `participant_ref` (e.g. `69d24108`) is the first 8 characters of an
  irreversible hash, not a code anyone typed.
* Free-text feedback **is** stored verbatim. If a participant typed personal
  information, it is in there. Delete that row.
* `PILOT_LOG_RAW_PARAMETERS=true` stores the values participants type. It is off
  by default; if it is on, both the consent page and the admin area say so.

If something genuinely leaked, delete the affected rows, and note that anonymity
means you cannot identify who was affected in order to tell them.
