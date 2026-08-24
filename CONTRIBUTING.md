# Contributing

Thanks for your interest in the Orbital Rendezvous Simulator. This is research
and educational software; contributions are welcome, and the bar is
correctness, clarity and honest documentation.

## Ground rules

- **The physics engine is protected.** Nothing in `orbital_core/` should change
  its numerical behaviour without a compelling, reproducible reason. Any change
  that could move a result must come with a regression test and a note in the
  changelog explaining what moved and why. The published figures are locked by
  `tests/test_regression_baseline.py` at `rtol = 1e-9`.
- **No overstated claims.** The software has not been benchmarked against GMAT,
  STK or any external tool, and is not NASA/ISRO/flight-certified. Please keep
  documentation and UI copy consistent with that.
- **Privacy by design.** The analytics layer stores no names, emails, exact
  ages, IP addresses or raw orbital parameters. Do not add fields that would.

## Development setup

```bash
python -m venv .venv
# Windows:  .venv\Scripts\Activate.ps1
# Unix:     source .venv/bin/activate
python -m pip install -r requirements.txt

# Web API
python -m pip install -r web-platform/backend/requirements.txt

# Web frontend
cd web-platform/frontend && npm install
```

Run the platform locally per the [README](README.md#4-running-the-platform-locally).

## Before opening a pull request

Run the full suite. All of it must pass.

```bash
# Engine + Streamlit platform (repository root)
python -m pytest -q

# Web API (includes bit-exact engine parity)
cd web-platform/backend && python -m pytest -q

# Web frontend
cd web-platform/frontend && npx tsc --noEmit && npx eslint src --ext .ts,.tsx && npx vitest run && npm run build
```

Please also:

- Keep commits focused and their messages descriptive.
- Match the surrounding code style; the frontend enforces ESLint + `jsx-a11y`,
  the number-formatting rule (no raw numbers in the UI — use the shared
  formatter), and TypeScript `strict`.
- Add or update tests for any behaviour you change.
- Update the relevant document in `docs/` if you change how something works.

## Reporting bugs and requesting features

Use the issue templates. For a suspected numerical defect, include the exact
inputs, the observed output, and the expected output with its source.

## Branching

Active work happens on `v2-platform`. `main` holds the historical V1 Streamlit
MVP and is not a development branch; please do not target pull requests at it.

## Licence

By contributing you agree that your contributions are licensed under the
project's [MIT Licence](LICENSE).
