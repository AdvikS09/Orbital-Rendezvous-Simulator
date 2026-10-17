# Start here — CEAS manuscript and reproduction

## Scope of this paper

The main question is whether a saturation-aware sensitivity calculated along a nominal CW trajectory provides a useful modelling-adequacy screen under a fixed finite-family conditional-risk rule. The result is conditional on the stated six-cell simulation distribution and numerical assessment contract. Established saturation, variational and sampled-data ideas are applied together; no universal optimum, certified continuous-time error enclosure, flight-safety result or runtime saving is asserted.

Cells D0-D2 use continuous PD at 400/600/800 km; D3-D4 use PD held for 1/10 s at 400 km; D5 uses a CW-LQR gain at 400 km. The primary matched-physics experiment excludes J2 and drag. Continuous propagation uses the documented SciPy DOP853 route; held propagation and estimator grids are documented in Online Resource 1. A case is adverse if its full-horizon position discrepancy exceeds the target, it exits physically, or its truth outcome is incomplete/unresolved. An unavailable nominal feature causes referral. Truth failure alone must not erase an already available screening feature.

## Get the selected code

```bash
git clone --branch research-engine-v3.1.1 --single-branch https://github.com/AdvikS09/Orbital-Rendezvous-Simulator.git
cd Orbital-Rendezvous-Simulator/research-engine
git rev-parse HEAD
```

The expected commit is `254b07d9abf0ff647f71093ade69fee99e5fb6d1`. The public release URL is https://github.com/AdvikS09/Orbital-Rendezvous-Simulator/releases/tag/research-engine-v3.1.1. Do not mistake its code ZIP for the complete campaign-data archive.

## Get the matching evidence

The evidence package is identified by release [`ceas-evidence-2026-10-17`](https://github.com/AdvikS09/Orbital-Rendezvous-Simulator/releases/tag/ceas-evidence-2026-10-17):

- [Online Resource 1 — supplementary methods (`ESM_1.pdf`)](https://github.com/AdvikS09/Orbital-Rendezvous-Simulator/releases/download/ceas-evidence-2026-10-17/ESM_1.pdf)
- [Online Resource 2 — complete code and stored evidence (`ESM_2.zip`)](https://github.com/AdvikS09/Orbital-Rendezvous-Simulator/releases/download/ceas-evidence-2026-10-17/ESM_2.zip)
- [SHA-256 checksums (`SHA256SUMS.txt`)](https://github.com/AdvikS09/Orbital-Rendezvous-Simulator/releases/download/ceas-evidence-2026-10-17/SHA256SUMS.txt)

`ESM_2.zip` includes the editable supplementary source, all 64 calibrated families, the 45,000-case calibration/test feature and status index, exact manifests and selected cutoffs, corrected overlays, and the complete original evidence. Verify downloaded bytes against `SHA256SUMS.txt`. The selected scientific code tag remains unchanged; this separate release packages the paper's evidence and reader support.

Extract `ESM_2.zip` into a new directory and follow its root `START_HERE.md` and `revision_2026_10_17/REPRODUCE.md`. Its assembler places `code/` and `v3_extension/` as siblings in a disposable working tree, the layout expected by `v3data.py`, `v31paths.py` and `v311paths.py`. That archive contains 5000 calibration and 2500 test cases per cell, the frozen plan/candidates/cutoffs, exact random draws and gains, all original interruptions, 341 held-exit overlays, separately named sensitivity completions and numerical-audit records.

Stored-data reconstruction checks the frozen selection without a new propagation campaign. The C07 correction harness is the supported route for rebuilding closed v3 results; the standalone v3 writers intentionally refuse to overwrite preserved outputs. The current figure code exports all plotted data. A separate command replays one exact manifest case. A new campaign belongs in a new output tree and needs its own protocol; never regenerate a prospective plan from inspected test outcomes.

## Environment and test scope

The paper depends on NumPy, SciPy, pandas and Matplotlib; pytest is required for tests. The archive records both the original Windows environment and the actual later verification environment. The old claim that the engine only needs three libraries or uses RK4 throughout is historical. The repository-wide collected tests and the eight-file 151-test follow-up scope are different. Use the complete commands and actual logs in the evidence archive. Both the earlier integration and 17 October repeat of the focused 151-test scope gave 145 passed, four skipped and two stored-platform comparison failures, with no tolerance widening. Do not advertise an all-pass result based on that run.

## Historical material

`docs/historical/README_before_2026-10-17.md` preserves the earlier platform introduction. Older RK4/J2/drag comparisons, a few-percent-separation heuristic, and old test counts are not the protocol of the present paper. LQR is implemented in the current selected engine. The previous OJSS paper remains cited and is mapped result by result in Online Resource 1; this guide does not decide editorial eligibility.
