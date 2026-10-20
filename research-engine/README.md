# Research engine

Code for *Saturation-Aware Trajectory-Error Screening for Controlled LEO Rendezvous* (Advik Sushanth, CEAS Space Journal manuscript).

The study compares nominal Clohessy–Wiltshire (CW) sensitivity features with matched nonlinear two-body trajectories under continuous PD, held PD (1 s and 10 s) and CW-LQR control. Frozen screening rules are calibrated in six simulation cells for a full-orbit position tolerance of 1 m or 0.1 m. The study does not claim a runtime saving or an operational safety certificate.

## Layout

| Folder | Contents |
|---|---|
| `cwvalid/` | CW, quadratic and nonlinear relative dynamics, closed-loop simulation and first-order estimates |
| `cwvalid3/` | Held-command engine, sensitivity maps, calibration, record contracts and serialization |
| `indverify/` | Separately coded numerical verifier |
| `experiments/` | Campaign drivers and analyses: `ext/` and `v2/` for the earlier studies, `v3/` for the calibrated study, `v31/` and `v311/` for the later corrections |
| `src/` | The earlier platform engine (RK4, J2, drag) used by the historical comparisons |
| `tests/` | Regression tests, with exact-line fixtures from the stored records |
| `app.py` | Streamlit dashboard for the earlier platform engine |

## Running the tests

Python 3.12 or later:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

Expect 316 passed and 5 skipped. The skipped tests need the complete data archive next to this folder.

## Data and versions

The stored campaign records, and the scripts that rebuild every table and figure in the paper, are in Online Resource 2 of the paper. The code there is the same as this folder (engine identity `66f0b10a9d842165`).

This folder is a tidied form of the code at tag [`research-engine-v3.1.1`](https://github.com/AdvikS09/Orbital-Rendezvous-Simulator/tree/research-engine-v3.1.1/research-engine) (commit `254b07d9abf0`), which the paper cites. Identifiers, files and folders were renamed and unused code was removed, but the logic is unchanged. The original source stays available at that tag.
