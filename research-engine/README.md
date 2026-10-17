# Orbital Rendezvous Research Engine

**Start with [the CEAS paper guide](START_HERE.md)** for the scientific scope, evidence dependencies and reproduction route.

This directory contains the code for *Saturation-Aware Trajectory-Error Screening for Controlled LEO Rendezvous*. The study compares nominal CW sensitivity features against matched nonlinear two-body trajectories under continuous PD, held PD (1 s and 10 s) and CW-LQR control, using six cells and frozen conditional-risk selection. It assesses a full-orbit position-discrepancy tolerance of 1 m or 0.1 m. It does not claim a runtime saving or operational safety certification.

The scientific code selected for the paper is tag `research-engine-v3.1.1`, commit `254b07d9abf0ff647f71093ade69fee99e5fb6d1`. The code tag includes selected regression fixtures. The [paper evidence release](https://github.com/AdvikS09/Orbital-Rendezvous-Simulator/releases/tag/ceas-evidence-2026-10-17) separately provides the [supplementary methods](https://github.com/AdvikS09/Orbital-Rendezvous-Simulator/releases/download/ceas-evidence-2026-10-17/ESM_1.pdf), [complete `ESM_2.zip` archive](https://github.com/AdvikS09/Orbital-Rendezvous-Simulator/releases/download/ceas-evidence-2026-10-17/ESM_2.zip), and [checksums](https://github.com/AdvikS09/Orbital-Rendezvous-Simulator/releases/download/ceas-evidence-2026-10-17/SHA256SUMS.txt). A code download alone is insufficient for the paper's stored-data analyses.

The historical platform introduction, including its older RK4/J2/drag examples, is preserved in [docs/historical/README_before_2026-10-17.md](docs/historical/README_before_2026-10-17.md). Those descriptions are not the methods or acceptance criterion of the present six-cell paper. Existing CW-LQR and held-command modules are part of the selected engine, not planned future additions.

This documentation update does not change the scientific engine, selected thresholds, original records or old tag. The accompanying guide distinguishes stored-data reproduction, one-case numerical replay, regression tests and a genuinely new campaign.
