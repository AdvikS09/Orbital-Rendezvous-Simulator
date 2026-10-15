"""C05: v3.1 LaTeX result snippets: v3.0 reports/handoff_snippets.tex (unchanged) with [R1]-[R4] reworded.
v3.1.1 (R03): the held timing in [R2] is stated as one 40-case batch per held cell, timed as the median of three
repetitions (no dispersion archived), instead of 'a single timing'. Writes handoff_snippets_v3.1.1.tex into --out
(default v3_1_1_followup/corrected_docs/reports; the released v3.1 file is kept unchanged).
Usage: python code/experiments/v3_1/make_snippets_v31.py [--out DIR]
"""
import argparse
import os

import v31paths as P

R = P.V3
s = open(os.path.join(R, "reports", "handoff_snippets.tex"), encoding="utf-8").read()
head = ("% v3.1.1 corrected copy (follow-up R03: held timing provenance in [R2]); otherwise as the v3.1 release.\n"
        "% v3.1 correction release: copy of reports/handoff_snippets.tex (v3.0, unchanged) with the result snippets\n"
        "% [R1]-[R4] reworded per CORRECTION_REPORT C05: 'same conditional-risk ceiling' instead of 'certified risk',\n"
        "% 'no observed adverse acceptance' instead of 'no adverse case', observational transfer, descriptive held\n"
        "% screening comparison, cost basis of the held timings. [S1]-[S4] are unchanged. Numbers are unchanged.\n")
reps = [
 ("58 of the 64 method--cell--target families reached the stated certificate on 5000 calibration cases per cell; the six saturated-time-fraction rules for the 0.1 m target did not and refer every case.",
  "58 of the 64 method--cell--target families had an admissible candidate ($U\\leq\\alpha$) on 5000 calibration cases per cell; the six saturated-time-fraction families for the 0.1 m target had none and refer every case."),
 ("the selected A threshold is 0.7998 m, which is also the archived threshold and the largest admissible candidate",
  "the best admissible frozen A candidate is 0.7998 m, which is also the archived threshold (the archived padding itself carries no certificate)"),
 ("B's selected threshold is 0.0271 m", "B's best admissible frozen candidate is 0.0271 m"),
 ("Applied once to 2500 untouched test cases, A accepts 1645 cases with no adverse case and B accepts 1421 with seven trajectories exceeding 1 m.",
  "Applied once to 2500 untouched test cases, A accepts 1645 cases with no observed adverse acceptance and B accepts 1421, seven of which exceed 1 m."),
 ("At 0.1 m the corresponding acceptances are 1440 (no adverse case) and 1184 (three). Test intervals are pointwise and descriptive; the simultaneous statement is the calibration certificate.",
  "At 0.1 m the corresponding acceptances are 1440 (no observed adverse acceptance) and 1184 (three). Test intervals are pointwise and descriptive; the simultaneous statement is the calibration bound, which holds with probability at least $1-\\delta$ over calibration sampling under the simulation distribution."),
 ("give the same accept/refer decisions in all but one of 5000 test cases, at the same certified risk.",
  "made the same accept/refer decisions in all but one of 5000 test cases, each selected under the same conditional-risk ceiling. No equivalence margin was set in advance, so this is a description rather than an equivalence test."),
 ("Timed in one process, screening costs 0.47 s per case from scratch (0.33 s when the CW run is already available), against 0.17 s for propagating the nonlinear relative model directly, so the screen offers no computational saving.",
  "Timed in one process, screening costs 0.47 s per D0 case from scratch (0.33 s when the CW run is already available), including the referrals, against 0.17 s for propagating the nonlinear relative model directly, so the screen offers no computational saving for any of the timed features (A, B and C); in one 40-case batch per held cell, timed as the median of three repetitions, the screen alone, before any referral work, also took slightly longer than direct propagation."),
 ("with no case exceeding 1 m in any of these sets. Recalibrating in each cell selected the same threshold except for the LQR gain, where a 1.024 m threshold was certified and admitted three test cases between 1.009 and 1.022 m.",
  "with no observed adverse acceptance in any of these sets. Recalibrating in each cell selected the same threshold except for the LQR gain, where a 1.024 m threshold was selected under the same ceiling and admitted three test cases between 1.009 and 1.022 m."),
 ("Calibrated against an explicit conditional-risk target, the saturation-aware estimate retains its advantage: at the same certified risk it accepts about nine percentage points more of a fresh random sample than the unsaturated-sensitivity alternative, with no observed 1 m violation among its acceptances.",
  "Calibrated against an explicit conditional-risk ceiling, the saturation-aware estimate retains its advantage: under the same ceiling its best admissible frozen threshold accepts about nine percentage points more of a fresh random sample than the unsaturated-sensitivity alternative, with no observed adverse acceptance."),
 ("The fixed threshold carried over to the tested altitudes, hold periods and LQR gain without an observed violation.",
  "Applied unchanged at the tested altitudes, hold periods and with the LQR gain, the threshold showed no observed adverse acceptance; this is observational evidence in those distributions, not a certificate there."),
]
for a, b in reps:
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)
ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.join(P.V311_DOCS, "reports"))
out = ap.parse_args().out
os.makedirs(P.guard_docs(out), exist_ok=True)
open(P.guard_docs(os.path.join(out, "handoff_snippets_v3.1.1.tex")), "w", encoding="utf-8", newline="\n").write(head + s)
print("ok")
