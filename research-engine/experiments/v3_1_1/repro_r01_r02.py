"""Reproduce the follow-up findings R01 and R02 on the counterexample D4 test index 0 (Caf93a5f0b849d73a) with the workers
of a given code tree (--code-root: a worktree of tag v3.1 for 'before', the v3.1.1 tree for 'after').

The planned case is the exact fixture line of the original manifest (tests/fixtures/v31 of this tree); the frozen cutoff is
the fixture copy of calibration_freeze.json. Nothing is written except the JSON on stdout.
  normal  worker output without any fault: worker_status, A_H, its frozen 1 m decision, truth;
  R01     RuntimeError injected only in held_metrics.estimator_feature(cv, 'A_naive');
  R02     a BaseException raised right after the combined (cw + nerm + eci) pass returns, standing in for a process loss in a
          single process: are there checkpoints, and is A_H available in the crash record? With a v3.1.1 tree the
          driver-level nominal-only recovery is then run in a fresh process (as cmd_run does) and merged.
Usage: python repro_r01_r02.py --code-root <code dir>
"""

import argparse
import json
import os
import sys
import tempfile

FIXTURES = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tests", "fixtures", "v31")


class ProcessLoss(BaseException):
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code-root", required=True)
    a = ap.parse_args()
    root = os.path.abspath(a.code_root)
    sys.path.insert(0, root)
    sys.path.insert(0, os.path.join(root, "experiments", "v3"))
    import campaign as CP
    import cwvalid3.held_metrics as HM
    import cwvalid3.sampled as SA
    from cwvalid3 import contracts as C
    m = json.loads(open(os.path.join(FIXTURES, "lines", "raw", "manifests", "D4_test.jsonl"), encoding="utf-8").readline())
    assert m["case_id"] == "Caf93a5f0b849d73a"
    thr = json.load(open(os.path.join(FIXTURES, "whole", "raw", "calibration_freeze.json"), encoding="utf-8"))["selection"]["D4|A_H|1m"]["threshold"]
    os.environ.pop(CP.CHECKPOINT_ENV, None)
    out = {"code_root": root, "engine_sha": CP.engine_version(), "contract": C.CONTRACT_VERSION, "case_id": m["case_id"],
           "frozen_D4_A_H_1m_cutoff": thr}

    def summary(o):
        v, st, why = C.feature_availability(o, "A_H", "held")
        return {"worker_status": o.get("worker_status"), "failed_stage": o.get("failed_stage"), "A_H_peak_m_in_record": o.get("A_H_peak_m"),
                "A_H_availability": st, "A_H_reason": why, "A_H_accepted_at_1m": bool(st == "ok" and v <= thr),
                "A_naive_status": o.get("A_naive_status"), "truth": C.truth_record(o, "held")["truth_status"],
                "reference_metrics_present": "cw_vs_eci_peak_m" in o, "feature_errors": sorted(o.get("feature_errors", {})),
                "feature_recovery": o.get("feature_recovery")}
    out["normal"] = summary(CP.work_held([m])[0])
    real = HM.estimator_feature

    def fail_naive(cv, e):
        if e == "A_naive":
            raise RuntimeError("injected A_naive extraction failure")
        return real(cv, e)
    HM.estimator_feature = fail_naive
    try:
        out["R01_A_naive_extraction_failure"] = summary(CP.work_held([m])[0])
    finally:
        HM.estimator_feature = real
    ck = tempfile.mkdtemp(prefix="r02_ck_")
    os.environ[CP.CHECKPOINT_ENV] = ck
    real_run = SA.HeldBatch.run

    def run(self):
        r = real_run(self)
        if self.has_eci:
            raise ProcessLoss("simulated process loss after the combined pass")
        return r
    SA.HeldBatch.run = run
    try:
        CP.work_held([m])
        lost = False
    except ProcessLoss:
        lost = True
    finally:
        SA.HeldBatch.run = real_run
        os.environ.pop(CP.CHECKPOINT_ENV, None)
    crash = CP._crash_record(m, RuntimeError("BrokenProcessPool (simulated)"), ck)
    r02 = {"loss_simulated": lost, "checkpoint_files": os.listdir(ck), "crash_record": summary(crash)}
    if hasattr(CP, "_recover_in_fresh_process"):
        got = CP._recover_in_fresh_process([m])[m["run_id"]]
        r02["after_driver_nominal_recovery"] = summary(CP._merge_features(crash, got, "driver_nominal_recovery")) \
            if "feature_recovery_error" not in got else {"feature_recovery_error": got["feature_recovery_error"]}
    out["R02_process_loss_after_combined_pass"] = r02
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
