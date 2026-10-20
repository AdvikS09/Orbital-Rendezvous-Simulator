import argparse
import json
import os
import sys
import tempfile

FIXTURES = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tests", "fixtures", "v31")


class ProcLoss(BaseException):
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code-root", required=True)
    a = ap.parse_args()
    root = os.path.abspath(a.code_root)
    sys.path.insert(0, root)
    sys.path.insert(0, os.path.join(root, "experiments", "v3"))
    import campaign as CP
    import cwvalid3.heldmet as HM
    import cwvalid3.sampled as SA
    from cwvalid3 import contracts as C
    m = json.loads(open(os.path.join(FIXTURES, "lines", "raw", "manifests", "D4test.jsonl"), encoding="utf-8").readline())
    assert m["case_id"] == "Caf93a5f0b849d73a"
    thr = json.load(open(os.path.join(FIXTURES, "whole", "raw", "freeze.json"), encoding="utf-8"))["selection"]["D4|A_H|1m"]["threshold"]
    os.environ.pop(CP.CKPTENV, None)
    out = {"code_root": root, "engine_sha": CP.enginever(), "contract": C.CONTRVER, "case_id": m["case_id"],
           "frozen_D4_A_H_1m_cutoff": thr}

    def summary(o):
        v, st, why = C.featavail(o, "A_H", "held")
        return {"worker_status": o.get("worker_status"), "failed_stage": o.get("failed_stage"), "A_H_peak_m_in_record": o.get("A_H_peak_m"),
                "A_H_availability": st, "A_H_reason": why, "A_H_accepted_at_1m": bool(st == "ok" and v <= thr),
                "A_naive_status": o.get("A_naive_status"), "truth": C.truth_record(o, "held")["truth_status"],
                "reference_metrics_present": "cw_vs_eci_peak_m" in o, "feature_errors": sorted(o.get("feature_errors", {})),
                "feature_recovery": o.get("feature_recovery")}
    out["normal"] = summary(CP.workheld([m])[0])
    real = HM.estimfeat

    def failnaive(cv, e):
        if e == "A_naive":
            raise RuntimeError("injected A_naive extraction failure")
        return real(cv, e)
    HM.estimfeat = failnaive
    try:
        out["R01_A_naive_extraction_failure"] = summary(CP.workheld([m])[0])
    finally:
        HM.estimfeat = real
    ck = tempfile.mkdtemp(prefix="r02_ck_")
    os.environ[CP.CKPTENV] = ck
    realrun = SA.HeldBatch.run

    def run(self):
        r = realrun(self)
        if self.haseci:
            raise ProcLoss("simulated process loss after the combined pass")
        return r
    SA.HeldBatch.run = run
    try:
        CP.workheld([m])
        lost = False
    except ProcLoss:
        lost = True
    finally:
        SA.HeldBatch.run = realrun
        os.environ.pop(CP.CKPTENV, None)
    crash = CP.crashrec(m, RuntimeError("BrokenProcessPool (simulated)"), ck)
    r02 = {"loss_simulated": lost, "checkpoint_files": os.listdir(ck), "crash_record": summary(crash)}
    if hasattr(CP, "recproc"):
        got = CP.recproc([m])[m["run_id"]]
        r02["after_driver_nominal_recovery"] = summary(CP.mergefeats(crash, got, "driver_nominal_recovery")) \
            if "feature_recovery_error" not in got else {"feature_recovery_error": got["feature_recovery_error"]}
    out["R02_process_loss_after_combined_pass"] = r02
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
