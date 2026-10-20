import numpy as np
import pandas as pd

from . import cw, lvlh, orbit
from .controller import PDCtrl


def trap(y, x):
    func = getattr(np, "trapezoid", None) or np.trapz
    return func(y, x)


def energyseries(states, mu=orbit.MUEARTH):
    return np.array([orbit.specenergy(s, mu) for s in states])


def angmomseries(states):
    return np.array([orbit.specangmom(s) for s in states])


def reldrift(series):
    series = np.asarray(series, dtype=float)
    ref = series[0]
    return series - ref if ref == 0.0 else (series - ref) / abs(ref)


def checkorbit(times, states, mu=orbit.MUEARTH, analperiod=None):
    energy = energyseries(states, mu)
    angmom = angmomseries(states)
    measured = orbit.measperiod(times, states)
    row = {
        "max_energy_drift_rel": float(np.max(np.abs(reldrift(energy)))),
        "max_ang_mom_drift_rel": float(np.max(np.abs(reldrift(angmom)))),
        "measured_period_s": measured,
        "analytic_period_s": analperiod if analperiod is not None else np.nan,
    }
    if analperiod is not None and np.isfinite(measured):
        row["period_rel_err"] = abs(measured - analperiod) / analperiod
    else:
        row["period_rel_err"] = np.nan
    return pd.DataFrame([row])


def poserrseries(states, target=None):
    pos = np.asarray(states)[:, :3]
    tgt = np.zeros(3) if target is None else np.asarray(target, dtype=float)[:3]
    return np.linalg.norm(pos - tgt, axis=1)


def velerrseries(states, target=None):
    vel = np.asarray(states)[:, 3:]
    tgt = np.zeros(3) if target is None else np.asarray(target, dtype=float)[3:]
    return np.linalg.norm(vel - tgt, axis=1)


def ctrlaccseries(controls):
    return np.linalg.norm(np.asarray(controls, dtype=float), axis=1)


def cumdeltav(times, controls):
    return float(trap(ctrlaccseries(controls), np.asarray(times, float)))


def convtime(times, pos_error, tol):
    times = np.asarray(times, dtype=float)
    pos_error = np.asarray(pos_error, dtype=float)
    above = np.where(pos_error > tol)[0]
    if above.size == 0:
        return 0.0
    last = above[-1]
    if last >= times.size - 1:
        return np.nan
    return float(times[last + 1])


def rdvmetrics(times, states, controls, target=None, pos_tol=1.0):
    perr = poserrseries(states, target)
    verr = velerrseries(states, target)
    accel = ctrlaccseries(controls)
    return pd.DataFrame([{
        "initial_pos_error_m": float(perr[0]),
        "final_pos_error_m": float(perr[-1]),
        "final_vel_error_mps": float(verr[-1]),
        "convergence_time_s": convtime(times, perr, pos_tol),
        "max_control_accel_mps2": float(np.max(accel)),
        "delta_v_mps": cumdeltav(times, controls),
    }])


def highfidrel(tgtstate0, relstate0, times, force_model, controller=None):
    tgtstate0 = np.asarray(tgtstate0, dtype=float)
    dep = lvlh.lvlhinert(tgtstate0, relstate0)
    times = np.asarray(times, dtype=float)

    def combderiv(z, t):
        ts, ds = z[:6], z[6:]
        at = force_model.acc3(ts, t)
        ad = force_model.acc3(ds, t)
        if controller is not None:
            rel = lvlh.inertlvlh(ts, ds)
            ulvlh = np.asarray(controller(rel, t), dtype=float)
            ad = ad + lvlh.lvlhrotmat(ts).T @ ulvlh
        return np.concatenate((ts[3:], at, ds[3:], ad))

    z0 = np.concatenate((tgtstate0, dep))
    z = orbit.propagate(combderiv, z0, times)
    tgtstates, depstates = z[:, :6], z[:, 6:]

    relstates = np.empty((times.size, 6), dtype=float)
    controls = np.zeros((times.size, 3), dtype=float)
    for k in range(times.size):
        relstates[k] = lvlh.inertlvlh(tgtstates[k], depstates[k])
        if controller is not None:
            controls[k] = np.asarray(controller(relstates[k], times[k]), dtype=float)
    return relstates, controls, tgtstates, depstates


def gainsweep2(initrelstate, times, n, kp_values, kd_values,
               target=None, max_accel=None, pos_tol=1.0):
    rows = []
    for kp in kp_values:
        for kd in kd_values:
            pdctrl = PDCtrl(kp, kd, target=target, max_accel=max_accel)
            states, controls = cw.propcw(initrelstate, times, n, pdctrl)
            perr = poserrseries(states, target)
            verr = velerrseries(states, target)
            accel = ctrlaccseries(controls)
            rows.append({
                "kp": kp,
                "kd": kd,
                "final_pos_error_m": float(perr[-1]),
                "final_vel_error_mps": float(verr[-1]),
                "convergence_time_s": convtime(times, perr, pos_tol),
                "max_control_accel_mps2": float(np.max(accel)),
                "delta_v_mps": cumdeltav(times, controls),
            })
    return pd.DataFrame(rows)


def cmpfid(tgtstate0, relstate0, times, force_model,
           n=None, controller=None):
    tgtstate0 = np.asarray(tgtstate0, dtype=float)
    if n is None:
        n = cw.cwmeanmotion(np.linalg.norm(tgtstate0[:3]), force_model.mu)

    cwstates, cwcontr = cw.propcw(relstate0, times, n, controller)
    hfstates, hfcontr, _, _ = highfidrel(
        tgtstate0, relstate0, times, force_model, controller
    )

    poserr = np.linalg.norm(cwstates[:, :3] - hfstates[:, :3], axis=1)
    velerr = np.linalg.norm(cwstates[:, 3:] - hfstates[:, 3:], axis=1)

    summary = pd.DataFrame([{
        "force_model": force_model.label,
        "initial_sep_m": float(np.linalg.norm(np.asarray(relstate0)[:3])),
        "max_pos_error_m": float(poserr.max()),
        "final_pos_error_m": float(poserr[-1]),
        "max_vel_error_mps": float(velerr.max()),
        "final_vel_error_mps": float(velerr[-1]),
    }])

    return {
        "cw_states": cwstates,
        "hf_states": hfstates,
        "cw_controls": cwcontr,
        "hf_controls": hfcontr,
        "pos_error": poserr,
        "vel_error": velerr,
        "summary": summary,
    }
