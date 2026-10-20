import numpy as np
import pandas as pd

from . import collision, cw, extval, orbit, validation
from .controller import PDCtrl
from .perturbations import ForceMod

MAXSTEPS = 200_000


def radalt(altitude_km):
    return orbit.REARTH + float(altitude_km) * 1000.0


def timegrid(duration_s, dt_s):
    duration_s = float(duration_s)
    dt_s = float(dt_s)
    nsteps = int(round(duration_s / dt_s))
    nsteps = max(1, min(nsteps, MAXSTEPS))
    return np.linspace(0.0, nsteps * dt_s, nsteps + 1)


def relstate(relpos, relvel):
    return np.array([*map(float, relpos), *map(float, relvel)], dtype=float)


def relres(times, states, controls, model, mean_motion, period_s, target=None):
    poserr = validation.poserrseries(states, target)
    velerr = validation.velerrseries(states, target)
    ctrlmag = validation.ctrlaccseries(controls)
    metr = validation.rdvmetrics(times, states, controls, target=target, pos_tol=1.0)
    history = pd.DataFrame({
        "t_s": times,
        "x_m": states[:, 0], "y_m": states[:, 1], "z_m": states[:, 2],
        "vx_mps": states[:, 3], "vy_mps": states[:, 4], "vz_mps": states[:, 5],
        "ax_mps2": controls[:, 0], "ay_mps2": controls[:, 1], "az_mps2": controls[:, 2],
        "pos_err_m": poserr, "vel_err_mps": velerr, "ctrl_mag_mps2": ctrlmag,
    })
    return {
        "model": model,
        "times": times,
        "states": states,
        "controls": controls,
        "pos_error": poserr,
        "vel_error": velerr,
        "control_mag": ctrlmag,
        "metrics": metr,
        "history": history,
        "mean_motion": mean_motion,
        "period_s": period_s,
    }


def runorbit(altitude_km=400.0, inclination_deg=51.6, norbits=2.0,
             use_j2=False, use_drag=False, stepsperorbit=2000,
             cd=2.2, area=1.0, mass=100.0):
    radius = radalt(altitude_km)
    mu = orbit.MUEARTH
    period = orbit.orbperiod(radius, mu)
    state0 = orbit.circorbitstate(radius, mu, np.radians(float(inclination_deg)))

    nsteps = max(1, min(int(round(norbits * stepsperorbit)), MAXSTEPS))
    times = np.linspace(0.0, float(norbits) * period, nsteps + 1)

    fm = ForceMod(use_j2=use_j2, use_drag=use_drag, cd=cd, area=area, mass=mass)
    states = orbit.propagate(fm.derivatives, state0, times)

    altseries = (np.linalg.norm(states[:, :3], axis=1) - orbit.REARTH) / 1000.0
    return {
        "label": fm.label,
        "times": times,
        "states": states,
        "period_s": period,
        "radius_m": radius,
        "altitude_km": altseries,
        "energy_drift": validation.reldrift(validation.energyseries(states, mu)),
        "validation": validation.checkorbit(times, states, mu, analperiod=period),
    }


def cmporbitpert(altitude_km=400.0, inclination_deg=51.6, norbits=5.0,
                 use_j2=False, use_drag=True, stepsperorbit=1000,
                 cd=2.2, area=1.0, mass=100.0):
    base = runorbit(altitude_km, inclination_deg, norbits, False, False, stepsperorbit)
    pert = runorbit(altitude_km, inclination_deg, norbits, use_j2, use_drag,
                    stepsperorbit, cd=cd, area=area, mass=mass)
    sep = np.linalg.norm(pert["states"][:, :3] - base["states"][:, :3], axis=1)
    return {
        "times": base["times"],
        "baseline": base,
        "perturbed": pert,
        "separation_m": sep,
    }


def runcwrel(relpos, relvel, altitude_km=400.0, duration_s=None, dt_s=10.0):
    radius = radalt(altitude_km)
    n = cw.cwmeanmotion(radius, orbit.MUEARTH)
    period = orbit.orbperiod(radius)
    times = timegrid(period if duration_s is None else duration_s, dt_s)
    states, controls = cw.propcw(relstate(relpos, relvel), times, n, None)
    return relres(times, states, controls, "CW (uncontrolled)", n, period)


def rendezvous(relpos, relvel, altitude_km=400.0, duration_s=None, dt_s=5.0,
               kp=None, kd=None, max_accel=0.05, omeganfactor=6.0):
    radius = radalt(altitude_km)
    n = cw.cwmeanmotion(radius, orbit.MUEARTH)
    period = orbit.orbperiod(radius)
    times = timegrid(period if duration_s is None else duration_s, dt_s)
    if kp is None or kd is None:
        pdctrl = PDCtrl.critdamped(omeganfactor * n, max_accel=max_accel)
    else:
        pdctrl = PDCtrl(float(kp), float(kd), max_accel=max_accel)
    states, controls = cw.propcw(relstate(relpos, relvel), times, n, pdctrl)
    return relres(times, states, controls, "CW + PD", n, period)


def gainsweep(relpos, relvel, altitude_km=400.0, duration_s=None, dt_s=5.0,
              kp_values=None, kd_values=None, max_accel=None):
    radius = radalt(altitude_km)
    n = cw.cwmeanmotion(radius, orbit.MUEARTH)
    period = orbit.orbperiod(radius)
    times = timegrid(period if duration_s is None else duration_s, dt_s)
    if kp_values is None:
        kp_values = [1.0e-5, 4.6e-5, 1.0e-4, 2.0e-4]
    if kd_values is None:
        kd_values = [5.0e-3, 1.0e-2, 1.36e-2, 3.0e-2]
    table = validation.gainsweep2(
        relstate(relpos, relvel), times, n,
        list(kp_values), list(kd_values), max_accel=max_accel, pos_tol=1.0,
    )
    return {
        "table": table,
        "kp_values": [float(v) for v in kp_values],
        "kd_values": [float(v) for v in kd_values],
        "mean_motion": n,
        "period_s": period,
    }


def runrelscen(relpos, relvel, altitude_km=400.0, inclination_deg=51.6,
               duration_s=None, dt_s=5.0, model="comparison",
               controlled=True, kp=None, kd=None, max_accel=0.05,
               omeganfactor=6.0, use_j2=True, use_drag=True,
               cd=2.2, area=1.0, mass=100.0):
    radius = radalt(altitude_km)
    mu = orbit.MUEARTH
    n = cw.cwmeanmotion(radius, mu)
    period = orbit.orbperiod(radius, mu)
    times = timegrid(period if duration_s is None else duration_s, dt_s)
    target0 = orbit.circorbitstate(radius, mu, np.radians(float(inclination_deg)))
    rel0 = relstate(relpos, relvel)
    fm = ForceMod(use_j2=use_j2, use_drag=use_drag, cd=cd, area=area, mass=mass)

    if controlled:
        if kp is None or kd is None:
            ctrl = PDCtrl.critdamped(omeganfactor * n, max_accel=max_accel)
        else:
            ctrl = PDCtrl(float(kp), float(kd), max_accel=max_accel)
    else:
        ctrl = None

    res = {"model": model, "times": times, "mean_motion": n, "period_s": period,
           "force_model": fm.label, "controlled": controlled}

    if model in ("cw", "comparison"):
        cwstates, cwcontr = cw.propcw(rel0, times, n, ctrl)
        res["cw"] = relres(times, cwstates, cwcontr, "CW", n, period)
    if model in ("high_fidelity", "comparison"):
        hfstates, hfcontr, _, _ = validation.highfidrel(target0, rel0, times, fm, ctrl)
        res["high_fidelity"] = relres(
            times, hfstates, hfcontr, f"high-fidelity ({fm.label})", n, period)
    if model == "comparison":
        cws, hfs = res["cw"]["states"], res["high_fidelity"]["states"]
        poserr = np.linalg.norm(cws[:, :3] - hfs[:, :3], axis=1)
        velerr = np.linalg.norm(cws[:, 3:] - hfs[:, 3:], axis=1)
        res["comparison"] = {
            "pos_error": poserr,
            "vel_error": velerr,
            "summary": pd.DataFrame([{
                "force_model": fm.label,
                "max_pos_error_m": float(poserr.max()),
                "final_pos_error_m": float(poserr[-1]),
                "max_vel_error_mps": float(velerr.max()),
                "final_vel_error_mps": float(velerr[-1]),
            }]),
        }

    res["primary"] = res.get("high_fidelity") or res.get("cw")
    return res


def buildvalscen(altitude_km=400.0, inclination_deg=51.6, duration_s=None,
                 dt_s=10.0, use_j2=True, use_drag=False,
                 cd=2.2, area=1.0, mass=100.0):
    radius = radalt(altitude_km)
    mu = orbit.MUEARTH
    period = orbit.orbperiod(radius, mu)
    times = timegrid(period if duration_s is None else duration_s, dt_s)
    state0 = orbit.circorbitstate(radius, mu, np.radians(float(inclination_deg)))
    fm = ForceMod(use_j2=use_j2, use_drag=use_drag, cd=cd, area=area, mass=mass)
    states = orbit.propagate(fm.derivatives, state0, times)

    eph = pd.DataFrame({
        "t_s": times,
        "x_m": states[:, 0], "y_m": states[:, 1], "z_m": states[:, 2],
        "vx_mps": states[:, 3], "vy_mps": states[:, 4], "vz_mps": states[:, 5],
    })
    metadata = {
        "frame": "ECI",
        "units": "SI (m, m/s, s)",
        "mu_m3_s2": mu,
        "initial_state_m_mps": [float(v) for v in state0],
        "altitude_km": float(altitude_km),
        "inclination_deg": float(inclination_deg),
        "dt_s": float(dt_s),
        "duration_s": float(times[-1]),
        "n_samples": int(times.size),
        "force_model": fm.label,
        "gravity": "point-mass + J2 (zonal)" if use_j2 else "point-mass (two-body)",
        "atmosphere": ("exponential rho0=3.9e-12 kg/m^3 @400km, H=60km"
                       if use_drag else "none"),
        "drag_cd": float(cd), "drag_area_m2": float(area), "mass_kg": float(mass),
    }
    return {
        "metadata": metadata,
        "ephemeris": eph,
        "times": times,
        "states": states,
        "csv_text": extval.tocsvtext(eph, metadata),
        "json_text": extval.jsontext(eph, metadata),
    }


def loadexteph(source, **kwargs):
    return extval.loadext(source, **kwargs)


def cmpexteph(inteph, exteph):
    return extval.cmpeph(inteph, exteph)


def runcollscen(relpos, relvel, altitude_km=400.0, duration_s=None, dt_s=5.0,
                controlled=True, kp=None, kd=None, max_accel=0.05,
                omeganfactor=6.0, keep_out_radius=25.0, collision_radius=5.0,
                debrisrelpos=None, debrisrelvel=None):
    radius = radalt(altitude_km)
    n = cw.cwmeanmotion(radius, orbit.MUEARTH)
    period = orbit.orbperiod(radius)
    times = timegrid(period if duration_s is None else duration_s, dt_s)

    if controlled:
        if kp is None or kd is None:
            ctrl = PDCtrl.critdamped(omeganfactor * n, max_accel=max_accel)
        else:
            ctrl = PDCtrl(float(kp), float(kd), max_accel=max_accel)
    else:
        ctrl = None

    chasstates, chascontr = cw.propcw(relstate(relpos, relvel), times, n, ctrl)

    debrstates = None
    if debrisrelpos is not None:
        debris0 = relstate(debrisrelpos, debrisrelvel if debrisrelvel is not None else [0, 0, 0])
        debrstates, _ = cw.propcw(debris0, times, n, None)

    analysis = collision.analyze(times, chasstates, keep_out_radius,
                                 debris_states=debrstates, collision_radius=collision_radius)
    chaser = relres(times, chasstates, chascontr, "chaser", n, period)
    return {
        "chaser": chaser,
        "debris_states": debrstates,
        "analysis": analysis,
        "times": times,
        "mean_motion": n,
        "period_s": period,
        "keep_out_radius": float(keep_out_radius),
        "collision_radius": float(collision_radius),
    }
