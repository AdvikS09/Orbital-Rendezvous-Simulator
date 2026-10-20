"""Capture canonical engine outputs to lock as regression baselines."""
import json
import numpy as np
from src import controller, cw, orbit, perturbations, validation
from src import scenarios as sc

out = {}

# 1. Two-body orbit validation (experiment 3, part 1)
radius = orbit.DEFAULT_RADIUS
mu = orbit.MU_EARTH
period = orbit.orbital_period(radius, mu)
inc = np.radians(51.6)
state0 = orbit.circular_orbit_state(radius, mu, inc)
times2 = np.linspace(0.0, 2.0 * period, 4001)
states2 = orbit.propagate(orbit.two_body_derivatives, state0, times2)
v = validation.verify_orbit(times2, states2, mu, analytic_period=period).iloc[0]
out["two_body"] = {
    "analytic_period_s": float(period),
    "max_energy_drift_rel": float(v["max_energy_drift_rel"]),
    "max_ang_mom_drift_rel": float(v["max_ang_mom_drift_rel"]),
    "measured_period_s": float(v["measured_period_s"]),
    "period_rel_err": float(v["period_rel_err"]),
}

# 2. CW mean motion
n = cw.cw_mean_motion(radius, mu)
out["mean_motion_rad_s"] = float(n)

# 3. CW RK4 vs closed-form STM (integration error)
t1 = np.linspace(0.0, period, 2001)
s0 = np.array([1000.0, 0.0, 500.0, 0.0, 0.0, 0.0])
rk4, _ = cw.propagate_cw(s0, t1, n, None)
exact = cw.propagate_cw_analytic(s0, t1, n)
out["cw_rk4_vs_stm_max_err_m"] = float(np.max(np.linalg.norm(rk4[:, :3] - exact[:, :3], axis=1)))

# 4. Baseline PD rendezvous (experiment 1 settings)
tr = np.linspace(0.0, period, 1112)   # dt ~= 5 s, as run_cw_rendezvous.py
ctrl = controller.PDController.critically_damped(6.0 * n, max_accel=0.05)
rst, rct = cw.propagate_cw(np.array([200.0, -300.0, 80.0, 0.0, 0.0, 0.0]), tr, n, ctrl)
m = validation.rendezvous_metrics(tr, rst, rct, pos_tol=1.0).iloc[0]
out["pd_rendezvous"] = {k: float(m[k]) for k in m.index}

# 5. CW vs high-fidelity, perturbation toggles (experiment 3, part 3)
tt = np.linspace(0.0, period, 1001)
rel0 = np.array([0.0, 1000.0, 0.0, 0.0, 0.0, 0.0])
toggles = {}
for name, fm in [("two_body", perturbations.ForceModel.two_body()),
                 ("j2", perturbations.ForceModel.j2_only()),
                 ("drag", perturbations.ForceModel.drag_only()),
                 ("j2_drag", perturbations.ForceModel.j2_drag())]:
    c = validation.compare_cw_high_fidelity(state0, rel0, tt, fm, n, controller=None)
    s = c["summary"].iloc[0]
    toggles[name] = {"max_pos_error_m": float(s["max_pos_error_m"]),
                     "final_pos_error_m": float(s["final_pos_error_m"]),
                     "max_vel_error_mps": float(s["max_vel_error_mps"])}
out["cw_vs_hf_toggles"] = toggles

# 6. Gain sweep corner values
tg = np.linspace(0.0, period, 556)  # dt ~= 10 s
gs = validation.gain_sweep(np.array([200.0, -300.0, 80.0, 0.0, 0.0, 0.0]), tg, n,
                           [1.0e-5, 4.6e-5, 1.0e-4, 2.0e-4],
                           [5.0e-3, 1.0e-2, 1.36e-2, 3.0e-2], max_accel=None, pos_tol=1.0)
out["gain_sweep_rows"] = int(len(gs))
sel = gs[(gs.kp == 4.6e-5) & (gs.kd == 1.36e-2)].iloc[0]
out["gain_sweep_critdamp"] = {k: (None if not np.isfinite(sel[k]) else float(sel[k]))
                              for k in ["final_pos_error_m", "convergence_time_s",
                                        "max_control_accel_mps2", "delta_v_mps"]}

# 7. Scenario-API level (what the UI actually calls)
r = sc.run_cw_rendezvous([200, -300, 80], [0, 0, 0])
out["api_run_cw_rendezvous"] = {k: float(r["metrics"].iloc[0][k]) for k in r["metrics"].columns}
o = sc.run_orbit(400.0, 51.6, 2.0)
out["api_run_orbit"] = {"period_s": float(o["period_s"]),
                        "max_energy_drift_rel": float(o["validation"].iloc[0]["max_energy_drift_rel"])}
cmpres = sc.run_relative_scenario([0, 1000, 0], [0, 0, 0], model="comparison",
                                  controlled=False, use_j2=True, use_drag=False, dt_s=10.0)
out["api_cw_vs_hf_j2"] = {k: float(cmpres["comparison"]["summary"].iloc[0][k])
                          for k in ["max_pos_error_m", "final_pos_error_m"]}

print(json.dumps(out, indent=2))
