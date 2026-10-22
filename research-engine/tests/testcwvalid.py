import math

import numpy as np
import pytest
from scipy.linalg import expm

from cwvalid import dynamics as dyn
from cwvalid.constants import DEFRAD, MUEARTH, mean_motion, period
from cwvalid.control import CmdSpec, PDLaw
from cwvalid.experiment import RunCfg, runscen
from cwvalid.scenario import (Scenario, axisscen, velradial,
                              velmatch)
from cwvalid.simulate import SolverCfg, simulate
from src import cw as src_cw
from src import lvlh as src_lvlh
from src import orbit as src_orbit
from src import perturbations as src_pert
from src import validation as src_val
from src.controller import PDCtrl

R = DEFRAD
N = mean_motion(R)
T = period(R)
RNG = np.random.default_rng(12345)


def randstates(k, scale):
    pos = RNG.normal(size=(k, 3)) * scale
    vel = RNG.normal(size=(k, 3)) * N * scale
    return np.hstack((pos, vel))


def testnermstableform():
    for s in randstates(200, 20_000.0):
        a = np.array(dyn.accnerm(N, R, *s))
        b = np.array(dyn.accnermnaive(N, R, *s))
        assert np.allclose(a, b, rtol=1e-8, atol=1e-13)


def testnermredcw():
    u = np.array([0.6, -0.5, 0.62])
    u /= np.linalg.norm(u)
    errscw, errsso = [], []
    for rho in (100.0, 200.0, 400.0, 800.0):
        s = np.concatenate((u * rho, [0.0, 0.0, 0.0]))
        an = np.array(dyn.accnerm(N, R, *s))
        errscw.append(np.linalg.norm(an - np.array(dyn.acccw(N, R, *s))))
        errsso.append(np.linalg.norm(an - np.array(dyn.accso(N, R, *s))))
    rcw = np.array(errscw[1:]) / np.array(errscw[:-1])
    rso = np.array(errsso[1:]) / np.array(errsso[:-1])
    assert np.allclose(rcw, 4.0, rtol=0.02)
    assert np.allclose(rso, 8.0, rtol=0.05)
    assert errscw[0] < 3.0 * N * N * 100.0**2 / R


def testvecrelacc():
    S = randstates(50, 5000.0)
    for m in ("cw", "so", "nerm"):
        V = dyn.relaccvec(m, N, R, S)
        for i, s in enumerate(S):
            assert np.allclose(V[i], dyn.RELMODELS[m](N, R, *s), rtol=1e-12, atol=1e-15)


def testinertforcematch():
    fm = src_pert.ForceMod.j2_drag()
    f = dyn.InertForce(use_j2=True, use_drag=True)
    tgt = src_orbit.circorbitstate(R, MUEARTH, math.radians(51.6))
    for k in range(20):
        st = tgt + np.concatenate((RNG.normal(size=3) * 3e4, RNG.normal(size=3) * 30))
        asrc = fm.acc3(st)
        anew = np.array(f.acc3(*st))
        assert np.allclose(asrc, anew, rtol=1e-12, atol=1e-18)


def testlvlhroundtripagr():
    tgt = src_orbit.circorbitstate(R, MUEARTH, math.radians(51.6))
    for rel in randstates(20, 3000.0):
        ch = dyn.lvlhtoeci(tgt, rel)
        back, _ = dyn.ecilvlhscalar(tuple(tgt), tuple(ch))
        assert np.allclose(back, rel, rtol=1e-9, atol=1e-9)
        assert np.allclose(src_lvlh.inertlvlh(tgt, ch), back, rtol=1e-12, atol=1e-9)


def testexactframerate():
    fm = src_pert.ForceMod.j2only()
    tgt0 = src_orbit.circorbitstate(R, MUEARTH, math.radians(51.6))
    rel0 = np.array([0.0, 0.0, 20_000.0, 0.0, 0.0, 0.0])
    ch0 = src_lvlh.lvlhinert(tgt0, rel0)
    tt = np.linspace(0.0, 1500.0, 15001)
    tg = src_orbit.propagate(fm.derivatives, tgt0, tt)
    cg = src_orbit.propagate(fm.derivatives, ch0, tt)
    f = dyn.InertForce(use_j2=True)
    ap = np.array([f.pert2(*s) for s in tg])
    relexact, _ = dyn.ecilvlhvec(tg, cg, ap)
    relbase, _ = dyn.ecilvlhvec(tg, cg, None)
    h = tt[1] - tt[0]
    fd = (relexact[2:, :3] - relexact[:-2, :3]) / (2 * h)
    eexact = np.abs(fd - relexact[1:-1, 3:]).max()
    ebase = np.abs(fd - relbase[1:-1, 3:]).max()
    assert eexact < 1e-6
    assert ebase > 100 * eexact


def testsatpresdir():
    law = PDLaw.bandw(6 * N, 0.05)
    for s in randstates(100, 50_000.0):
        raw = np.array(law.raw(*s))
        a = np.array(law.command(*s))
        if np.linalg.norm(raw) > 0.05:
            assert abs(np.linalg.norm(a) - 0.05) < 1e-15
            assert np.allclose(a / np.linalg.norm(a), raw / np.linalg.norm(raw), atol=1e-14)
        else:
            assert np.array_equal(a, raw)
    ctrl = PDCtrl.critdamped(6 * N, max_accel=0.05)
    s = np.array([3000.0, -2000.0, 500.0, 1.0, -0.5, 0.2])
    assert np.allclose(ctrl(s), law.command(*s), rtol=1e-15)


def testsatdistamax():
    law = PDLaw.bandw(6 * N, 0.05)
    assert law.saturdist == pytest.approx(0.05 / (36 * N * N))
    assert 1080 < law.saturdist < 1090


@pytest.mark.parametrize("x0", [-20_000.0, -500.0, 500.0, 5_000.0, 20_000.0])
def testexactenergymatching(x0):
    tgt = src_orbit.circorbitstate(R, MUEARTH, 0.0)
    ydot = velradial(x0, R)
    ch = dyn.lvlhtoeci(tgt, [x0, 0, 0, 0, ydot, 0])
    et = src_orbit.specenergy(tgt)
    ec = src_orbit.specenergy(ch)
    assert abs(ec - et) < 1e-9 * abs(et)
    approx = -2 * N * x0 + N * x0**2 / (2 * R)
    assert abs(ydot - approx) < 5 * N * abs(x0) ** 3 / R**2 + 1e-12
    assert abs(velmatch(x0, N) - (-2 * N * x0)) == 0.0


def testexactenergymatch():
    with pytest.raises(ValueError):
        velradial(R * 1.01, R)


def testunccwmatch():
    s0 = np.array([300.0, -800.0, 200.0, 0.1, -0.2, 0.05])
    sim = simulate("cw", s0, CmdSpec("none"), T, n=N, R=R)
    t = np.linspace(0, T, 101)
    ref = src_cw.propcwanal(s0, t, N)
    assert np.abs(sim.states(t) - ref)[:, :3].max() < 1e-7


def testunsatclosedloop():
    law = PDLaw.bandw(6 * N, None)
    A, B = src_cw.cwsystemmats(N)
    K = np.hstack((law.kp * np.eye(3), law.kd * np.eye(3)))
    Acl = A - B @ K
    s0 = np.array([200.0, -300.0, 80.0, 0.0, 0.0, 0.0])
    sim = simulate("cw", s0, CmdSpec("continuous", law=law), T, n=N, R=R)
    for t in (100.0, 1000.0, 3000.0):
        assert np.abs(sim.states([t])[0] - expm(Acl * t) @ s0).max() < 1e-7


def rk4nodeerrs(sep, ks):
    law = PDLaw.bandw(6 * N, 0.05)
    s0 = np.array([sep, 0, 0, 0, 0, 0])
    cmd = CmdSpec("continuous", law=law)
    ref = simulate("nerm", s0, cmd, T, n=N, R=R, solver=SolverCfg(rtol=1e-13, max_step=5.0))
    tn = np.linspace(0, T, 1001)
    out = []
    for k in ks:
        r4 = simulate("nerm", s0, cmd, T, n=N, R=R, solver=SolverCfg(method="RK4", rk4_dt=T / k))
        out.append(np.abs(r4.segments[0].sol.y[::k // 1000, :3] - ref.states(tn)[:, :3]).max())
    return out


def testrk4fourthord():
    e = rk4nodeerrs(1000.0, (1000, 2000, 4000))
    assert 12 < e[0] / e[1] < 20 and 12 < e[1] / e[2] < 20


def testrk4redorder():
    e = rk4nodeerrs(5000.0, (1000, 2000))
    assert e[0] / e[1] < 4
    assert e[1] < 5e-3


def testecibranchmatch():
    law = PDLaw.bandw(6 * N, 0.05)
    s0 = np.array([0.0, 1000.0, 0.0, 0.0, 0.0, 0.0])
    sim = simulate("eci", s0, CmdSpec("continuous", law=law), T, n=N, R=R)
    tgt0 = src_orbit.circorbitstate(R, MUEARTH, math.radians(51.6))
    t = np.linspace(0, T, 8001)
    ctrl = PDCtrl.critdamped(6 * N, max_accel=0.05)
    rel, _, _, _ = src_val.highfidrel(tgt0, s0, t, src_pert.ForceMod.twobody(), ctrl)
    assert np.abs(sim.states(t)[:, :3] - rel[:, :3]).max() < 1e-5


def testnermeciagree():
    law = PDLaw.bandw(6 * N, 0.05)
    s0 = np.array([-8000.0, 3000.0, 2000.0, 0.5, -1.0, 0.2])
    cmd = CmdSpec("continuous", law=law)
    a = simulate("nerm", s0, cmd, T, n=N, R=R)
    b = simulate("eci", s0, cmd, T, n=N, R=R)
    t = np.linspace(0, T, 2001)
    assert np.abs(a.states(t)[:, :3] - b.states(t)[:, :3]).max() < 1e-4


def testdvaccmatch():
    law = PDLaw.bandw(6 * N, 0.05)
    s0 = np.array([3000.0, 0, 0, 0, 0, 0])
    sim = simulate("cw", s0, CmdSpec("continuous", law=law), T, n=N, R=R)
    t = np.linspace(0, T, 200_001)
    _, a = sim.commands(t)
    from scipy.integrate import simpson
    assert abs(sim.dvfinal - simpson(np.linalg.norm(a, axis=1), x=t)) < 1e-6


def testsampctrlholds():
    law = PDLaw.bandw(6 * N, 0.05)
    s0 = np.array([2000.0, 0, 0, 0, 0, 0])
    cont = simulate("cw", s0, CmdSpec("continuous", law=law), T, n=N, R=R)
    samp = simulate("cw", s0, CmdSpec("sampled", law=law, sample_period=10.0), T, n=N, R=R)
    assert len(samp.segments) == math.ceil(T / 10.0)
    _, a = samp.commands(np.array([20.0, 25.0, 29.999]))
    assert np.allclose(a[0], a[1]) and np.allclose(a[1], a[2])
    fine = simulate("cw", s0, CmdSpec("sampled", law=law, sample_period=0.25), T, n=N, R=R)
    t = np.linspace(0, T, 501)
    e10 = np.abs(samp.states(t)[:, :3] - cont.states(t)[:, :3]).max()
    e025 = np.abs(fine.states(t)[:, :3] - cont.states(t)[:, :3]).max()
    assert e025 < e10 / 10


def testprescrcmdreplay():
    law = PDLaw.bandw(6 * N, 0.05)
    s0 = np.array([4000.0, 0, 0, 0, 0, 0])
    cl = simulate("cw", s0, CmdSpec("continuous", law=law), T, n=N, R=R)
    pres = lambda t: tuple(law.command(*cl.states([t])[0]))
    presvec = lambda tt: law.applvec(cl.states(tt))
    rep = simulate("cw", s0, CmdSpec("prescribed", prescribed=pres, prescribed_vec=presvec), T, n=N, R=R)
    t = np.linspace(0, T, 401)
    assert np.abs(rep.states(t)[:, :3] - cl.states(t)[:, :3]).max() < 1e-4


def testscenidsdet():
    a = axisscen("main", "+R", 1000.0, 6.0, 0.05)
    b = axisscen("main", "+R", 1000.0, 6.0, 0.05)
    c = axisscen("main", "-R", 1000.0, 6.0, 0.05)
    d = axisscen("main", "+R", 1000.0, 6.0, 0.05, velocity="em1")
    assert a.sid == b.sid
    assert len({a.sid, c.sid, d.sid}) == 3
    assert Scenario.from_dict(a.to_dict()).sid == a.sid


def testpairrunrepr():
    s = axisscen("test", "+V", 1000.0, 6.0, 0.05)
    row, _ = runscen(s, RunCfg(models=("cw", "nerm"), reference="nerm", crosscheck=None))
    assert row["cw_vs_nerm_max_pos_disc_m"] == pytest.approx(2.263e-3, rel=1e-3)
    assert row["outcome_terminal_cw"] == "both_succeed"


def testtermarrtime():
    s = axisscen("test", "+R", 3000.0, 6.0, 0.05)
    cfg = RunCfg(models=("cw", "nerm"), reference="nerm", crosscheck=None)
    from cwvalid.experiment import runbranch
    sims, _ = runbranch(s, cfg)
    row, _ = runscen(s, cfg)
    ta = row["nerm_terminal_arrival_time_s"]
    x = sims["nerm"].states([ta])[0]
    g = max(np.linalg.norm(x[:3]) / 1.0, np.linalg.norm(x[3:]) / 0.01)
    assert abs(g - 1.0) < 1e-6


def testsatswitchevents():
    law = PDLaw.bandw(4 * N, 0.01)
    s0 = np.array([0.0, 0.0, 14471.330624, 0.0, 0.0, 0.0])
    sim = simulate("nerm", s0, CmdSpec("continuous", law=law), T, n=N, R=R)
    assert len(sim.satswitchtimes) >= 2
    S = sim.states(np.array(sim.satswitchtimes))
    g = np.linalg.norm(law.rawvec(S), axis=1) - law.a_max
    assert np.abs(g).max() < 1e-9


def testeventrestbring():
    law = PDLaw.bandw(4 * N, 0.01)
    s0 = np.array([0.0, 0.0, 14471.330624, 0.0, 0.0, 0.0])
    cmd = CmdSpec("continuous", law=law)
    a = simulate("nerm", s0, cmd, T, n=N, R=R)
    b = simulate("eci", s0, cmd, T, n=N, R=R)
    t = np.linspace(0, T, 4001)
    assert np.abs(a.states(t)[:, :3] - b.states(t)[:, :3]).max() < 1e-5


def testrk4branchstops():
    law = PDLaw.bandw(8 * N, 0.05)
    s0 = np.array([-17275.536473, 0.0, 0.0, 0.0, 0.0, 0.0])
    cmd = CmdSpec("continuous", law=law)
    dop = simulate("nerm", s0, cmd, T, n=N, R=R)
    rk4 = simulate("nerm", s0, cmd, T, n=N, R=R, solver=SolverCfg(method="RK4", rk4_dt=T / 2000))
    assert dop.termination == "altitude_floor" and rk4.termination == "altitude_floor"
    assert abs(rk4.tfinal - dop.tfinal) <= T / 2000 + 1e-9


def testtracebranchwith():
    from cwvalid import metrics as M
    from cwvalid.simulate import SimRes
    law = PDLaw.bandw(6 * N, 0.05)
    sim = SimRes("cw", CmdSpec("continuous", law=law), N, R, T, SolverCfg(), None, [1000.0, 0, 0, 0, 0, 0])
    sim.status, sim.termination = "failed", "solver_failure"
    tr = M.maketrace(sim, M.commongrid(T))
    d = M.branchdiags(sim, tr, T, M.OutcCrit())
    assert tr.t.size == 1 and d["numerical_failure"] and not d["terminal_arrived"]


def testlegacygridper():
    from cwvalid import metrics as M
    sim = simulate("cw", np.array([100.0, 0, 0, 0, 0, 0]), CmdSpec("none"), 2 * T, n=N, R=R)
    g = M.legacygrid(sim, 2 * T, M.OutcCrit())
    assert g.size == 4001 and abs(g[1] - T / 2000) < 1e-9


def testrunnerrefmixed(tmp_path):
    from cwvalid.runner import readjsonl2, runcamp
    s = axisscen("t", "+V", 200.0, 6.0, 0.05)
    cfg = {"models": ["cw", "nerm"], "reference": "nerm", "crosscheck": None}
    runcamp(str(tmp_path), [s], cfg, workers=1, keephist=False, log=lambda *a: None)
    with open(tmp_path / "results.jsonl", "a") as fh:
        fh.write('{"sid": "partial')
    assert len(readjsonl2(str(tmp_path / "results.jsonl"), log=lambda *a: None)) == 1
    with pytest.raises(RuntimeError):
        runcamp(str(tmp_path), [s], dict(cfg, dt_eval=1.0), workers=1, keephist=False, log=lambda *a: None)
