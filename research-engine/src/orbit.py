import numpy as np

G = 6.67430e-11
MUEARTH = 3.986004418e14
MEARTH = MUEARTH / G
REARTH = 6_378_137.0
J2EARTH = 1.08262668e-3
OMEGAEARTH = 7.2921159e-5

DEFALT = 400_000.0
DEFRAD = REARTH + DEFALT


def twobodyacc(position, mu=MUEARTH):
    position = np.asarray(position, dtype=float)
    r = np.linalg.norm(position)
    if r == 0.0:
        raise ValueError("two_body_acceleration: zero position vector")
    return -mu * position / r**3


def twobodyderivs(state, t, mu=MUEARTH):
    state = np.asarray(state, dtype=float)
    return np.concatenate((state[3:], twobodyacc(state[:3], mu)))


def rk4step(deriv, state, t, dt):
    state = np.asarray(state, dtype=float)
    k1 = np.asarray(deriv(state, t), dtype=float)
    k2 = np.asarray(deriv(state + 0.5 * dt * k1, t + 0.5 * dt), dtype=float)
    k3 = np.asarray(deriv(state + 0.5 * dt * k2, t + 0.5 * dt), dtype=float)
    k4 = np.asarray(deriv(state + dt * k3, t + dt), dtype=float)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def propagate(deriv, state0, tgrid):
    tgrid = np.asarray(tgrid, dtype=float)
    state0 = np.asarray(state0, dtype=float)
    states = np.empty((tgrid.size, state0.size), dtype=float)
    states[0] = state0
    for i in range(tgrid.size - 1):
        dt = tgrid[i + 1] - tgrid[i]
        states[i + 1] = rk4step(deriv, states[i], tgrid[i], dt)
    return states


def circspeed(radius, mu=MUEARTH):
    return np.sqrt(mu / radius)


def orbperiod(semimajoraxis, mu=MUEARTH):
    return 2.0 * np.pi * np.sqrt(semimajoraxis**3 / mu)


def mean_motion(semimajoraxis, mu=MUEARTH):
    return np.sqrt(mu / semimajoraxis**3)


def circorbitstate(radius, mu=MUEARTH, inclination=0.0):
    v = circspeed(radius, mu)
    return np.array(
        [radius, 0.0, 0.0, 0.0, v * np.cos(inclination), v * np.sin(inclination)],
        dtype=float,
    )


def specenergy(state, mu=MUEARTH):
    state = np.asarray(state, dtype=float)
    r = np.linalg.norm(state[:3])
    v = np.linalg.norm(state[3:])
    return 0.5 * v * v - mu / r


def angmomvec(state):
    state = np.asarray(state, dtype=float)
    return np.cross(state[:3], state[3:])


def specangmom(state):
    return np.linalg.norm(angmomvec(state))


TOL = 1e-11


def rvelem(state, mu=MUEARTH):
    state = np.asarray(state, dtype=float)
    rvec, vvec = state[:3], state[3:]
    r = np.linalg.norm(rvec)
    v = np.linalg.norm(vvec)

    hvec = np.cross(rvec, vvec)
    h = np.linalg.norm(hvec)
    nodevec = np.cross([0.0, 0.0, 1.0], hvec)
    node = np.linalg.norm(nodevec)

    evec = ((v * v - mu / r) * rvec - np.dot(rvec, vvec) * vvec) / mu
    e = np.linalg.norm(evec)

    energy = 0.5 * v * v - mu / r
    a = -mu / (2.0 * energy) if abs(energy) > TOL else np.inf

    inc = np.arccos(np.clip(hvec[2] / h, -1.0, 1.0)) if h > TOL else 0.0

    if node > TOL:
        raan = np.arccos(np.clip(nodevec[0] / node, -1.0, 1.0))
        if nodevec[1] < 0.0:
            raan = 2.0 * np.pi - raan
    else:
        raan = 0.0

    if node > TOL and e > TOL:
        argp = np.arccos(np.clip(np.dot(nodevec, evec) / (node * e), -1.0, 1.0))
        if evec[2] < 0.0:
            argp = 2.0 * np.pi - argp
    else:
        argp = 0.0

    if e > TOL:
        nu = np.arccos(np.clip(np.dot(evec, rvec) / (e * r), -1.0, 1.0))
        if np.dot(rvec, vvec) < 0.0:
            nu = 2.0 * np.pi - nu
    else:
        ref = nodevec if node > TOL else np.array([1.0, 0.0, 0.0])
        refn = np.linalg.norm(ref)
        nu = np.arccos(np.clip(np.dot(ref, rvec) / (refn * r), -1.0, 1.0))
        if rvec[2] < 0.0:
            nu = 2.0 * np.pi - nu

    period = orbperiod(a, mu) if np.isfinite(a) and a > 0.0 else np.inf
    return {
        "a": a, "e": e, "i": inc, "raan": raan, "argp": argp, "nu": nu,
        "h": h, "energy": energy, "period": period,
    }


def elemrv(a, e, i, raan, argp, nu, mu=MUEARTH):
    p = a * (1.0 - e * e)
    r = p / (1.0 + e * np.cos(nu))

    rpf = np.array([r * np.cos(nu), r * np.sin(nu), 0.0])
    vpf = np.sqrt(mu / p) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])

    cO, sO = np.cos(raan), np.sin(raan)
    ci, si = np.cos(i), np.sin(i)
    c, sw = np.cos(argp), np.sin(argp)

    rot = np.array([
        [cO * c - sO * sw * ci, -cO * sw - sO * c * ci, sO * si],
        [sO * c + cO * sw * ci, -sO * sw + cO * c * ci, -cO * si],
        [sw * si, c * si, ci],
    ])
    return np.concatenate((rot @ rpf, rot @ vpf))


def measperiod(t, states):
    t = np.asarray(t, dtype=float)
    pos = np.asarray(states)[:, :3]
    r0 = pos[0]
    h0 = np.cross(r0, np.asarray(states)[0, 3:])
    ex = r0 / np.linalg.norm(r0)
    ez = h0 / np.linalg.norm(h0)
    ey = np.cross(ez, ex)

    projx = pos @ ex
    projy = pos @ ey
    theta = np.unwrap(np.arctan2(projy, projx))
    if theta[-1] < theta[0]:
        theta = -theta
    target = theta[0] + 2.0 * np.pi
    if theta[-1] < target:
        return np.nan
    return float(np.interp(target, theta, t))
