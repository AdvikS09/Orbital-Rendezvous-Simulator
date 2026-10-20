import numpy as np



def lvlhrotmat(tgtstate):
    tgtstate = np.asarray(tgtstate, dtype=float)
    rvec, vvec = tgtstate[:3], tgtstate[3:]
    xhat = rvec / np.linalg.norm(rvec)
    hvec = np.cross(rvec, vvec)
    zhat = hvec / np.linalg.norm(hvec)
    yhat = np.cross(zhat, xhat)
    return np.vstack((xhat, yhat, zhat))


def lvlhangvel(tgtstate):
    tgtstate = np.asarray(tgtstate, dtype=float)
    rvec, vvec = tgtstate[:3], tgtstate[3:]
    r2 = np.dot(rvec, rvec)
    return np.cross(rvec, vvec) / r2


def inertlvlh(tgtstate, depstate):
    tgtstate = np.asarray(tgtstate, dtype=float)
    depstate = np.asarray(depstate, dtype=float)

    rot = lvlhrotmat(tgtstate)
    omegabody = rot @ lvlhangvel(tgtstate)

    dr = depstate[:3] - tgtstate[:3]
    dv = depstate[3:] - tgtstate[3:]

    rho = rot @ dr
    rhodot = rot @ dv - np.cross(omegabody, rho)
    return np.concatenate((rho, rhodot))


def lvlhinert(tgtstate, relstate):
    tgtstate = np.asarray(tgtstate, dtype=float)
    relstate = np.asarray(relstate, dtype=float)

    rot = lvlhrotmat(tgtstate)
    omegabody = rot @ lvlhangvel(tgtstate)

    rho = relstate[:3]
    rhodot = relstate[3:]

    dr = rot.T @ rho
    dv = rot.T @ (rhodot + np.cross(omegabody, rho))
    return np.concatenate((tgtstate[:3] + dr, tgtstate[3:] + dv))


def relatpos(tgtstate, depstate):
    return inertlvlh(tgtstate, depstate)[:3]


def relatvel(tgtstate, depstate):
    return inertlvlh(tgtstate, depstate)[3:]
