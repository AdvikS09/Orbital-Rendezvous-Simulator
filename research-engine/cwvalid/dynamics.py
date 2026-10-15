"""Relative-motion and inertial dynamics used by the extension.

Four translational models are provided. All relative models use the target
(chief) LVLH frame of the existing engine: x radial (away from Earth), y
along-track, z cross-track (orbit normal); state ``[x, y, z, vx, vy, vz]``
where (vx, vy, vz) are time derivatives of the LVLH coordinates (i.e. the
relative velocity *as seen in the rotating frame*). The chief orbit is circular
with radius ``R`` and mean motion ``n = sqrt(mu / R^3)``.

``cw``    Clohessy-Wiltshire (first order in rho/R)::

    x'' =  3 n^2 x + 2 n y' + ax
    y'' = -2 n x'           + ay
    z'' = -  n^2 z          + az

``so``    Second-order (quadratic) expansion of the exact relative equations
          about the circular chief. Obtained by expanding
          mu (R + x) / r^3 and mu y / r^3, mu z / r^3 with
          r^2 = (R + x)^2 + y^2 + z^2 to second order in rho/R::

    x'' - 2 n y' - 3 n^2 x = -(3 n^2 / (2 R)) (2 x^2 - y^2 - z^2) + ax
    y'' + 2 n x'           =  (3 n^2 / R) x y                     + ay
    z'' +   n^2 z          =  (3 n^2 / R) x z                     + az

          This is the standard quadratic ("second-order CW") Cartesian model;
          see e.g. Alfriend, Vadali, Gurfil, How & Breger, *Spacecraft Formation
          Flying* (Elsevier, 2010), nonlinear/second-order relative-motion
          chapter, and London (1963), AIAA J. 1(7). [CITE-CHECK: equation
          numbers not verified; derivation reproduced in methods notes.]

``nerm``  Exact nonlinear equations of relative motion (NERM) for an
          unperturbed two-body circular chief, written directly in the
          rotating LVLH frame (no inertial propagation, no frame
          reconstruction)::

    x'' = 2 n y' + n^2 (R + x) - mu (R + x) / r^3 + ax
    y'' = -2 n x' + n^2 y      - mu y / r^3       + ay
    z'' =                      - mu z / r^3       + az

          with r = |(R + x, y, z)|. This is the circular-chief special case
          (f' = n, f'' = 0, r_c = R) of the general nonlinear relative equations
          in e.g. Schaub & Junkins, *Analytical Mechanics of Space Systems*
          (AIAA), relative-motion chapter, and Alfriend et al. (2010).
          [CITE-CHECK]. It is implemented in a cancellation-free form:
          with q = (2 R x + x^2 + y^2 + z^2) / R^2 = r^2/R^2 - 1 and
          f = 1 - (R/r)^3 = -expm1(-1.5 log1p(q)),
          x'' = 2 n y' + n^2 (R + x) f, y'' = -2 n x' + n^2 y f,
          z'' = -n^2 z (1 - f).

``eci``   Two independent inertial propagations (target + chaser) under a
          selectable force model (two-body, optional J2, drag), with the
          relative state reconstructed in the target LVLH frame. This is the
          manuscript's nonlinear reference formulation (re-implemented here in
          scalar arithmetic for speed and checked against ``src.validation``).

All functions here are scalar (pure-Python float) for speed inside the ODE
right-hand side; vectorised helpers are provided for post-processing.
"""

import math

import numpy as np

from .constants import J2_EARTH, MU_EARTH, OMEGA_EARTH, R_EARTH

# ---------------------------------------------------------------------------
# Relative-model free (uncontrolled) accelerations, scalar
# ---------------------------------------------------------------------------


def acc_cw(n, R, x, y, z, vx, vy, vz):
    """CW natural acceleration (first-order)."""
    n2 = n * n
    return (3.0 * n2 * x + 2.0 * n * vy, -2.0 * n * vx, -n2 * z)


def acc_so(n, R, x, y, z, vx, vy, vz):
    """Second-order (quadratic) relative acceleration about a circular chief."""
    n2 = n * n
    c = 3.0 * n2 / R
    return (
        3.0 * n2 * x + 2.0 * n * vy - 0.5 * c * (2.0 * x * x - y * y - z * z),
        -2.0 * n * vx + c * x * y,
        -n2 * z + c * x * z,
    )


def acc_nerm(n, R, x, y, z, vx, vy, vz):
    """Exact nonlinear relative acceleration (circular two-body chief).

    Cancellation-free form; mathematically identical to
    ``2 n vy + n^2 (R+x) - mu (R+x)/r^3`` etc. with ``mu = n^2 R^3``.
    """
    n2 = n * n
    q = (2.0 * R * x + x * x + y * y + z * z) / (R * R)
    f = -math.expm1(-1.5 * math.log1p(q))  # 1 - (R/r)^3
    return (
        2.0 * n * vy + n2 * (R + x) * f,
        -2.0 * n * vx + n2 * y * f,
        -n2 * z * (1.0 - f),
    )


def acc_nerm_naive(n, R, x, y, z, vx, vy, vz, mu=MU_EARTH):
    """Direct (cancellation-prone) form of the NERM, used only in tests."""
    rx = R + x
    r = math.sqrt(rx * rx + y * y + z * z)
    k = mu / (r * r * r)
    n2 = n * n
    return (2.0 * n * vy + n2 * rx - k * rx, -2.0 * n * vx + n2 * y - k * y, -k * z)


RELATIVE_MODELS = {"cw": acc_cw, "so": acc_so, "nerm": acc_nerm}


def relative_acc_vectorized(model, n, R, states):
    """Vectorised natural acceleration for an (N, 6) array of relative states."""
    s = np.asarray(states, dtype=float)
    x, y, z, vx, vy, vz = (s[:, i] for i in range(6))
    n2 = n * n
    if model == "cw":
        return np.column_stack((3 * n2 * x + 2 * n * vy, -2 * n * vx, -n2 * z))
    if model == "so":
        c = 3 * n2 / R
        return np.column_stack((
            3 * n2 * x + 2 * n * vy - 0.5 * c * (2 * x * x - y * y - z * z),
            -2 * n * vx + c * x * y,
            -n2 * z + c * x * z,
        ))
    if model == "nerm":
        q = (2 * R * x + x * x + y * y + z * z) / (R * R)
        f = -np.expm1(-1.5 * np.log1p(q))
        return np.column_stack((2 * n * vy + n2 * (R + x) * f, -2 * n * vx + n2 * y * f, -n2 * z * (1 - f)))
    raise ValueError(f"unknown relative model {model!r}")


# ---------------------------------------------------------------------------
# Inertial force model (scalar), consistent with src.perturbations
# ---------------------------------------------------------------------------


class InertialForce:
    """Scalar two-body (+J2, +drag) acceleration matching ``src.perturbations``.

    Parameters mirror :class:`src.perturbations.ForceModel`. ``cd*area/mass``
    is the ballistic parameter; it can differ between target and chaser to
    study differential drag.
    """

    __slots__ = ("mu", "use_j2", "use_drag", "r_earth", "j2", "cd", "area", "mass",
                 "omega_earth", "rho0", "h_ref", "scale_height", "_bc")

    def __init__(self, use_j2=False, use_drag=False, mu=MU_EARTH, r_earth=R_EARTH,
                 j2=J2_EARTH, cd=2.2, area=1.0, mass=100.0, omega_earth=OMEGA_EARTH,
                 rho0=3.9e-12, h_ref=400_000.0, scale_height=60_000.0):
        self.mu = mu
        self.use_j2 = bool(use_j2)
        self.use_drag = bool(use_drag)
        self.r_earth = r_earth
        self.j2 = j2
        self.cd = cd
        self.area = area
        self.mass = mass
        self.omega_earth = omega_earth
        self.rho0 = rho0
        self.h_ref = h_ref
        self.scale_height = scale_height
        self._bc = cd * area / mass

    @property
    def label(self):
        parts = ["two-body"]
        if self.use_j2:
            parts.append("J2")
        if self.use_drag:
            parts.append(f"drag(Cd*A/m={self._bc:.4g} m^2/kg)")
        return " + ".join(parts)

    def as_dict(self):
        return {k: getattr(self, k) for k in self.__slots__ if not k.startswith("_")}

    def perturbation(self, px, py, pz, vx, vy, vz):
        """Non-Keplerian acceleration (J2 + drag) only."""
        ax = ay = az = 0.0
        r2 = px * px + py * py + pz * pz
        r = math.sqrt(r2)
        if self.use_j2:
            factor = -1.5 * self.j2 * self.mu * self.r_earth ** 2 / (r2 * r2 * r)
            zr2 = 5.0 * pz * pz / r2
            ax += factor * px * (1.0 - zr2)
            ay += factor * py * (1.0 - zr2)
            az += factor * pz * (3.0 - zr2)
        if self.use_drag:
            alt = r - self.r_earth
            if alt >= -self.h_ref:
                rho = self.rho0 * math.exp(-(alt - self.h_ref) / self.scale_height)
                we = self.omega_earth
                # v_rel = v - omega_e x r, omega_e = (0, 0, we)
                wx = vx + we * py
                wy = vy - we * px
                wz = vz
                sp = math.sqrt(wx * wx + wy * wy + wz * wz)
                k = -0.5 * rho * self._bc * sp
                ax += k * wx
                ay += k * wy
                az += k * wz
        return ax, ay, az

    def acceleration(self, px, py, pz, vx, vy, vz):
        r2 = px * px + py * py + pz * pz
        r = math.sqrt(r2)
        k = -self.mu / (r2 * r)
        ax, ay, az = k * px, k * py, k * pz
        if self.use_j2 or self.use_drag:
            bx, by, bz = self.perturbation(px, py, pz, vx, vy, vz)
            ax += bx
            ay += by
            az += bz
        return ax, ay, az


# ---------------------------------------------------------------------------
# LVLH frame (scalar), with optional exact frame rate for non-Keplerian target
# ---------------------------------------------------------------------------


def lvlh_basis(tx, ty, tz, tvx, tvy, tvz):
    """Return (xhat, yhat, zhat, r, h) for the target state (scalars/tuples)."""
    r = math.sqrt(tx * tx + ty * ty + tz * tz)
    xh = (tx / r, ty / r, tz / r)
    hx = ty * tvz - tz * tvy
    hy = tz * tvx - tx * tvz
    hz = tx * tvy - ty * tvx
    h = math.sqrt(hx * hx + hy * hy + hz * hz)
    zh = (hx / h, hy / h, hz / h)
    yh = (zh[1] * xh[2] - zh[2] * xh[1], zh[2] * xh[0] - zh[0] * xh[2], zh[0] * xh[1] - zh[1] * xh[0])
    return xh, yh, zh, r, h


def frame_rate(r, h, a_perp_h):
    """LVLH angular velocity components (wx, wz) in LVLH axes.

    wz = h / r^2 (in-plane rate); wx = r * a_h / h, where a_h is the target's
    non-Keplerian acceleration along the orbit normal. The baseline engine uses
    wx = 0 (exact only for an unperturbed target).
    """
    return r * a_perp_h / h, h / (r * r)


def eci_to_lvlh_scalar(t6, c6, a_target_pert=None):
    """Chaser ECI state -> LVLH relative state; exact frame rate if a_target_pert given."""
    tx, ty, tz, tvx, tvy, tvz = t6
    xh, yh, zh, r, h = lvlh_basis(tx, ty, tz, tvx, tvy, tvz)
    dx, dy, dz = c6[0] - tx, c6[1] - ty, c6[2] - tz
    dvx, dvy, dvz = c6[3] - tvx, c6[4] - tvy, c6[5] - tvz
    px = xh[0] * dx + xh[1] * dy + xh[2] * dz
    py = yh[0] * dx + yh[1] * dy + yh[2] * dz
    pz = zh[0] * dx + zh[1] * dy + zh[2] * dz
    if a_target_pert is None:
        wx, wz = 0.0, h / (r * r)
    else:
        ah = zh[0] * a_target_pert[0] + zh[1] * a_target_pert[1] + zh[2] * a_target_pert[2]
        wx, wz = frame_rate(r, h, ah)
    # rho_dot = C dv - omega x rho, omega = (wx, 0, wz)
    vx = xh[0] * dvx + xh[1] * dvy + xh[2] * dvz + wz * py
    vy = yh[0] * dvx + yh[1] * dvy + yh[2] * dvz - (wz * px - wx * pz)
    vz = zh[0] * dvx + zh[1] * dvy + zh[2] * dvz - wx * py
    return (px, py, pz, vx, vy, vz), (xh, yh, zh)


def lvlh_to_eci(target6, rel6, a_target_pert=None):
    """LVLH relative state -> chaser ECI state (numpy arrays)."""
    t = np.asarray(target6, dtype=float)
    rel = np.asarray(rel6, dtype=float)
    xh, yh, zh, r, h = lvlh_basis(*t)
    C = np.array([xh, yh, zh])  # rows: LVLH axes in ECI; v_lvlh = C v_eci
    if a_target_pert is None:
        wx, wz = 0.0, h / (r * r)
    else:
        ah = float(np.dot(zh, a_target_pert))
        wx, wz = frame_rate(r, h, ah)
    w = np.array([wx, 0.0, wz])
    rho, rhod = rel[:3], rel[3:]
    dr = C.T @ rho
    dv = C.T @ (rhod + np.cross(w, rho))
    return np.concatenate((t[:3] + dr, t[3:] + dv))


def eci_to_lvlh_vectorized(target_states, chaser_states, a_target_pert=None):
    """(N,6) target and chaser ECI arrays -> (N,6) LVLH relative states.

    ``a_target_pert`` (N,3) enables the exact frame rate (wx term)."""
    T = np.asarray(target_states, dtype=float)
    Cs = np.asarray(chaser_states, dtype=float)
    r_vec, v_vec = T[:, :3], T[:, 3:]
    r = np.linalg.norm(r_vec, axis=1)
    xh = r_vec / r[:, None]
    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec, axis=1)
    zh = h_vec / h[:, None]
    yh = np.cross(zh, xh)
    dr = Cs[:, :3] - r_vec
    dv = Cs[:, 3:] - v_vec
    px = np.einsum("ij,ij->i", xh, dr)
    py = np.einsum("ij,ij->i", yh, dr)
    pz = np.einsum("ij,ij->i", zh, dr)
    wz = h / r**2
    if a_target_pert is None:
        wx = np.zeros_like(wz)
    else:
        ah = np.einsum("ij,ij->i", zh, np.asarray(a_target_pert, dtype=float))
        wx = r * ah / h
    vx = np.einsum("ij,ij->i", xh, dv) + wz * py
    vy = np.einsum("ij,ij->i", yh, dv) - (wz * px - wx * pz)
    vz = np.einsum("ij,ij->i", zh, dv) - wx * py
    return np.column_stack((px, py, pz, vx, vy, vz)), (xh, yh, zh)


__all__ = [
    "acc_cw", "acc_so", "acc_nerm", "acc_nerm_naive", "RELATIVE_MODELS",
    "relative_acc_vectorized", "InertialForce", "lvlh_basis", "frame_rate",
    "eci_to_lvlh_scalar", "lvlh_to_eci", "eci_to_lvlh_vectorized",
]
