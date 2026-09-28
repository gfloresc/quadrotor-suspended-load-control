#!/usr/bin/env python3
"""
quadrotor_core.py — clean simulation core for the AESCTE revision.

Implements the manuscript equations literally:

  Outer loop, Eq. (25):
      u* = (I + mu q q^T)(g e3 + a_d - Kp ep - Kd ev - Kc s) - mu l ||s||^2 q
  Thrust allocation, Eq. (28):
      T* = mq (u* . b3)          [projected form; see NOTE 1]
      b3* = u*/||u*||  ->  Rd    [computed ONLINE from the current state]
  Inner loop, Eqs. (30)-(31) without the saturation functions:
      eR  = vee(Rd^T R - R^T Rd) / (2 sqrt(1 + tr(Rd^T R)))
      eOm = Om - R^T Rd Om_d
      tau = -KR eR - KOm eOm + Om x J Om - J (hat(Om) R^T Rd Om_d)
  Tension monitor, Eq. (24):
      lambda_hat = -(mL/(1+mu)) (q . u_applied - l ||s||^2)

NOTE 1: the released code used T = mq ||u*||. With an online Rd that form is
        unstable, because a misaligned b3 keeps full thrust magnitude pointing
        the wrong way. The projected form self-corrects and is standard.
NOTE 2: Om_d is obtained by a directional derivative of Rd along the flow
        (one extra evaluation, H = 1e-6). Om_d_dot is set to zero.

Plant: full 24-state model, Eqs. (21a)-(21f).
"""
import numpy as np
from scipy.integrate import solve_ivp

# ----------------------------------------------------------------- parameters
mq, mL, ell, g, ct = 1.20, 0.35, 0.80, 9.81, 0.25
mu = mL / mq
J = np.diag([1.5e-2, 1.5e-2, 2.5e-2])
Jinv = np.linalg.inv(J)
e3 = np.array([0., 0., 1.])
I3 = np.eye(3)

# gains (defaults = tuned set; the manuscript set is Kp=45, Kd=12, Kc=.5, KR=6, KOm=2)
Kp, Kd, Kc, KR, KOm = 140., 12., 6., 250., 20.

# trajectory
A_AMP, T8, ZH, TRAMP, T_END = 1.4, 10.0, 1.5, 2.5, 22.0
THETA0, PHI0 = np.radians(20.0), np.radians(30.0)
H_FD = 1e-6

# ------------------------------------------------------------------- helpers
def hat(x):
    return np.array([[0., -x[2], x[1]], [x[2], 0., -x[0]], [-x[1], x[0], 0.]])

def vee(M):
    return np.array([M[2, 1], M[0, 2], M[1, 0]])

def nrm(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v

def PT(q):
    return I3 - np.outer(q, q)

def prj(q, s):
    return PT(q) @ s

def reorth(R):
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    return R if np.linalg.det(R) > 0 else U @ np.diag([1., 1., -1.]) @ Vt

def DR(b):
    """Desired rotation from a unit thrust direction (yaw fixed by b1c)."""
    b = nrm(b)
    bc = np.array([1., 0., 0.]) if abs(b[0]) < 0.9 else np.array([0., 1., 0.])
    b2 = nrm(np.cross(b, bc))
    return np.column_stack([np.cross(b2, b), b2, b])

def ref(t):
    """Lemniscate of Bernoulli with a quintic start-up envelope, Eq. (82).
    Analytic derivatives (finite differencing here is numerically fatal)."""
    tau = min(max(t / TRAMP, 0.), 1.)
    S = 10. * tau**3 - 15. * tau**4 + 6. * tau**5
    Sd = (30. * tau**2 - 60. * tau**3 + 30. * tau**4) / TRAMP if t < TRAMP else 0.
    Sd2 = (60. * tau - 180. * tau**2 + 120. * tau**3) / TRAMP**2 if t < TRAMP else 0.
    w = 2. * np.pi / T8
    c, s_ = np.cos(w * t), np.sin(w * t)
    D = 1. + s_**2
    D3 = D**3
    X, Y = A_AMP * c / D, A_AMP * c * s_ / D
    dX = -A_AMP * w * (s_ + 2. * s_ * c**2) / D**2
    dY = A_AMP * w * (c**2 - s_**2) / D**2
    dX2 = -A_AMP * w**2 * (c - 4. * c**3 + 2. * c * s_**2 * (3. - 4. * s_**2)) / D3
    dY2 = -A_AMP * w**2 * (s_ * (1. + 4. * c**2 - 4. * s_**2 * c**2)) / D3
    pd = np.array([S * X, S * Y, ZH])
    vd = np.array([Sd * X + S * dX, Sd * Y + S * dY, 0.])
    ad = np.array([Sd2 * X + 2. * Sd * dX + S * dX2,
                   Sd2 * Y + 2. * Sd * dY + S * dY2, 0.])
    return pd, vd, ad


def outer_law(q, s, ep, ev, ad):
    """Eq. (25)."""
    v = g * e3 + ad - Kp * ep - Kd * ev - Kc * s
    return (I3 + mu * np.outer(q, q)) @ v - mu * ell * float(np.dot(s, s)) * q

def q_init():
    return np.array([np.sin(THETA0) * np.cos(PHI0),
                     np.sin(THETA0) * np.sin(PHI0), -np.cos(THETA0)])

def x0(align_attitude=True):
    """Initial state. With align_attitude=True the vehicle starts aligned with
    the commanded thrust direction, R(0)=Rd(0), removing the artificial
    attitude mismatch at t=0 caused by the initial cable deflection."""
    q0 = q_init(); s0 = np.zeros(3)
    pd, vd, ad = ref(0.)
    u0 = outer_law(q0, s0, np.array([0., 0., ZH]) - pd, -vd, ad)
    R0 = DR(nrm(u0)) if align_attitude else I3
    Om0 = np.zeros(3)
    if align_attitude:
        # also match the desired angular velocity, so that e_Omega(0)=0
        _, _, _, _, _, Rd0, vdot0, sdot0 = ctrl(0., np.array([0., 0., ZH]),
                                                np.zeros(3), R0, np.zeros(3), q0, s0)
        pd2, vd2, ad2 = ref(H_FD)
        u2 = outer_law(nrm(q0 + H_FD*s0), s0 + H_FD*sdot0,
                       np.array([0., 0., ZH]) - pd2, H_FD*vdot0 - vd2, ad2)
        Om0 = vee(Rd0.T @ ((DR(nrm(u2)) - Rd0) / H_FD))
    return np.concatenate([np.array([0., 0., ZH]), np.zeros(3),
                           R0.flatten(), Om0, q0, np.zeros(3)])

# --------------------------------------------------------------- control law
def ctrl(t, pq, vq, R, Om, q, s):
    """Returns T, tau, u_applied, lambda_hat, eR, Rd."""
    pd, vd, ad = ref(t)
    u = outer_law(q, s, pq - pd, vq - vd, ad)
    b3 = R @ e3
    T = mq * float(np.dot(u, b3))
    s2 = float(np.dot(s, s))
    # plant accelerations needed for the Om_d directional derivative
    vdot = (I3 - mu / (1 + mu) * np.outer(q, q)) @ (T / mq * b3 - g * e3) \
        - mu / (1 + mu) * (np.dot(q, g * e3) - ell * s2) * q
    sdot = (-1 / ell) * PT(q) @ (vdot + g * e3) \
        - ct / (mL * ell) * PT(q) @ (vq + ell * s) - s2 * q
    Rd = DR(nrm(u))
    pd2, vd2, ad2 = ref(t + H_FD)
    u2 = outer_law(nrm(q + H_FD * s), s + H_FD * sdot,
                   pq + H_FD * vq - pd2, vq + H_FD * vdot - vd2, ad2)
    Omd = vee(Rd.T @ ((DR(nrm(u2)) - Rd) / H_FD))
    M = Rd.T @ R
    eR = vee(M - M.T) / (2. * np.sqrt(max(1. + np.trace(M), 1e-9)))   # Eq. (30)
    eOm = Om - M @ Omd
    tau = -KR * eR - KOm * eOm + np.cross(Om, J @ Om) - J @ (hat(Om) @ M @ Omd)
    u_app = T / mq * b3
    lam = -(mL / (1 + mu)) * (float(np.dot(q, u_app)) - ell * s2)      # Eq. (24)
    return T, tau, u_app, lam, eR, Rd, vdot, sdot

# ---------------------------------------------------------------------- plant
def ode(t, X):
    pq, vq = X[:3], X[3:6]
    R = X[6:15].reshape(3, 3)
    Om = X[15:18]
    q = nrm(X[18:21]); s = prj(q, X[21:24])
    _, tau, _, _, _, _, vdot, sdot = ctrl(t, pq, vq, R, Om, q, s)
    Omdot = Jinv @ (tau - np.cross(Om, J @ Om))
    return np.concatenate([vq, vdot, (R @ hat(Om)).flatten(), Omdot, s, sdot])

def simulate(t_end=None, n=2200, rtol=1e-6, atol=1e-8):
    te_end = T_END if t_end is None else t_end
    te = np.linspace(0., te_end, n)
    sol = solve_ivp(ode, [0., te_end], x0(), method='RK45',
                    rtol=rtol, atol=atol, t_eval=te)
    if not sol.success:
        raise RuntimeError(sol.message)
    t = sol.t
    PQ, VQ = sol.y[:3].T, sol.y[3:6].T
    QA = sol.y[18:21].T; QA = QA / np.linalg.norm(QA, axis=1, keepdims=True)
    SA = np.array([prj(QA[k], sol.y[21:24].T[k]) for k in range(len(t))])
    PD = np.array([ref(float(x))[0] for x in t])
    VD = np.array([ref(float(x))[1] for x in t])
    n_t = len(t)
    LAM = np.zeros(n_t); TT = np.zeros(n_t); ERN = np.zeros(n_t)
    TAU = np.zeros((n_t, 3)); UA = np.zeros((n_t, 3))
    for k in range(n_t):
        R = reorth(sol.y[6:15].T[k].reshape(3, 3))
        T, tau, u_app, lam, eR, _, _, _ = ctrl(float(t[k]), PQ[k], VQ[k], R,
                                               sol.y[15:18].T[k], QA[k], SA[k])
        LAM[k], TT[k] = lam, T
        ERN[k] = np.degrees(np.linalg.norm(eR))
        TAU[k], UA[k] = tau, u_app
    return dict(t=t, pq=PQ, vq=VQ, q=QA, s=SA, pd=PD, vd=VD,
                pL=PQ + ell * QA,
                ep=PQ - PD, ev=VQ - VD,
                ep_norm=np.linalg.norm(PQ - PD, axis=1),
                ev_norm=np.linalg.norm(VQ - VD, axis=1),
                theta_deg=np.degrees(np.arccos(np.clip(-QA[:, 2], -1, 1))),
                lam=LAM, T=TT, tau=TAU, u=UA, eR_deg=ERN,
                Om=sol.y[15:18].T)

def metrics(d):
    t = d["t"]; last = t >= (T_END - T8)
    e, th, lam = d["ep_norm"], d["theta_deg"], d["lam"]
    return {
        "ep peak [m]": e.max(), "ep final [m]": e[-1],
        "ep rms last cycle [m]": float(np.sqrt(np.mean(e[last]**2))),
        "theta(0) [deg]": th[0], "theta peak [deg]": th.max(),
        "theta final [deg]": th[-1],
        "theta rms last cycle [deg]": float(np.sqrt(np.mean(th[last]**2))),
        "lambda min [N]": lam.min(), "lambda final [N]": lam[-1],
        "mL g [N]": mL * g,
        "eR peak [deg]": d["eR_deg"].max(),
        "T max [N]": d["T"].max(), "T min [N]": d["T"].min(),
        "|tau| max [N m]": np.linalg.norm(d["tau"], axis=1).max(),
    }

if __name__ == "__main__":
    d = simulate()
    print(f"Gains: Kp={Kp:g} Kd={Kd:g} Kc={Kc:g} KR={KR:g} KOm={KOm:g}\n")
    for k, v in metrics(d).items():
        print(f"  {k:32s} {v:10.4f}")
