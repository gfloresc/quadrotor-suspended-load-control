#!/usr/bin/env python3
"""
benchmarks.py — Lee (TCST 2018) and Yang & Xian (TCST 2020) implemented on the
SAME plant, reference and initial condition as quadrotor_core.py, for the
comparative study requested by both reviewers.

Frame convention of this work: e3 points UP, gravity is -g e3, and q is the
unit vector from the drone to the load (q = -e3 at rest).

LEE (2018), Section III, point-mass specialisation (n=1, rho=0), integral terms
removed (the exponential variant discussed after Proposition 1):
    F_d   = mL(-k_x e_pL - k_v e_vL + pLd_ddot + g e3)            [Eq. (20)]
    q_d   = -F_d/||F_d||                                          [Eq. (25)]
    a     = F_d/mL                       (commanded payload accel, Eq. (18))
    u_par = q q^T F_d + mq l ||s||^2 q + mq q q^T a               [Eq. (17)]
    e_q   = q_d x q,   e_w = w + hat(q)^2 w_d,  w = q x s         [Sec. III-B]
    u_perp= mq l hat(q){-kq e_q - kw e_w - (q.w_d) qdot} - mq hat(q)^2 a
                                                                  [Eq. (27)]
    u     = u_par + u_perp,  T = u.b3                             [Eqs. (29),(37)]

YANG & XIAN (2020), Section IV-C:
    u_Q   = -KpQ e_p - KdQ e_v                                    [Eq. (41)]
    u_L   = -KpL gamma - KdL gamma_dot                            [Eq. (42)]
    w_Q   = -g e3 + (mL l/(mL+mq))||s||^2 q - pd_ddot             [Eq. (15)]
    F_d   = g_Q^{-1}(-w_Q + u_Q + l [c_gy q2, q3] u_L)            [Eq. (28)]
    g_Q^{-1} = mq (I + mu q q^T)                                  [Eq. (29)]
with gamma = (gamma_x, gamma_y) the swing angles of
    q = [s_gx c_gy, s_gy, -c_gx c_gy]^T.
"""
import numpy as np
from scipy.integrate import solve_ivp
import quadrotor_core as c

H = 1e-6


# ----------------------------------------------------------------- Lee (2018)
def lee_outer(t, pq, vq, q, s, gains):
    k_x, k_v, k_q, k_w = gains
    mL, mq, ell, g, e3 = c.mL, c.mq, c.ell, c.g, c.e3
    pd, vd, ad = c.ref(t)
    pLd, vLd, aLd = pd - ell * e3, vd, ad          # load reference
    pL, vL = pq + ell * q, vq + ell * s
    Fd = mL * (-k_x * (pL - pLd) - k_v * (vL - vLd) + aLd + g * e3)
    nF = np.linalg.norm(Fd)
    qd = -Fd / nF if nF > 1e-9 else -e3
    a = Fd / mL
    w = np.cross(q, s)                              # cable angular velocity
    # desired cable angular velocity by directional derivative
    pd2, vd2, ad2 = c.ref(t + H)
    pL2, vL2 = pL + H * vL, vL + H * a
    Fd2 = mL * (-k_x * (pL2 - (pd2 - ell * e3)) - k_v * (vL2 - vd2) + ad2 + g * e3)
    qd2 = -Fd2 / max(np.linalg.norm(Fd2), 1e-9)
    qd_dot = (qd2 - qd) / H
    wd = np.cross(qd, qd_dot)
    e_q = np.cross(qd, q)
    e_w = w + c.hat(q) @ c.hat(q) @ wd
    u_par = np.outer(q, q) @ Fd + mq * ell * float(s @ s) * q + mq * np.outer(q, q) @ a
    inner = -k_q * e_q - k_w * e_w - float(q @ wd) * s
    u_perp = mq * ell * (c.hat(q) @ inner) - mq * (c.hat(q) @ c.hat(q)) @ a
    return u_par + u_perp                           # force [N]


# -------------------------------------------------------- Yang & Xian (2020)
def swing_angles(q):
    gy = np.arcsin(np.clip(q[1], -1.0, 1.0))
    gx = np.arctan2(q[0], -q[2])
    return gx, gy


def yang_outer(t, pq, vq, q, s, gains):
    KpQ, KdQ, KpL, KdL = gains
    mL, mq, ell, g, e3, mu = c.mL, c.mq, c.ell, c.g, c.e3, c.mu
    pd, vd, ad = c.ref(t)
    gx, gy = swing_angles(q)
    cx, sx, cy, sy = np.cos(gx), np.sin(gx), np.cos(gy), np.sin(gy)
    q2 = np.array([cx, 0.0, sx])                    # dq/dgx / c_gy
    q3 = np.array([-sx * sy, cy, cx * sy])          # dq/dgy
    B = np.column_stack([cy * q2, q3])              # 3x2
    # gamma_dot from s = dq/dgx gx_dot + dq/dgy gy_dot  (B is orthogonal basis)
    gdx = float(q2 @ s) / max(cy, 1e-6)
    gdy = float(q3 @ s)
    u_Q = -KpQ * (pq - pd) - KdQ * (vq - vd)
    u_L = -KpL * np.array([gx, gy]) - KdL * np.array([gdx, gdy])
    w_Q = -g * e3 + (mL * ell / (mL + mq)) * float(s @ s) * q - ad
    return mq * (np.eye(3) + mu * np.outer(q, q)) @ (-w_Q + u_Q + ell * (B @ u_L))


# --------------------------------------------------------------- common plant
def simulate(outer, gains, KR=250., KOm=20., t_end=22.0, n=2200, theta0=20.0):
    mL, mq, ell, g, ct, mu, e3 = c.mL, c.mq, c.ell, c.g, c.ct, c.mu, c.e3
    c.THETA0 = np.radians(theta0)
    q0 = c.q_init()

    def force(t, pq, vq, q, s):
        return outer(t, pq, vq, q, s, gains)

    def ode(t, X):
        pq, vq = X[:3], X[3:6]
        R = X[6:15].reshape(3, 3); Om = X[15:18]
        q = c.nrm(X[18:21]); s = c.prj(q, X[21:24])
        F = force(t, pq, vq, q, s)
        b3 = R @ e3
        T = float(F @ b3)
        s2 = float(s @ s)
        vdot = (np.eye(3) - mu / (1 + mu) * np.outer(q, q)) @ (T / mq * b3 - g * e3) \
            - mu / (1 + mu) * (float(q @ (g * e3)) - ell * s2) * q
        sdot = (-1 / ell) * c.PT(q) @ (vdot + g * e3) \
            - ct / (mL * ell) * c.PT(q) @ (vq + ell * s) - s2 * q
        Rd = c.DR(c.nrm(F))
        F2 = force(t + H, pq + H * vq, vq + H * vdot, c.nrm(q + H * s), s + H * sdot)
        Omd = c.vee(Rd.T @ ((c.DR(c.nrm(F2)) - Rd) / H))
        Mx = Rd.T @ R
        eR = c.vee(Mx - Mx.T) / (2 * np.sqrt(max(1 + np.trace(Mx), 1e-9)))
        eOm = Om - Mx @ Omd
        tau = -KR * eR - KOm * eOm + np.cross(Om, c.J @ Om) - c.J @ (c.hat(Om) @ Mx @ Omd)
        return np.concatenate([vq, vdot, (R @ c.hat(Om)).flatten(),
                               c.Jinv @ (tau - np.cross(Om, c.J @ Om)), s, sdot])

    # aligned initial attitude, as in quadrotor_core.x0()
    F0 = force(0., np.array([0., 0., c.ZH]), np.zeros(3), q0, np.zeros(3))
    R0 = c.DR(c.nrm(F0))
    X0 = np.concatenate([np.array([0., 0., c.ZH]), np.zeros(3),
                         R0.flatten(), np.zeros(3), q0, np.zeros(3)])
    te = np.linspace(0, t_end, n)
    sol = solve_ivp(ode, [0, t_end], X0, method='RK45', rtol=1e-6, atol=1e-8, t_eval=te)
    if not sol.success:
        return None
    t = sol.t
    PQ, VQ = sol.y[:3].T, sol.y[3:6].T
    QA = sol.y[18:21].T; QA /= np.linalg.norm(QA, axis=1, keepdims=True)
    SA = np.array([c.prj(QA[k], sol.y[21:24].T[k]) for k in range(len(t))])
    PD = np.array([c.ref(float(x))[0] for x in t])
    epn = np.linalg.norm(PQ - PD, axis=1)
    th = np.degrees(np.arccos(np.clip(-QA[:, 2], -1, 1)))
    LAM = np.zeros(len(t)); TT = np.zeros(len(t))
    for k in range(len(t)):
        R = c.reorth(sol.y[6:15].T[k].reshape(3, 3)); b3 = R @ e3
        F = force(float(t[k]), PQ[k], VQ[k], QA[k], SA[k])
        T = float(F @ b3); TT[k] = T
        LAM[k] = -(mL / (1 + mu)) * (float(QA[k] @ (T / mq * b3)) - ell * float(SA[k] @ SA[k]))
    last = t >= (t_end - c.T8)
    return dict(t=t, ep=epn, th=th, lam=LAM, T=TT,
                ep_rms=float(np.sqrt(np.mean(epn[last] ** 2))), ep_peak=float(epn.max()),
                ep_fin=float(epn[-1]),
                th_rms=float(np.sqrt(np.mean(th[last] ** 2))), th_peak=float(th.max()),
                th_fin=float(th[-1]), lam_min=float(LAM.min()), T_max=float(TT.max()))


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("lee", "both"):
        for gs in [(8., 5., 12., 6.), (16., 8., 20., 9.)]:
            r = simulate(lee_outer, gs)
            tag = f"Lee  kx={gs[0]:g} kv={gs[1]:g} kq={gs[2]:g} kw={gs[3]:g}"
            print(f"{tag:42s} " + ("DIVERGE" if r is None else
                  f"ep rms={r['ep_rms']:.4f} peak={r['ep_peak']:.4f} | th rms={r['th_rms']:5.2f} "
                  f"fin={r['th_fin']:5.2f} | lam={r['lam_min']:.3f} T={r['T_max']:.1f}"), flush=True)
    if which in ("yang", "both"):
        for gs in [(16., 8., 4., 2.), (36., 12., 9., 4.)]:
            r = simulate(yang_outer, gs)
            tag = f"Yang KpQ={gs[0]:g} KdQ={gs[1]:g} KpL={gs[2]:g} KdL={gs[3]:g}"
            print(f"{tag:42s} " + ("DIVERGE" if r is None else
                  f"ep rms={r['ep_rms']:.4f} peak={r['ep_peak']:.4f} | th rms={r['th_rms']:5.2f} "
                  f"fin={r['th_fin']:5.2f} | lam={r['lam_min']:.3f} T={r['T_max']:.1f}"), flush=True)


# ------------------------------------------- Goodarzi, Lee & Lee (IJCAS 2015)
def goodarzi_outer(t, pq, vq, q, s, gains):
    """Linear state feedback on the linearised dynamics, n=1, Eq. (26) without
    the integral term:  A = -K_x x - K_xdot xdot + M00 g e3.
    The linearised link state is the horizontal part of q and of s.
    No feedforward is added: the law is used exactly as published, which
    targets regulation about the hanging equilibrium."""
    k_x, k_v, k_q, k_w = gains
    mL, mq, e3 = c.mL, c.mq, c.e3
    pd, vd, ad = c.ref(t)
    Ph = np.diag([1.0, 1.0, 0.0])
    return (-k_x * (pq - pd) - k_v * (vq - vd)
            + k_q * (Ph @ q) + k_w * (Ph @ s)
            + (mq + mL) * g_e3())


def g_e3():
    return c.g * c.e3


# ---------------------------------- Sreenath, Lee & Kumar (CDC 2013), Prop. 3
def sreenath_outer(t, pq, vq, q, s, gains):
    """Load-position controlled flight mode. Same frame convention as this
    work (gravity -g e3, thrust f R e3 upward), so no sign conversion.
        A     = -kx ex - kv ev + (mQ+mL)(aLd + g e3) + mQ l (qdot.qdot) q  (37)
        q_c   = -A/||A||                                                  (36)
        e_q   = hat(q)^2 q_d,   e_qdot = qdot - (q_d x qdot_d) x q     (17),(18)
        F_n   = (A.q) q                                                   (38)
        F_pd  = -kq e_q - kw e_qdot                                       (29)
        F_ff  = mQ l <q, q_d x qdot_d>(q x qdot) + mQ l (q_d x qddot_d) x q (30)
        F     = F_n - F_pd - F_ff                                         (27)
    qddot_d is taken as zero, i.e. the simpler implementation the authors
    themselves report (Section V-B).
    """
    k_x, k_v, k_q, k_w = gains
    mL, mq, ell, g, e3 = c.mL, c.mq, c.ell, c.g, c.e3
    pd, vd, ad = c.ref(t)
    pLd, vLd, aLd = pd - ell * e3, vd, ad
    pL, vL = pq + ell * q, vq + ell * s
    s2 = float(s @ s)

    def Avec(pL_, vL_, q_, s2_, pLd_, vLd_, aLd_):
        return (-k_x * (pL_ - pLd_) - k_v * (vL_ - vLd_)
                + (mq + mL) * (aLd_ + g * e3) + mq * ell * s2_ * q_)

    A = Avec(pL, vL, q, s2, pLd, vLd, aLd)
    qd = -A / max(np.linalg.norm(A), 1e-9)
    # qdot_d by directional derivative along the flow
    pd2, vd2, ad2 = c.ref(t + H)
    A2 = Avec(pL + H * vL, vL + H * (A / (mq + mL)), c.nrm(q + H * s), s2,
              pd2 - ell * e3, vd2, ad2)
    qd2 = -A2 / max(np.linalg.norm(A2), 1e-9)
    qd_dot = (qd2 - qd) / H
    wd = np.cross(qd, qd_dot)
    e_q = (c.hat(q) @ c.hat(q)) @ qd
    e_qd = s - np.cross(wd, q)
    F_n = float(A @ q) * q
    F_pd = -k_q * e_q - k_w * e_qd
    F_ff = mq * ell * float(q @ wd) * np.cross(q, s)
    return F_n - F_pd - F_ff
