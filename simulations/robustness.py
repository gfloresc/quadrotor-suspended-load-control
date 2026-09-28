#!/usr/bin/env python3
"""
robustness.py — noise and parametric-uncertainty study requested by R1.5/R1.6.

The plant uses the true parameters and the true state; the controller sees
(i) noisy cable measurements and (ii) perturbed parameters. Integration is
fixed-step RK4 at 1 kHz with control and noise held at the control rate, since
white noise inside an adaptive solver is not well posed.
"""
import numpy as np
import quadrotor_core as c

g, e3 = c.g, c.e3


def ctrl_force(t, pq, vq, q_m, s_m, P):  # pq,vq son MEDIDOS
    """Eq. (25) evaluated with the controller's parameters P and measurements."""
    mLh, ellh, mqh = P['mL'], P['ell'], c.mq
    muh = mLh / mqh
    pd, vd, ad = c.ref(t)
    v = g * e3 + ad - c.Kp * (pq - pd) - c.Kd * (vq - vd) - c.Kc * s_m
    u = (np.eye(3) + muh * np.outer(q_m, q_m)) @ v \
        - muh * ellh * float(s_m @ s_m) * q_m
    return c.mq * u


def deriv(t, X, P, q_m, s_m, KR, KOm, Omd, pq_m=None, vq_m=None,
          fdist=None, Tmax=None, taumax=None):
    pq, vq = X[:3], X[3:6]
    R = X[6:15].reshape(3, 3); Om = X[15:18]
    q = c.nrm(X[18:21]); s = c.prj(q, X[21:24])
    F = ctrl_force(t, pq if pq_m is None else pq_m,
                   vq if vq_m is None else vq_m, q_m, s_m, P)
    b3 = R @ e3
    T = float(F @ b3)
    if Tmax is not None:
        T = float(np.clip(T, 0.0, Tmax))
    s2 = float(s @ s); mu = c.mu
    vdot = (np.eye(3) - mu / (1 + mu) * np.outer(q, q)) @ (T / c.mq * b3 - g * e3) \
        - mu / (1 + mu) * (float(q @ (g * e3)) - c.ell * s2) * q
    if fdist is not None:                       # fuerza externa sobre la carga
        fd = fdist(t)
        vdot = vdot + c.mu / (1 + c.mu) * (np.eye(3) - np.outer(q, q)) @ fd / c.mL
    sdot = (-1 / c.ell) * c.PT(q) @ (vdot + g * e3) \
        - c.ct / (c.mL * c.ell) * c.PT(q) @ (vq + c.ell * s) - s2 * q
    Rd = c.DR(c.nrm(F))
    Mx = Rd.T @ R
    eR = c.vee(Mx - Mx.T) / (2 * np.sqrt(max(1 + np.trace(Mx), 1e-9)))
    tau = -KR * eR - KOm * (Om - Mx @ Omd) + np.cross(Om, c.J @ Om)
    if taumax is not None:
        nt = np.linalg.norm(tau)
        if nt > taumax: tau = tau * (taumax / nt)
    Omdot = c.Jinv @ (tau - np.cross(Om, c.J @ Om))
    return np.concatenate([vq, vdot, (R @ c.hat(Om)).flatten(), Omdot, s, sdot]), T, tau


def run(sigma_q_deg=0.0, sigma_s=0.0, sigma_p=0.0, sigma_v=0.0,
        dmL=0.0, dell=0.0, dct=0.0, wind=0.0, Tmax=None, taumax=None,
        KR=250., KOm=20., t_end=22.0, dt=1e-3, ctrl_dt=5e-3, seed=0,
        theta0=20.0, series=False, obs=None):
    rng = np.random.default_rng(seed)
    c.THETA0 = np.radians(theta0)
    P = {'mL': c.mL * (1 + dmL), 'ell': c.ell * (1 + dell), 'ct': c.ct * (1 + dct)}
    sq = np.radians(sigma_q_deg)
    X = c.x0()
    n = int(round(t_end / dt)); every = int(round(ctrl_dt / dt))
    q_m = c.nrm(X[18:21]); s_m = c.prj(q_m, X[21:24])
    pq_m = X[:3].copy(); vq_m = X[3:6].copy()
    # observador de Luenberger (86) sobre el doble integrador del dron
    ph = X[:3].copy(); vh = X[3:6].copy(); aq = np.zeros(3)
    if obs is not None:
        L1, L2 = 2 * obs['zeta'] * obs['wo'], obs['wo'] ** 2
    fd = (lambda tt: wind * np.array([np.sin(0.7 * tt), np.cos(1.1 * tt), 0.0])) if wind else None
    Rd_prev = None; Omd = np.zeros(3)
    ts, EP, TH, LAM, TT, TAU = [], [], [], [], [], []
    for k in range(n + 1):
        t = k * dt
        if k % every == 0:                       # sample-and-hold of measurements
            q_t = c.nrm(X[18:21]); s_t = c.prj(q_t, X[21:24])
            q_m = c.nrm(q_t + sq * rng.standard_normal(3))
            s_m = c.prj(q_m, s_t + sigma_s * rng.standard_normal(3))
            pq_raw = X[:3] + sigma_p * rng.standard_normal(3)
            vq_raw = X[3:6] + sigma_v * rng.standard_normal(3)
            if obs is None:
                pq_m, vq_m = pq_raw, vq_raw
            else:                                   # una actualizacion por paso de control
                e = pq_raw - ph
                ph = ph + ctrl_dt * (vh + L1 * e)
                vh = vh + ctrl_dt * (aq + L2 * e)
                pq_m, vq_m = ph.copy(), vh.copy()
            Rd_now = c.DR(c.nrm(ctrl_force(t, pq_m, vq_m, q_m, s_m, P)))
            if Rd_prev is not None:
                Omd = c.vee(Rd_now.T @ ((Rd_now - Rd_prev) / ctrl_dt))
            Rd_prev = Rd_now
        d1, T, tau = deriv(t, X, P, q_m, s_m, KR, KOm, Omd, pq_m, vq_m, fd, Tmax, taumax)
        aq = d1[3:6]                                # aceleracion medida (IMU del dron)
        if k % 10 == 0:
            q = c.nrm(X[18:21]); s = c.prj(q, X[21:24])
            pd = c.ref(t)[0]
            lam = -(c.mL / (1 + c.mu)) * (float(q @ (T / c.mq * (X[6:15].reshape(3, 3) @ e3)))
                                          - c.ell * float(s @ s))
            ts.append(t); EP.append(np.linalg.norm(X[:3] - pd))
            TH.append(np.degrees(np.arccos(np.clip(-q[2], -1, 1))))
            LAM.append(lam); TT.append(T); TAU.append(np.linalg.norm(tau))
        d2, _, _ = deriv(t + dt/2, X + dt/2*d1, P, q_m, s_m, KR, KOm, Omd, pq_m, vq_m, fd, Tmax, taumax)
        d3, _, _ = deriv(t + dt/2, X + dt/2*d2, P, q_m, s_m, KR, KOm, Omd, pq_m, vq_m, fd, Tmax, taumax)
        d4, _, _ = deriv(t + dt, X + dt*d3, P, q_m, s_m, KR, KOm, Omd, pq_m, vq_m, fd, Tmax, taumax)
        X = X + dt / 6 * (d1 + 2 * d2 + 2 * d3 + d4)
        R = c.reorth(X[6:15].reshape(3, 3)); X[6:15] = R.flatten()
        q = c.nrm(X[18:21]); X[18:21] = q; X[21:24] = c.prj(q, X[21:24])
        if not np.all(np.isfinite(X)) or np.linalg.norm(X[:3]) > 50:
            return None
    ts = np.array(ts); EP = np.array(EP); TH = np.array(TH)
    LAM = np.array(LAM); TT = np.array(TT); TAU = np.array(TAU)
    last = ts >= (t_end - c.T8)
    if series:
        return dict(t=ts, ep=EP, th=TH, lam=LAM, T=TT, tau=TAU)
    return dict(ep_rms=float(np.sqrt(np.mean(EP[last] ** 2))), ep_peak=float(EP.max()),
                th_rms=float(np.sqrt(np.mean(TH[last] ** 2))), th_fin=float(TH[-1]),
                lam_min=float(LAM.min()), T_max=float(TT.max()), tau_max=float(TAU.max()))


if __name__ == "__main__":
    import sys
    c.Kp, c.Kd, c.Kc = 140., 12., 3.
    cases = [
        ("nominal",                  dict()),
        ("noise q 0.5 deg, s 0.05",  dict(sigma_q_deg=0.5, sigma_s=0.05)),
        ("noise q 1.0 deg, s 0.10",  dict(sigma_q_deg=1.0, sigma_s=0.10)),
        ("mL +20%",                  dict(dmL=+0.20)),
        ("mL -20%",                  dict(dmL=-0.20)),
        ("ell +10%",                 dict(dell=+0.10)),
        ("ct -50%",                  dict(dct=-0.50)),
        ("all: noise + mL+20% + ell+10%",
         dict(sigma_q_deg=0.5, sigma_s=0.05, dmL=0.20, dell=0.10)),
    ]
    sel = int(sys.argv[1]) if len(sys.argv) > 1 else None
    for i, (name, kw) in enumerate(cases):
        if sel is not None and i != sel:
            continue
        r = run(**kw)
        print(f"{name:34s} " + ("DIVERGE" if r is None else
              f"ep_rms={r['ep_rms']:.4f} peak={r['ep_peak']:.4f} | th_rms={r['th_rms']:5.2f} "
              f"fin={r['th_fin']:5.2f} | lam={r['lam_min']:.3f} | T={r['T_max']:.1f} "
              f"tau={r['tau_max']:.1f}"), flush=True)
