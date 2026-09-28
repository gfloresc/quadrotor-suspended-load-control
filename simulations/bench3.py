#!/usr/bin/env python3
"""
bench3.py — Sreenath, Lee & Kumar (CDC 2013) and Lee (TCST 2018) with the
desired cable direction q_d computed online from the current state and
omega_d = 0, i.e. the simplified implementation the original authors
themselves report (Sreenath et al., Sec. V-B, where the command derivatives
are not evaluated). Used for the comparison of Table 5.
"""
import numpy as np, time
from scipy.integrate import solve_ivp
import quadrotor_core as c
g,e3,mL,mq,ell,ct,mu = c.g,c.e3,c.mL,c.mq,c.ell,c.ct,c.mu

def outer(law, t, pq, vq, q, s, gains):
    k_x,k_v,k_q,k_w = gains
    pd,vd,ad = c.ref(t); s2=float(s@s)
    pL,vL = pq+ell*q, vq+ell*s
    pLd,vLd,aLd = pd-ell*e3, vd, ad
    if law=="sreenath":
        A = (-k_x*(pL-pLd) - k_v*(vL-vLd) + (mq+mL)*(aLd+g*e3) + mq*ell*s2*q)
        qd = -A/max(np.linalg.norm(A),1e-9)
        e_q  = (c.hat(q)@c.hat(q))@qd          # Ec. (17)
        e_qd = s                                # Ec. (18) con omega_d = 0
        return float(A@q)*q + k_q*e_q + k_w*e_qd
    else:                                       # lee 2018
        Fd = mL*(-k_x*(pL-pLd) - k_v*(vL-vLd) + aLd + g*e3)
        qd = -Fd/max(np.linalg.norm(Fd),1e-9)
        a  = Fd/mL
        w  = np.cross(q,s)
        e_q = np.cross(qd,q)                    # Sec. III-B
        e_w = w                                  # omega_d = 0
        u_par = np.outer(q,q)@Fd + mq*ell*s2*q + mq*np.outer(q,q)@a
        inner = -k_q*e_q - k_w*e_w
        return u_par + mq*ell*(c.hat(q)@inner) - mq*(c.hat(q)@c.hat(q))@a

def run(law,gains,KR=250.,KOm=20.,t_end=22.,theta0=20.,n=2200,method='RK45'):
    c.THETA0=np.radians(theta0); H=1e-5
    F=lambda t,pq,vq,q,s: outer(law,t,pq,vq,q,s,gains)
    def ode(t,X):
        pq,vq=X[:3],X[3:6]; R=X[6:15].reshape(3,3); Om=X[15:18]
        q=c.nrm(X[18:21]); s=c.prj(q,X[21:24])
        Fv=F(t,pq,vq,q,s); b3=R@e3; T=float(Fv@b3); s2=float(s@s)
        vdot=(np.eye(3)-mu/(1+mu)*np.outer(q,q))@(T/mq*b3-g*e3)-mu/(1+mu)*(float(q@(g*e3))-ell*s2)*q
        sdot=(-1/ell)*c.PT(q)@(vdot+g*e3)-ct/(mL*ell)*c.PT(q)@(vq+ell*s)-s2*q
        Rd=c.DR(c.nrm(Fv))
        F2=F(t+H,pq+H*vq,vq+H*vdot,c.nrm(q+H*s),s+H*sdot)
        Omd=c.vee(Rd.T@((c.DR(c.nrm(F2))-Rd)/H))
        Mx=Rd.T@R; eR=c.vee(Mx-Mx.T)/(2*np.sqrt(max(1+np.trace(Mx),1e-9)))
        tau=-KR*eR-KOm*(Om-Mx@Omd)+np.cross(Om,c.J@Om)-c.J@(c.hat(Om)@Mx@Omd)
        return np.concatenate([vq,vdot,(R@c.hat(Om)).flatten(),c.Jinv@(tau-np.cross(Om,c.J@Om)),s,sdot])
    q0=c.q_init(); R0=c.DR(c.nrm(F(0.,np.array([0,0,c.ZH]),np.zeros(3),q0,np.zeros(3))))
    X0=np.concatenate([np.array([0.,0.,c.ZH]),np.zeros(3),R0.flatten(),np.zeros(3),q0,np.zeros(3)])
    te=np.linspace(0,t_end,n)
    sol=solve_ivp(ode,[0,t_end],X0,method=method,rtol=1e-5,atol=1e-7,t_eval=te)
    if not sol.success: return None
    t=sol.t; PQ,VQ=sol.y[:3].T,sol.y[3:6].T
    QA=sol.y[18:21].T; QA/=np.linalg.norm(QA,axis=1,keepdims=True)
    SA=np.array([c.prj(QA[k],sol.y[21:24].T[k]) for k in range(len(t))])
    PD=np.array([c.ref(float(x))[0] for x in t])
    epn=np.linalg.norm(PQ-PD,axis=1); th=np.degrees(np.arccos(np.clip(-QA[:,2],-1,1)))
    LAM=np.zeros(len(t)); TT=np.zeros(len(t))
    for k in range(len(t)):
        R=c.reorth(sol.y[6:15].T[k].reshape(3,3)); b3=R@e3
        Fv=F(float(t[k]),PQ[k],VQ[k],QA[k],SA[k]); T=float(Fv@b3); TT[k]=T
        LAM[k]=-(mL/(1+mu))*(float(QA[k]@(T/mq*b3))-ell*float(SA[k]@SA[k]))
    last=t>=(t_end-c.T8)
    return dict(ep_rms=float(np.sqrt(np.mean(epn[last]**2))),ep_peak=float(epn.max()),
        ep_fin=float(epn[-1]),th_rms=float(np.sqrt(np.mean(th[last]**2))),
        th_peak=float(th.max()),th_fin=float(th[-1]),lam_min=float(LAM.min()),T_max=float(TT.max()))
