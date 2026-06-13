"""
Quadrotor + Cable-Suspended Load  —  Full System, Single Window
================================================================
Layout: ventana única matplotlib  (como quadrotor_full_sim.py)
Dron 3D: Poly3DCollection  →  cuerpo, brazos, motores, aspas
Mouse:   arrastra el panel 3D para rotar libremente

Instalación:  pip install numpy scipy matplotlib
Uso:          python quadrotor_single_window.py

Authors: Gerardo Flores, Aldo Muñoz-Vázquez
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D                   # noqa
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import time

# ═══════════════════════════════════════════════════════════════════
# 1.  PARÁMETROS
# ═══════════════════════════════════════════════════════════════════
mq  = 1.20;  mL  = 0.35;  ell = 0.80;  ct = 0.25;  g = 9.81
Jxx = 1.5e-2; Jzz = 2.5e-2
J   = np.diag([Jxx, Jxx, Jzz]);  Jinv = np.diag([1/Jxx, 1/Jxx, 1/Jzz])
mu  = mL / mq;  e3 = np.array([0., 0., 1.])

Kp = 45.;  Kd = 12.;  Kc = 0.5;  KR = 6.;  KOm = 2.0
T_END = 22.0;  ZH = 1.5;  T8 = 10.0;  A = 1.4
TRAIL = 60      # puntos de trail
ANIM_INTERVAL_MS = 50   # ms entre frames (~20 fps)
ANIM_SPEED = 1.0        # 1.0=tiempo real, 2.0=doble

# ═══════════════════════════════════════════════════════════════════
# 2.  MATH
# ═══════════════════════════════════════════════════════════════════
def hat(v):   return np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
def vee(S):   return np.array([S[2,1],S[0,2],S[1,0]])
def PT(q):    return np.eye(3)-np.outer(q,q)
def nrm(v):   n=np.linalg.norm(v); return v/n if n>1e-10 else v
def prj(q,s): return s-np.dot(q,s)*q
def reorth(R): U,_,Vt=np.linalg.svd(R); return U@Vt
def DR(b):
    b=nrm(b)
    bc=np.array([1.,0.,0.]) if abs(b[0])<0.9 else np.array([0.,1.,0.])
    b2=nrm(np.cross(b,bc)); return np.column_stack([np.cross(b2,b),b2,b])

def ref(t):
    Tr=2.5; tau=np.clip(t/Tr,0.,1.)
    S  =10*tau**3-15*tau**4+6*tau**5
    Sd =(30*tau**2-60*tau**3+30*tau**4)/Tr   if t<Tr else 0.
    Sd2=(60*tau-180*tau**2+120*tau**3)/Tr**2  if t<Tr else 0.
    w=2*np.pi/T8; c,s_=np.cos(w*t),np.sin(w*t); D=1+s_**2; D3=D**3
    X=A*c/D; Y=A*c*s_/D
    dX=-A*w*(s_+2*s_*c**2)/D**2; dY=A*w*(c**2-s_**2)/D**2
    dX2=-A*w**2*(c-4*c**3+2*c*s_**2*(3-4*s_**2))/D3
    dY2=-A*w**2*(s_*(1+4*c**2-4*s_**2*c**2))/D3
    pd=np.array([S*X,S*Y,ZH]); vd=np.array([Sd*X+S*dX,Sd*Y+S*dY,0.])
    ad=np.array([Sd2*X+2*Sd*dX+S*dX2,Sd2*Y+2*Sd*dY+S*dY2,0.])
    return pd,vd,ad

# ═══════════════════════════════════════════════════════════════════
# 3.  SIMULATION
# ═══════════════════════════════════════════════════════════════════
def ode_outer(t,X):
    pq=X[:3];vq=X[3:6];q=nrm(X[6:9]);s=prj(q,X[9:12])
    pd,vd,ad=ref(t);ep=pq-pd;ev=vq-vd
    u=(1+mu)*g*e3+(ad-Kp*ep-Kd*ev)-Kc*(PT(q)@s)
    s2=float(np.dot(s,s));Pqq=np.outer(q,q)
    vdot=(np.eye(3)-mu/(1+mu)*Pqq)@(u-g*e3)-mu/(1+mu)*(np.dot(q,g*e3)-ell*s2)*q
    Pq=PT(q);sdot=(-1/ell)*Pq@(vdot+g*e3)-ct/(mL*ell)*Pq@(vq+ell*s)-s2*q
    return np.concatenate([vq,vdot,s,sdot])

th0=np.radians(20.); ph0=np.radians(30.)
q0=nrm(np.array([np.sin(th0)*np.cos(ph0),np.sin(th0)*np.sin(ph0),-np.cos(th0)]))

print("━"*60)
print("PASO 1/3 — Pre-computando Rd(t) …")
t_grid=np.linspace(0.,T_END,4400)
sol_o=solve_ivp(ode_outer,[0.,T_END],
    np.concatenate([np.array([0.,0.,ZH]),np.zeros(3),q0,np.zeros(3)]),
    t_eval=t_grid,method='RK45',rtol=1e-7,atol=1e-9)
PQr=sol_o.y[:3].T;VQr=sol_o.y[3:6].T
QAr=sol_o.y[6:9].T;SAr=sol_o.y[9:12].T
QAr/=np.linalg.norm(QAr,axis=1,keepdims=True)
Rdf=np.zeros((len(t_grid),9));Omg=np.zeros((len(t_grid),3))
for k in range(len(t_grid)):
    pd,vd,ad=ref(t_grid[k]);ep=PQr[k]-pd;ev=VQr[k]-vd
    u=(1+mu)*g*e3+(ad-Kp*ep-Kd*ev)-Kc*(PT(QAr[k])@SAr[k])
    Rdf[k]=DR(nrm(u)).flatten()
for k in range(len(t_grid)):
    k0=max(0,k-1);k1=min(len(t_grid)-1,k+1)
    Rdk=Rdf[k].reshape(3,3)
    Omg[k]=vee(Rdk.T@((Rdf[k1]-Rdf[k0]).reshape(3,3)/(t_grid[k1]-t_grid[k0])))
iRd =interp1d(t_grid,Rdf, axis=0,kind='linear',fill_value='extrapolate')
iOmd=interp1d(t_grid,Omg, axis=0,kind='linear',fill_value='extrapolate')
print("  ✓")

def ctrl(t,pq,vq,R,Om,q,s):
    pd,vd,ad=ref(t);ep=pq-pd;ev=vq-vd
    u=(1+mu)*g*e3+(ad-Kp*ep-Kd*ev)-Kc*(PT(q)@s)
    T=mq*np.linalg.norm(u); Rd=iRd(t).reshape(3,3); Omd_=iOmd(t)
    eR=0.5*vee(Rd.T@R-R.T@Rd); eOm=Om-R.T@Rd@Omd_
    tau=-KR*eR-KOm*eOm+np.cross(Om,J@Om)
    b3=R@e3; s2=float(np.dot(s,s))
    lest=-(mL/(1+mu))*(np.dot(q,T/mq*b3)-ell*s2)
    return T,tau,b3,Rd[:,2],lest

def ode_full(t,X):
    pq=X[:3];vq=X[3:6];R=X[6:15].reshape(3,3);Om=X[15:18]
    q=nrm(X[18:21]);s=prj(q,X[21:24])
    T,tau,b3,_,_=ctrl(t,pq,vq,R,Om,q,s)
    s2=float(np.dot(s,s));Pqq=np.outer(q,q)
    vdot=(np.eye(3)-mu/(1+mu)*Pqq)@(T/mq*b3-g*e3)-mu/(1+mu)*(np.dot(q,g*e3)-ell*s2)*q
    Rdot=R@hat(Om); Omdot=Jinv@(tau-np.cross(Om,J@Om))
    Pq=PT(q); sdot=(-1/ell)*Pq@(vdot+g*e3)-ct/(mL*ell)*Pq@(vq+ell*s)-s2*q
    return np.concatenate([vq,vdot,Rdot.flatten(),Omdot,s,sdot])

X0=np.concatenate([np.array([0.,0.,ZH]),np.zeros(3),
                   np.eye(3).flatten(),np.zeros(3),q0,np.zeros(3)])
t_ev=np.linspace(0.,T_END,2200)
print("PASO 2/3 — Integrando sistema completo (24 estados) …")
_t0=time.time()
sol=solve_ivp(ode_full,[0.,T_END],X0,t_eval=t_ev,method='RK45',rtol=1e-5,atol=1e-7)
print(f"  {time.time()-_t0:.1f}s  |  {sol.message}")

Ts=sol.t; PQ=sol.y[:3].T; VQ=sol.y[3:6].T
RS=sol.y[6:15].T; OMS=sol.y[15:18].T
QA=sol.y[18:21].T; SA=sol.y[21:24].T
for k in range(len(Ts)): RS[k]=reorth(RS[k].reshape(3,3)).flatten()
QA/=np.linalg.norm(QA,axis=1,keepdims=True); PL=PQ+ell*QA
PD_a=np.array([ref(t)[0] for t in Ts])
LAM=np.zeros(len(Ts)); B3a=np.zeros_like(PQ); eRd=np.zeros(len(Ts))
for k,t in enumerate(Ts):
    R_k=RS[k].reshape(3,3); _,_,b3k,_,le=ctrl(t,PQ[k],VQ[k],R_k,OMS[k],QA[k],SA[k])
    LAM[k]=le; B3a[k]=b3k
    Rd_k=iRd(t).reshape(3,3)
    eRd[k]=np.degrees(np.linalg.norm(0.5*vee(Rd_k.T@R_k-R_k.T@Rd_k)))
EP_n =np.linalg.norm(PQ-PD_a,axis=1)
theta=np.degrees(np.arccos(np.clip(-QA[:,2],-1.,1.)))

# velocidades deseadas y errores
VD_a  = np.array([ref(t)[1] for t in Ts])
EV    = VQ - VD_a                              # error de velocidad (N,3)
EV_n  = np.linalg.norm(EV, axis=1)

# empuje y torques
T_arr   = np.zeros(len(Ts))
TAU_arr = np.zeros((len(Ts), 3))
eOm_arr = np.zeros(len(Ts))
for k, t in enumerate(Ts):
    R_k  = RS[k].reshape(3,3)
    pd,vd,ad = ref(t); ep = PQ[k]-pd; ev = VQ[k]-vd
    #u    = (1+mu)*g*e3 + (ad-Kp*ep-Kd*ev) - Kc*(PT(QA[k])@SA[k])
    v = (1+mu)*g*e3 + (ad-Kp*ep-Kd*ev) - Kc*(PT(QA[k])@SA[k])
    u = v + mu*QA[k]*(QA[k]@v)      # aplica (I + mu*q*q^T)

    T_arr[k] = mq * np.linalg.norm(u)
    Rd   = iRd(t).reshape(3,3); Omd_ = iOmd(t)
    eR_v = 0.5*vee(Rd.T@R_k - R_k.T@Rd)
    eOm_v= OMS[k] - R_k.T@Rd@Omd_
    TAU_arr[k] = -KR*eR_v - KOm*eOm_v + np.cross(OMS[k], J@OMS[k])
    eOm_arr[k] = np.degrees(np.linalg.norm(eOm_v))

print(f"  max|ep|={EP_n.max():.3f}m  θ_f={theta[-1]:.1f}°  max eR={eRd.max():.2f}°")
print("━"*60)


def generate_paper_figures():
    """
    Genera figuras estilo IEEE Transactions / Elsevier AST.
    Fuente Times, LaTeX, vectorizadas en PDF.
    Ancho: 7.16 in  (ocupa ambas columnas IEEE  /  texto completo AST)
    """
    import matplotlib
    _backend_orig = matplotlib.get_backend()

    # ── estilo paper ──────────────────────────────────────────────
    try:
        usetex = True
        import subprocess
        subprocess.run(['latex','--version'],
                       capture_output=True, check=True)
    except Exception:
        usetex = False   # LaTeX no instalado → usa mathtext

    paper_rc = {
        'text.usetex':        usetex,
        'font.family':        'serif',
        'font.serif':         ['Times New Roman','Times','DejaVu Serif'],
        'font.size':          9,
        'axes.labelsize':     9,
        'axes.titlesize':     9,
        'xtick.labelsize':    8,
        'ytick.labelsize':    8,
        'legend.fontsize':    8,
        'legend.framealpha':  0.9,
        'lines.linewidth':    1.2,
        'axes.linewidth':     0.7,
        'grid.linewidth':     0.4,
        'grid.alpha':         0.35,
        'grid.linestyle':     '--',
        'figure.dpi':         300,
        'savefig.dpi':        300,
        'savefig.bbox':       'tight',
        'savefig.pad_inches': 0.02,
        #'axes.spines.top':    False,
        #'axes.spines.right':  False,
        'axes.spines.top':    True,
        'axes.spines.right':  True,
        'xtick.direction':    'in',
        'ytick.direction':    'in',
        'xtick.top':          True,
        'ytick.right':        True,
        #'xtick.minor.visible': True,
        #'ytick.minor.visible': True,
        # negro puro en las graficas:
        'text.color':      '#000000',   
        'axes.labelcolor': '#000000',
        'xtick.color':     '#000000',
        'ytick.color':     '#000000',
        'axes.edgecolor':  '#000000',
        'grid.color':      '#bbbbbb',   # grid gris medio (no tan claro)
        'grid.alpha':      0.6,
    }
    with matplotlib.rc_context(paper_rc):

        W  = 7.16    # ancho figura [in] — dos columnas IEEE / texto AST
        lw = 1.2     # linewidth principal
        ls = 0.8     # linewidth secundario (referencia, grid)

        # paleta de colores accesible (colorblind-friendly)
        C = {
            'blue':   '#0072B2',
            'orange': '#E69F00',
            'green':  '#009E73',
            'red':    '#D55E00',
            'purple': '#CC79A7',
            'gray':   '#999999',
            'black':  '#000000',
        }

        tl = r'$' if usetex else ''   # delimitador math
        tr = r'$' if usetex else ''

        def math(s):
            return f'${s}$' if usetex else s

        # ══════════════════════════════════════════════════════════
        # FIGURA 1 — Trayectorias 3D
        # ══════════════════════════════════════════════════════════
        fig1 = plt.figure(figsize=(W, W*0.55))
        ax   = fig1.add_subplot(111, projection='3d')
        ax.plot(*PD_a.T, '--', color=C['gray'],   lw=ls,
                label=math(r'p_d') + ' (reference)')
        ax.plot(*PQ.T,   '-',  color=C['blue'],   lw=lw,
                label=math(r'p_q') + ' (drone)')
        ax.plot(*PL.T,   '-',  color=C['red'],    lw=ls,
                label=math(r'p_L') + ' (load)', alpha=0.8)
        # cable snapshots every ~1s
        #skip = max(1, len(Ts)//22)
        #for k in range(0, len(Ts), skip):
        #    ax.plot([PQ[k,0],PL[k,0]],[PQ[k,1],PL[k,1]],
        #            [PQ[k,2],PL[k,2]],
        #            '-', color=C['gray'], lw=0.4, alpha=0.4)

        # snapshots del dron y carga con efecto de sombra (trail)
        n_snaps = 18
        snap_idx = np.linspace(0, len(Ts)-1, n_snaps, dtype=int)
        for i, k in enumerate(snap_idx):
            alpha = 0.15 + 0.55 * (i / n_snaps)   # más opaco = más reciente
            size  = 18  + 30  * (i / n_snaps)
            # cable en ese instante
            ax.plot([PQ[k,0], PL[k,0]],
                    [PQ[k,1], PL[k,1]],
                    [PQ[k,2], PL[k,2]],
                    '-', color=C['gray'], lw=0.5, alpha=alpha*0.7)
            # dron
            ax.scatter(*PQ[k], s=size, color=C['blue'],
                    alpha=alpha, depthshade=True, zorder=5)
            # carga
            ax.scatter(*PL[k], s=size*0.7, color=C['red'],
                    alpha=alpha, depthshade=True, zorder=5)

        ax.set_xlabel('$x$ [m]'); ax.set_ylabel('$y$ [m]')
        ax.set_zlabel('$z$ [m]')
        ax.legend(loc='upper left', fontsize=8)
        ax.view_init(25, -45)
        fig1.tight_layout()
        fig1.savefig('DronCable-March2026/render/fig1_trajectories.pdf', format='pdf')
        plt.close(fig1)
        print('  Saved fig1_trajectories.pdf')

        # ══════════════════════════════════════════════════════════
        # FIGURA 2 — Errores de tracking y cable  (2×2)
        # ══════════════════════════════════════════════════════════
        fig2, axs = plt.subplots(2, 2, figsize=(W, W*0.62))
        fig2.subplots_adjust(hspace=0.42, wspace=0.32)

        EP = PQ - PD_a   # (N,3)

        # (a) error de posición
        ax = axs[0,0]
        ax.plot(Ts, EP[:,0], color=C['blue'],   lw=lw, label=math(r'e_x'))
        ax.plot(Ts, EP[:,1], color=C['orange'], lw=lw, label=math(r'e_y'))
        ax.plot(Ts, EP[:,2], color=C['green'],  lw=lw, label=math(r'e_z'))
        ax.plot(Ts, EP_n,   color=C['black'],   lw=ls,
                ls='--', label=math(r'\|e_p\|'))
        ax.axhline(0, color=C['gray'], lw=0.5)
        ax.set_ylabel('[m]'); ax.grid(True)
        ax.legend(ncol=2); ax.set_title('(a) ' + math(r'e_p'))
        ax.set_xlim(0, T_END)

        # (b) error de velocidad
        ax = axs[0,1]
        ax.plot(Ts, EV[:,0], color=C['blue'],   lw=lw, label=math(r'e_{v_x}'))
        ax.plot(Ts, EV[:,1], color=C['orange'], lw=lw, label=math(r'e_{v_y}'))
        ax.plot(Ts, EV[:,2], color=C['green'],  lw=lw, label=math(r'e_{v_z}'))
        ax.plot(Ts, EV_n,   color=C['black'],   lw=ls,
                ls='--', label=math(r'\|e_v\|'))
        ax.axhline(0, color=C['gray'], lw=0.5)
        ax.set_ylabel('[m/s]'); ax.grid(True)
        ax.legend(ncol=2); ax.set_title('(b) ' + math(r'e_v'))
        ax.set_xlim(0, T_END)

        # (c) oscilación del cable
        ax = axs[1,0]
        ax.plot(Ts, theta, color=C['red'], lw=lw)
        ax.axhline(0, color=C['gray'], lw=0.5)
        ax.set_xlabel('Time [s]'); ax.set_ylabel('[deg]'); ax.grid(True)
        ax.set_title('(c) Cable swing ' + math(r'\theta'))
        ax.set_xlim(0, T_END)

        # (d) tensión vs referencia
        ax = axs[1,1]
        ax.axhline(mL*g, color=C['gray'], lw=ls, ls='--',
                   label=math(r'\lambda_d = m_L g'))
        ax.plot(Ts, LAM, color=C['purple'], lw=lw,
                label=math(r'\hat{\lambda}'))
        ax.set_ylim(bottom=0)
        ax.set_xlabel('Time [s]'); ax.set_ylabel('[N]'); ax.grid(True)
        ax.legend(); ax.set_title('(d) Cable tension ' + math(r'\hat{\lambda}'))
        ax.set_xlim(0, T_END)

        fig2.savefig('DronCable-March2026/render/fig2_errors.pdf', format='pdf')
        plt.close(fig2)
        print('  Saved fig2_errors.pdf')

        # ══════════════════════════════════════════════════════════
        # FIGURA 3 — Errores de actitud  (1×2)
        # ══════════════════════════════════════════════════════════
        fig3, axs = plt.subplots(1, 2, figsize=(W, W*0.32))
        fig3.subplots_adjust(wspace=0.32)

        ax = axs[0]
        ax.plot(Ts, eRd, color=C['blue'], lw=lw)
        ax.axhline(0, color=C['gray'], lw=0.5)
        ax.set_xlabel('Time [s]'); ax.set_ylabel('[deg]'); ax.grid(True)
        ax.set_title('(a) Attitude error ' + math(r'\|e_R\|'))
        ax.set_xlim(0, T_END)

        ax = axs[1]
        ax.plot(Ts, eOm_arr, color=C['orange'], lw=lw)
        ax.axhline(0, color=C['gray'], lw=0.5)
        ax.set_xlabel('Time [s]'); ax.set_ylabel('[deg/s]'); ax.grid(True)
        ax.set_title('(b) Angular velocity error ' + math(r'\|e_\Omega\|'))
        ax.set_xlim(0, T_END)

        fig3.savefig('DronCable-March2026/render/fig3_attitude.pdf', format='pdf')
        plt.close(fig3)
        print('  Saved fig3_attitude.pdf')

        # ══════════════════════════════════════════════════════════
        # FIGURA 4 — Señales de control  (2×2)
        # ══════════════════════════════════════════════════════════
        fig4, axs = plt.subplots(2, 2, figsize=(W, W*0.62))
        fig4.subplots_adjust(hspace=0.42, wspace=0.32)

        T_hover = (mq + mL) * g

        # (a) empuje T
        ax = axs[0,0]
        ax.axhline(T_hover, color=C['gray'], lw=ls, ls='--',
                   label=math(r'(m_q+m_L)g'))
        ax.plot(Ts, T_arr, color=C['blue'], lw=lw, label=math(r'T(t)'))
        ax.set_ylabel('[N]'); ax.grid(True)
        ax.legend(); ax.set_title('(a) Thrust ' + math(r'T'))
        ax.set_xlim(0, T_END)

        # (b) norma del torque
        ax = axs[0,1]
        ax.plot(Ts, np.linalg.norm(TAU_arr, axis=1),
                color=C['orange'], lw=lw)
        ax.axhline(0, color=C['gray'], lw=0.5)
        ax.set_ylabel('[N·m]'); ax.grid(True)
        ax.set_title('(b) Torque norm ' + math(r'\|\tau\|'))
        ax.set_xlim(0, T_END)

        # (c) componentes del torque
        ax = axs[1,0]
        ax.plot(Ts, TAU_arr[:,0], color=C['blue'],   lw=lw,
                label=math(r'\tau_x'))
        ax.plot(Ts, TAU_arr[:,1], color=C['orange'], lw=lw,
                label=math(r'\tau_y'))
        ax.plot(Ts, TAU_arr[:,2], color=C['green'],  lw=lw,
                label=math(r'\tau_z'))
        ax.axhline(0, color=C['gray'], lw=0.5)
        ax.set_xlabel('Time [s]'); ax.set_ylabel('[N·m]'); ax.grid(True)
        ax.legend(ncol=3)
        ax.set_title('(c) Torque components')
        ax.set_xlim(0, T_END)

        # (d) control u* (loop externo diseñado)
        u_arr = np.zeros((len(Ts), 3))
        for k, t in enumerate(Ts):
            pd,vd,ad = ref(t); ep=PQ[k]-pd; ev=VQ[k]-vd
            u_arr[k] = (1+mu)*g*e3 + (ad-Kp*ep-Kd*ev) \
                       - Kc*(PT(QA[k])@SA[k])
        ax = axs[1,1]
        ax.plot(Ts, u_arr[:,0], color=C['blue'],   lw=lw,
                label=math(r'u^*_x'))
        ax.plot(Ts, u_arr[:,1], color=C['orange'], lw=lw,
                label=math(r'u^*_y'))
        ax.plot(Ts, u_arr[:,2], color=C['green'],  lw=lw,
                label=math(r'u^*_z'))
        ax.axhline(0, color=C['gray'], lw=0.5)
        ax.set_xlabel('Time [s]')
        #ax.set_ylabel('[m/s' + (r'$^2$]' if not usetex else r'^2$]'))
        ax.set_ylabel(r'[m/s$^2$]' if not usetex else r'[m/s$^2$]')
        ax.grid(True); ax.legend(ncol=3)
        ax.set_title('(c) Virtual control ' + math(r'u^*'))
        ax.set_xlim(0, T_END)

        fig4.savefig('DronCable-March2026/render/fig4_controls.pdf', format='pdf')
        plt.close(fig4)
        print('  Saved fig4_controls.pdf')

    print('━'*60)
    print('Figures saved:')
    print('  fig1_trajectories.pdf')
    print('  fig2_errors.pdf')
    print('  fig3_attitude.pdf')
    print('  fig4_controls.pdf')

generate_paper_figures()

# ═══════════════════════════════════════════════════════════════════
# 4.  DRONE GEOMETRY  (correctly oriented arms, scaled small)
# ═══════════════════════════════════════════════════════════════════

def box_faces(cx,cy,cz,dx,dy,dz):
    x0,x1=cx-dx,cx+dx; y0,y1=cy-dy,cy+dy; z0,z1=cz-dz,cz+dz
    return [
        [[x0,x1,x1,x0],[y0,y0,y1,y1],[z0,z0,z0,z0]],
        [[x0,x1,x1,x0],[y0,y0,y1,y1],[z1,z1,z1,z1]],
        [[x0,x0,x1,x1],[y0,y1,y1,y0],[z0,z1,z1,z0]],
        [[x0,x0,x1,x1],[y0,y1,y1,y0],[z1,z0,z0,z1]],
        [[x0,x0,x0,x0],[y0,y1,y1,y0],[z0,z0,z1,z1]],
        [[x1,x1,x1,x1],[y0,y1,y1,y0],[z0,z0,z1,z1]],
    ]

def disc_faces(cx,cy,cz,r,n=16):
    a=np.linspace(0,2*np.pi,n,endpoint=False)
    return [[[float(cx+r*np.cos(ai)) for ai in a],
              [float(cy+r*np.sin(ai)) for ai in a],
              [cz]*n]]

def rotated_box_faces(cx,cy,cz,length,width,height,angle_rad):
    """Thin box elongated along angle_rad direction in xy-plane."""
    c,s=np.cos(angle_rad),np.sin(angle_rad)
    pts=np.array([[-length,-width,-height],[length,-width,-height],
                   [length, width,-height],[-length, width,-height],
                   [-length,-width, height],[length,-width, height],
                   [length, width, height],[-length, width, height]])
    rot=np.array([[c,-s,0],[s,c,0],[0,0,1]])
    pts=(rot@pts.T).T; pts[:,0]+=cx; pts[:,1]+=cy; pts[:,2]+=cz
    fi=[[0,1,2,3],[4,5,6,7],[0,1,5,4],[2,3,7,6],[0,3,7,4],[1,2,6,5]]
    return [[[float(pts[i,j]) for i in f] for j in range(3)] for f in fi]

def build_drone_polys():
    parts=[]
    SC=0.40   # scale: arm-to-arm ~0.22m (looks small vs 1.4m figure-8)

    BDY='#252525'; ARM_F='#2a4a8a'; ARM_B='#8a2a2a'
    MOT='#1e1e1e'; BLADE='#0d0d0d'; CAP_F='#58a6ff'; CAP_B='#f85149'
    LED='#00ccff'; CAM='#383838'

    def AB(faces,color,alpha=0.95,ec='#333'):
        for f in faces: parts.append(([f],color,alpha,ec))

    # body
    AB(box_faces(0,0,0,         .11*SC,.055*SC,.016*SC), BDY)
    AB(box_faces(0,0,.022*SC,   .05*SC,.030*SC,.016*SC), '#303030')
    AB(box_faces(.10*SC,0,.008*SC,.055*SC,.030*SC,.012*SC), '#1e1e1e')
    AB(box_faces(.148*SC,0,.010*SC,.008*SC,.015*SC,.005*SC), LED, 0.9, LED)

    # arms: each at 45° toward its motor corner
    # angle_rad = direction FROM body centre TOWARD motor
    ARM_MOTORS=[
        ( .22*SC,  .20*SC, CAP_F, ARM_F, np.arctan2( .20, .22)),   # front-right
        ( .22*SC, -.20*SC, CAP_F, ARM_F, np.arctan2(-.20, .22)),   # front-left
        (-.18*SC,  .20*SC, CAP_B, ARM_B, np.arctan2( .20,-.18)),   # rear-right
        (-.18*SC, -.20*SC, CAP_B, ARM_B, np.arctan2(-.20,-.18)),   # rear-left
    ]
    for (ex,ey,cap_col,arm_col,angle) in ARM_MOTORS:
        cx,cy=ex*.55,ey*.55
        arm_L=np.sqrt(ex**2+ey**2)*0.47  # half-length along diagonal
        AB(rotated_box_faces(cx,cy,.006*SC,arm_L,.013*SC,.011*SC,angle),
           arm_col,0.92,'#222')
        AB(disc_faces(ex,ey,.005*SC,.032*SC), MOT)
        AB(disc_faces(ex,ey,.038*SC,.032*SC), MOT)
        AB(disc_faces(ex,ey,.042*SC,.023*SC), cap_col,1.0,cap_col)
        for ang_deg in [0,90]:
            ang=np.radians(ang_deg); c_b,s_b=np.cos(ang),np.sin(ang)
            BL=.100*SC; BW=.010*SC; BH=.003*SC
            pts2=np.array([[-BL,-BW,-BH],[BL,-BW,-BH],[BL,BW,-BH],[-BL,BW,-BH],
                            [-BL,-BW, BH],[BL,-BW, BH],[BL,BW, BH],[-BL,BW, BH]])
            r2=np.array([[c_b,-s_b,0],[s_b,c_b,0],[0,0,1]])
            pts2=(r2@pts2.T).T; pts2[:,0]+=ex; pts2[:,1]+=ey; pts2[:,2]+=.050*SC
            for idx in [[0,1,2,3],[4,5,6,7]]:
                face=[[float(pts2[i,j]) for i in idx] for j in range(3)]
                parts.append(([face],BLADE,0.88,'#111'))

    # skids
    for sx,sy in [(1,1),(1,-1),(-1,1),(-1,-1)]:
        AB(box_faces(sx*.07*SC,sy*.045*SC,-.052*SC,.007*SC,.007*SC,.022*SC),'#1a1a1a',0.9,'#333')
    AB(box_faces(0,0,-.060*SC,.13*SC,.008*SC,.007*SC),'#1a1a1a',0.9)
    AB(box_faces(0,0,-.060*SC,.008*SC,.09*SC,.007*SC),'#1a1a1a',0.9)

    # camera
    AB(box_faces(.135*SC,0,-.040*SC,.018*SC,.017*SC,.017*SC),CAM,0.9,'#444')
    AB(disc_faces(.135*SC,-.022*SC,-.038*SC,.012*SC),'#0a0a0a',1.0,'#111')

    return parts

DRONE_POLYS=build_drone_polys()

def transform_face(face,R,t):
    pts=np.array(face); pts=R@pts; pts=pts+t[:,None]; return pts.tolist()

def make_poly3d(faces_list,R,t,facecolor,alpha,edgecolor):
    transformed=[]
    for f in faces_list:
        tf=transform_face(f,R,t)
        verts=list(zip(tf[0],tf[1],tf[2]))
        transformed.append(verts)
    return Poly3DCollection(transformed,alpha=alpha,facecolor=facecolor,
                             edgecolor=edgecolor,linewidth=0.3,zsort='average')
# 5.  FIGURE LAYOUT  (single window, same style as quadrotor_full_sim)
# ═══════════════════════════════════════════════════════════════════
BG  = "#ffffff"; TXT = "#6D6060"; GRD = "#2d2a21"

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': "#ffffff",
    'text.color': TXT, 'axes.labelcolor': TXT,
    'xtick.color': TXT, 'ytick.color': TXT,
    'axes.edgecolor': GRD, 'grid.color': GRD,
    'grid.linewidth': 0.5, 'font.family': 'serif',
    'text.usetex': True,
})

fig = plt.figure(figsize=(17, 9))
fig.patch.set_facecolor(BG)
fig.canvas.manager.set_window_title('Full System: Quadrotor + Suspended Load')

gs = gridspec.GridSpec(4, 2, figure=fig,
                        left=0.04, right=0.98,
                        top=0.90, bottom=0.06,
                        wspace=0.30, hspace=0.55)

# ── 3D axes (left, spans all 4 rows) ─────────────────────────────
ax3 = fig.add_subplot(gs[:, 0], projection='3d')
ax3.set_facecolor("#ffffff")
ax3.xaxis.pane.fill = False; ax3.yaxis.pane.fill = False; ax3.zaxis.pane.fill = False
for pane in [ax3.xaxis.pane,ax3.yaxis.pane,ax3.zaxis.pane]:
    pane.set_edgecolor(GRD)
ax3.tick_params(colors=TXT, labelsize=7)
ax3.xaxis.label.set_color(TXT); ax3.yaxis.label.set_color(TXT); ax3.zaxis.label.set_color(TXT)
ax3.set_xlabel('x [m]'); ax3.set_ylabel('y [m]'); ax3.set_zlabel('z [m]')
ax3.grid(True, color=GRD, lw=0.4)

zall = np.concatenate([PQ[:,2],PL[:,2]])
ax3.set_xlim(-1.6,1.6); ax3.set_ylim(-1.1,1.1)
ax3.set_zlim(zall.min()-.2, zall.max()+.3)

# static reference path
pd_pts = np.array([ref(t)[0] for t in Ts])
ax3.plot(*pd_pts.T, '--', color='#1a4a1a', lw=1.0, alpha=0.5, label='Ref $p_d$')

# ── 4 right panels ───────────────────────────────────────────────
plot_cfg = [
    (EP_n,  'Position error  $\\|e_p\\|$',          '[m]',  '#58a6ff', EP_n.max()*1.15+.02),
    (theta, 'Cable swing  $\\theta$',               '[°]',  '#ff7b72', theta.max()*1.2+1),
    (LAM,   'Cable tension  $\\hat{\\lambda}$',      '[N]',  '#d2a8ff', max(LAM.max(),mL*g)*1.3+.1),
    (eRd,   'Attitude error  $\\|e_R\\|$',           '[°]',  '#c792ea', max(eRd.max()*1.2,.5)),
]
axes_r = []; lines_r = []; fills_r = []; vlines_r = []
for i,(data,title,ylabel,color,ymax) in enumerate(plot_cfg):
    ax = fig.add_subplot(gs[i,1])
    ax.set_xlim(0,T_END); ax.set_ylim(-ymax*.04, ymax)
    ax.set_title(title, fontsize=9, pad=3); ax.set_ylabel(ylabel, fontsize=8)
    ax.axhline(0, color=GRD, lw=0.7)
    if i==2: ax.axhline(mL*g, color='#555', lw=0.8, ls='--')
    if i==3: ax.set_xlabel('Time [s]', fontsize=8)
    ax.grid(True)
    # static full shading
    ax.fill_between(Ts, data, alpha=0.07, color=color)
    ln, = ax.plot([],[],'-', color=color, lw=1.8)
    vl  = ax.axvline(0, color='white', lw=0.9, alpha=0.5)
    axes_r.append(ax); lines_r.append(ln); vlines_r.append(vl)

# ── title ─────────────────────────────────────────────────────────
fig.text(0.5, 0.955,
         'Full System: Quadrotor + Suspended Load  —  '
         'Outer (Lyapunov) + Inner (Geometric SO(3))',
         ha='center', fontsize=11, fontweight='bold', color=TXT)
fig.text(0.5, 0.930,
         f'Drag to rotate  ·  Scroll to zoom  |  '
         f'$K_p={Kp}$, $K_d={Kd}$, $K_c={Kc}$  |  '
         f'$K_R={KR}$, $K_\\Omega={KOm}$',
         ha='center', fontsize=9, color='#8b949e')

time_txt = ax3.text2D(0.02, 0.78, 't = 0.00 s',
                      transform=ax3.transAxes,
                      ha='left', fontsize=12,
                      fontweight='bold', color='#f0b429')

# ── legend ────────────────────────────────────────────────────────
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
legend_elems = [
    Line2D([0],[0], color='#2d07eb', lw=2.5, label='Cable'),
    Line2D([0],[0], color='#58a6ff',     lw=0,  marker='o', ms=7, label='Drone'),
    Line2D([0],[0], color='#cc3300',     lw=0,  marker='o', ms=7, label='Load'),
    Line2D([0],[0], color='#ffa640',     lw=2.0,        label='Thrust $b_3$'),
    Line2D([0],[0], color='#c792ea',     lw=1.5, ls='--', label='Desired $b_3^*$'),
]
ax3.legend(handles=legend_elems, fontsize=8, loc='upper left',
           facecolor="#c6daf7", edgecolor=GRD, labelcolor=TXT)

# ═══════════════════════════════════════════════════════════════════
# 6.  ANIMATION ARTISTS
# ═══════════════════════════════════════════════════════════════════

# trails
trail_dr, = ax3.plot([],[],[], '-', color='#1a3a6a', lw=1.2, alpha=0.6)
trail_ld, = ax3.plot([],[],[], '-', color='#5a1010', lw=0.9, alpha=0.5)

# cable
cable_ln, = ax3.plot([],[],[], '-', color="#2d07eb", lw=1.0,
                      solid_capstyle='round')

# drone / load markers (for hover text & visibility)
drone_dot, = ax3.plot([],[],[], 'o', color='#58a6ff', ms=9,
                       markeredgecolor='white', markeredgewidth=0.7, zorder=10)
load_dot,  = ax3.plot([],[],[], 'o', color='#cc3300', ms=8, zorder=10)

# thrust arrow (quiver)
quiv_thrust = ax3.quiver(0,0,ZH, 0,0,0.3,
                          color='#ffa640', linewidth=2.0,
                          arrow_length_ratio=0.35, normalize=False)
# desired b3 (dashed quiver)
quiv_b3d    = ax3.quiver(0,0,ZH, 0,0,0.3,
                          color='#c792ea', linewidth=1.5,
                          arrow_length_ratio=0.35, normalize=False,
                          linestyle='dashed')

# drone body Poly3DCollection actors — we rebuild each frame
# (matplotlib doesn't support inplace update of Poly3D vertices easily)
_poly_actors = []   # cleared and re-added each frame

# ═══════════════════════════════════════════════════════════════════
# 7.  ANIMATION
# ═══════════════════════════════════════════════════════════════════
N  = len(Ts)
DT = Ts[1]-Ts[0]
state = {'frame':0, 'last_wall':time.time()}

def update(ani_frame):
    # ── advance simulation time ──────────────────────────────────
    now  = time.time()
    dtw  = now - state['last_wall']
    state['last_wall'] = now
    steps = max(1, int(dtw * ANIM_SPEED / DT))
    state['frame'] = (state['frame'] + steps) % N
    k   = state['frame']
    t_  = Ts[k]
    pq  = PQ[k]; pl_ = PL[k]
    R_k = RS[k].reshape(3,3)
    b3  = B3a[k]
    b3d_= iRd(t_).reshape(3,3)[:,2]

    # ── trails ──────────────────────────────────────────────────
    k0 = max(0, k-TRAIL)
    trail_dr.set_data(PQ[k0:k+1,0], PQ[k0:k+1,1])
    trail_dr.set_3d_properties(PQ[k0:k+1,2])
    trail_ld.set_data(PL[k0:k+1,0], PL[k0:k+1,1])
    trail_ld.set_3d_properties(PL[k0:k+1,2])

    # ── cable ────────────────────────────────────────────────────
    cable_ln.set_data([pq[0],pl_[0]], [pq[1],pl_[1]])
    cable_ln.set_3d_properties([pq[2],pl_[2]])

    # ── dots ─────────────────────────────────────────────────────
    drone_dot.set_data([pq[0]], [pq[1]])
    drone_dot.set_3d_properties([pq[2]])
    load_dot.set_data([pl_[0]], [pl_[1]])
    load_dot.set_3d_properties([pl_[2]])

    # ── thrust arrows (remove old, add new quiver) ────────────────
    global quiv_thrust, quiv_b3d
    quiv_thrust.remove(); quiv_b3d.remove()
    sc = 0.28
    quiv_thrust = ax3.quiver(pq[0],pq[1],pq[2],
                              b3[0]*sc, b3[1]*sc, b3[2]*sc,
                              color='#ffa640', linewidth=2.0,
                              arrow_length_ratio=0.35, normalize=False)
    quiv_b3d = ax3.quiver(pq[0],pq[1],pq[2],
                           b3d_[0]*sc, b3d_[1]*sc, b3d_[2]*sc,
                           color='#c792ea', linewidth=1.5,
                           arrow_length_ratio=0.35, normalize=False,
                           linestyle='dashed')

    # ── 3D drone body (Poly3DCollection) ─────────────────────────
    for actor in _poly_actors:
        actor.remove()
    _poly_actors.clear()

    for (faces, facecolor, alpha, edgecolor) in DRONE_POLYS:
        poly = make_poly3d(faces, R_k, pq, facecolor, alpha, edgecolor)
        ax3.add_collection3d(poly)
        _poly_actors.append(poly)

    # ── time label ───────────────────────────────────────────────
    time_txt.set_text(f't = {t_:.2f} s')

    # ── right panels ─────────────────────────────────────────────
    ts_ = Ts[:k+1]
    for i,(data,_,_,color,_) in enumerate(plot_cfg):
        lines_r[i].set_data(ts_, data[:k+1])
        vlines_r[i].set_xdata([t_, t_])

    return (trail_dr, trail_ld, cable_ln, drone_dot, load_dot,
            quiv_thrust, quiv_b3d, time_txt, *lines_r, *vlines_r)

generate_paper_figures()
print("PASO 3/3 — Iniciando animación …")
print("  Arrastra el panel 3D para rotar  |  cierra la ventana para salir")

anim = FuncAnimation(fig, update, frames=200, #frames=None, 400 frames 
                      interval=ANIM_INTERVAL_MS, blit=False)
#anim.save('DronCable-March2026/render/quadrotor.gif', writer='pillow', fps=7) #fps=20

anim.save('DronCable-March2026/render/quadrotor.mp4', writer='ffmpeg', fps=20, dpi=100,
           extra_args=['-vcodec', 'libx264', '-crf', '23'])

# ── snapshot for paper ────────────────────────────────────────────
SNAP_FRAME = 190          # ← elige el frame que se vea bien (0–199)
update(SNAP_FRAME)
fig.savefig('DronCable-March2026/render/fig_snapshot.pdf',
            format='pdf', dpi=300, bbox_inches='tight')
#ax3d.view_init(elev=20, azim=-60)   # ajusta a tu gusto

plt.show()
