"""
super_sim.py - Bo mo phong hop nhat va kiem dinh thuc nghiem quy mo sieu lon (SCUDEM XI 2026 Problem B).
Tich hop tat ca cac co che tu Giai doan 1 den Giai doan 5, ho tro Common Random Numbers (CRN),
chay mau sieu lon (R len den 10.000+), thong ke sai so chuan, khoang tin cay 95%, va kiem dinh TOST.
"""

import sys
import time
import os
import numpy as np
from typing import Dict, Any, Tuple, List, Optional

# Tham so mac dinh co so (Giai doan 1)
BASE_PARAMS = dict(
    L=600.0,        # Chu vi vong tron (m)
    K=6,            # So luong vung thuc an
    smin=40.0,      # Khoang cach toi thieu giua cac tam patch (m)
    s=10.0,         # Do rong thuc cua vung thuc an Gauss (m)
    sig=15.0,       # Tam cam nhan cuc bo cua ca the (m)
    Bg=100.0,       # Toc do bam gradient thuc an (m^2/phut)
    D=50.0,         # He so khuech tan Brown (m^2/phut)
    e=1.0,          # Toc do tieu thu thuc an toi da (don vi/phut)
    Gm=100.0,       # Ky vong suc chua thuc an ban dau (don vi)
    Gh=10.0,        # Hang so Michaelis-Menten
    r=0.001,        # Toc do tai tao thuc an tu nhien (phut^-1)
    T=600.0,        # Thoi gian mot ngay hoat dong (phut)
    nu=0.5,         # Do di the log-normal cua cac vung thuc an
    dt=0.25,        # Buoc thoi gian tich phan Euler-Maruyama (phut)
)

# Tham so tuong tac hai ca the (Giai doan 2)
STAGE2_PARAMS = dict(
    Bc=4.0,         # Cuong do dap ung theo tin hieu giao tiep (m/phut)
    Bs=0.2,         # Do cung lo xo giu khoang cach xa hoi (phut^-1)
    d0=50.0,        # Khoang cach ua thich (m)
    Rc=50.0,        # Ban kinh suy giam tin hieu am thanh (m)
    rho0=1.0,       # Cuong do tin hieu tai nguon
    g0=0.05,        # Nguong nhan thuc tai cho
    delta0=2.0,     # Do rong lam tron ham dau (m)
    lam=0.0,        # He so canh tranh can thiep truc tiep
    kappa=1.0,      # He so giu vung an MVT (1 = thu dong, 0.3 = co giu vung)
    t0=30.0,        # Thoi gian bat dau kich hoat co che MVT (phut)
    kernel='gauss', # Ham nhan suy giam tin hieu: 'gauss', 'exp', 'power'
)

# Tham so chat luong thong tin & chi phi (Giai doan 3)
STAGE3_PARAMS = dict(
    b=3,            # Bac chi tiet thong diep: 0, 1, 2, 3, 4
    sigc=0.0,       # Do lech chuan nhiễu Ornstein-Uhlenbeck (m)
    tauc=10.0,      # Thoi gian tuong quan cua nhiễu (phut)
    recv='naive',   # Kieu nguoi nhan: 'naive' (wB=1) hoac 'bayes'
    sigp=50.0,      # Do bat dinh tien nghiem cho Bayes (m)
    c0=0.004,       # Chi phi phat tin co so (don vi thuc an/(bac * phut))
    Gq=0.3,         # Nguong chuan hoa chat luong thuc an o bac b=4
    gate='soft',    # Cong cam nhan: 'soft' (gamma), 'off' (gamma=0)
    gamma=1.0,      # So mu cong cam nhan (1 - q_i)^gamma
    noise_mask=(1.0, 1.0), # Mat na nhieu cho hai ca the (A, B)
)

# Tham so lua doi & niem tin (Giai doan 4)
STAGE4_PARAMS = dict(
    t=1.0,          # Ti le noi that cua nguoi gui (t in [0, 1])
    Delta=60.0,     # Do lech toa do khi noi doi (m)
    taul=10.0,      # Thoi gian duy tri trang thai noi doi (phut)
    liars='A',      # Ca the noi doi: 'A', 'B', hoac 'both'
    trust=1.0,      # Trong so niem tin ban dau cua nguoi nhan
    learn=False,    # Co bat quy tac hoc cap nhat niem tin khong
    nup=0.2,        # Toc do hoc niem tin
    dchk=10.0,      # Ban kinh kiem chung tai vi tri bao tin (m)
    qmsg=0.3,       # Nguong thuc an toi thieu de coi la co thuc an
    shared=True,    # Dung chung kho thuc an (True) hay kho rieng (False)
    couple=0.0,     # Canh tranh kho lien tuc giua hai ca the s in [0, 1]
    lietype='default', # Huong noi doi: 'default'/'toward', 'away', 'rand', 'withhold', 'flip'
    forget=0.0,     # He so quen niem tin (phut^-1)
    ptrust0=1.0,    # Muc neo niem tin mac dinh khi quen
    pstep=100,      # Chu ky ghi quy dao niem tin (buoc)
    cl=0.0,         # Chi phi noi tai cua hanh vi noi doi (don vi/phut)
    surrogate=False, # Che do mo hinh rut gon surrogate
    alpha_eff=None, # He so hieu luc tin hieu ngoai sinh (neu chay surrogate)
)

def wrap(d: np.ndarray, L: float) -> np.ndarray:
    """Khoang cach ngan nhat co dau tren vong tron tuan hoan chu vi L."""
    return (d + L / 2.0) % L - L / 2.0

def make_env(p: Dict[str, Any], R: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sinh ngau nhien moi truong: vi tri patch C (R, K), suc chua Gcap (R, K), vi tri ban dau x0 (R)."""
    rng = np.random.default_rng(seed)
    K = int(p['K'])
    L = float(p['L'])
    smin = float(p['smin'])
    C = np.zeros((R, K), dtype=np.float64)
    for i in range(R):
        cs: List[float] = []
        tries = 0
        while len(cs) < K:
            tries += 1
            if tries > 600:
                cs = []
                tries = 0
            c = rng.uniform(0.0, L)
            if all(min(abs(c - q), L - abs(c - q)) >= smin for q in cs):
                cs.append(c)
        C[i] = cs
    z = rng.standard_normal((R, K))
    nu = float(p['nu'])
    Gcap = float(p['Gm']) * np.exp(nu * z - nu**2 / 2.0)
    x0 = rng.uniform(0.0, L, R)
    return C, Gcap, x0

def run_simulation(p_in: Dict[str, Any], R: int = 1000, seed: int = 1, noise_seed: int = 2, msg_seed: int = 7) -> Dict[str, Any]:
    """
    Chay mo phong toan phan (Full Vectorized Simulation) cho R ban sao.
    Tra ve day du: food, net_E, qint, lint, dmean, ptraj, pend.
    """
    p = {**BASE_PARAMS, **STAGE2_PARAMS, **STAGE3_PARAMS, **STAGE4_PARAMS, **p_in}
    
    C, Gcap, xa = make_env(p, R, seed)
    rng = np.random.default_rng(noise_seed)
    rm = np.random.default_rng(msg_seed)
    rl = np.random.default_rng(p.get('lie_seed', 11))
    
    L, s, sig, dt = float(p['L']), float(p['s']), float(p['sig']), float(p['dt'])
    seff2 = s**2 + sig**2
    amp = s / np.sqrt(seff2)
    Gref = float(p['Gm'])
    e, Gh, r = float(p['e']), float(p['Gh']), float(p['r'])
    Bg, Bc, Bs = float(p['Bg']), float(p['Bc']), float(p['Bs'])
    d0, Rc, rho0 = float(p['d0']), float(p['Rc']), float(p['rho0'])
    g0, del0 = float(p['g0']), float(p['delta0'])
    b = int(p['b'])
    sigc, tauc, c0, cl = float(p['sigc']), float(p['tauc']), float(p['c0']), float(p['cl'])
    shared = bool(p['shared'])
    Delta, taul, qmsg = float(p['Delta']), float(p['taul']), float(p['qmsg'])
    
    # Ty le noi that cua tung ca the
    if p['liars'] == 'A':
        tt = np.array([float(p['t']), 1.0])
    elif p['liars'] == 'B':
        tt = np.array([1.0, float(p['t'])])
    else:
        tt = np.array([float(p['t']), float(p['t'])])
        
    pt = np.ones((2, R), dtype=np.float64)
    pt[1] = float(p['trust']) # Niem tin cua B
    if p['liars'] == 'both':
        pt[0] = float(p['trust'])
        
    learn = bool(p['learn'])
    lam, kappa, couple = float(p['lam']), float(p['kappa']), float(p['couple'])
    lt = str(p['lietype'])
    forget, t0k = float(p['forget']), float(p['t0'])
    ptrust0 = float(p['ptrust0'])
    kernel = str(p['kernel'])
    surrogate = bool(p.get('surrogate', False))
    alpha_eff = p.get('alpha_eff', None)
    
    # Khoi tao vi tri ban dau
    side = rng.choice([-1.0, 1.0], R)
    x = np.stack([xa, (xa + side * d0) % L]) # (2, R)
    G = Gcap.copy() if shared else np.stack([Gcap, Gcap]).copy() # (R, K) hoac (2, R, K)
    
    mask = np.array(p['noise_mask'])[:, None] # (2, 1)
    eta = (sigc * rm.standard_normal((2, R))) * mask
    ou_a = dt / tauc
    ou_s = sigc * np.sqrt(2.0 * dt / tauc)
    
    food = np.zeros((2, R), dtype=np.float64)
    qsum = np.zeros((2, R), dtype=np.float64)
    lsum = np.zeros((2, R), dtype=np.float64)
    dsum = np.zeros(R, dtype=np.float64)
    
    gam = 0.0 if p['gate'] == 'off' else float(p['gamma'])
    wB = (float(p['sigp'])**2 / (float(p['sigp'])**2 + sigc**2)) if p['recv'] == 'bayes' else 1.0
    
    n_steps = int(round(float(p['T']) / dt))
    sd = np.sqrt(2.0 * float(p['D']) * dt)
    ar = np.arange(R)
    nref = max(1, int(round(taul / dt)))
    lie = np.zeros((2, R), dtype=bool)
    xi = np.ones((2, R), dtype=np.float64)
    active = (Bc > 0.0) and (b >= 1)
    
    ptraj = []
    pstep = int(p.get('pstep', 100))
    
    for t_step in range(n_steps):
        # 1. Tinh khoang cach toi cac patch
        d_patch = wrap(x[:, :, None] - C[None, :, :], L) # (2, R, K)
        w = np.exp(-d_patch**2 / (2.0 * s**2)) # (2, R, K)
        
        Gc = G[None, :, :] if shared else G # (2, R, K)
        cont = (Gc / Gref) * amp * np.exp(-d_patch**2 / (2.0 * seff2))
        gs = cont.sum(axis=2) # (2, R)
        grad = (cont * (-d_patch / seff2)).sum(axis=2) # (2, R)
        
        h = Gc / (Gc + Gh)
        raw = e * w * h * dt # (2, R, K)
        
        if lam > 0.0:
            raw = raw / (1.0 + lam * w[::-1])
            
        if shared:
            tot = raw.sum(axis=0) # (R, K)
            sc = np.minimum(1.0, G / np.maximum(tot, 1e-12))
            take = raw * sc[None, :, :]
            G = G - take.sum(axis=0) + r * (Gcap - G) * dt
        else:
            take = np.minimum(raw, G)
            if couple > 0.0:
                G = np.maximum(G - take - couple * take[::-1] + r * (Gcap[None, :, :] - G) * dt, 0.0)
            else:
                G = G - take + r * (Gcap[None, :, :] - G) * dt
                
        food += take.sum(axis=2)
        Iin = raw.sum(axis=2) / dt
        
        # 2. Khoang cach giua hai ca the
        dsep = wrap(x[0] - x[1], L)
        dabs = np.abs(dsep)
        dsum += dabs
        
        vel = Bg * grad
        q = gs / (gs + g0)
        
        # 3. Luc tuong tac giao tiep
        if active:
            idx = cont.argmax(axis=2)
            cdom = np.stack([C[ar, idx[0]], C[ar, idx[1]]]) # (2, R)
            
            # Ham nhan suy giam tin hieu rho(d)
            if kernel == 'gauss':
                rho = rho0 * np.exp(-dabs**2 / (2.0 * Rc**2))
            elif kernel == 'exp':
                rho = rho0 * np.exp(-dabs * np.log(2.0) / (1.1774 * Rc))
            else: # power law
                rho = rho0 / (1.0 + (dabs / (1.1774 * Rc))**2)
                
            # Cap nhat nhiễu Ornstein-Uhlenbeck
            if sigc > 0.0:
                eta = (eta * (1.0 - ou_a) + ou_s * rm.standard_normal((2, R))) * mask
                
            # Cap nhat trang thai noi doi theo block taul
            if t_step % nref == 0:
                lie = rl.random((2, R)) > tt[:, None]
                if lt == 'rand':
                    xi = np.where(rl.random((2, R)) > 0.5, 1.0, -1.0)
                    
            if surrogate and alpha_eff is not None:
                # Che do surrogate rut gon: thay the toan bo giao tiep bang alpha_eff
                # giu nguyen cong gam va sgn
                for i in (0, 1):
                    j = 1 - i
                    base = cdom[j]
                    dl = wrap(base - x[i], L)
                    resp = np.sign(dl)
                    gate = (1.0 - q[i])**gam if gam != 1.0 else (1.0 - q[i])
                    vel[i] = vel[i] + alpha_eff * Bc * rho * q[j] * gate * resp
            else:
                # Che do mo hinh goc day du
                for i in (0, 1):
                    j = 1 - i
                    base = x[j] if b == 1 else cdom[j]
                    a = wrap(base + eta[j] - x[i], L)
                    shift = lt in ('default', 'toward', 'away', 'rand')
                    
                    if b == 1:
                        resp0 = np.tanh(a / del0)
                        dl = a
                    elif b == 2:
                        resp0 = np.sign(a)
                        dl = a
                    else:
                        if shift:
                            if lt == 'away':
                                dls = a + np.sign(a) * Delta
                            elif lt == 'rand':
                                dls = a + xi[j] * Delta
                            else: # default / toward
                                dls = a - np.sign(a) * Delta
                            dl = np.where(lie[j], dls, a)
                        else:
                            dl = a
                        resp0 = np.tanh(dl / del0) if shift else np.tanh(a / del0)
                        
                    if b == 4:
                        Ghat = G[ar, idx[j]] / Gref if shared else G[i][ar, idx[j]] / Gref
                        resp0 = resp0 * np.minimum(1.0, Ghat / p['Gq'])
                        
                    if (b < 3) or (not shift):
                        mode = lt if lt in ('withhold', 'flip') else ('withhold' if b == 1 else 'flip')
                        resp = np.where(lie[j], 0.0 if mode == 'withhold' else -resp0, resp0)
                    else:
                        resp = resp0
                        
                    gate = (1.0 - q[i])**gam if gam != 1.0 else (1.0 - q[i])
                    vel[i] = vel[i] + pt[i] * wB * Bc * rho * q[j] * gate * resp
                    
                    # Quy tac hoc niem tin
                    if learn and i == 1 and b >= 3:
                        near = (np.abs(dl) < p['dchk']) & (q[j] > qmsg)
                        conf = near & (q[i] > qmsg)
                        disc = near & (q[i] <= qmsg)
                        pt[i] = np.clip(
                            pt[i] + dt * p['nup'] * ((1.0 - pt[i]) * conf - pt[i] * disc)
                            + dt * forget * (ptrust0 - pt[i]),
                            0.0, 1.0
                        )
            qsum += q
            lsum += lie * q
            
        if learn and t_step % pstep == 0:
            ptraj.append(pt[1].mean())
            
        # 4. Luc lo xo giu khoang cach
        if Bs > 0.0:
            sp = Bs * (d0 - dabs) * np.sign(dsep)
            vel[0] = vel[0] + sp
            vel[1] = vel[1] - sp
            
        # 5. Co che giu vung an MVT
        mult = 1.0
        if kappa < 1.0:
            cur_time = (t_step + 1) * dt
            mult = np.where((Iin > food / cur_time) & (cur_time > t0k), kappa, 1.0)
            
        x = (x + vel * dt + mult * sd * rng.standard_normal((2, R))) % L
        
    Tm = float(p['T'])
    gross_E = food / Tm # (2, R)
    comm_cost = (c0 * b * qsum * dt) / Tm if active else np.zeros((2, R), dtype=np.float64)
    lie_cost = (cl * lsum * dt) / Tm if active else np.zeros((2, R), dtype=np.float64)
    net_E = gross_E - comm_cost - lie_cost
    
    return dict(
        food=gross_E,
        net_E=net_E,
        comm_cost=comm_cost,
        lie_cost=lie_cost,
        qint=qsum * dt,
        lint=lsum * dt,
        dmean=dsum / n_steps,
        ptraj=np.array(ptraj),
        pend=pt[1].copy(),
        b=b,
        c0=c0,
        R=R,
    )

def stats_ci(x: np.ndarray, alpha: float = 0.05) -> Tuple[float, float, float, float]:
    """Tra ve (mean, se, ci_low, ci_high) voi 95% CI."""
    m = float(np.mean(x))
    n = len(x)
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    z = 1.96
    return m, se, m - z * se, m + z * se

def paired_diff(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, float, float]:
    """Tinh hieu so cap CRN: mean, se, ci_low, ci_high."""
    diff = a - b
    return stats_ci(diff)
