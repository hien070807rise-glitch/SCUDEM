"""Giai doan 4: lua doi co huong (ti le noi that t, do lech Delta), niem tin p, canh tranh bat/tat."""
import numpy as np
from sim1 import BASE, make_env
from sim_gd2 import P2, wrap
from sim_gd3 import P3

P4 = dict(t=1.0, Delta=60., taul=10., liars='A', trust=1.0, learn=False, nup=0.2, dchk=10.,
          shared=True, noise_mask=(1., 1.), lie_seed=11, pstep=100, qmsg=0.3,
          lam=0., kappa=1., couple=0., lietype='default', forget=0., ptrust0=1.)

def run4(p, R=1000, seed=1, noise_seed=2, msg_seed=7):
    p = {**BASE, **P2, **P3, **P4, **p}
    C, Gcap, xa = make_env(p, R, seed)
    rng = np.random.default_rng(noise_seed); rm = np.random.default_rng(msg_seed)
    rl = np.random.default_rng(p['lie_seed'])
    L, s, sig, dt = p['L'], p['s'], p['sig'], p['dt']
    seff2 = s**2 + sig**2; amp = s / np.sqrt(seff2); Gref = p['Gm']
    e, Gh, r, Bg, Bc, Bs = p['e'], p['Gh'], p['r'], p['Bg'], p['Bc'], p['Bs']
    d0, Rc, rho0, g0, del0 = p['d0'], p['Rc'], p['rho0'], p['g0'], p['delta0']
    b, sigc, tauc, c0 = int(p['b']), p['sigc'], p['tauc'], p['c0']
    shared = p['shared']
    Delta, taul, qmsg = p['Delta'], p['taul'], p['qmsg']
    tt = np.array([p['t'], 1.0]) if p['liars'] == 'A' else np.array([p['t'], p['t']])
    pt = np.ones((2, R))
    pt[1] = p['trust']           # niem tin cua B (nguoi nhan bi lua); A tin B hoan toan
    if p['liars'] == 'both': pt[0] = p['trust']
    learn = p['learn']
    lam, kappa, couple, lt, forget, t0k = p['lam'], p['kappa'], p['couple'], p['lietype'], p['forget'], p['t0']
    xi = np.ones((2, R))
    side = rng.choice([-1., 1.], R)
    x = np.stack([xa, (xa + side * d0) % L])
    G = Gcap.copy() if shared else np.stack([Gcap, Gcap]).copy()
    mask = np.array(p['noise_mask'])[:, None]
    eta = (sigc * rm.standard_normal((2, R))) * mask
    ou_a = dt / tauc; ou_s = sigc * np.sqrt(2 * dt / tauc)
    food = np.zeros((2, R)); qsum = np.zeros((2, R)); lsum = np.zeros((2, R)); dsum = np.zeros(R)
    gam = 0.0 if p['gate'] == 'off' else float(p['gamma'])
    n = int(round(p['T'] / dt)); sd = np.sqrt(2 * p['D'] * dt); ar = np.arange(R)
    nref = max(1, int(round(taul / dt)))
    lie = np.zeros((2, R), bool)
    active = (Bc > 0) and (b >= 1)
    ptraj = []; ntrial = np.zeros(R)
    for t in range(n):
        d = wrap(x[:, :, None] - C[None], L)
        w = np.exp(-d**2 / (2 * s**2))
        Gc = G[None] if shared else G
        cont = (Gc / Gref) * amp * np.exp(-d**2 / (2 * seff2))
        gs = cont.sum(2); grad = (cont * (-d / seff2)).sum(2)
        h = Gc / (Gc + Gh)
        raw = e * w * h * dt
        if lam > 0:
            raw = raw / (1 + lam * w[::-1])
        if shared:
            tot = raw.sum(0); sc = np.minimum(1., G / np.maximum(tot, 1e-12))
            take = raw * sc[None]
            G = G - take.sum(0) + r * (Gcap - G) * dt
        else:
            take = np.minimum(raw, G)
            if couple > 0:
                G = np.maximum(G - take - couple * take[::-1] + r * (Gcap[None] - G) * dt, 0.)
            else:
                G = G - take + r * (Gcap[None] - G) * dt
        food += take.sum(2)
        Iin = raw.sum(2) / dt
        dsep = wrap(x[0] - x[1], L); dabs = np.abs(dsep)
        vel = Bg * grad; q = gs / (gs + g0)
        if active:
            idx = cont.argmax(2)
            cdom = np.stack([C[ar, idx[0]], C[ar, idx[1]]])
            rho = rho0 * np.exp(-dabs**2 / (2 * Rc**2))
            if sigc > 0:
                eta = (eta * (1 - ou_a) + ou_s * rm.standard_normal((2, R))) * mask
            if t % nref == 0:
                lie = rl.random((2, R)) > tt[:, None]
                if lt == 'rand': xi = np.where(rl.random((2, R)) > 0.5, 1., -1.)
            for i in (0, 1):
                j = 1 - i
                base = x[j] if b == 1 else cdom[j]
                a = wrap(base + eta[j] - x[i], L)
                shift = lt in ('default', 'toward', 'away', 'rand')
                if b == 1:
                    resp0 = np.tanh(a / del0); dl = a
                elif b == 2:
                    resp0 = np.sign(a); dl = a
                else:
                    if shift:
                        if lt == 'away': dls = a + np.sign(a) * Delta
                        elif lt == 'rand': dls = a + xi[j] * Delta
                        else: dls = a - np.sign(a) * Delta
                        dl = np.where(lie[j], dls, a)
                    else:
                        dl = a
                    resp0 = np.tanh(dl / del0) if shift else np.tanh(a / del0)
                    if b == 4:
                        Ghat = G[ar, idx[j]] / Gref if shared else G[i][ar, idx[j]] / Gref
                        resp0 = resp0 * np.minimum(1., Ghat / p['Gq'])
                if (b < 3) or (not shift):
                    mode = lt if lt in ('withhold', 'flip') else ('withhold' if b == 1 else 'flip')
                    resp = np.where(lie[j], 0. if mode == 'withhold' else -resp0, resp0)
                else:
                    resp = resp0
                gate = (1 - q[i]) ** gam if gam != 1.0 else (1 - q[i])
                vel[i] = vel[i] + pt[i] * Bc * rho * q[j] * gate * resp
                if learn and i == 1 and b >= 3:
                    near = (np.abs(dl) < p['dchk']) & (q[j] > qmsg)
                    conf = near & (q[i] > qmsg); disc = near & (q[i] <= qmsg)
                    pt[i] = np.clip(pt[i] + dt * p['nup'] * ((1 - pt[i]) * conf - pt[i] * disc) + dt * forget * (p['ptrust0'] - pt[i]), 0., 1.)
            qsum += q
            lsum += lie * q
        if learn and t % p['pstep'] == 0: ptraj.append(pt[1].mean())
        if Bs > 0:
            sp = Bs * (d0 - dabs) * np.sign(dsep)
            vel[0] = vel[0] + sp; vel[1] = vel[1] - sp
        mult = 1.
        if kappa < 1.:
            tt_ = (t + 1) * dt
            mult = np.where((Iin > food / tt_) & (tt_ > t0k), kappa, 1.)
        x = (x + vel * dt + mult * sd * rng.standard_normal((2, R))) % L
        dsum += dabs
    Tm = p['T']
    return dict(food=food / Tm, qint=qsum * dt, lint=lsum * dt, ptraj=np.array(ptraj), pend=pt[1].copy(),
                dmean=(dsum / n).mean(), b=b, c0=c0, Gfin=G, Gcap=Gcap)
