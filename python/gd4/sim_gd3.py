import numpy as np
from sim1 import BASE, make_env
from sim_gd2 import P2, wrap
P3 = dict(b=3, sigc=0., tauc=10., recv='naive', c0=0.004, sigp=50., Gq=0.3, gate='soft', gamma=1.0, qthr=0.5)
def run3(p, R=1000, seed=1, noise_seed=2, msg_seed=7):
    p = {**BASE, **P2, **P3, **p}
    C, Gcap, xa = make_env(p, R, seed)
    rng = np.random.default_rng(noise_seed); rm = np.random.default_rng(msg_seed)
    L, s, sig, dt = p['L'], p['s'], p['sig'], p['dt']
    seff2 = s**2 + sig**2; amp = s / np.sqrt(seff2); Gref = p['Gm']
    e, Gh, r, Bg, Bc, Bs = p['e'], p['Gh'], p['r'], p['Bg'], p['Bc'], p['Bs']
    d0, Rc, rho0, g0, del0 = p['d0'], p['Rc'], p['rho0'], p['g0'], p['delta0']
    b, sigc, tauc, recv, c0 = int(p['b']), p['sigc'], p['tauc'], p['recv'], p['c0']
    wB = p['sigp']**2 / (p['sigp']**2 + sigc**2) if recv == 'bayes' else 1.0
    side = rng.choice([-1., 1.], R)
    x = np.stack([xa, (xa + side * d0) % L]); G = Gcap.copy()
    eta = sigc * rm.standard_normal((2, R))
    ou_a = dt / tauc; ou_s = sigc * np.sqrt(2 * dt / tauc)
    food = np.zeros((2, R)); qsum = np.zeros((2, R)); dsum = np.zeros(R)
    gam = 0.0 if p['gate'] == 'off' else float(p['gamma'])
    n = int(round(p['T'] / dt)); sd = np.sqrt(2 * p['D'] * dt); ar = np.arange(R)
    active = (Bc > 0) and (b >= 1)
    for t in range(n):
        d = wrap(x[:, :, None] - C[None], L)
        w = np.exp(-d**2 / (2 * s**2))
        cont = (G[None] / Gref) * amp * np.exp(-d**2 / (2 * seff2))
        gs = cont.sum(2); grad = (cont * (-d / seff2)).sum(2)
        h = G[None] / (G[None] + Gh)
        raw = e * w * h * dt
        tot = raw.sum(0); sc = np.minimum(1., G / np.maximum(tot, 1e-12))
        take = raw * sc[None]
        G = G - take.sum(0) + r * (Gcap - G) * dt
        food += take.sum(2)
        dsep = wrap(x[0] - x[1], L); dabs = np.abs(dsep)
        vel = Bg * grad; q = gs / (gs + g0)
        if active:
            idx = cont.argmax(2)
            cdom = np.stack([C[ar, idx[0]], C[ar, idx[1]]])
            rho = rho0 * np.exp(-dabs**2 / (2 * Rc**2))
            if sigc > 0:
                eta = eta * (1 - ou_a) + ou_s * rm.standard_normal((2, R))
            for i in (0, 1):
                j = 1 - i
                base = x[j] if b == 1 else cdom[j]
                dl = wrap(base + eta[j] - x[i], L)
                if b == 2: resp = np.sign(dl)
                else: resp = np.tanh(dl / del0)
                if b == 4:
                    Ghat = G[ar, idx[j]] / Gref
                    resp = resp * np.minimum(1., Ghat / p['Gq'])
                gate = (1 - q[i]) ** gam if gam != 1.0 else (1 - q[i])
                vel[i] = vel[i] + wB * Bc * rho * q[j] * gate * resp
            qsum += q
        if Bs > 0:
            sp = Bs * (d0 - dabs) * np.sign(dsep)
            vel[0] = vel[0] + sp; vel[1] = vel[1] - sp
        x = (x + vel * dt + sd * rng.standard_normal((2, R))) % L
        dsum += dabs
    cost = (c0 * b * qsum * dt) if active else np.zeros((2, R))
    Eg = food / p['T']; En = (food - cost) / p['T']
    return dict(E=En.mean(0), Eg=Eg.mean(0), EA=En[0], EB=En[1], EgA=Eg[0], EgB=Eg[1], qint=(qsum*dt/p['T']).mean(0), dmean=(dsum/n).mean())
