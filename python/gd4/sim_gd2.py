import numpy as np
from sim1 import BASE, make_env
P2 = dict(Bc=4.0, Bs=0.2, d0=50., Rc=50., rho0=1., g0=0.05, delta0=2., lam=0., kappa=1., t0=30.)
def wrap(d, L):
    return (d + L / 2) % L - L / 2
def run2(p, R=1000, seed=1, noise_seed=2, shared=True, report=False):
    p = {**BASE, **P2, **p}
    C, Gcap, xa = make_env(p, R, seed)
    rng = np.random.default_rng(noise_seed)
    L, s, sig, dt = p['L'], p['s'], p['sig'], p['dt']
    seff2 = s**2 + sig**2; amp = s / np.sqrt(seff2); Gref = p['Gm']
    e, Gh, r, Bg, Bc, Bs = p['e'], p['Gh'], p['r'], p['Bg'], p['Bc'], p['Bs']
    d0, Rc, rho0, g0, del0, lam = p['d0'], p['Rc'], p['rho0'], p['g0'], p['delta0'], p['lam']
    kappa, t0 = p['kappa'], p['t0']
    side = rng.choice([-1., 1.], R)
    x = np.stack([xa, (xa + side * d0) % L])
    G = Gcap.copy() if shared else np.stack([Gcap, Gcap]).copy()
    food = np.zeros((2, R)); dsum = np.zeros(R); ovl = np.zeros(R)
    n = int(round(p['T'] / dt)); sd = np.sqrt(2 * p['D'] * dt)
    ar = np.arange(R)
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
            tot = raw.sum(0)
            sc = np.minimum(1., G / np.maximum(tot, 1e-12))
            take = raw * sc[None]
            G = G - take.sum(0) + r * (Gcap - G) * dt
        else:
            take = np.minimum(raw, G)
            G = G - take + r * (Gcap[None] - G) * dt
        food += take.sum(2)
        I = raw.sum(2) / dt
        dsep = wrap(x[0] - x[1], L); dabs = np.abs(dsep)
        vel = Bg * grad
        if Bc > 0:
            idx = cont.argmax(2)
            cdom = np.stack([C[ar, idx[0]], C[ar, idx[1]]])
            q = gs / (gs + g0)
            rho = rho0 * np.exp(-dabs**2 / (2 * Rc**2))
            for i in (0, 1):
                j = 1 - i
                vel[i] = vel[i] + Bc * rho * q[j] * (1 - q[i]) * np.tanh(wrap(cdom[j] - x[i], L) / del0)
        if Bs > 0:
            sp = Bs * (d0 - dabs) * np.sign(dsep)
            vel[0] = vel[0] + sp; vel[1] = vel[1] - sp
        mult = 1.
        if kappa < 1.:
            tt = (t + 1) * dt
            mult = np.where((I > food / tt) & (tt > t0), kappa, 1.)
        x = (x + vel * dt + mult * sd * rng.standard_normal((2, R))) % L
        dsum += dabs; ovl += (w[0] * w[1]).sum(1)
    E = food / p['T']
    out = dict(E=E.mean(0), EA=E[0], EB=E[1])
    return out
