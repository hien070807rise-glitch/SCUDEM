import numpy as np
BASE = dict(L=600., K=6, smin=40., s=10., sig=15., Bg=100., D=50., e=1., Gm=100., Gh=10.,
            r=0.001, T=600., nu=0.5, dt=0.25)

def make_env(p, R, seed):
    rng = np.random.default_rng(seed)
    K = int(p['K']); L = p['L']
    C = np.zeros((R, K))
    for i in range(R):
        cs = []; tries = 0
        while len(cs) < K:
            tries += 1
            if tries > 500: cs = []; tries = 0
            c = rng.uniform(0, L)
            if all(min(abs(c - q), L - abs(c - q)) >= p['smin'] for q in cs): cs.append(c)
        C[i] = cs
    z = rng.standard_normal((R, K))
    nu = p['nu']
    Gcap = p['Gm'] * np.exp(nu * z - nu**2 / 2)
    x0 = rng.uniform(0, L, R)
    return C, Gcap, x0
