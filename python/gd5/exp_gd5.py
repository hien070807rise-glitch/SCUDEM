import numpy as np, pickle, os, sys, time
from sim_gd4 import run4
from sim_gd5 import run5
F='res_gd5.pkl'
res=pickle.load(open(F,'rb')) if os.path.exists(F) else {}
t00=time.time()
def log(m): print(f"[{time.time()-t00:6.0f}s] {m}",flush=True)
def save(): pickle.dump(res,open(F,'wb'))
BASEP=dict(Bs=0.2,Bc=8.,c0=0.,b=3,d0=50.,Rc=50.)
def hk(d): return tuple(sorted((k,str(v)) for k,v in d.items()))
def get(p,ns=3,R=1000,fn=run5,off=0):
    q={**BASEP,**p}
    key=(hk(q),ns,R,fn.__name__,off)
    if key in res: return res[key]
    outs=[fn({**q,'lie_seed':11+k+off},R=R,seed=1+k+off,noise_seed=2+k+off,msg_seed=7+k+off) for k in range(ns)]
    d=dict(food=np.concatenate([o['food'] for o in outs],1).astype(np.float32),
           qint=np.concatenate([o['qint'] for o in outs],1).astype(np.float32),
           lint=np.concatenate([o['lint'] for o in outs],1).astype(np.float32),
           dmean=float(np.mean([o['dmean'] for o in outs])))
    res[key]=d; return d
def net(o,b,c0=0.004):
    return o['food']-c0*b*o['qint']/600.
def ref(p,ns=3,R=1000,off=0):
    q={k:v for k,v in p.items() if k not in('b','Bc','Rc','sigc','t','Delta','gamma','gate','g0','trust','kernel')}
    return get({**q,'b':0},ns,R,off=off)
if __name__=='__main__':
    stage=sys.argv[1]
    if stage=='tornado':
        base={}
        L=[('base',{}),
           ('d0lo',dict(d0=20.)),('d0hi',dict(d0=140.)),
           ('Rclo',dict(Rc=25.)),('Rchi',dict(Rc=100.)),
           ('b1',dict(b=1)),('b4',dict(b=4)),
           ('Bc4',dict(Bc=4.)),('Bc16',dict(Bc=16.)),
           ('sig60',dict(sigc=60.)),('sig100',dict(sigc=100.)),
           ('t0',dict(t=0.,Delta=60.)),('t05',dict(t=0.5,Delta=60.)),
           ('gam0',dict(gate='off')),('gam05',dict(gamma=0.5)),
           ('g002',dict(g0=0.02)),('g05',dict(g0=0.5)),
           ('Bg0',dict(Bg=0.)),('Bg300',dict(Bg=300.)),
           ('s5',dict(s=5.)),('s20',dict(s=20.)),
           ('K3',dict(K=3,Gm=200.)),('K11',dict(K=11,Gm=600./11)),
           ('D25',dict(D=25.)),('D100',dict(D=100.)),
           ('sg5',dict(sig=5.)),('sg40',dict(sig=40.)),
           ('kap',dict(kappa=0.3)),('lam',dict(lam=1.)),
           ('trust0',dict(t=0.,Delta=60.,trust=0.)),
           ('Bs1',dict(Bs=1.)),('Bs005',dict(Bs=0.05))]
        for name,p in L:
            get(p); ref(p); save(); log(name)
        log('TORNADO DONE')
    if stage=='inter':
        lev={'d0':dict(d0=100.),'sig':dict(sigc=60.),'t':dict(t=0.,Delta=60.),'Rc':dict(Rc=25.),
             'b':dict(b=1),'Bg':dict(Bg=300.),'kap':dict(kappa=0.3),'lam':dict(lam=1.),'gam':dict(gamma=0.5),'g0':dict(g0=0.5)}
        pairs=[('d0','sig'),('d0','t'),('sig','t'),('b','sig'),('Rc','d0'),('Bg','t'),('kap','t'),('lam','t'),('gam','sig'),('g0','t'),('gam','t'),('Rc','sig')]
        for k,p in lev.items(): get(p); ref(p)
        get({}); ref({})
        for a,b_ in pairs:
            p={**lev[a],**lev[b_]}; get(p); ref(p); save(); log((a,b_))
        log('INTER DONE')
    if stage=='kernel':
        for ker in ('gauss','exp','power'):
            for d0 in (0.,20.,40.,60.,80.,100.,140.,200.):
                p=dict(kernel=ker,d0=d0); get(p,ns=2); ref(p,ns=2)
            save(); log(ker+' spring')
            for nm,p in [('H',dict(Bs=0.)),('L',dict(Bs=0.,t=0.,Delta=60.)),('N30',dict(Bs=0.,sigc=30.)),
                         ('N60',dict(Bs=0.,sigc=60.)),('N100',dict(Bs=0.,sigc=100.)),('P0',dict(Bs=0.,t=0.,Delta=60.,trust=0.)),
                         ('L100',dict(Bs=0.,t=0.,Delta=100.))]:
                get({**p,'kernel':ker},ns=3); ref({**p,'kernel':ker},ns=3)
            save(); log(ker+' free')
        log('KERNEL DONE')
    if stage=='dstar':
        vars_={'base':{},'Bc2':dict(Bc=2.),'Bc4':dict(Bc=4.),'Bc16':dict(Bc=16.),'Bs005':dict(Bs=0.05),'Bs1':dict(Bs=1.),
               'lam1':dict(lam=1.),'kap03':dict(kappa=0.3),'Bg25':dict(Bg=25.),'Bg300':dict(Bg=300.),'g05':dict(g0=0.5),'gam0':dict(gate='off')}
        for nm,v in vars_.items():
            for d0 in (0.,20.,40.,60.,80.,100.,140.,200.):
                p={**v,'d0':d0}; get(p,ns=2); ref(p,ns=2)
            save(); log('dstar '+nm)
        log('DSTAR DONE')
