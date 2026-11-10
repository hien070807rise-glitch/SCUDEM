import numpy as np, pickle, os, sys, time
from sim_gd4 import run4
F='res4.pkl'; res=pickle.load(open(F,'rb')) if os.path.exists(F) else {}
BUD=float(sys.argv[1]); t0=time.time()
NS=5
def save(): pickle.dump(res,open(F,'wb'))
BASEP=dict(Bs=0.,Bc=8.,c0=0.,b=3)
def cell(key,**kw):
    if key in res: return
    if time.time()-t0>BUD: save(); print('stop',len(res),flush=True); sys.exit(0)
    outs=[run4({**BASEP,**kw,'lie_seed':11+k},R=1000,seed=1+k,noise_seed=2+k,msg_seed=7+k) for k in range(NS)]
    d=dict(food=np.concatenate([o['food'] for o in outs],1).astype(np.float32),
           qint=np.concatenate([o['qint'] for o in outs],1).astype(np.float32),
           lint=np.concatenate([o['lint'] for o in outs],1).astype(np.float32),
           pend=np.concatenate([o['pend'] for o in outs]).astype(np.float32),
           dmean=np.mean([o['dmean'] for o in outs]))
    if kw.get('learn'): d['ptraj']=np.mean([o['ptraj'] for o in outs],0)
    res[key]=d; save(); print(round(time.time()-t0),key,flush=True)
cell('E0',b=0); cell('E0sep',b=0,shared=False)
cell('H',t=1.); cell('Hsep',t=1.,shared=False)
for sh in (True,False):
    for D in (30.,60.,100.):
        for t in (0.8,0.6,0.4,0.2,0.):
            cell(('L',t,D,sh),t=t,Delta=D,shared=sh)
for s in (10.,20.,30.,45.,60.,80.,100.):
    cell(('N',s),sigc=s,noise_mask=(1.,0.))
for D in (10.,20.,45.,80.):
    cell(('L',0.,D,True),t=0.,Delta=D)
cell('P0',t=0.,trust=0.)
for p in (0.25,0.5,0.75): cell(('TR',p),t=0.,trust=p)
for b in (1,2,3,4):
    cell(('Hb',b),b=b,t=1.)
    for t in (0.5,0.): cell(('B',b,t),b=b,t=t,Delta=60.)
for g in ('off',0.5):
    kw=dict(gate='off') if g=='off' else dict(gamma=0.5)
    for t in (1.,0.5,0.): cell(('GAM',g,t),t=t,Delta=60.,**kw)
for D in (60.,100.):
    for t in (1.,0.8,0.5,0.2,0.): cell(('LEARN',t,D),t=t,Delta=D,learn=True)
for d0 in (0.,40.,80.,140.):
    cell(('SPb0',d0),b=0,Bs=0.2,d0=d0)
    for t in (1.,0.5): cell(('SP',d0,t),Bs=0.2,d0=d0,t=t,Delta=60.)
for tl in (2.,60.): cell(('TL',tl),t=0.5,Delta=60.,taul=tl)
for t in (1.,0.): cell(('Bc4',t),Bc=4.,t=t,Delta=60.)
save(); print('DONE',len(res),flush=True)
