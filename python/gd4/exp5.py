import numpy as np, pickle, os, sys, time
from sim_gd4 import run4
F='res5.pkl'; res=pickle.load(open(F,'rb')) if os.path.exists(F) else {}
BUD=float(sys.argv[1]); t0=time.time(); NS=5
def save(): pickle.dump(res,open(F,'wb'))
BASEP=dict(Bs=0.,Bc=8.,c0=0.,b=3)
def cell(key,off=0,**kw):
    if key in res: return
    if time.time()-t0>BUD: save(); print('stop',len(res),flush=True); sys.exit(0)
    outs=[run4({**BASEP,**kw,'lie_seed':11+k+off},R=1000,seed=1+k+off,noise_seed=2+k+off,msg_seed=7+k+off) for k in range(NS)]
    d=dict(food=np.concatenate([o['food'] for o in outs],1).astype(np.float32),
           qint=np.concatenate([o['qint'] for o in outs],1).astype(np.float32),
           lint=np.concatenate([o['lint'] for o in outs],1).astype(np.float32),
           pend=np.concatenate([o['pend'] for o in outs]).astype(np.float32))
    if kw.get('learn'): d['ptraj']=np.mean([o['ptraj'] for o in outs],0)
    res[key]=d; save(); print(round(time.time()-t0),key,flush=True)
# 1. nguoi nhan biet t: luoi p x t
for t in (.8,.6,.4,.2):
    for p in (.2,.4,.6,.8): cell(('PT',p,t),t=t,Delta=60.,trust=p)
# 2. kiem tra ngoai mau (hat giong 11..15)
O=10
cell(('OOS','E0'),O,b=0); cell(('OOS','H'),O,t=1.); cell(('OOS','P0'),O,t=0.,trust=0.)
cell(('OOS','L0'),O,t=0.,Delta=60.); cell(('OOS','L4'),O,t=.4,Delta=60.); cell(('OOS','N60'),O,sigc=60.,noise_mask=(1.,0.))
cell(('OOS','Hsep'),O,t=1.,shared=False); cell(('OOS','L0sep'),O,t=0.,Delta=60.,shared=False)
# 3. canh tranh lien tuc
for s in (.25,.5,.75):
    for t in (1.,0.): cell(('CP',s,t),t=t,Delta=60.,shared=False,couple=s)
for t in (1.,0.): cell(('CL',t),t=t,Delta=60.,lam=1.)
for t in (1.,0.): cell(('CK',t),t=t,Delta=60.,kappa=0.3)
# 4. kieu noi doi
for lt in ('away','rand'):
    for t in (.5,0.): cell(('LY',lt,t),t=t,Delta=60.,lietype=lt)
for lt in ('withhold','flip'):
    for b in (1,2,3,4):
        for t in (.5,0.): cell(('LC',lt,b,t),t=t,Delta=60.,lietype=lt,b=b)
# 5. hoc niem tin: bien the
for nu in (.05,.8):
    for t in (.5,.2): cell(('LV',nu,t),t=t,Delta=60.,learn=True,nup=nu)
for dc in (5.,20.): cell(('LVd',dc),t=.2,Delta=60.,learn=True,dchk=dc)
for t in (.5,.2): cell(('LVf',t),t=t,Delta=60.,learn=True,forget=0.02)
cell(('LT','H'),T=1800.,t=1.)
for t in (.5,.2):
    cell(('LT','F',t),T=1800.,t=t,Delta=60.); cell(('LT','L',t),T=1800.,t=t,Delta=60.,learn=True)
pass
cell(('CL','P0'),t=0.,Delta=60.,lam=1.,trust=0.); cell(('CK','P0'),t=0.,Delta=60.,kappa=0.3,trust=0.)
save(); print('DONE2',len(res),flush=True)
