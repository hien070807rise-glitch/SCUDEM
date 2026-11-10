import pickle, numpy as np
from analyze_gd5 import *
import exp_gd5
vars_={'Bc25':dict(Bc=2.5),'Bc3':dict(Bc=3.),'Bg100':dict(Bg=100.),'Bg200':dict(Bg=200.)}
out={}
for nm,v in vars_.items():
    Ms=[];Vv=[]
    for d0 in D0G:
        n,r=cellstat({**v,'d0':d0},2,1000); Ms.append(n.mean(0)); Vv.append(n.mean(0)-r.mean(0))
    Ms=np.array(Ms);Vv=np.array(Vv); d,ci=dstar_fit(Ms)
    gi=int(np.argmin(abs(np.array(D0G)-d)))
    out[nm]=dict(d=d,ci=ci,Vstar=Vv[gi].mean(),Vse=se(Vv[gi]),M=Ms.mean(1),V=Vv.mean(1))
    print(nm,'d*=%.1f CI[%.1f,%.1f] V*=%+.4f±%.4f'%(d,ci[0],ci[1],Vv[gi].mean(),se(Vv[gi])),'M',np.round(Ms.mean(1),4),flush=True)
pickle.dump(out,open('scan.pkl','wb'))
