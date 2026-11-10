import numpy as np, pickle, time, sys
from scipy.stats import qmc
from sim_gd5 import run5
from sim_gd4 import run4
N=64; R=250; k=8
names=['d0','Rc','Bc','sigc','lie','Delta','gamma','g0']
lo=np.array([0,15,4,0,0,20,0,np.log(0.02)]); hi=np.array([140,100,16,100,1,100,1,np.log(0.5)])
S=qmc.Sobol(2*k,scramble=True,seed=5).random(N)
A=S[:,:k]; B=S[:,k:]
def scale(U):
    X=lo+U*(hi-lo); X[:,7]=np.exp(X[:,7]); return X
rows=[A,B]+[np.where(np.arange(k)==i,B[:,[i]],A) for i in range(k)]  # AB_i: A with column i from B
rows=[A,B]+[np.concatenate([A[:,:i],B[:,[i]],A[:,i+1:]],1) for i in range(k)]
refc={}
def V(x,seedoff=0):
    d0,Rc,Bc,sg,lie,Dl,gm,g0=x
    p=dict(Bs=0.2,Bc=Bc,c0=0.,b=3,d0=d0,Rc=Rc,sigc=sg,t=1-lie,Delta=Dl,gamma=gm,g0=g0)
    o=run5(p,R=R,seed=1+seedoff,noise_seed=2+seedoff,msg_seed=7+seedoff)
    key=(round(d0,6),seedoff)
    if key not in refc:
        refc[key]=run5(dict(Bs=0.2,c0=0.,b=0,d0=d0),R=R,seed=1+seedoff,noise_seed=2+seedoff,msg_seed=7+seedoff)['food'].mean(1)
    r=refc[key]
    c=0.004*3*o['qint']/600.
    fn=(o['food']-c).mean(1)
    return (fn.mean()-r.mean(), fn[1]-r[1])
import os
t0=time.time(); BUD=float(sys.argv[1]) if len(sys.argv)>1 else 1e9
ck=pickle.load(open('sobol_ck.pkl','rb')) if os.path.exists('sobol_ck.pkl') else {}
for ri,U in enumerate(rows):
    if ri in ck: continue
    X=scale(U); part=[V(x) for x in X]
    ck[ri]=np.array(part); pickle.dump(ck,open('sobol_ck.pkl','wb')); print(ri,round(time.time()-t0),flush=True)
    if time.time()-t0>BUD and len(ck)<len(rows): print('PAUSE',flush=True); sys.exit(0)
Y=np.array([ck[i] for i in range(len(rows))])
# noise check: replicate baseline point over seeds
xb=scale(np.array([[0.36,0.3,0.5,0.3,0.5,0.5,0.5,0.5]]))[0]
rep=np.array([V(xb,s) for s in range(6)])
pickle.dump(dict(Y=Y,rep=rep,names=names,A=A,B=B),open('sobol.pkl','wb'))
print('SOBOL DONE',rep.std(0),flush=True)
