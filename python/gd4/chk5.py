import numpy as np, pickle
from sim_gd3 import run3
from sim_gd4 import run4
import sim_gd4_v1 as v1
B=dict(Bs=0.,Bc=8.,c0=0.)
for kw in (dict(sigc=30.),dict(Bs=0.2,d0=40.),dict(b=2),dict(b=4,sigc=20.)):
    a=run3({**B,**kw},R=300); b=run4({**B,**kw,'t':1.},R=300)
    print('t=1 vs run3',kw,np.abs(a['EgA']-b['food'][0]).max())
for kw in (dict(t=0.),dict(t=0.4,b=2),dict(t=0.5,b=1),dict(t=0.3,learn=True),dict(t=0.5,shared=False)):
    a=v1.run4({**B,**kw},R=300); b=run4({**B,**kw},R=300)
    print('regress v1',kw,np.abs(a['food']-b['food']).max())
res=pickle.load(open('res4.pkl','rb'))
r=run4({**B,'t':0.,'Delta':60.,'lie_seed':11},R=1000,seed=1,noise_seed=2,msg_seed=7)
print('vs stored seed0', np.abs(r['food']-res[('L',0.,60.,True)]['food'][:,:1000]).max())
for kw in (dict(lietype='away'),dict(lietype='rand'),dict(lietype='withhold',b=3),dict(lietype='flip',b=3),dict(lam=1.),dict(kappa=0.3),dict(shared=False,couple=0.5)):
    o=run4({**B,'t':0.,**kw},R=300); print(kw,o['food'].mean(1).round(4))
# separate worlds with couple=1 vs shared
a=run4({**B,'t':0.5},R=500); b=run4({**B,'t':0.5,'shared':False,'couple':1.},R=500)
print('couple=1 vs shared (mean A,B)',a['food'].mean(1).round(4),b['food'].mean(1).round(4))
