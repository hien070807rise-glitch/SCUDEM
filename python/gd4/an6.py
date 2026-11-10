import numpy as np, pickle
r4=pickle.load(open('res4.pkl','rb')); r5=pickle.load(open('res5.pkl','rb')); T=600.; c0=0.004
def net(r,i): return r['food'][i]-c0*3*r['qint'][i]/T
def pr(x,y): z=np.asarray(x,float)-np.asarray(y,float); return z.mean(), z.std()/np.sqrt(z.size)
def coefs(Hc,Lc,Pc):
    a=net(Lc,0)-net(Hc,0); bT=net(Hc,1)-net(Pc,1); bL=net(Lc,1)-net(Pc,1); L=Lc['lint'][0]/T
    return a,bT,bL,L
cfgs={'co so (khai thac)':(r4['H'],r4[('L',0.,60.,True)],r4['P0']),
      'can thiep lam=1':(r5[('CL',1.)],r5[('CL',0.)],r5[('CL','P0')]),
      'giu patch kappa=.3':(r5[('CK',1.)],r5[('CK',0.)],r5[('CK','P0')])}
rng=np.random.default_rng(1); out={}
for name,(Hc,Lc,Pc) in cfgs.items():
    a,bT,bL,L=coefs(Hc,Lc,Pc); n=len(a)
    ts=-bL.mean()/(bT.mean()-bL.mean()); cs=a.mean()/L.mean()
    bs=[];bt=[]
    for _ in range(400):
        i=rng.integers(0,n,n); al,t1,t2,l=a[i].mean(),bT[i].mean(),bL[i].mean(),L[i].mean()
        bs.append(al/l); bt.append(-t2/(t1-t2))
    print(f"{name:20s} alpha={a.mean():+.4f} bT={bT.mean():+.4f} bL={bL.mean():+.4f} L={L.mean():.3f} t*={ts:.3f} CI{np.percentile(bt,[2.5,97.5]).round(3)} c_l*={cs:.4f} CI{np.percentile(bs,[2.5,97.5]).round(4)} c*/c0={cs/c0:.1f}")
    out[name]=(a.mean(),bT.mean(),bL.mean(),L.mean(),ts,cs)
# best response of sender: net gain G(t)-c_l L(t), base config
H=r4['H']
print('best response t* vs c_l (base):')
ts_=[1.,.8,.6,.4,.2,0.]
G=[];Lt=[]
for t in ts_:
    k='H' if t==1. else ('L',t,60.,True)
    G.append(pr(net(r4[k],0),net(H,0))[0]); Lt.append((r4[k]['lint'][0]/T).mean())
for cl in (0.,0.002,0.005,0.01):
    v=[g-cl*l for g,l in zip(G,Lt)]; print(' c_l',cl,'best t',ts_[int(np.argmax(v))],np.round(v,4))
# stochastic replicator (finite population drift) quick: Euler-Maruyama with noise sigma/sqrt(N)
a,bT,bL,L,ts,cs=out['co so (khai thac)']
def sim(cl,N=200,steps=40000,dtau=0.5,seed=0):
    r=np.random.default_rng(seed); t,y=ts,cl*L/a; tr=[]
    for _ in range(steps):
        dt=t*(1-t)*(cl*L-y*a); dy=y*(1-y)*(t*bT+(1-t)*bL)
        t+=dtau*dt+np.sqrt(max(t*(1-t),0)/N*dtau)*r.normal()*0.02; y+=dtau*dy+np.sqrt(max(y*(1-y),0)/N*dtau)*r.normal()*0.02
        t=min(max(t,1e-6),1-1e-6); y=min(max(y,1e-6),1-1e-6)
    return t,y
print('drift end state (N=200) ',[tuple(np.round(sim(cl,seed=s),3)) for cl in (0.005,) for s in range(4)])
# significance helpers
f60=r4[('L',0.,60.,True)]['food'][1]; f100=r4[('L',0.,100.,True)]['food'][1]; f80=r4[('L',0.,80.,True)]['food'][1]
print('D100 vs D60 B',pr(f100,f60),'D80 vs D60',pr(f80,f60))
# ruler: harm B vs rms and vs mean|err|
H1=r4['H']['food'][1]
pts=[]
for s in (10,20,30,45,60,80,100): pts.append(('noise',s,0.8*s,pr(r4[('N',float(s))]['food'][1],H1)[0]))
for D in (10.,20.,30.,45.,60.,80.,100.): pts.append(('lie',D,D,pr(r4[('L',0.,D,True)]['food'][1],H1)[0]))
for D in (30.,60.,100.):
    for t in (.8,.6,.4,.2): pts.append(('lie_t',D*np.sqrt(1-t),D*(1-t),pr(r4[('L',t,D,True)]['food'][1],H1)[0]))
pickle.dump(pts,open('ruler.pkl','wb'))
# interpolate noise harm at rms=x and lie harm at rms x
nz=sorted([(p[1],p[3]) for p in pts if p[0]=='noise']); lz=sorted([(p[1],p[3]) for p in pts if p[0]=='lie'])
for x in (20,30,45,60):
    hn=np.interp(x,[a for a,b in nz],[b for a,b in nz]); hl=np.interp(x,[a for a,b in lz],[b for a,b in lz]); print('rms',x,'noise',round(hn,4),'lie',round(hl,4),'ratio',round(hl/hn,2))
# ruler mean|err|: noise at sigma s has mean|eta|=0.8 s ; lie |err|=D
nz2=sorted([(0.8*a,b) for a,b in nz])
for x in (24,36,48):
    hn=np.interp(x,[a for a,b in nz2],[b for a,b in nz2]); hl=np.interp(x,[a for a,b in lz],[b for a,b in lz]); print('mean|err|',x,'noise',round(hn,4),'lie',round(hl,4),'ratio',round(hl/hn,2))
