import sys
import numpy as np, pickle, itertools
from exp_gd5 import *
def pair(o,b,c0=0.004,idx=None):
    n=o['food']-c0*b*o['qint']/600.
    return n if idx is None else n[idx]
def cellstat(p,ns=3,R=1000,off=0,b=None):
    q={**BASEP,**p}; b=q.get('b',3) if b is None else b
    o=get(p,ns,R,off=off); r=ref(p,ns,R,off=off)
    n=o['food']-0.004*b*o['qint']/600.
    return n, r['food']
def V(p,ns=3,R=1000):
    n,r=cellstat(p,ns,R)
    return n.mean(0)-r.mean(0)   # per replicate pair-mean V
def VB(p,ns=3,R=1000):
    n,r=cellstat(p,ns,R); return n[1]-r[1]
def se(x): return x.std(ddof=1)/np.sqrt(len(x))
if __name__=='__main__':
    pass
    if 'tornado' in sys.argv:
        names=['d0lo','d0hi','Rclo','Rchi','b1','b4','Bc4','Bc16','sig60','sig100','t0','t05','gam0','gam05','g002','g05','Bg0','Bg300','s5','s20','K3','K11','D25','D100','sg5','sg40','kap','lam','trust0','Bs1','Bs005']
        spec={'d0lo':dict(d0=20.),'d0hi':dict(d0=140.),'Rclo':dict(Rc=25.),'Rchi':dict(Rc=100.),'b1':dict(b=1),'b4':dict(b=4),'Bc4':dict(Bc=4.),'Bc16':dict(Bc=16.),'sig60':dict(sigc=60.),'sig100':dict(sigc=100.),'t0':dict(t=0.,Delta=60.),'t05':dict(t=0.5,Delta=60.),'gam0':dict(gate='off'),'gam05':dict(gamma=0.5),'g002':dict(g0=0.02),'g05':dict(g0=0.5),'Bg0':dict(Bg=0.),'Bg300':dict(Bg=300.),'s5':dict(s=5.),'s20':dict(s=20.),'K3':dict(K=3,Gm=200.),'K11':dict(K=11,Gm=600./11),'D25':dict(D=25.),'D100':dict(D=100.),'sg5':dict(sig=5.),'sg40':dict(sig=40.),'kap':dict(kappa=0.3),'lam':dict(lam=1.),'trust0':dict(t=0.,Delta=60.,trust=0.),'Bs1':dict(Bs=1.),'Bs005':dict(Bs=0.05)}
        n0,r0=cellstat({})
        Mb=n0.mean(0); Vb=Mb-r0.mean(0); EBb=n0[1]
        print('base M %.4f V %.4f EB %.4f ref %.4f'%(Mb.mean(),Vb.mean(),EBb.mean(),r0.mean()))
        out={}
        for nm in names:
            p=spec[nm]; n,r=cellstat(p)
            M=n.mean(0); v=M-r.mean(0)
            out[nm]=dict(dM=(M-Mb).mean(),seM=se(M-Mb),Mabs=M.mean(),dV=(v-Vb).mean(),seV=se(v-Vb),dEB=(n[1]-EBb).mean(),seEB=se(n[1]-EBb),Eref=r.mean())
            o=out[nm]; print('%-7s M %.4f dM %+.4f±%.4f | dV %+.4f±%.4f | dEB %+.4f±%.4f | ref %.4f'%(nm,o['Mabs'],o['dM'],o['seM'],o['dV'],o['seV'],o['dEB'],o['seEB'],o['Eref']))
        pickle.dump(dict(out=out,Mb=Mb.mean(),Vb=Vb.mean(),EBb=EBb.mean(),rb=r0.mean()),open('tornado.pkl','wb'))

def Vs(p,ns=3,R=1000):
    n,r=cellstat(p,ns,R); return n.mean(0)-r.mean(0), n[1]-r[1]
LEV={'d0':dict(d0=100.),'sig':dict(sigc=60.),'t':dict(t=0.,Delta=60.),'Rc':dict(Rc=25.),'b':dict(b=1),'Bg':dict(Bg=300.),'kap':dict(kappa=0.3),'lam':dict(lam=1.),'gam':dict(gamma=0.5),'g0':dict(g0=0.5)}
PAIRS=[('d0','sig'),('d0','t'),('sig','t'),('b','sig'),('Rc','d0'),('Bg','t'),('kap','t'),('lam','t'),('gam','sig'),('g0','t'),('gam','t'),('Rc','sig')]
def inter():
    v0,b0=Vs({}); out=[]
    for a,b_ in PAIRS:
        va,ba=Vs(LEV[a]); vb,bb=Vs(LEV[b_]); vab,bab=Vs({**LEV[a],**LEV[b_]})
        I=vab-va-vb+v0; IB=bab-ba-bb+b0
        da=(va-v0).mean(); db=(vb-v0).mean(); dab=(vab-v0).mean()
        out.append((a,b_,da,db,dab,I.mean(),se(I),(ba-b0).mean(),(bb-b0).mean(),(bab-b0).mean(),IB.mean(),se(IB)))
        print('%-4s x %-4s | V: da %+.4f db %+.4f dab %+.4f  I %+.4f±%.4f (%.1f se) | EB: da %+.4f db %+.4f dab %+.4f I %+.4f±%.4f (%.1f se)'%(a,b_,da,db,dab,I.mean(),se(I),I.mean()/se(I),(ba-b0).mean(),(bb-b0).mean(),(bab-b0).mean(),IB.mean(),se(IB),IB.mean()/se(IB)))
    pickle.dump(out,open('inter.pkl','wb'))
D0G=(0.,20.,40.,60.,80.,100.,140.,200.)
def dstar_fit(Mrep,B=300,seed=0):
    g=np.array(D0G); R=Mrep.shape[1]
    def est(M):
        c=np.polyfit(g/100,M,4); xs=np.linspace(0,2,401); y=np.polyval(c,xs); i=y.argmax(); return xs[i]*100
    d=est(Mrep.mean(1)); rng=np.random.default_rng(seed)
    bs=[est(Mrep[:,rng.integers(0,R,R)].mean(1)) for _ in range(B)]
    return d,np.percentile(bs,[2.5,97.5])
def dstar_all():
    vars_={'base':{},'Bc2':dict(Bc=2.),'Bc4':dict(Bc=4.),'Bc16':dict(Bc=16.),'Bs005':dict(Bs=0.05),'Bs1':dict(Bs=1.),'lam1':dict(lam=1.),'kap03':dict(kappa=0.3),'Bg25':dict(Bg=25.),'Bg300':dict(Bg=300.),'g05':dict(g0=0.5),'gam0':dict(gate='off')}
    out={}
    for nm,v in vars_.items():
        Ms=[];Vv=[]
        for d0 in D0G:
            n,r=cellstat({**v,'d0':d0},2,1000); Ms.append(n.mean(0)); Vv.append(n.mean(0)-r.mean(0))
        Ms=np.array(Ms); Vv=np.array(Vv)
        d,ci=dstar_fit(Ms)
        gi=int(np.argmin(abs(np.array(D0G)-d))); vs=Vv[gi].mean(); vse=se(Vv[gi])
        pk=int(np.argmax(Ms.mean(1)))
        interior=(10<d<190); sig=vs>4*vse
        out[nm]=dict(d=d,ci=ci,Vstar=vs,Vse=vse,Mgrid=Ms.mean(1),Vgrid=Vv.mean(1),interior=interior,sig=sig,gridargmax=D0G[pk])
        print('%-6s d*=%6.1f CI[%5.1f,%5.1f] gridmax=%5.0f  Vstar %+.4f±%.4f  interior=%s sig=%s | M(d0)='%(nm,d,ci[0],ci[1],D0G[pk],vs,vse,interior,sig),np.round(Ms.mean(1),4))
    pickle.dump(out,open('dstar.pkl','wb'))
def kernels():
    out={}
    for ker in ('gauss','exp','power'):
        Vk=[];Mk=[]
        for d0 in D0G:
            n,r=cellstat(dict(kernel=ker,d0=d0),2,1000); Vk.append((n.mean(0)-r.mean(0)).mean()); Mk.append(n.mean())
        def fr(p): 
            return cellstat({**p,'kernel':ker,'Bs':0.},3,1000)
        nH,rH=fr({}); nL,_=fr(dict(t=0.,Delta=60.)); nL100,_=fr(dict(t=0.,Delta=100.))
        res_={}
        res_['VH']=(nH.mean(0)-rH.mean(0)).mean()
        res_['harmL']=(nL[1]-nH[1]).mean(); res_['harmL_se']=se(nL[1]-nH[1]); res_['harmL100']=(nL100[1]-nH[1]).mean()
        res_['gainA']=(nL[0]-nH[0]).mean()
        for s in (30,60,100):
            nN,_=fr(dict(sigc=float(s),noise_mask=(1.,0.))); res_['harmN%d'%s]=(nN[1]-nH[1]).mean(); res_['VN%d'%s]=(nN.mean(0)-rH.mean(0)).mean()
        res_['ratio60']=res_['harmL']/res_['harmN60']
        res_['Vgrid']=np.array(Vk); res_['Mgrid']=np.array(Mk)
        out[ker]=res_
        print(ker,'V(d0)',np.round(Vk,4),'| VH %.4f harmL %.4f harmN60 %.4f ratio %.2f gainA %+.4f VN30/60/100 %.4f %.4f %.4f'%(res_['VH'],res_['harmL'],res_['harmN60'],res_['ratio60'],res_['gainA'],res_['VN30'],res_['VN60'],res_['VN100']))
    pickle.dump(out,open('kernels.pkl','wb'))
if __name__=='__main__' and 'inter' in sys.argv: inter()
if __name__=='__main__' and 'dstar' in sys.argv: dstar_all()
if __name__=='__main__' and 'kernel' in sys.argv: kernels()
