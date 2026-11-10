import numpy as np, pickle
r4=pickle.load(open('res4.pkl','rb')); r5=pickle.load(open('res5.pkl','rb')); T=600.
def pr(x,y): z=np.asarray(x,float)-np.asarray(y,float); return z.mean(), z.std()/np.sqrt(z.size)
def fm(a): return f"{a[0]:+.4f}±{a[1]:.4f}"
H=r4['H']['food'];E0=r4['E0']['food'];P0=r4['P0']['food']
print('== PT: E_B - E_bo (B) and dA vs H ; rows t, cols p=0,.2,.4,.6,.8,1')
for t in (.8,.6,.4,.2):
    row=[]
    for p in (0,.2,.4,.6,.8,1):
        if p==0: f=P0
        elif p==1: f=r4[('L',t,60.,True)]['food']
        else: f=r5[('PT',p,t)]['food']
        row.append(pr(f[1],P0[1])[0])
    print(t,' '.join(f"{v:+.4f}" for v in row))
print('tstar approx 0.40; p=t diag:',[round(pr(r5[('PT',t,t)]['food'][1],P0[1])[0],4) for t in (.8,.6,.4,.2)])
print('== OOS vs in-sample')
def stats(E0_,H_,P0_,L0_,L4_,N_,Hs,Ls):
    return dict(HmE0B=pr(H_[1],E0_[1]),Hbo=pr(H_[1],P0_[1]),L0dB=pr(L0_[1],H_[1]),L0dA=pr(L0_[0],H_[0]),L4dB=pr(L4_[1],H_[1]),L4dA=pr(L4_[0],H_[0]),NdB=pr(N_[1],H_[1]),sepdA=pr(Ls[0],Hs[0]))
ins=stats(E0,H,P0,r4[('L',0.,60.,True)]['food'],r4[('L',.4,60.,True)]['food'],r4[('N',60.)]['food'],r4['Hsep']['food'],r4[('L',0.,60.,False)]['food'])
oo=stats(*[r5[('OOS',k)]['food'] for k in ('E0','H','P0','L0','L4','N60','Hsep','L0sep')])
for k in ins: print(f"{k:6s} in={fm(ins[k])}  oos={fm(oo[k])}")
print('ratio lie/noise oos',oo['L0dB'][0]/oo['NdB'][0])
print('== CP: competition continuous, A gain (t=0 vs t=1), B harm')
for s in (0.,.25,.5,.75,1.):
    if s==0.: h=r4['Hsep']['food']; l=r4[('L',0.,60.,False)]['food']
    elif s==1.: h=H; l=r4[('L',0.,60.,True)]['food']
    else: h=r5[('CP',s,1.)]['food']; l=r5[('CP',s,0.)]['food']
    print(s,'dA',fm(pr(l[0],h[0])),'dB',fm(pr(l[1],h[1])))
for name,k1,k0 in (('lam1',('CL',1.),('CL',0.)),('kappa.3',('CK',1.),('CK',0.))):
    pass
h=r5[('CL',1.)]['food'];l=r5[('CL',0.)]['food'];print('lam=1 dA',fm(pr(l[0],h[0])),'dB',fm(pr(l[1],h[1])))
h=r5[('CK',1.)]['food'];l=r5[('CK',0.)]['food'];print('kappa=.3 dA',fm(pr(l[0],h[0])),'dB',fm(pr(l[1],h[1])))
print('== lie types (vs honest b=3 H)')
for lt in ('away','rand'):
    for t in (.5,0.):
        f=r5[('LY',lt,t)]['food']; print(lt,t,'dB',fm(pr(f[1],H[1])),'dA',fm(pr(f[0],H[0])))
f=r4[('L',0.,60.,True)]['food'];print('toward t0 dB',fm(pr(f[1],H[1])),'dA',fm(pr(f[0],H[0])))
print('== common lie across b (B harm vs same-b honest, B-E0)')
for lt in ('withhold','flip'):
    for b in (1,2,3,4):
        Hb=r4[('Hb',b)]['food']; row=[]
        for t in (.5,0.):
            f=r5[('LC',lt,b,t)]['food']; row.append(f"t={t} dB {fm(pr(f[1],Hb[1]))} B-E0 {pr(f[1],E0[1])[0]:+.4f}")
        print(lt,'b',b,' | '.join(row),' honest B-E0',f"{pr(Hb[1],E0[1])[0]:+.4f}")
print('== learning variants dB vs H ; compare fixed')
for t in (.5,.2):
    fx=r4[('B',3,.5)]['food'] if t==.5 else r4[('L',.2,60.,True)]['food']
    base=r4[('LEARN',t,60.)]['food']
    print('t',t,'fixed',fm(pr(fx[1],H[1])),'learn(nu.2)',fm(pr(base[1],H[1])))
    for nu in (.05,.8):
        f=r5[('LV',nu,t)]['food']; print('  nu',nu,fm(pr(f[1],H[1])),'learn-fixed',fm(pr(f[1],fx[1])),'pend',r5[('LV',nu,t)]['pend'].mean().round(3))
    f=r5[('LVf',t)]['food']; print('  forget',fm(pr(f[1],H[1])),'learn-fixed',fm(pr(f[1],fx[1])),'pend',r5[('LVf',t)]['pend'].mean().round(3))
for dc in (5.,20.):
    f=r5[('LVd',dc)]['food']; fx=r4[('L',.2,60.,True)]['food']; print('dchk',dc,fm(pr(f[1],H[1])),'learn-fixed',fm(pr(f[1],fx[1])),'pend',r5[('LVd',dc)]['pend'].mean().round(3))
print('== T=1800 (3 days): harm of B vs honest T=1800, learn vs fixed')
Hl=r5[('LT','H')]['food']
for t in (.5,.2):
    F_=r5[('LT','F',t)]['food'];L_=r5[('LT','L',t)]['food']
    print(t,'fixed',fm(pr(F_[1],Hl[1])),'learn',fm(pr(L_[1],Hl[1])),'learn-fixed',fm(pr(L_[1],F_[1])),'pend',r5[('LT','L',t)]['pend'].mean().round(3),'ptraj end',r5[('LT','L',t)]['ptraj'][[0,10,20,-1]].round(2))
