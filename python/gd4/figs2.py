import numpy as np, pickle, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
r4=pickle.load(open('res4.pkl','rb')); r5=pickle.load(open('res5.pkl','rb')); T=600.; c0=0.004
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.25,'legend.frameon':False,'figure.dpi':150})
CA,CB,CG='#c0392b','#1f6fb4','#2e8b57'
def pr(x,y): z=np.asarray(x,float)-np.asarray(y,float); return z.mean(), z.std()/np.sqrt(z.size)
H=r4['H']['food'];P0=r4['P0']['food']
def FB(p,t):
    if p==0: return P0
    if p==1: return r4[('L',t,60.,True)]['food']
    return r5[('PT',p,t)]['food']
# f8
fig,ax=plt.subplots(1,2,figsize=(7.2,2.9)); ps=[0,.2,.4,.6,.8,1]
for t,c in zip((.8,.6,.4,.2),('#2e8b57','#7fb3d5','#a04000','#c0392b')):
    m=[pr(FB(p,t)[1],P0[1]) for p in ps]; ax[0].errorbar(ps,[a for a,_ in m],yerr=[2*b for _,b in m],color=c,marker='o',ms=3,capsize=2,label=f'$t={t}$')
ax[0].axhline(0,color='k',lw=.6); ax[0].set_xlabel('trọng số niềm tin $p$'); ax[0].set_ylabel('$E_B-E_{\\rm bỏ}$'); ax[0].legend(fontsize=7); ax[0].set_title('(A) tuyến tính theo $p$',fontsize=9)
tg=[1.,.8,.6,.4,.2,0.]
def Bv(p,t):
    if t==1.: return pr(H[1],P0[1])[0]*(p if p in (0,1) else p)
    return pr(FB(p,t)[1],P0[1])[0]
naive=[pr((H if t==1. else r4[('L',t,60.,True)]['food'])[1],P0[1])[0] for t in tg]
calib=[]
for t in tg:
    if t in (1.,0.): calib.append(naive[tg.index(t)] if t==1. else 0.)
    else: calib.append(pr(r5[('PT',t,t)]['food'][1],P0[1])[0])
best=[max(0,v) for v in naive]
ax[1].plot(tg,naive,color=CB,marker='o',ms=3,label='ngây thơ $p=1$'); ax[1].plot(tg,calib,color='#a04000',marker='s',ms=3,label='hiệu chuẩn $p=t$'); ax[1].plot(tg,best,color=CG,marker='^',ms=3,ls='--',label='biết $t$: $p\\in\\{0,1\\}$')
ax[1].axhline(0,color='k',lw=.6); ax[1].set_xlabel('tỉ lệ nói thật $t$'); ax[1].set_ylabel('$E_B-E_{\\rm bỏ}$'); ax[1].legend(fontsize=7); ax[1].set_title('(B) người nhận biết $t$',fontsize=9)
plt.tight_layout(); plt.savefig('fig/f8.pdf'); plt.close()
# f9
fig,ax=plt.subplots(1,2,figsize=(7.2,2.9))
ss=[0,.25,.5,.75,1.]; dA=[];dB=[]
for s in ss:
    if s==0: h=r4['Hsep']['food']; l=r4[('L',0.,60.,False)]['food']
    elif s==1: h=H; l=r4[('L',0.,60.,True)]['food']
    else: h=r5[('CP',s,1.)]['food']; l=r5[('CP',s,0.)]['food']
    dA.append(pr(l[0],h[0])); dB.append(pr(l[1],h[1]))
ax[0].errorbar(ss,[a for a,_ in dA],yerr=[2*b for _,b in dA],color=CA,marker='o',ms=3,capsize=2,label='$\\Delta E_A$ (người nói dối)')
ax[0].errorbar(ss,[a for a,_ in dB],yerr=[2*b for _,b in dB],color=CB,marker='o',ms=3,capsize=2,label='$\\Delta E_B$ (người bị lừa)')
ax[0].axhline(0,color='k',lw=.6); ax[0].set_xlabel('phần kho dùng chung $s$'); ax[0].set_ylabel('$\\Delta E$ ($t=0$ so $t=1$)'); ax[0].legend(fontsize=7,loc='center left'); ax[0].set_title('(A) cạnh tranh liên tục, $\\Delta=60$',fontsize=9)
names=['khai thác\n(cơ sở)','can thiệp\n$\\lambda=1$','giữ patch\n$\\kappa=0{,}3$']; cs=[0.0093,0.0359,0.0669]; ts=[0.404,0.475,0.662]
ax[1].bar(range(3),[c/c0 for c in cs],color=['#9db7d1',CB,'#0b3d6b']); ax[1].set_xticks(range(3)); ax[1].set_xticklabels(names,fontsize=7); ax[1].set_ylabel('$c_l^*/c_0$')
for i,(c,t) in enumerate(zip(cs,ts)): ax[1].text(i,c/c0+0.4,f'$t^*={t}$',ha='center',fontsize=7)
ax[1].set_title('(B) ngưỡng chi phí theo kiểu cạnh tranh',fontsize=9)
plt.tight_layout(); plt.savefig('fig/f9.pdf'); plt.close()
# f10
fig,ax=plt.subplots(1,2,figsize=(7.2,2.9))
lab=['dịch về phía\nngười nhận','hướng\nngẫu nhiên','dịch ra xa\nngười nhận']
for t,c,sh in ((.5,'#7fb3d5',-.18),(0.,CB,.18)):
    v=[pr((r4[('L',0.,60.,True)]['food'] if t==0. else r4[('B',3,.5)]['food'])[1],H[1]), pr(r5[('LY','rand',t)]['food'][1],H[1]), pr(r5[('LY','away',t)]['food'][1],H[1])]
    ax[0].bar(np.arange(3)+sh,[a for a,_ in v],.34,yerr=[2*b for _,b in v],color=c,capsize=2,label=f'$t={t}$')
ax[0].set_xticks(range(3)); ax[0].set_xticklabels(lab,fontsize=7); ax[0].set_ylabel('$\\Delta E_B$ so với nói thật'); ax[0].legend(fontsize=7,loc='lower right'); ax[0].set_title('(A) hướng của lời nói dối ($b=3$, $\\Delta=60$)',fontsize=8)
bs=[1,2,3,4]
for lt,c,mk in (('withhold','#7fb3d5','o'),('flip',CB,'s')):
    r=[]
    for b in bs:
        Hb=r4[('Hb',b)]['food']; val=pr(Hb[1],r4['E0']['food'][1])[0]; harm=pr(r5[('LC',lt,b,0.)]['food'][1],Hb[1])[0]; r.append(-harm/val)
    ax[1].plot(bs,r,color=c,marker=mk,ms=4,label={'withhold':'giữ lại','flip':'đảo'}[lt])
ax[1].set_xticks(bs); ax[1].set_xlabel('độ chi tiết $b$'); ax[1].set_ylabel('thiệt hại / giá trị tin thật ($t=0$)'); ax[1].legend(fontsize=7); ax[1].set_title('(B) cùng kiểu nói dối cho mọi $b$',fontsize=9); ax[1].set_ylim(1,2.4)
plt.tight_layout(); plt.savefig('fig/f10.pdf'); plt.close()
# f11
fig,ax=plt.subplots(figsize=(7.2,2.8))
vars_=[('$\\nu=0{,}05$',lambda t:r5[('LV',.05,t)]['food'],lambda t:None),('$\\nu=0{,}2$ (cơ sở)',lambda t:r4[('LEARN',t,60.)]['food'],None),('$\\nu=0{,}8$',lambda t:r5[('LV',.8,t)]['food'],None),('có quên',lambda t:r5[('LVf',t)]['food'],None)]
fx={.5:r4[('B',3,.5)]['food'],.2:r4[('L',.2,60.,True)]['food']}
xl=[];data={.5:[],.2:[]}
for nm,fn,_ in vars_:
    xl.append(nm)
    for t in (.5,.2): data[t].append(pr(fn(t)[1],fx[t][1]))
xl+=['$\\delta_{\\rm chk}=5$','$\\delta_{\\rm chk}=20$','3 ngày']
data[.5]+= [(np.nan,0),(np.nan,0),pr(r5[('LT','L',.5)]['food'][1],r5[('LT','F',.5)]['food'][1])]
data[.2]+= [pr(r5[('LVd',5.)]['food'][1],fx[.2][1]),pr(r5[('LVd',20.)]['food'][1],fx[.2][1]),pr(r5[('LT','L',.2)]['food'][1],r5[('LT','F',.2)]['food'][1])]
x=np.arange(len(xl))
for t,c,sh in ((.5,'#7fb3d5',-.18),(.2,CB,.18)):
    ax.bar(x+sh,[a for a,_ in data[t]],.34,yerr=[2*b for _,b in data[t]],color=c,capsize=2,label=f'$t={t}$')
ax.axhline(0,color='k',lw=.6); ax.set_xticks(x); ax.set_xticklabels(xl,fontsize=7); ax.set_ylabel('$E_B$(học) $-$ $E_B$(cố định $p=1$)'); ax.legend(fontsize=7); ax.set_title('Độ bền của kết luận H3 theo biến thể quy tắc học',fontsize=9)
plt.tight_layout(); plt.savefig('fig/f11.pdf'); plt.close()
# f12
pts=pickle.load(open('ruler.pkl','rb'))
fig,ax=plt.subplots(1,2,figsize=(7.2,2.8))
nz=sorted([(p[1],p[3]) for p in pts if p[0]=='noise']); lz=sorted([(p[1],p[3]) for p in pts if p[0]=='lie'])
ax[0].plot([a for a,_ in nz],[b for _,b in nz],color=CB,ls='--',marker='o',ms=3,label='nhiễu (RMS $=\\sigma_c$)')
ax[0].plot([a for a,_ in lz],[b for _,b in lz],color=CB,marker='o',ms=3,label='lừa dối $t=0$ (RMS $=\\Delta$)')
lt_=[(p[1],p[3]) for p in pts if p[0]=='lie_t']; ax[0].scatter([a for a,_ in lt_],[b for _,b in lt_],color='#a04000',s=14,zorder=3,label='lừa dối $t>0$: RMS $=\\Delta\\sqrt{1-t}$')
ax[0].set_xlabel('sai số RMS của thông điệp (m)'); ax[0].set_ylabel('$\\Delta E_B$ so với nói thật'); ax[0].legend(fontsize=7); ax[0].set_title('(A) thước RMS',fontsize=9)
nz2=sorted([(.8*a,b) for a,b in nz]); 
ax[1].plot([a for a,_ in nz2],[b for _,b in nz2],color=CB,ls='--',marker='o',ms=3,label='nhiễu ($|\\eta|$ TB $=0{,}8\\sigma_c$)'); ax[1].plot([a for a,_ in lz],[b for _,b in lz],color=CB,marker='o',ms=3,label='lừa dối ($|{\\rm lỗi}|=\\Delta$)')
ax[1].set_xlabel('|sai số| trung bình (m)'); ax[1].legend(fontsize=7); ax[1].set_title('(B) thước $|$sai số$|$ trung bình',fontsize=9)
plt.tight_layout(); plt.savefig('fig/f12.pdf'); plt.close(); print('ok')
