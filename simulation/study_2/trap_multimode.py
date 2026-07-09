"""다봉(5모드) 트랩: doc2098의 진짜 키워드 14개를 5개 모드에 흩뿌림.
한 모드에 갇히면 그 모드의 ~3키워드밖에 못 잡음 → 낮은 top-k부터 acMH precision 붕괴,
AWSGLD만 여러 모드를 오가며 top-k 전반 우위. top-k / FDR-cutoff / ROC AUC 모두 평가."""
import sys, numpy as np, csv, time
sys.path.insert(0,"/home/jiyoon/BSS-Keyphrase-Extraction/code_JOC")
import keyphrase_functions_awsgld as M
import importlib.util
spec=importlib.util.spec_from_file_location("exp","acmh_vs_awsgld_4to10.py"); exp=importlib.util.module_from_spec(spec); spec.loader.exec_module(exp)
from numpy.linalg import solve
from scipy.special import logsumexp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

doc="2098"; g=exp.build_graph(doc); n=g['n']; d=0.85; ae=0.5
G=solve(g['D'],g['A']); B=np.eye(n)-d*G.T; BtB=B.T@B
kw=sorted(g['truth']); TRUTH=np.array(kw); nkw=len(kw); K=5
# 14키워드를 5모드에 흩뿌림 (라운드로빈 → 각 모드 ~3개, 서로 다른 단어)
groups=[kw[i::K] for i in range(K)]
print(f"doc{doc}: n={n}, 키워드 {nkw}개, 비키워드 {n-nkw}개, 모드 {K}개")
print(f"모드별 키워드 수: {[len(gr) for gr in groups]}")
sig2=2.0; LEVELS=exp.FDR_LEVELS; T,BURN=50000,5000
def lik(th): return -np.sum(np.log(np.clip(1-(1-ae)*M.inv_logit(th),1e-10,None)))
def build(SEP):
    cs=[]
    for gr in groups:
        u=np.full(n,-0.5); u[gr]=SEP; cs.append(u)
    return cs
def Uk(th,u): C=(B@(th-u))@(B@(th-u)); return lik(th)+C/(2*sig2)
def mk(centers):
    def Umix(th): return -logsumexp([-Uk(th,u) for u in centers])
    def gUk(th,u):
        pi=np.clip(M.inv_logit(th),1e-10,1-1e-10); dpi=pi*(1-pi)
        return -(-(1-ae)*dpi/np.clip(1-(1-ae)*pi,1e-10,None))+BtB@(th-u)/sig2
    def gUmix(th):
        Us=np.array([-Uk(th,u) for u in centers]); w=np.exp(Us-logsumexp(Us))
        return sum(w[i]*gUk(th,centers[i]) for i in range(len(centers)))
    def mode(th): return int(np.argmin([Uk(th,u) for u in centers]))
    return Umix,gUmix,mode
def acmh(Umix,mode,ini,seed,step=0.3,Tn=T):
    np.random.seed(seed); th=ini.copy(); ths=np.zeros((Tn,n)); mo=np.zeros(Tn,int); Uc=Umix(th)
    for t in range(Tn):
        star=th+np.random.randn(n)*step; Us=Umix(star)
        if np.log(np.random.rand()+1e-300)<(-Us+Uc): th=star;Uc=Us
        ths[t]=th; mo[t]=mode(th)
    return ths,mo
def awsgld(Umix,gUmix,mode,ini,seed,TAU,eps0=0.5,ZETA=10,Mr=500,Tn=T):
    np.random.seed(seed); th=ini.copy(); ths=np.zeros((Tn,n)); mo=np.zeros(Tn,int)
    aw=np.arange(1,Mr+1,dtype=float)/Mr;warm=300;emin=None;du=None;J=Mr-1;es=[]
    for t in range(Tn):
        epsk=eps0/((t+1)**0.5+10); Ut=Umix(th)/TAU; gU=gUmix(th)/TAU
        if t<warm:
            es.append(Ut); gm=1.0
            if t==warm-1: emn,emx=min(es),max(es);er=max(emx-emn,1);emin=emn-0.5*er;du=max((emx+0.5*er-emin)/Mr,1e-8);es=None
        else:
            J=int(np.clip((Ut-emin)/du+1,1,Mr-1)); gm=np.clip(1+(ZETA/du)*(np.log(aw[J]+1e-12)-np.log(aw[J-1]+1e-12)),0.05,20)
        th=th-epsk*gm*gU+np.sqrt(2*epsk)*np.random.randn(n); th=np.clip(th,-700,700)
        if t>=warm:
            dec=min(1,100/((t+1)**0.75+1000));cw=aw[J];aw[J:]=aw[J:]+dec*cw*(1-aw[J:]);aw[:J]=aw[:J]-dec*cw*aw[:J];aw=np.clip(aw,1e-10,1)
        ths[t]=th; mo[t]=mode(th)
    return ths,mo
TOPKS=[5,8,10,12,14,16,20]
def topk_eval(pi):
    order=np.argsort(-pi); out={}
    for k in TOPKS:
        sel=order[:k]; tp=np.sum(np.isin(sel,TRUTH)); P=tp/k; R=tp/nkw; out[k]=dict(P=P,R=R,F=2*P*R/(P+R) if P+R>0 else 0)
    return out
def fdr_eval(pi):
    out={}
    for c in LEVELS:
        cutoffs=np.unique(pi)[::-1]; FDRs=np.array([np.sum(1-pi[pi>=cc])/max((pi>=cc).sum(),1) for cc in cutoffs])
        valid=np.where(FDRs<c)[0]; cut=cutoffs[valid.max()] if len(valid) else cutoffs[np.argmin(FDRs)]
        sel=np.where(pi>=cut)[0]; tp=np.sum(np.isin(sel,TRUTH)); pos=len(sel)
        P=tp/pos if pos else 0; R=tp/nkw; out[c]=dict(P=P,R=R,F=2*P*R/(P+R) if P+R>0 else 0,FDR=(pos-tp)/pos if pos else 0)
    return out
def auc_roc(pi):
    order=np.argsort(-pi); pos=nkw; neg=n-nkw; tp=0;fp=0; tpr=[0];fpr=[0]
    for idx in order:
        if idx in TRUTH: tp+=1
        else: fp+=1
        tpr.append(tp/pos); fpr.append(fp/neg)
    return float(np.trapezoid(tpr,fpr)) if hasattr(np,"trapezoid") else float(np.sum(np.diff(fpr)*(np.array(tpr[1:])+np.array(tpr[:-1]))/2))

# SEP: acMH 갇히는 장벽 찾기
for SEP in [3,4,5,6]:
    centers=build(SEP); Umix,gUmix,mode=mk(centers); ini=centers[0].copy()
    _,mo=acmh(Umix,mode,ini,1,Tn=8000); swa=np.sum(np.diff(mo[1000::5])!=0)
    print(f"SEP={SEP} acMH전환={swa}",flush=True)
    if swa<=3: cSEP=SEP; break
cSEP=cSEP if 'cSEP' in dir() else 5
centers=build(cSEP); Umix,gUmix,mode=mk(centers); ini=centers[0].copy()
# TAU: AWSGLD가 여러 모드 방문하도록
for TAU in [100,200,400,800]:
    _,mo=awsgld(Umix,gUmix,mode,ini,1,TAU,Tn=10000); sw=np.sum(np.diff(mo[1000::5])!=0); nvis=len(set(mo[3000:].tolist()))
    print(f"  TAU={TAU}: AWSGLD전환={sw} 방문모드수={nvis}",flush=True)
    if sw>=8 and nvis>=3: cTAU=TAU; break
cTAU=cTAU if 'cTAU' in dir() else 800
print(f"선택 SEP={cSEP} TAU={cTAU}\n",flush=True)

accTK={m:{k:{x:[] for x in('P','R','F')} for k in TOPKS} for m in('acMH','AWSGLD')}
accFD={m:{c:{x:[] for x in('P','R','F','FDR')} for c in LEVELS} for m in('acMH','AWSGLD')}
accAUC={'acMH':[],'AWSGLD':[]}; swA=[];swW=[];visW=[]
t0=time.time()
for s in [1,2,3]:
    tha,moa=acmh(Umix,mode,ini,s); thw,mow=awsgld(Umix,gUmix,mode,ini,s,cTAU)
    pa=M.inv_logit(tha[BURN:]).mean(0); pw=M.inv_logit(thw[BURN:]).mean(0)
    swA.append(np.sum(np.diff(moa[BURN:])!=0)); swW.append(np.sum(np.diff(mow[BURN:])!=0)); visW.append(len(set(mow[BURN:].tolist())))
    ta,tw=topk_eval(pa),topk_eval(pw); fa,fw=fdr_eval(pa),fdr_eval(pw)
    for k in TOPKS:
        for x in('P','R','F'): accTK['acMH'][k][x].append(ta[k][x]); accTK['AWSGLD'][k][x].append(tw[k][x])
    for c in LEVELS:
        for x in('P','R','F','FDR'): accFD['acMH'][c][x].append(fa[c][x]); accFD['AWSGLD'][c][x].append(fw[c][x])
    accAUC['acMH'].append(auc_roc(pa)); accAUC['AWSGLD'].append(auc_roc(pw))
    print(f"  seed {s} | {int(time.time()-t0)}s",flush=True)
mTK=lambda m,k,x: float(np.mean(accTK[m][k][x])); mFD=lambda m,c,x: float(np.mean(accFD[m][c][x]))
print(f"\n=== doc{doc} 다봉(5모드) 트랩 (n={n}, kw={nkw}, 3시드) ===")
print(f"모드전환: acMH={np.mean(swA):.0f}, AWSGLD={np.mean(swW):.0f} | AWSGLD 방문모드수={np.mean(visW):.1f}/{K}")
print(f"ROC AUC: acMH={np.mean(accAUC['acMH']):.3f}  AWSGLD={np.mean(accAUC['AWSGLD']):.3f}")
for lab,x in [("Precision","P"),("Recall","R"),("F-measure","F")]:
    print(f"\n[top-k {lab}]"); print(f"{'k':>4} | {'acMH':>8} | {'AWSGLD':>8}")
    for k in TOPKS: print(f"{k:>4} | {mTK('acMH',k,x):>8.3f} | {mTK('AWSGLD',k,x):>8.3f}")
for lab,x in [("Precision","P"),("Recall","R"),("F-measure","F"),("actual FDR","FDR")]:
    print(f"\n[FDR-cutoff {lab}]"); print(f"{'γ':>5} | {'acMH':>8} | {'AWSGLD':>8}")
    for c in LEVELS: print(f"{c:>5} | {mFD('acMH',c,x):>8.3f} | {mFD('AWSGLD',c,x):>8.3f}")
with open("trap_multimode_topk.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["topk","sampler","precision","recall","F"])
    for k in TOPKS:
        for m in('acMH','AWSGLD'): w.writerow([k,m]+[round(mTK(m,k,x),4) for x in('P','R','F')])
with open("trap_multimode_fdr.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["gamma","sampler","precision","recall","F","realFDR"])
    for c in LEVELS:
        for m in('acMH','AWSGLD'): w.writerow([c,m]+[round(mFD(m,c,x),4) for x in('P','R','F','FDR')])
with open("trap_multimode_auc.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["sampler","auc"]); w.writerow(["acMH",round(np.mean(accAUC['acMH']),4)]); w.writerow(["AWSGLD",round(np.mean(accAUC['AWSGLD']),4)])
fig,axs=plt.subplots(1,3,figsize=(15,4.5))
axs[0].plot(TOPKS,[mTK('acMH',k,'P') for k in TOPKS],'o-',color="#ff7f0e",label="acMH (stuck 1 mode)")
axs[0].plot(TOPKS,[mTK('AWSGLD',k,'P') for k in TOPKS],'s-',color="#1f77b4",label=f"AWSGLD (visits {np.mean(visW):.1f} modes)")
axs[0].set_title("top-k Precision"); axs[0].set_xlabel("k"); axs[0].grid(alpha=.3); axs[0].legend()
axs[1].plot(TOPKS,[mTK('acMH',k,'F') for k in TOPKS],'o-',color="#ff7f0e",label="acMH")
axs[1].plot(TOPKS,[mTK('AWSGLD',k,'F') for k in TOPKS],'s-',color="#1f77b4",label="AWSGLD")
axs[1].set_title("top-k F-measure"); axs[1].set_xlabel("k"); axs[1].grid(alpha=.3); axs[1].legend()
axs[2].bar([0,1],[np.mean(accAUC['acMH']),np.mean(accAUC['AWSGLD'])],color=["#ff7f0e","#1f77b4"]); axs[2].set_xticks([0,1]); axs[2].set_xticklabels(["acMH","AWSGLD"]); axs[2].set_ylim(0,1); axs[2].set_title("ROC AUC"); axs[2].grid(axis='y',alpha=.3)
plt.suptitle(f"MULTIMODE (K={K}) trap, doc{doc}: {nkw}kw spread across {K} modes",y=1.02)
plt.tight_layout(); plt.savefig("trap_multimode_result.png",dpi=110,bbox_inches="tight")
print("\n저장: trap_multimode_topk.csv, trap_multimode_fdr.csv, trap_multimode_auc.csv, trap_multimode_result.png")
