"""lever ②(b): 5모드 트랩에서 저온(TAU↓)+강한 flat-histogram(ZETA↑)로
탈출은 유지하되 π를 날카롭게 → top-k 선명도 회복 시도.
(TAU,ZETA) 격자를 1시드로 스캔(탈출 nvis + AUC)해 최적 선택 후 3시드 전체 평가."""
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
groups=[kw[i::K] for i in range(K)]
print(f"doc{doc}: n={n}, 키워드 {nkw}개, 모드 {K}개, 모드별={[len(gr) for gr in groups]}")
sig2=2.0; LEVELS=exp.FDR_LEVELS; T,BURN=50000,5000; cSEP=4  # 5모드 실험에서 acMH 갇히는 장벽
def lik(th): return -np.sum(np.log(np.clip(1-(1-ae)*M.inv_logit(th),1e-10,None)))
centers=[]
for gr in groups:
    u=np.full(n,-0.5); u[gr]=cSEP; centers.append(u)
def Uk(th,u): C=(B@(th-u))@(B@(th-u)); return lik(th)+C/(2*sig2)
def Umix(th): return -logsumexp([-Uk(th,u) for u in centers])
def gUk(th,u):
    pi=np.clip(M.inv_logit(th),1e-10,1-1e-10); dpi=pi*(1-pi)
    return -(-(1-ae)*dpi/np.clip(1-(1-ae)*pi,1e-10,None))+BtB@(th-u)/sig2
def gUmix(th):
    Us=np.array([-Uk(th,u) for u in centers]); w=np.exp(Us-logsumexp(Us))
    return sum(w[i]*gUk(th,centers[i]) for i in range(len(centers)))
def mode(th): return int(np.argmin([Uk(th,u) for u in centers]))
ini=centers[0].copy()
def acmh(seed,step=0.3,Tn=T):
    np.random.seed(seed); th=ini.copy(); ths=np.zeros((Tn,n)); mo=np.zeros(Tn,int); Uc=Umix(th)
    for t in range(Tn):
        star=th+np.random.randn(n)*step; Us=Umix(star)
        if np.log(np.random.rand()+1e-300)<(-Us+Uc): th=star;Uc=Us
        ths[t]=th; mo[t]=mode(th)
    return ths,mo
def awsgld(seed,TAU,ZETA,eps0=0.5,Mr=500,Tn=T):
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
def auc_roc(pi):
    order=np.argsort(-pi); pos=nkw; neg=n-nkw; tp=0;fp=0; tpr=[0];fpr=[0]
    for idx in order:
        if idx in TRUTH: tp+=1
        else: fp+=1
        tpr.append(tp/pos); fpr.append(fp/neg)
    return float(np.trapezoid(tpr,fpr))
def topk(pi):
    order=np.argsort(-pi); out={}
    for k in TOPKS: sel=order[:k]; tp=np.sum(np.isin(sel,TRUTH)); P=tp/k; R=tp/nkw; out[k]=(P,R,2*P*R/(P+R) if P+R>0 else 0)
    return out
def fdr(pi):
    out={}
    for c in LEVELS:
        cf=np.unique(pi)[::-1]; FD=np.array([np.sum(1-pi[pi>=cc])/max((pi>=cc).sum(),1) for cc in cf])
        v=np.where(FD<c)[0]; cut=cf[v.max()] if len(v) else cf[np.argmin(FD)]
        sel=np.where(pi>=cut)[0]; tp=np.sum(np.isin(sel,TRUTH)); pos=len(sel)
        P=tp/pos if pos else 0; R=tp/nkw; out[c]=(P,R,2*P*R/(P+R) if P+R>0 else 0,(pos-tp)/pos if pos else 0)
    return out

# --- (TAU,ZETA) 격자 스캔: 1시드로 탈출(nvis) + AUC ---
print("\n[격자 스캔] TAU↓ ZETA↑ 로 탈출 유지+선명 π 탐색 (1시드, T=15000)")
print(f"{'TAU':>5} {'ZETA':>5} | {'nvis':>4} {'sw':>4} | {'AUC':>6} {'top10P':>7}")
cand=[]
for TAU in [5,10,20,50]:
    for ZETA in [30,60,120]:
        th,mo=awsgld(1,TAU,ZETA,Tn=15000); pi=M.inv_logit(th[3000:]).mean(0)
        nvis=len(set(mo[3000:].tolist())); sw=np.sum(np.diff(mo[3000::5])!=0); a=auc_roc(pi); t10=topk(pi)[10][0]
        print(f"{TAU:>5} {ZETA:>5} | {nvis:>4} {sw:>4} | {a:>6.3f} {t10:>7.3f}",flush=True)
        if nvis>=3: cand.append((a,TAU,ZETA,nvis))
if cand:
    cand.sort(reverse=True); _,cTAU,cZETA,_=cand[0]
else:
    cTAU,cZETA=50,120
print(f"\n선택(탈출하면서 AUC 최고): TAU={cTAU} ZETA={cZETA}\n",flush=True)

# --- 3시드 전체 평가: acMH vs AWSGLD(sharp) ---
accTK={m:{k:[[],[],[]] for k in TOPKS} for m in('acMH','AWSGLD')}
accFD={m:{c:[[],[],[],[]] for c in LEVELS} for m in('acMH','AWSGLD')}
accA={'acMH':[],'AWSGLD':[]}; swA=[];swW=[];visW=[]
t0=time.time()
for s in [1,2,3]:
    tha,moa=acmh(s); thw,mow=awsgld(s,cTAU,cZETA)
    pa=M.inv_logit(tha[BURN:]).mean(0); pw=M.inv_logit(thw[BURN:]).mean(0)
    swA.append(np.sum(np.diff(moa[BURN:])!=0)); swW.append(np.sum(np.diff(mow[BURN:])!=0)); visW.append(len(set(mow[BURN:].tolist())))
    for m,pi in [('acMH',pa),('AWSGLD',pw)]:
        tk=topk(pi); fd=fdr(pi)
        for k in TOPKS:
            for i in range(3): accTK[m][k][i].append(tk[k][i])
        for c in LEVELS:
            for i in range(4): accFD[m][c][i].append(fd[c][i])
        accA[m].append(auc_roc(pi))
    print(f"  seed {s} | {int(time.time()-t0)}s",flush=True)
mTK=lambda m,k,i: float(np.mean(accTK[m][k][i])); mFD=lambda m,c,i: float(np.mean(accFD[m][c][i]))
print(f"\n=== doc{doc} 5모드 트랩 + AWSGLD sharp(TAU={cTAU},ZETA={cZETA}) 3시드 ===")
print(f"모드전환: acMH={np.mean(swA):.0f}, AWSGLD={np.mean(swW):.0f} | AWSGLD 방문모드={np.mean(visW):.1f}/{K}")
print(f"ROC AUC: acMH={np.mean(accA['acMH']):.3f}  AWSGLD={np.mean(accA['AWSGLD']):.3f}")
for lab,i in [("Precision",0),("Recall",1),("F-measure",2)]:
    print(f"\n[top-k {lab}]"); print(f"{'k':>4} | {'acMH':>8} | {'AWSGLD':>8}")
    for k in TOPKS: print(f"{k:>4} | {mTK('acMH',k,i):>8.3f} | {mTK('AWSGLD',k,i):>8.3f}")
for lab,i in [("Precision",0),("Recall",1),("F-measure",2),("actual FDR",3)]:
    print(f"\n[FDR-cutoff {lab}]"); print(f"{'γ':>5} | {'acMH':>8} | {'AWSGLD':>8}")
    for c in LEVELS: print(f"{c:>5} | {mFD('acMH',c,i):>8.3f} | {mFD('AWSGLD',c,i):>8.3f}")
with open("trap_multimode_sharp_topk.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["topk","sampler","precision","recall","F"])
    for k in TOPKS:
        for m in('acMH','AWSGLD'): w.writerow([k,m]+[round(mTK(m,k,i),4) for i in range(3)])
with open("trap_multimode_sharp_fdr.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["gamma","sampler","precision","recall","F","realFDR"])
    for c in LEVELS:
        for m in('acMH','AWSGLD'): w.writerow([c,m]+[round(mFD(m,c,i),4) for i in range(4)])
fig,axs=plt.subplots(1,3,figsize=(15,4.5))
axs[0].plot(TOPKS,[mTK('acMH',k,0) for k in TOPKS],'o-',color="#ff7f0e",label="acMH (stuck)")
axs[0].plot(TOPKS,[mTK('AWSGLD',k,0) for k in TOPKS],'s-',color="#1f77b4",label=f"AWSGLD sharp (TAU={cTAU})")
axs[0].set_title("top-k Precision"); axs[0].set_xlabel("k"); axs[0].grid(alpha=.3); axs[0].legend()
axs[1].plot(TOPKS,[mTK('acMH',k,2) for k in TOPKS],'o-',color="#ff7f0e",label="acMH")
axs[1].plot(TOPKS,[mTK('AWSGLD',k,2) for k in TOPKS],'s-',color="#1f77b4",label="AWSGLD sharp")
axs[1].set_title("top-k F-measure"); axs[1].set_xlabel("k"); axs[1].grid(alpha=.3); axs[1].legend()
axs[2].bar([0,1],[np.mean(accA['acMH']),np.mean(accA['AWSGLD'])],color=["#ff7f0e","#1f77b4"]); axs[2].set_xticks([0,1]); axs[2].set_xticklabels(["acMH","AWSGLD"]); axs[2].set_ylim(0,1); axs[2].set_title("ROC AUC"); axs[2].grid(axis='y',alpha=.3)
plt.suptitle(f"MULTIMODE(K={K}) + AWSGLD SHARP (low TAU={cTAU}, high ZETA={cZETA}), doc{doc}",y=1.02)
plt.tight_layout(); plt.savefig("trap_multimode_sharp_result.png",dpi=110,bbox_inches="tight")
print("\n저장: trap_multimode_sharp_topk.csv, trap_multimode_sharp_fdr.csv, trap_multimode_sharp_result.png")
