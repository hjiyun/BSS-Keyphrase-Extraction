"""왜곡 없는 정당한 튜닝: ZETA=1 고정 + eps 스텝 스케줄 스윕.
eps_k = eps_scale/((t+1)^eps_pow + 10). eps↓ → 이산화 편향↓(AUC 개선 기대), 단 과소면 mixing 저하.
sigma2_floor는 baseline 기본값(0.5) 그대로 — 왜곡 레버로 쓰지 않음. acMH는 기준선."""
import os, sys, csv, time, numpy as np
sys.path.insert(0,"/home/jiyoon/BSS-Keyphrase-Extraction/code_JOC")
import keyphrase_functions_awsgld as M
import awsgld_tunable as AT
import importlib.util
spec=importlib.util.spec_from_file_location("exp","acmh_vs_awsgld_4to10.py"); exp=importlib.util.module_from_spec(spec); spec.loader.exec_module(exp)
from numpy.linalg import solve

M.ZETA=1.0   # 튜닝: flat-histogram 강도 (스윕에서 확인한 최적)
d=exp.d; grid=exp.grid; T,BURN=exp.T,exp.BURN_IN; GAMMA=0.20; NSEED=5
BASE=exp.BASE
dense=[r[0] for r in csv.reader(open(os.path.join(BASE,"selected_ids.txt")))]
picker=np.random.RandomState(20260711); docs=sorted(picker.choice(sorted(dense,key=int),5,replace=False).tolist(),key=int)
# (eps_scale, eps_pow) 후보
CFG=[(0.3,0.6),(0.5,0.6),(0.8,0.6),(0.5,0.5)]
print(f"문서: {docs} | ZETA=1 고정 | eps(scale,pow) {CFG} | {NSEED}시드\n",flush=True)

def auc_roc(pi,truth,n):
    Tt=set(truth); order=np.argsort(-pi); pos=len(Tt); neg=n-pos; tp=fp=0; tpr=[0.];fpr=[0.]
    for i in order:
        if i in Tt: tp+=1
        else: fp+=1
        tpr.append(tp/pos); fpr.append(fp/neg)
    return float(np.trapezoid(tpr,fpr))
def met(pi,Y,truth,n):
    ev=exp.eval_metrics(pi,Y,truth)[GAMMA]; nk=len(truth)
    return dict(Recall=ev['recall'],F=ev['F'],topk=float(np.isin(np.argsort(-pi)[:nk],truth).sum()/nk),AUC=auc_roc(pi,truth,n))
def setup(graph,seed):
    n,truth=graph['n'],graph['truth']; rng=np.random.RandomState(seed)
    G=solve(graph['D'],graph['A']); B=np.eye(n)-d*G.T
    w=np.diag(1.0/np.sqrt(np.diag(graph['D']))); Bs=np.eye(n)-d*w@graph['A']@w
    k=max(1,len(truth)//2); Y=np.zeros(n); Y[list(rng.choice(truth,k,replace=False))]=1
    u_0=solve(B,np.ones(n)*(1-d)); ini=M.base_to_start(solve(Bs,Y)); a=M.alpha_find(u_0,Y,grid)
    return n,truth,B,Y,u_0,ini,a

MK=('Recall','F','topk','AUC')
labels=['acMH']+[f'AW_eps{s:g}_p{p:g}' for s,p in CFG]
allv={L:{m:[] for m in MK} for L in labels}
t0=time.time()
for dc in docs:
    g=exp.build_graph(dc)
    for s in range(NSEED):
        n,truth,B,Y,u_0,ini,a=setup(g,s)
        cm=exp.componentwise_mcmc_numba(T,ini,n,grid,a,u_0,B,Y,seed=s+200000)
        pi_ac=np.mean(M.inv_logit(cm['theta'])[BURN:T,:],axis=0)
        for m,v in met(pi_ac,Y,truth,n).items(): allv['acMH'][m].append(v)
        for (esc,epw) in CFG:
            np.random.seed(s+100000)
            pi_aw=AT.gibbs_mh_eps(BURN,T,ini,n,g,Y,B,u_0,a,grid,eps_scale=esc,eps_pow=epw,verbose=False)['poster_pi_mn']
            for m,v in met(pi_aw,Y,truth,n).items(): allv[f'AW_eps{esc:g}_p{epw:g}'][m].append(v)
    print(f"  doc{dc} 완료 | {int(time.time()-t0)}s",flush=True)
def ms(v): return f"{np.mean(v):.3f}({np.std(v):.3f})"
print(f"\n=== eps 튜닝 결과 (ZETA=1, 5문서×{NSEED}시드=25 pooled, FDR γ={GAMMA}) ===")
print(f"{'구성':>16} | {'Recall':>13} {'F':>13} {'top-k':>13} {'ROC AUC':>13}")
print("-"*78)
for L in labels:
    star=" ★" if L!='acMH' and np.mean(allv[L]['F'])>np.mean(allv['acMH']['F']) and np.mean(allv[L]['AUC'])>=np.mean(allv['acMH']['AUC']) else ""
    print(f"{L:>16} | {ms(allv[L]['Recall']):>13} {ms(allv[L]['F']):>13} {ms(allv[L]['topk']):>13} {ms(allv[L]['AUC']):>13}{star}")
with open("hulth_eps_tune.csv","w",newline="") as f:
    wr=csv.writer(f); wr.writerow(["config","Recall","F","topk","AUC"])
    for L in labels: wr.writerow([L]+[round(np.mean(allv[L][m]),4) for m in MK])
print("\n저장: hulth_eps_tune.csv")
