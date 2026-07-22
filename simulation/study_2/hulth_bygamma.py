"""Hulth dense 랜덤5, acMH vs AWSGLD(ZETA=1), γ별(0.05~0.30) FDR-cutoff 수치.
각 γ: Precision/Recall/F/actual-FDR 평균(표준편차). γ무관 top-k·ROC AUC는 별도. 5문서×10시드."""
import os, sys, csv, time, numpy as np
sys.path.insert(0,"/home/jiyoon/BSS-Keyphrase-Extraction/code_JOC")
import keyphrase_functions_awsgld as M
import importlib.util
spec=importlib.util.spec_from_file_location("exp","acmh_vs_awsgld_4to10.py"); exp=importlib.util.module_from_spec(spec); spec.loader.exec_module(exp)
from numpy.linalg import solve

M.ZETA=1.0
d=exp.d; grid=exp.grid; T,BURN=exp.T,exp.BURN_IN; NSEED=10; LEVELS=exp.FDR_LEVELS
BASE=exp.BASE
dense=[r[0] for r in csv.reader(open(os.path.join(BASE,"selected_ids.txt")))]
picker=np.random.RandomState(20260711); docs=sorted(picker.choice(sorted(dense,key=int),5,replace=False).tolist(),key=int)
print(f"문서: {docs} | acMH vs AWSGLD(ZETA=1) | {NSEED}시드 | γ={list(LEVELS)}\n",flush=True)

def auc_roc(pi,truth,n):
    Tt=set(truth); order=np.argsort(-pi); pos=len(Tt); neg=n-pos; tp=fp=0; tpr=[0.];fpr=[0.]
    for i in order:
        if i in Tt: tp+=1
        else: fp+=1
        tpr.append(tp/pos); fpr.append(fp/neg)
    return float(np.trapezoid(tpr,fpr))
def setup(graph,seed):
    n,truth=graph['n'],graph['truth']; rng=np.random.RandomState(seed)
    G=solve(graph['D'],graph['A']); B=np.eye(n)-d*G.T
    w=np.diag(1.0/np.sqrt(np.diag(graph['D']))); Bs=np.eye(n)-d*w@graph['A']@w
    k=max(1,len(truth)//2); Y=np.zeros(n); Y[list(rng.choice(truth,k,replace=False))]=1
    u_0=solve(B,np.ones(n)*(1-d)); ini=M.base_to_start(solve(Bs,Y)); a=M.alpha_find(u_0,Y,grid)
    return n,truth,B,Y,u_0,ini,a

FM=('precision','recall','F','realFDR')
gam={s:{g:{m:[] for m in FM} for g in LEVELS} for s in ('acMH','AWSGLD')}
ind={s:{'topk':[],'AUC':[]} for s in ('acMH','AWSGLD')}
t0=time.time()
for dc in docs:
    g=exp.build_graph(dc)
    for s in range(NSEED):
        n,truth,B,Y,u_0,ini,a=setup(g,s); nk=len(truth)
        cm=exp.componentwise_mcmc_numba(T,ini,n,grid,a,u_0,B,Y,seed=s+200000)
        pi_ac=np.mean(M.inv_logit(cm['theta'])[BURN:T,:],axis=0)
        np.random.seed(s+100000)
        pi_aw=M.gibbs_mh(BURN,T,ini,n,g,Y,B,u_0,a,grid,verbose=False)['poster_pi_mn']
        for name,pi in (('acMH',pi_ac),('AWSGLD',pi_aw)):
            ev=exp.eval_metrics(pi,Y,truth)
            for gg in LEVELS:
                for m in FM: gam[name][gg][m].append(ev[gg][m])
            ind[name]['topk'].append(float(np.isin(np.argsort(-pi)[:nk],truth).sum()/nk))
            ind[name]['AUC'].append(auc_roc(pi,truth,n))
    print(f"  doc{dc} 완료 | {int(time.time()-t0)}s",flush=True)
def ms(v): return f"{np.mean(v):.3f}({np.std(v):.3f})"
print(f"\n=== γ별 FDR-cutoff (5문서×{NSEED}시드=50 pooled) ===")
print(f"{'γ':>5} {'샘플러':>7} | {'Precision':>13} {'Recall':>13} {'F':>13} {'actualFDR':>13}")
print("-"*74)
for gg in LEVELS:
    for s in ('acMH','AWSGLD'):
        r=gam[s][gg]
        print(f"{gg:>5} {s:>7} | {ms(r['precision']):>13} {ms(r['recall']):>13} {ms(r['F']):>13} {ms(r['realFDR']):>13}")
    print("-"*74)
print(f"\n=== γ무관 지표 ===")
print(f"{'샘플러':>7} | {'top-k':>13} {'ROC AUC':>13}")
for s in ('acMH','AWSGLD'):
    print(f"{s:>7} | {ms(ind[s]['topk']):>13} {ms(ind[s]['AUC']):>13}")
with open("hulth_bygamma.csv","w",newline="") as f:
    wr=csv.writer(f); wr.writerow(["gamma","sampler","precision","recall","F","realFDR"])
    for gg in LEVELS:
        for s in ('acMH','AWSGLD'): wr.writerow([gg,s]+[round(np.mean(gam[s][gg][m]),4) for m in FM])
    for s in ('acMH','AWSGLD'): wr.writerow(["topk/auc",s,round(np.mean(ind[s]['topk']),4),round(np.mean(ind[s]['AUC']),4),"",""])
print("\n저장: hulth_bygamma.csv")
