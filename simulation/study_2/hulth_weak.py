"""약지도(비율 관측): k=max(1,round(RATIO*truth)). 원논문 고정 k=4 스케일과 유사.
같은 dense 5문서에서 관측만 약지도로 바꿔 acMH vs AWSGLD(Z5) vs AWSGLD(Z1) 비교.
γ별 FDR-cutoff(P/R/F/actualFDR) + γ무관 top-k·ROC AUC. 5문서×10시드."""
import os, sys, csv, time, numpy as np
sys.path.insert(0,"/home/jiyoon/BSS-Keyphrase-Extraction/code_JOC")
import keyphrase_functions_awsgld as M
import importlib.util
spec=importlib.util.spec_from_file_location("exp","acmh_vs_awsgld_4to10.py"); exp=importlib.util.module_from_spec(spec); spec.loader.exec_module(exp)
from numpy.linalg import solve

d=exp.d; grid=exp.grid; T,BURN=exp.T,exp.BURN_IN; NSEED=10; LEVELS=exp.FDR_LEVELS; RATIO=0.20
BASE=exp.BASE
dense=[r[0] for r in csv.reader(open(os.path.join(BASE,"selected_ids.txt")))]
picker=np.random.RandomState(20260711); docs=sorted(picker.choice(sorted(dense,key=int),5,replace=False).tolist(),key=int)
print(f"문서: {docs} | 약지도 비율={RATIO} | acMH / AWSGLD_Z5 / AWSGLD_Z1 | {NSEED}시드\n",flush=True)

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
    k=max(1,round(RATIO*len(truth))); Y=np.zeros(n); Y[list(rng.choice(truth,k,replace=False))]=1
    u_0=solve(B,np.ones(n)*(1-d)); ini=M.base_to_start(solve(Bs,Y)); a=M.alpha_find(u_0,Y,grid)
    return n,truth,B,Y,u_0,ini,a,k

FM=('precision','recall','F','realFDR'); SAMP=('acMH','AWSGLD_Z5','AWSGLD_Z1')
gam={s:{g:{m:[] for m in FM} for g in LEVELS} for s in SAMP}
ind={s:{'topk':[],'AUC':[]} for s in SAMP}; krec=[]
t0=time.time()
for dc in docs:
    g=exp.build_graph(dc)
    for s in range(NSEED):
        n,truth,B,Y,u_0,ini,a,k=setup(g,s); nk=len(truth); krec.append(k)
        cm=exp.componentwise_mcmc_numba(T,ini,n,grid,a,u_0,B,Y,seed=s+200000)
        pi={'acMH':np.mean(M.inv_logit(cm['theta'])[BURN:T,:],axis=0)}
        M.ZETA=5.0; np.random.seed(s+100000)
        pi['AWSGLD_Z5']=M.gibbs_mh(BURN,T,ini,n,g,Y,B,u_0,a,grid,verbose=False)['poster_pi_mn']
        M.ZETA=1.0; np.random.seed(s+100000)
        pi['AWSGLD_Z1']=M.gibbs_mh(BURN,T,ini,n,g,Y,B,u_0,a,grid,verbose=False)['poster_pi_mn']
        for name in SAMP:
            ev=exp.eval_metrics(pi[name],Y,truth)
            for gg in LEVELS:
                for m in FM: gam[name][gg][m].append(ev[gg][m])
            ind[name]['topk'].append(float(np.isin(np.argsort(-pi[name])[:nk],truth).sum()/nk))
            ind[name]['AUC'].append(auc_roc(pi[name],truth,n))
    print(f"  doc{dc} (truth={len(g['truth'])}, k≈{max(1,round(RATIO*len(g['truth'])))}) 완료 | {int(time.time()-t0)}s",flush=True)
M.ZETA=5.0
def ms(v): return f"{np.mean(v):.3f}({np.std(v):.3f})"
print(f"\n=== 약지도(비율{RATIO}, k평균={np.mean(krec):.1f}) γ별 FDR-cutoff (5문서×{NSEED}시드=50) ===")
print(f"{'γ':>5} {'샘플러':>10} | {'Precision':>13} {'Recall':>13} {'F':>13} {'actualFDR':>13}")
print("-"*80)
for gg in LEVELS:
    for s in SAMP:
        r=gam[s][gg]
        print(f"{gg:>5} {s:>10} | {ms(r['precision']):>13} {ms(r['recall']):>13} {ms(r['F']):>13} {ms(r['realFDR']):>13}")
    print("-"*80)
print(f"\n=== γ무관 ==="); print(f"{'샘플러':>10} | {'top-k':>13} {'ROC AUC':>13}")
for s in SAMP: print(f"{s:>10} | {ms(ind[s]['topk']):>13} {ms(ind[s]['AUC']):>13}")
with open("hulth_weak.csv","w",newline="") as f:
    wr=csv.writer(f); wr.writerow(["gamma","sampler","precision","recall","F","realFDR"])
    for gg in LEVELS:
        for s in SAMP: wr.writerow([gg,s]+[round(np.mean(gam[s][gg][m]),4) for m in FM])
    for s in SAMP: wr.writerow(["topk/auc",s,round(np.mean(ind[s]['topk']),4),round(np.mean(ind[s]['AUC']),4),"",""])
print("\n저장: hulth_weak.csv")
