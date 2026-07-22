"""왜곡 없는 최대 비교: acMH vs AWSGLD(ZETA=1, 튜닝) vs MALA_v2(MH 보정 → 이산화편향 제거).
동일 타깃(σ²-적분 posterior)·동일 Y·동일 init. 5문서×10시드, Recall/F(γ0.2)/top-k/ROC AUC."""
import os, sys, csv, time, numpy as np
sys.path.insert(0,"/home/jiyoon/BSS-Keyphrase-Extraction/code_JOC")
import keyphrase_functions_awsgld as M
import mala_keyphrase as MA
import importlib.util
spec=importlib.util.spec_from_file_location("exp","acmh_vs_awsgld_4to10.py"); exp=importlib.util.module_from_spec(spec); spec.loader.exec_module(exp)
from numpy.linalg import solve

M.ZETA=1.0   # AWSGLD 튜닝(왜곡 아님: flat-histogram 강도)
d=exp.d; grid=exp.grid; T,BURN=exp.T,exp.BURN_IN; GAMMA=0.20; NSEED=10
BASE=exp.BASE
dense=[r[0] for r in csv.reader(open(os.path.join(BASE,"selected_ids.txt")))]
picker=np.random.RandomState(20260711); docs=sorted(picker.choice(sorted(dense,key=int),5,replace=False).tolist(),key=int)
print(f"문서: {docs} | acMH / AWSGLD(ZETA=1) / MALA_v2 | {NSEED}시드\n",flush=True)

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

MK=('Recall','F','topk','AUC'); labels=['acMH','AWSGLD_Z1','MALA_v2']
per={dc:{L:{m:[] for m in MK} for L in labels} for dc in docs}
allv={L:{m:[] for m in MK} for L in labels}
t0=time.time()
for dc in docs:
    g=exp.build_graph(dc)
    for s in range(NSEED):
        n,truth,B,Y,u_0,ini,a=setup(g,s)
        # acMH
        cm=exp.componentwise_mcmc_numba(T,ini,n,grid,a,u_0,B,Y,seed=s+200000)
        pi={'acMH':np.mean(M.inv_logit(cm['theta'])[BURN:T,:],axis=0)}
        # AWSGLD ZETA=1
        np.random.seed(s+100000)
        pi['AWSGLD_Z1']=M.gibbs_mh(BURN,T,ini,n,g,Y,B,u_0,a,grid,verbose=False)['poster_pi_mn']
        # MALA v2 (MH 보정)
        np.random.seed(s+300000)
        pi['MALA_v2']=MA.mala_v2(T,ini,n,B,Y,u_0,a,grid,Burn_in=BURN,verbose=False)['poster_pi_mn']
        for L in labels:
            for m,v in met(pi[L],Y,truth,n).items(): per[dc][L][m].append(v); allv[L][m].append(v)
    print(f"  doc{dc} 완료 | {int(time.time()-t0)}s",flush=True)
def ms(v): return f"{np.mean(v):.3f}({np.std(v):.3f})"
print(f"\n=== 최종 (5문서×{NSEED}시드=50 pooled, FDR γ={GAMMA}) ===")
print(f"{'샘플러':>10} | {'Recall':>13} {'F':>13} {'top-k':>13} {'ROC AUC':>13}")
print("-"*72)
for L in labels:
    print(f"{L:>10} | {ms(allv[L]['Recall']):>13} {ms(allv[L]['F']):>13} {ms(allv[L]['topk']):>13} {ms(allv[L]['AUC']):>13}")
with open("hulth_final_bench.csv","w",newline="") as f:
    wr=csv.writer(f); wr.writerow(["scope","sampler","Recall_m","Recall_s","F_m","F_s","topk_m","topk_s","AUC_m","AUC_s"])
    for dc in docs:
        for L in labels: wr.writerow([dc,L]+[round(x,4) for m in MK for x in (np.mean(per[dc][L][m]),np.std(per[dc][L][m]))])
    for L in labels: wr.writerow(["ALL",L]+[round(x,4) for m in MK for x in (np.mean(allv[L][m]),np.std(allv[L][m]))])
print("\n저장: hulth_final_bench.csv")
