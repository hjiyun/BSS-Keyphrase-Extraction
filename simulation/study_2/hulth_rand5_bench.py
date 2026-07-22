"""Hulth dense 풀에서 랜덤 5문서, acMH/AWSGLD 각 10시드 → Recall/F(FDR γ=0.20)/top-k/ROC AUC 평균(표준편차)."""
import os, sys, csv, time, numpy as np
sys.path.insert(0,"/home/jiyoon/BSS-Keyphrase-Extraction/code_JOC")
import keyphrase_functions_awsgld as M
import importlib.util
spec=importlib.util.spec_from_file_location("exp","acmh_vs_awsgld_4to10.py"); exp=importlib.util.module_from_spec(spec); spec.loader.exec_module(exp)
from numpy.linalg import solve

d=exp.d; grid=exp.grid; T,BURN=exp.T,exp.BURN_IN; GAMMA=0.20; NSEED=10
BASE=exp.BASE
# dense(>=10) 문서 목록
dense=[r[0] for r in csv.reader(open(os.path.join(BASE,"selected_ids.txt")))] if os.path.exists(os.path.join(BASE,"selected_ids.txt")) else None
if dense is None or not dense:
    rows=[r for r in csv.reader(open(os.path.join(BASE,"doc_stats.csv")))][1:]
    dense=[r[0] for r in rows if int(r[2])>=10]
# 재현성: 고정 시드로 5개 랜덤 추출
picker=np.random.RandomState(20260711); docs=sorted(picker.choice(sorted(dense,key=int),5,replace=False).tolist(),key=int)
print(f"dense 풀 {len(dense)}개 중 랜덤 5문서: {docs}\n",flush=True)

def auc_roc(pi,truth,n):
    T_=set(truth); order=np.argsort(-pi); pos=len(T_); neg=n-pos; tp=fp=0; tpr=[0.];fpr=[0.]
    for i in order:
        if i in T_: tp+=1
        else: fp+=1
        tpr.append(tp/pos); fpr.append(fp/neg)
    return float(np.trapezoid(tpr,fpr)) if hasattr(np,"trapezoid") else float(np.sum(np.diff(fpr)*(np.array(tpr[1:])+np.array(tpr[:-1]))/2))

def run(graph,seed):
    """run_doc와 동일 설정, 단 pi 벡터도 반환."""
    n,truth=graph['n'],graph['truth']; rng=np.random.RandomState(seed)
    G=solve(graph['D'],graph['A']); B=np.eye(n)-d*G.T
    w=np.diag(1.0/np.sqrt(np.diag(graph['D']))); Bs=np.eye(n)-d*w@graph['A']@w
    k=max(1,len(truth)//2); Y=np.zeros(n); Y[list(rng.choice(truth,k,replace=False))]=1
    u_0=solve(B,np.ones(n)*(1-d)); ini=M.base_to_start(solve(Bs,Y)); a=M.alpha_find(u_0,Y,grid)
    np.random.seed(seed+100000)
    pi_aw=M.gibbs_mh(BURN,T,ini,n,graph,Y,B,u_0,a,grid,verbose=False)['poster_pi_mn']
    cm=exp.componentwise_mcmc_numba(T,ini,n,grid,a,u_0,B,Y,seed=seed+200000)
    pi_ac=np.mean(M.inv_logit(cm['theta'])[BURN:T,:],axis=0)
    return pi_aw,pi_ac,Y,truth,n

def metrics(pi,Y,truth,n):
    ev=exp.eval_metrics(pi,Y,truth)[GAMMA]; nk=len(truth)
    topk=float(np.isin(np.argsort(-pi)[:nk],truth).sum()/nk)  # k=정답수 → P=R=F
    return dict(Recall=ev['recall'],F=ev['F'],topk=topk,AUC=auc_roc(pi,truth,n))

MK=('Recall','F','topk','AUC')
per={dc:{s:{m:[] for m in MK} for s in ('acMH','AWSGLD')} for dc in docs}
allv={s:{m:[] for m in MK} for s in ('acMH','AWSGLD')}
t0=time.time()
for dc in docs:
    g=exp.build_graph(dc)
    for s in range(NSEED):
        pi_aw,pi_ac,Y,truth,n=run(g,s)
        for name,pi in (('AWSGLD',pi_aw),('acMH',pi_ac)):
            mm=metrics(pi,Y,truth,n)
            for m in MK: per[dc][name][m].append(mm[m]); allv[name][m].append(mm[m])
    print(f"  doc{dc} (n={g['n']}, truth={len(g['truth'])}) 완료 | {int(time.time()-t0)}s",flush=True)

def ms(v): return f"{np.mean(v):.3f}({np.std(v):.3f})"
print(f"\n=== Hulth dense 랜덤5, 각 문서 10시드 평균(표준편차) | FDR γ={GAMMA}, top-k=정답수 ===\n")
hdr=f"{'문서':>7} {'n':>4} {'kw':>3} {'샘플러':>7} | {'Recall':>13} {'F':>13} {'top-k':>13} {'ROC AUC':>13}"
print(hdr); print("-"*len(hdr))
for dc in docs:
    g_n=len(per[dc]['acMH']['Recall'])  # dummy
    for s in ('acMH','AWSGLD'):
        info=exp.build_graph(dc) if False else None
        print(f"{dc:>7} {'':>4} {'':>3} {s:>7} | {ms(per[dc][s]['Recall']):>13} {ms(per[dc][s]['F']):>13} {ms(per[dc][s]['topk']):>13} {ms(per[dc][s]['AUC']):>13}")
    print("-"*len(hdr))
print(f"\n=== 전체 pooled (5문서×10시드=50) ===")
print(f"{'샘플러':>7} | {'Recall':>13} {'F':>13} {'top-k':>13} {'ROC AUC':>13}")
for s in ('acMH','AWSGLD'):
    print(f"{s:>7} | {ms(allv[s]['Recall']):>13} {ms(allv[s]['F']):>13} {ms(allv[s]['topk']):>13} {ms(allv[s]['AUC']):>13}")

with open("hulth_rand5_bench.csv","w",newline="") as f:
    wr=csv.writer(f); wr.writerow(["doc","sampler","Recall_mean","Recall_std","F_mean","F_std","topk_mean","topk_std","AUC_mean","AUC_std"])
    for dc in docs:
        for s in ('acMH','AWSGLD'):
            wr.writerow([dc,s]+[round(x,4) for m in MK for x in (np.mean(per[dc][s][m]),np.std(per[dc][s][m]))])
    for s in ('acMH','AWSGLD'):
        wr.writerow(["ALL",s]+[round(x,4) for m in MK for x in (np.mean(allv[s][m]),np.std(allv[s][m]))])
print("\n저장: hulth_rand5_bench.csv")
