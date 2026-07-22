"""AWSGLD ZETA(flat-histogram 강도) 튜닝: 단봉 실데이터에 ZETA=5(기본) vs 1 vs 0 비교.
M.ZETA를 런타임 오버라이드(파일 수정 아님). acMH는 기준선으로 1회. 5문서×5시드."""
import os, sys, csv, time, numpy as np
sys.path.insert(0,"/home/jiyoon/BSS-Keyphrase-Extraction/code_JOC")
import keyphrase_functions_awsgld as M
import importlib.util
spec=importlib.util.spec_from_file_location("exp","acmh_vs_awsgld_4to10.py"); exp=importlib.util.module_from_spec(spec); spec.loader.exec_module(exp)
from numpy.linalg import solve

d=exp.d; grid=exp.grid; T,BURN=exp.T,exp.BURN_IN; GAMMA=0.20; NSEED=5
BASE=exp.BASE
dense=[r[0] for r in csv.reader(open(os.path.join(BASE,"selected_ids.txt")))]
picker=np.random.RandomState(20260711); docs=sorted(picker.choice(sorted(dense,key=int),5,replace=False).tolist(),key=int)
ZETAS=[5.0,1.0,0.0]
print(f"문서: {docs} | ZETA 스윕 {ZETAS} | {NSEED}시드\n",flush=True)

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
labels=['acMH']+[f'AWSGLD_Z{z:g}' for z in ZETAS]
allv={L:{m:[] for m in MK} for L in labels}
t0=time.time()
for dc in docs:
    g=exp.build_graph(dc)
    for s in range(NSEED):
        n,truth,B,Y,u_0,ini,a=setup(g,s)
        # acMH 기준
        cm=exp.componentwise_mcmc_numba(T,ini,n,grid,a,u_0,B,Y,seed=s+200000)
        pi_ac=np.mean(M.inv_logit(cm['theta'])[BURN:T,:],axis=0)
        for m,v in met(pi_ac,Y,truth,n).items(): allv['acMH'][m].append(v)
        # AWSGLD ZETA 스윕
        for z in ZETAS:
            M.ZETA=z
            np.random.seed(s+100000)
            pi_aw=M.gibbs_mh(BURN,T,ini,n,g,Y,B,u_0,a,grid,verbose=False)['poster_pi_mn']
            for m,v in met(pi_aw,Y,truth,n).items(): allv[f'AWSGLD_Z{z:g}'][m].append(v)
    print(f"  doc{dc} 완료 | {int(time.time()-t0)}s",flush=True)
M.ZETA=5.0  # 복원
def ms(v): return f"{np.mean(v):.3f}({np.std(v):.3f})"
print(f"\n=== ZETA 튜닝 결과 (5문서×{NSEED}시드=25 pooled, FDR γ={GAMMA}) ===")
print(f"{'구성':>13} | {'Recall':>13} {'F':>13} {'top-k':>13} {'ROC AUC':>13}")
print("-"*74)
for L in labels:
    print(f"{L:>13} | {ms(allv[L]['Recall']):>13} {ms(allv[L]['F']):>13} {ms(allv[L]['topk']):>13} {ms(allv[L]['AUC']):>13}")
with open("hulth_zeta_tune.csv","w",newline="") as f:
    wr=csv.writer(f); wr.writerow(["config","Recall","F","topk","AUC"])
    for L in labels: wr.writerow([L]+[round(np.mean(allv[L][m]),4) for m in MK])
print("\n저장: hulth_zeta_tune.csv")
