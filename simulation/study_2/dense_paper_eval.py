"""dense 문서 재평가 — 원논문(main_graph.R + precision.recall.auc) 방식 그대로.

우리 dense_test.py 와의 차이:
  (1) 집계: 문서별 precision 평균(macro) → TP/pos 를 문서 합산 후 나눔(pooled/micro)
        main_graph.R:  mean_precision = sum(TP)/sum(pos)
                       mean_recall    = sum(TP)/sum(total_key)
                       mean_f         = 2PR/(P+R)   (pooled P,R 로부터)
  (2) 추가 지표: ROC AUC + PR 곡선(recall 0.1~1.0 고정점의 precision)
        Keyphrase_functions.R: precision.recall.auc()
FDR-cutoff 내부의 force_obs_to_key2(Y==1 → π=1) 는 M.vec_FDR_cutoff 가 이미 적용.
"""
import os, csv, time, importlib.util
import numpy as np
from numpy.linalg import solve
HERE=os.path.dirname(os.path.abspath(__file__))
spec=importlib.util.spec_from_file_location("exp",os.path.join(HERE,"acmh_vs_awsgld_4to10.py")); exp=importlib.util.module_from_spec(spec); spec.loader.exec_module(exp)
M=exp.M; M.TAU=1.0; M.ZETA=5.0; M.DECAY_LR=100.0; M.M_REGIONS=1000
LEVELS=exp.FDR_LEVELS; T,BURN,N_SIM,KOBS=exp.T,exp.BURN_IN,8,5; d=0.85
DOCS=["1949","2007","2092","215","2017"]
RECALL_PTS=np.round(np.arange(0.1,1.01,0.1),2)

def setup(graph,seed):
    n,truth=graph['n'],graph['truth']
    rng=np.random.RandomState(seed); G=solve(graph['D'],graph['A']); B=np.eye(n)-d*G.T
    w=np.diag(1/np.sqrt(np.diag(graph['D']))); Bs=np.eye(n)-d*w@graph['A']@w
    Y=np.zeros(n); Y[list(rng.choice(truth,KOBS,replace=False))]=1
    u0=solve(B,np.ones(n)*(1-d)); ini=M.base_to_start(solve(Bs,Y))
    return B,Y,u0,ini,M.alpha_find(u0,Y,exp.grid)

def roc_auc(pi,truth_Y):
    """R: roc(truth_Y, poster_pi, direction='<')$auc"""
    pos=int(truth_Y.sum()); neg=len(truth_Y)-pos
    if pos==0 or neg==0: return np.nan
    order=np.argsort(-pi); tp=0;fp=0; tpr=[0.0];fpr=[0.0]
    for i in order:
        if truth_Y[i]==1: tp+=1
        else: fp+=1
        tpr.append(tp/pos); fpr.append(fp/neg)
    return float(np.trapezoid(tpr,fpr))

def pr_at_recall(pi,truth_Y):
    """R ROCR performance(pred,'prec','rec') 를 recall 고정점에서 읽는 것과 동형.
    각 recall 목표를 처음 달성하는 cutoff 의 precision."""
    pos=int(truth_Y.sum())
    if pos==0: return np.full(len(RECALL_PTS),np.nan)
    order=np.argsort(-pi); tp=0; out=[]; recs=[]; precs=[]
    for r,i in enumerate(order,1):
        if truth_Y[i]==1: tp+=1
        recs.append(tp/pos); precs.append(tp/r)
    recs=np.array(recs); precs=np.array(precs)
    for rt in RECALL_PTS:
        idx=np.where(recs>=rt-1e-12)[0]
        out.append(precs[idx[0]] if len(idx) else np.nan)
    return np.array(out)

# 누적: γ별 pooled TP/pos/key, 그리고 AUC·PR곡선(문서·시드 평균)
pool={m:{g:{'tp':0.0,'pos':0.0,'key':0.0} for g in LEVELS} for m in ["acMH","AWSGLD"]}
aucs={m:[] for m in ["acMH","AWSGLD"]}; prs={m:[] for m in ["acMH","AWSGLD"]}
exp.run_doc(exp.build_graph(DOCS[0]),1); t0=time.time()
for doc in DOCS:
    g=exp.build_graph(doc); n,truth=g['n'],g['truth']
    truth_Y=np.zeros(n,int); truth_Y[truth]=1
    for s in range(N_SIM):
        seed=hash((doc,s))%(2**31); B,Y,u0,ini,ae=setup(g,seed)
        cm=exp.componentwise_mcmc_numba(T,ini,n,exp.grid,ae,u0,B,Y,seed=seed+200000)
        pa=np.mean(M.inv_logit(cm['theta'])[BURN:T],axis=0)
        np.random.seed(seed+100000); rb=M.gibbs_mh(BURN,T,ini,n,g,Y,B,u0,ae,exp.grid,verbose=False)
        pw=rb['poster_pi_mn']
        for m,pi in [("acMH",pa),("AWSGLD",pw)]:
            ev=exp.eval_metrics(pi,Y,truth=truth)
            for gl in LEVELS:
                pool[m][gl]['tp']+=ev[gl]['tp']; pool[m][gl]['pos']+=ev[gl]['pos']; pool[m][gl]['key']+=len(truth)
            aucs[m].append(roc_auc(pi,truth_Y)); prs[m].append(pr_at_recall(pi,truth_Y))
    print(f"[{doc}] n={n} kw={len(truth)} | {int(time.time()-t0)}s",flush=True)

def pooled(m,g):
    tp,pos,key=pool[m][g]['tp'],pool[m][g]['pos'],pool[m][g]['key']
    P=tp/pos if pos>0 else 0.0; R=tp/key if key>0 else 0.0
    F=2*P*R/(P+R) if P+R>0 else 0.0
    return P,R,F,(pos-tp)/pos if pos>0 else 0.0

print("\n=== [원논문 방식] dense 5문서 pooled 집계 (sum TP / sum pos) ===")
print(f"{'γ':>5} | {'acMH-P':>7} {'AWS-P':>7} | {'acMH-R':>7} {'AWS-R':>7} | {'acMH-F':>7} {'AWS-F':>7} | {'acMH-FDR':>8} {'AWS-FDR':>8}")
for g in LEVELS:
    Pa,Ra,Fa,Da=pooled("acMH",g); Pw,Rw,Fw,Dw=pooled("AWSGLD",g)
    print(f"{g:>5} | {Pa:>7.3f} {Pw:>7.3f} | {Ra:>7.3f} {Rw:>7.3f} | {Fa:>7.3f} {Fw:>7.3f} | {Da:>8.3f} {Dw:>8.3f}")

print(f"\n=== [원논문 방식] ROC AUC (문서·시드 평균) ===")
print(f"acMH={np.nanmean(aucs['acMH']):.4f}   AWSGLD={np.nanmean(aucs['AWSGLD']):.4f}")

Pa=np.nanmean(np.array(prs['acMH']),axis=0); Pw=np.nanmean(np.array(prs['AWSGLD']),axis=0)
print(f"\n=== [원논문 방식] PR 곡선: recall 고정점별 precision ===")
print(f"{'recall':>7} | {'acMH':>7} | {'AWSGLD':>7}")
for i,rt in enumerate(RECALL_PTS): print(f"{rt:>7.1f} | {Pa[i]:>7.3f} | {Pw[i]:>7.3f}")

with open(os.path.join(HERE,"dense_paper_eval.csv"),"w",newline="") as fh:
    w=csv.writer(fh); w.writerow(["metric","gamma_or_recall","acMH","AWSGLD"])
    for g in LEVELS:
        a=pooled("acMH",g); b=pooled("AWSGLD",g)
        for nm,i in [("precision",0),("recall",1),("F",2),("realFDR",3)]:
            w.writerow([f"pooled_{nm}",g,round(a[i],4),round(b[i],4)])
    w.writerow(["roc_auc","-",round(float(np.nanmean(aucs['acMH'])),4),round(float(np.nanmean(aucs['AWSGLD'])),4)])
    for i,rt in enumerate(RECALL_PTS):
        w.writerow(["pr_precision_at_recall",rt,round(float(Pa[i]),4),round(float(Pw[i]),4)])
print("\n저장: dense_paper_eval.csv")
