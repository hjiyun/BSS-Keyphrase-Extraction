"""dense 문서 AWSGLD 하이퍼파라미터 sweep (모델/batch/floor 불변). N_SIM=8."""
import os, csv, time, importlib.util, sys
import numpy as np
from numpy.linalg import solve
HERE=os.path.dirname(os.path.abspath(__file__))
spec=importlib.util.spec_from_file_location("exp",os.path.join(HERE,"acmh_vs_awsgld_4to10.py")); exp=importlib.util.module_from_spec(spec); spec.loader.exec_module(exp)
sys.path.insert(0,"/home/jiyoon/BSS-Keyphrase-Extraction/code_JOC"); import awsgld_tunable as AT
M=exp.M
LEVELS=exp.FDR_LEVELS; T,BURN,N_SIM,KOBS=exp.T,exp.BURN_IN,8,5; d=0.85
DOCS=["1949","2007","2092","215","2017"]
BASE=dict(TAU=1.0,ZETA=5.0,DECAY_LR=100.0,M_REGIONS=1000)
# (이름, 모듈전역 override, eps_scale)  -- eps_scale은 awsgld_tunable로
CONFIGS=[("base",{},0.3),
         ("TAU1.5",{"TAU":1.5},0.3),("TAU2.0",{"TAU":2.0},0.3),
         ("ZETA1",{"ZETA":1.0},0.3),("ZETA10",{"ZETA":10.0},0.3),
         ("DECAY200",{"DECAY_LR":200.0},0.3),
         ("eps1.0",{},1.0),("eps2.0",{},2.0)]
graphs={doc:exp.build_graph(doc) for doc in DOCS}
def setup(g,seed):
    n,truth=g['n'],g['truth']; rng=np.random.RandomState(seed); G=solve(g['D'],g['A']); B=np.eye(n)-d*G.T
    w=np.diag(1/np.sqrt(np.diag(g['D']))); Bs=np.eye(n)-d*w@g['A']@w
    Y=np.zeros(n); Y[list(rng.choice(truth,KOBS,replace=False))]=1
    u0=solve(B,np.ones(n)*(1-d)); ini=M.base_to_start(solve(Bs,Y))
    return B,Y,u0,ini,M.alpha_find(u0,Y,exp.grid)
exp.run_doc(graphs[DOCS[0]],1)
# acMH 기준
acc_ac={g:[] for g in LEVELS}; t0=time.time()
for doc in DOCS:
    g=graphs[doc]; n=g['n']
    for s in range(N_SIM):
        seed=hash((doc,s))%(2**31); B,Y,u0,ini,ae=setup(g,seed)
        cm=exp.componentwise_mcmc_numba(T,ini,n,exp.grid,ae,u0,B,Y,seed=seed+200000)
        for gl in LEVELS: acc_ac[gl].append(exp.eval_metrics(np.mean(M.inv_logit(cm['theta'])[BURN:T],axis=0),Y,g['truth'])[gl])
print(f"acMH 기준 완료 | {int(time.time()-t0)}s",flush=True)
results={}
for tag,ov,es in CONFIGS:
    for kk,vv in BASE.items(): setattr(M,kk,vv)
    for kk,vv in ov.items(): setattr(M,kk,vv)
    acc={g:[] for g in LEVELS}
    for doc in DOCS:
        g=graphs[doc]; n=g['n']
        for s in range(N_SIM):
            seed=hash((doc,s))%(2**31); B,Y,u0,ini,ae=setup(g,seed)
            np.random.seed(seed+800000)
            r=AT.gibbs_mh_eps(BURN,T,ini,n,g,Y,B,u0,ae,exp.grid,eps_scale=es,verbose=False)  # full-batch, floor 0.5 기본
            for gl in LEVELS: acc[gl].append(exp.eval_metrics(r['poster_pi_mn'],Y,g['truth'])[gl])
    results[tag]=acc
    print(f"[{tag:9}] F@0.2={np.mean([x['F'] for x in acc[0.2]]):.3f} F@0.25={np.mean([x['F'] for x in acc[0.25]]):.3f} | {int(time.time()-t0)}s",flush=True)
for kk,vv in BASE.items(): setattr(M,kk,vv)
def mn(acc,g,key): return float(np.mean([x[key] for x in acc[g]]))
print(f"\n=== dense 5문서 HP sweep (F-measure) ===")
print("config".ljust(10)+"".join(f"g{g:<5}" for g in LEVELS))
print("acMH".ljust(10)+"".join(f"{mn(acc_ac,g,'F'):<6.3f}" for g in LEVELS))
for tag,acc in results.items(): print(tag.ljust(10)+"".join(f"{mn(acc,g,'F'):<6.3f}" for g in LEVELS))
with open(os.path.join(HERE,"dense_hp_tune.csv"),"w",newline="") as fh:
    w=csv.writer(fh); w.writerow(["config","gamma","precision","recall","F","realFDR"])
    w.writerows([["acMH",g]+[round(mn(acc_ac,g,k),4) for k in("precision","recall","F","realFDR")] for g in LEVELS])
    for tag,acc in results.items():
        for g in LEVELS: w.writerow([tag,g]+[round(mn(acc,g,k),4) for k in("precision","recall","F","realFDR")])
print("\n저장: dense_hp_tune.csv")
