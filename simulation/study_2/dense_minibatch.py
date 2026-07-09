"""dense 문서 AWSGLD 미니배치 크기 sweep (모델/floor 불변). N_SIM=8, k=5."""
import os, csv, time, importlib.util
import numpy as np
from numpy.linalg import solve
HERE=os.path.dirname(os.path.abspath(__file__))
spec=importlib.util.spec_from_file_location("exp",os.path.join(HERE,"acmh_vs_awsgld_4to10.py")); exp=importlib.util.module_from_spec(spec); spec.loader.exec_module(exp)
M=exp.M; M.TAU=1.0; M.ZETA=5.0; M.DECAY_LR=100.0; M.M_REGIONS=1000
LEVELS=exp.FDR_LEVELS; T,BURN,N_SIM,KOBS=exp.T,exp.BURN_IN,8,5; d=0.85
DOCS=["1949","2007","2092","215","2017"]
graphs={doc:exp.build_graph(doc) for doc in DOCS}
def setup(g,seed):
    n,truth=g['n'],g['truth']; rng=np.random.RandomState(seed); G=solve(g['D'],g['A']); B=np.eye(n)-d*G.T
    w=np.diag(1/np.sqrt(np.diag(g['D']))); Bs=np.eye(n)-d*w@g['A']@w
    Y=np.zeros(n); Y[list(rng.choice(truth,KOBS,replace=False))]=1
    u0=solve(B,np.ones(n)*(1-d)); ini=M.base_to_start(solve(Bs,Y))
    return B,Y,u0,ini,M.alpha_find(u0,Y,exp.grid)
# batch 비율: full, 1/2, 1/4, 1/8 (문서별 n에 비례)
FRACS=[("full",None),("half",0.5),("quarter",0.25),("eighth",0.125)]
exp.run_doc(graphs[DOCS[0]],1)
acc_ac={g:[] for g in LEVELS}; t0=time.time()
for doc in DOCS:
    g=graphs[doc]; n=g['n']
    for s in range(N_SIM):
        seed=hash((doc,s))%(2**31); B,Y,u0,ini,ae=setup(g,seed)
        cm=exp.componentwise_mcmc_numba(T,ini,n,exp.grid,ae,u0,B,Y,seed=seed+200000)
        for gl in LEVELS: acc_ac[gl].append(exp.eval_metrics(np.mean(M.inv_logit(cm['theta'])[BURN:T],axis=0),Y,g['truth'])[gl])
print(f"acMH 완료 | {int(time.time()-t0)}s",flush=True)
results={}
for tag,frac in FRACS:
    acc={g:[] for g in LEVELS}
    for doc in DOCS:
        g=graphs[doc]; n=g['n']; bs=None if frac is None else max(2,int(n*frac))
        for s in range(N_SIM):
            seed=hash((doc,s))%(2**31); B,Y,u0,ini,ae=setup(g,seed)
            np.random.seed(seed+100000)
            r=M.gibbs_mh(BURN,T,ini,n,g,Y,B,u0,ae,exp.grid,batch_size=bs,verbose=False)
            for gl in LEVELS: acc[gl].append(exp.eval_metrics(r['poster_pi_mn'],Y,g['truth'])[gl])
    results[tag]=acc
    print(f"[{tag:8}] F@0.2={np.mean([x['F'] for x in acc[0.2]]):.3f} F@0.25={np.mean([x['F'] for x in acc[0.25]]):.3f} | {int(time.time()-t0)}s",flush=True)
def mn(acc,g,key): return float(np.mean([x[key] for x in acc[g]]))
print(f"\n=== dense 미니배치 sweep (F-measure) ===")
print("config".ljust(10)+"".join(f"g{g:<5}" for g in LEVELS))
print("acMH".ljust(10)+"".join(f"{mn(acc_ac,g,'F'):<6.3f}" for g in LEVELS))
for tag,acc in results.items(): print(f"AW-{tag}".ljust(10)+"".join(f"{mn(acc,g,'F'):<6.3f}" for g in LEVELS))
print("\n[Recall / actual FDR @ 0.2 / 0.25]")
print(f"  {'acMH':<10} R={mn(acc_ac,0.2,'recall'):.3f}/{mn(acc_ac,0.25,'recall'):.3f}  FDR={mn(acc_ac,0.2,'realFDR'):.3f}/{mn(acc_ac,0.25,'realFDR'):.3f}")
for tag,acc in results.items():
    print(f"  AW-{tag:<7} R={mn(acc,0.2,'recall'):.3f}/{mn(acc,0.25,'recall'):.3f}  FDR={mn(acc,0.2,'realFDR'):.3f}/{mn(acc,0.25,'realFDR'):.3f}")
with open(os.path.join(HERE,"dense_minibatch.csv"),"w",newline="") as fh:
    w=csv.writer(fh); w.writerow(["config","gamma","precision","recall","F","realFDR"])
    w.writerows([["acMH",g]+[round(mn(acc_ac,g,k),4) for k in("precision","recall","F","realFDR")] for g in LEVELS])
    for tag,acc in results.items():
        for g in LEVELS: w.writerow([f"AW-{tag}",g]+[round(mn(acc,g,k),4) for k in("precision","recall","F","realFDR")])
print("\n저장: dense_minibatch.csv")
