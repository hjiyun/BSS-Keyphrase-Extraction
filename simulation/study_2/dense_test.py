"""dense 문서: 분포 진단(꼬리/단봉) + acMH vs AWSGLD(full-batch). N_SIM=8, k=5."""
import os, csv, time, importlib.util
import numpy as np
from numpy.linalg import solve
from scipy.stats import kurtosis
HERE=os.path.dirname(os.path.abspath(__file__))
spec=importlib.util.spec_from_file_location("exp",os.path.join(HERE,"acmh_vs_awsgld_4to10.py")); exp=importlib.util.module_from_spec(spec); spec.loader.exec_module(exp)
M=exp.M; M.TAU=1.0; M.ZETA=5.0; M.DECAY_LR=100.0; M.M_REGIONS=1000
LEVELS=exp.FDR_LEVELS; T,BURN,N_SIM,KOBS=exp.T,exp.BURN_IN,8,5; d=0.85
DOCS=["1949","2007","2092","215","2017"]
def setup(graph,seed):
    n,truth=graph['n'],graph['truth']
    rng=np.random.RandomState(seed); G=solve(graph['D'],graph['A']); B=np.eye(n)-d*G.T
    w=np.diag(1/np.sqrt(np.diag(graph['D']))); Bs=np.eye(n)-d*w@graph['A']@w
    Y=np.zeros(n); Y[list(rng.choice(truth,KOBS,replace=False))]=1
    u0=solve(B,np.ones(n)*(1-d)); ini=M.base_to_start(solve(Bs,Y))
    return B,Y,u0,ini,M.alpha_find(u0,Y,exp.grid)
def pc1(th): mu=th.mean(0);X=th-mu;_,_,Vt=np.linalg.svd(X,full_matrices=False);return X@Vt[0]
acc={m:{g:[] for g in LEVELS} for m in ["acMH","AWSGLD"]}
diag=[]
exp.run_doc(exp.build_graph(DOCS[0]),1); t0=time.time()
for doc in DOCS:
    g=exp.build_graph(doc); n,truth=g['n'],len(g['truth'])
    acstd=[]; ackurt=[]; awstd=[]
    for s in range(N_SIM):
        seed=hash((doc,s))%(2**31); B,Y,u0,ini,ae=setup(g,seed)
        cm=exp.componentwise_mcmc_numba(T,ini,n,exp.grid,ae,u0,B,Y,seed=seed+200000)
        p=pc1(cm['theta'][BURN:T]); acstd.append(p.std()); ackurt.append(kurtosis(p))
        for gl in LEVELS: acc["acMH"][gl].append(exp.eval_metrics(np.mean(M.inv_logit(cm['theta'])[BURN:T],axis=0),Y,truth=g['truth'])[gl])
        np.random.seed(seed+100000); rb=M.gibbs_mh(BURN,T,ini,n,g,Y,B,u0,ae,exp.grid,verbose=False)
        awstd.append(pc1(rb['theta_store'][BURN:T]).std())
        for gl in LEVELS: acc["AWSGLD"][gl].append(exp.eval_metrics(rb['poster_pi_mn'],Y,truth=g['truth'])[gl])
    diag.append((doc,n,truth,np.mean(acstd),np.mean(ackurt),np.mean(awstd)))
    print(f"[{doc}] n={n} kw={truth} | acMH std={np.mean(acstd):.0f} kurt={np.mean(ackurt):.1f} | AWSGLD std={np.mean(awstd):.1f} | {int(time.time()-t0)}s",flush=True)
def mn(m,g,key): return float(np.mean([x[key] for x in acc[m][g]]))
print("\n=== dense 5문서 분포 진단 ===")
print(f"{'doc':>6} {'n':>4} {'kw':>3} {'acMH탐색std':>11} {'kurtosis':>9} {'AWSGLD std':>11}")
for doc,n,tr,acs,ack,aws in diag: print(f"{doc:>6} {n:>4} {tr:>3} {acs:>11.0f} {ack:>9.1f} {aws:>11.1f}")
print("\n=== dense 5문서 평균 성능 (acMH vs AWSGLD) ===")
for label,key in [("Precision","precision"),("Recall","recall"),("F-measure","F"),("actual FDR","realFDR")]:
    print(f"\n[{label}]"); print(f"{'γ':>5} | {'acMH':>8} | {'AWSGLD':>8}")
    for g in LEVELS: print(f"{g:>5} | {mn('acMH',g,key):>8.3f} | {mn('AWSGLD',g,key):>8.3f}")
with open(os.path.join(HERE,"dense_test.csv"),"w",newline="") as fh:
    w=csv.writer(fh); w.writerow(["gamma","sampler","precision","recall","F","realFDR"])
    for g in LEVELS:
        for m in ["acMH","AWSGLD"]: w.writerow([g,m]+[round(mn(m,g,key),4) for key in("precision","recall","F","realFDR")])
print("\n저장: dense_test.csv")
