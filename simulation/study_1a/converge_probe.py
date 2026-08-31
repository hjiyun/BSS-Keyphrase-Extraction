"""Difficult 장기 수렴 궤적 — AWSGLD vs acMH(+qSGLD).
긴 horizon 에서 running split-R̂(med/q95/max)을 체크포인트마다 기록해
AWSGLD 가 acMH 보다 낮아지는(=더 잘 수렴하는) 지점을 찾는다.
"""
import sys, time
import numpy as np
import unified_scenarios as U
import langevin_methods_comparison as LMC

SCEN = int(sys.argv[1]) if len(sys.argv) > 1 else 2      # 0 Easy /1 Mod /2 Difficult /3 Sparse
TMAX = int(sys.argv[2]) if len(sys.argv) > 2 else 40000
CKPT = 4000
U.T_MAX = TMAX + 10; U.BURN = 2000
METHODS = ["acMH", "qSGLD", "AWSGLD"]

sc = LMC.SCENARIOS[SCEN]
graph, Y, B, u_0, ts, scen, a0 = U.gen_scenario(sc, U.SEED); n = U.N
BtB = B.T @ B; ridge = 1e-6 * np.trace(BtB) / n
P = np.linalg.solve(BtB + ridge * np.eye(n), np.eye(n)); P = 0.5 * (P + P.T)
Lc = np.linalg.cholesky(P + 1e-10 * np.eye(n))
BtB_inv = np.linalg.solve(BtB + 1e-8 * np.eye(n), np.eye(n))
name = scen["name"].replace("Controlled", "").replace("_v2_OptB", "").replace("_v2", "")


def McholScaled(s2):
    return np.linalg.cholesky(BtB_inv * s2 * 4.0 / n + 1e-10 * np.eye(n))


INITS = [scen["mu_N"], scen["mu_W"], scen["mu_S"], "rand"]


def mkini(ci, val):
    return (np.random.RandomState(7000 + ci).randn(n) * 1.5 if val == "rand" else np.full(n, float(val)))


ctx = dict(Y=Y, B=B, u_0=u_0, BtB=BtB, P=P, Lc=Lc, a0=a0, BtB_inv=BtB_inv,
           McholScaled=McholScaled, INITS=INITS, mkini=mkini)

print(f"[{name}] 장기 수렴 궤적  μ={scen['mu_S']}/{scen['mu_W']}/{scen['mu_N']} α={scen['alpha_true']}  T_MAX={TMAX}", flush=True)
print(f"{'T':>6} | " + " | ".join(f"{m:>18}" for m in METHODS))
print(f"{'':>6} | " + " | ".join(f"{'med/q95/max':>18}" for m in METHODS))

states = {m: [U.new_state(m, mkini(ci, v), ctx, seed=100 * U.SEED + ci) for ci, v in enumerate(INITS)] for m in METHODS}
t0 = time.time()
for T in range(CKPT, TMAX + 1, CKPT):
    line = f"{T:>6} | "
    cells = []
    for m in METHODS:
        for st in states[m]:
            U.advance(st, CKPT, ctx)
        posts, wts = U.collect(states[m])
        R = U.split_rhat_coords(posts, wts)
        cells.append(f"{np.median(R):.2f}/{np.quantile(R,0.95):.2f}/{np.nanmax(R):.2f}")
    print(line + " | ".join(f"{c:>18}" for c in cells) + f"   ({int(time.time()-t0)}s)", flush=True)
