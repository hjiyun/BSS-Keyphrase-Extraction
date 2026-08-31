"""원본 componentwise acMH의 Difficult 수렴(R̂)을 AWSGLD와 비교."""
import numpy as np, time
import unified_scenarios as U
import energy_diagnostics as E
import langevin_methods_comparison as LMC

sc = LMC.SCENARIOS[2]  # Difficult
graph, Y, B, u_0, ts, scen, a0 = U.gen_scenario(sc, 0); n = U.N
E.T = 20000; E.BURN = 2000; E.BATCH = 50
INITS = [scen["mu_N"], scen["mu_W"], scen["mu_S"], "rand"]


def mkini(ci, v):
    return (np.random.RandomState(7000 + ci).randn(n) * 1.5 if v == "rand" else np.full(n, float(v)))


t0 = time.time()
posts = []
for ci, v in enumerate(INITS):
    np.random.seed(ci)
    ths = E.run_acmh(graph, Y, B, u_0, mkini(ci, v), a0, ci)
    posts.append(ths[E.BURN:])
    print(f"  chain{ci} done ({int(time.time()-t0)}s)", flush=True)
R = U.split_rhat_coords(posts, [None] * 4)
print(f"원본 acMH(componentwise) Difficult T=20000: R̂ med/q95/max = "
      f"{np.median(R):.3f}/{np.quantile(R,0.95):.3f}/{np.nanmax(R):.3f}  ({int(time.time()-t0)}s)")
