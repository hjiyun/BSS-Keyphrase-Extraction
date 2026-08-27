"""acMH 4-chain 종합표 지표만 계산해 full_metrics_n100.csv 에 병합 (strip은 이미 있음).
full_metrics_n100.py 와 동일 설정(4-chain, T=20000, BURN=2000)."""
import os, sys, time, csv
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "_archive"))
import energy_diagnostics as E
import data_generator as DG
from local_trap_landscape import PARAMS
from extra_metrics import ess_per_node

HERE = os.path.dirname(os.path.abspath(__file__))
N = 100; E.T = 20000; E.BURN = 2000; E.BATCH = 50
T = E.T; BURN = E.BURN; SEED = 0
INITS = [-0.8, 1.0, 2.5, "rand"]


def gen(n, seed):
    rng = np.random.default_rng(seed)
    z, _ = DG.assign_groups(n, (PARAMS["rho_S"], PARAMS["rho_W"], PARAMS["rho_N"]), rng)
    ts = DG.sample_theta_star(z, PARAMS, rng)
    Yc, _ = DG.sample_Y(ts, PARAMS["alpha"], rng)
    Y, _ = DG.apply_label_conflict(Yc, z, DG.FLIP_RATE_S_TO_0, DG.FLIP_RATE_N_TO_1, rng)
    A = DG.build_sbm_graph(z, DG.P_IN, DG.P_OUT, rng)
    B, u_0, _ = DG.build_B_and_u0(A, DG.DAMPING)
    return {"n": n, "A": A, "D": np.diag(A.sum(1))}, Y.astype(float), B, u_0, ts


def ndcg_at_k(ts, th, k):
    rel = np.argsort(np.argsort(ts)).astype(float) / max(len(ts) - 1, 1)
    pred = np.argsort(th)[::-1][:k]; ideal = np.argsort(ts)[::-1][:k]
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    d = np.sum((2 ** rel[pred] - 1) * disc); i = np.sum((2 ** rel[ideal] - 1) * disc)
    return float(d / i) if i > 0 else 0.0


def split_rhat(posts):
    """split-R̂ (raw): 각 체인 반으로 갈라 2M sub-chain."""
    hm = []; hv = []; hL = None
    for post in posts:
        L = post.shape[0]; h = L // 2; hL = h
        for sl in (slice(0, h), slice(h, 2 * h)):
            seg = post[sl]; hm.append(seg.mean(0)); hv.append(seg.var(0, ddof=1))
    hm = np.array(hm); hv = np.array(hv); M2 = len(hm)
    gm = hm.mean(0); Bo = ((hm - gm) ** 2).sum(0) / (M2 - 1); W = hv.mean(0)
    return np.sqrt(np.clip(((hL - 1) / hL * W + Bo) / np.maximum(W, 1e-12), 0, None))


def main():
    t0 = time.time()
    graph, Y, B, u_0, ts = gen(N, SEED); n = N; a0 = E.alpha_find(u_0, Y, E.GRID); L = T - BURN
    posts = []; ess_list = []; pn = 0.0; pd = 0.0
    for ci, val in enumerate(INITS):
        ini = (np.random.RandomState(7000 + 100 * SEED + ci).randn(n) * 1.5 if val == "rand" else np.full(n, float(val)))
        ths = E.run_acmh(graph, Y, B, u_0, ini, a0, 100 * SEED + ci)
        post = ths[BURN:]; posts.append(post)
        ess_list.append(np.nanmedian(ess_per_node(post))); pn += post.sum(0); pd += post.shape[0]
        print(f"  chain {ci} done ({int(time.time()-t0)}s)", flush=True)
    R = split_rhat(posts)
    th = pn / pd; sp = spearmanr(ts, th).statistic; mse = float(np.mean((th - ts) ** 2))
    row = ["acMH", round(sp, 4), round(mse, 4), round(ndcg_at_k(ts, th, 5), 4), round(ndcg_at_k(ts, th, 10), 4),
           round(ndcg_at_k(ts, th, 20), 4), round(ndcg_at_k(ts, th, 50), 4),
           round(float(np.nanmax(R)), 4), round(float(np.median(R)), 4), round(float(np.median(ess_list)), 2), ""]

    p = os.path.join(HERE, "full_metrics_n100.csv"); lines = list(csv.reader(open(p)))
    header, body = lines[0], [r for r in lines[1:] if r and r[0] != "acMH"]
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(header); w.writerow(row); [w.writerow(r) for r in body]
    print(f"\nacMH(4-chain) 병합: Spear {sp:.3f} MSE {mse:.2f} R̂max {np.nanmax(R):.2f} R̂med {np.median(R):.3f} "
          f"ESS {np.median(ess_list):.1f} ({int(time.time()-t0)}s)")


if __name__ == "__main__":
    main()
