"""acMH만 n=100·T=20000 으로 계산해 두 CSV(full_metrics, strip6)에 병합.
느려서 별도 배경 실행용. 완료 후 plotter 재실행하면 acMH 포함 그림 생성.
"""
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
T = E.T; BURN = E.BURN
INITS = [-0.8, 1.0, 2.5]; DATA_SEEDS = [0, 1, 2, 3, 4]; SEED = 0


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


def main():
    t0 = time.time()
    # ── full_metrics 용: 3 chain seed 0 ──
    graph, Y, B, u_0, ts = gen(N, SEED); n = N; a0 = E.alpha_find(u_0, Y, E.GRID); L = T - BURN
    means = []; withins = []; ess_list = []; pn = 0.0; pd = 0.0
    for ci, val in enumerate(INITS):
        ths = E.run_acmh(graph, Y, B, u_0, np.full(n, float(val)), a0, 100 * SEED + ci)
        post = ths[BURN:]; means.append(post.mean(0)); withins.append(post.var(0, ddof=1))
        ess_list.append(np.nanmedian(ess_per_node(post))); pn += post.sum(0); pd += post.shape[0]
        print(f"  metrics chain {ci} done ({int(time.time()-t0)}s)", flush=True)
    means = np.array(means); withins = np.array(withins); gm = means.mean(0)
    Bo = ((means - gm) ** 2).sum(0) / 2; W = withins.mean(0)
    R = np.sqrt(np.clip(((L - 1) / L * W + Bo) / np.maximum(W, 1e-12), 0, None))
    th = pn / pd; sp = spearmanr(ts, th).statistic; mse = float(np.mean((th - ts) ** 2))
    row_m = ["acMH", round(sp, 4), round(mse, 4), round(ndcg_at_k(ts, th, 5), 4), round(ndcg_at_k(ts, th, 10), 4),
             round(ndcg_at_k(ts, th, 20), 4), round(ndcg_at_k(ts, th, 50), 4),
             round(float(np.nanmax(R)), 4), round(float(np.median(R)), 4), round(float(np.median(ess_list)), 2), ""]

    # ── strip 용: 5 seed 단일 chain (cutoff 는 기존 strip6 CSV 값 사용) ──
    with open(os.path.join(HERE, "strip6_cutoff_n100.csv")) as fh:
        CUT = float(next(csv.DictReader(fh))["cutoff"])
    strip_rows = []
    for s in DATA_SEEDS:
        g, Yy, Bb, uu, _ = gen(N, s); aa = E.alpha_find(uu, Yy, E.GRID)
        ths = E.run_acmh(g, Yy, Bb, uu, np.full(N, float(E.MU_N)), aa, 1000 + s)
        U = E.energy_trace_common(ths, Yy, Bb, uu, aa)
        minU = float(U[BURN:].min()); low = max(CUT, minU); ess = float(np.nanmedian(ess_per_node(ths[BURN:])))
        strip_rows.append(["acMH", s, round(low, 1), round(minU, 1), round(ess, 2), CUT])
        print(f"  strip seed {s} done ({int(time.time()-t0)}s)", flush=True)

    # ── CSV 병합 (acMH 를 맨 앞에) ──
    for path, newrows in [("full_metrics_n100.csv", [row_m]), ("strip6_cutoff_n100.csv", strip_rows)]:
        p = os.path.join(HERE, path); lines = list(csv.reader(open(p)))
        header, body = lines[0], [r for r in lines[1:] if r and r[0] != "acMH"]
        with open(p, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(header); [w.writerow(r) for r in newrows]; [w.writerow(r) for r in body]
    print(f"\nacMH 병합 완료: full_metrics_n100.csv, strip6_cutoff_n100.csv ({int(time.time()-t0)}s)")
    print(f"  acMH metrics: Spear {sp:.3f} MSE {mse:.2f} R̂max {np.nanmax(R):.2f} ESS {np.median(ess_list):.1f}")


if __name__ == "__main__":
    main()
