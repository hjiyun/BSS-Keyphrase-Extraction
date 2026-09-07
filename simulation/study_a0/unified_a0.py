"""Study A0 (원본 U) 통일표 — study_1a 와 동일 2표·10시드 mean±std.
A0 데이터(study_1b data_generator 기본 config, n=100)에 study_1a/unified_scenarios 의
파이프라인(6샘플러·4연쇄·split-R̂·π가중·조기종료·원본 acMH)을 그대로 적용.
지표: 표1 R̂med/q95/max·ESS·θ̂(S/W/N)·π̂(S/W/N)·LowU / 표2 Spearman·Kendall·Top-k·NDCG@50·MSE_all.

사용: python3 unified_a0.py <seed>   → unified_a0_s{seed}.csv
"""
import os, sys, csv
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_1A = os.path.join(os.path.dirname(_HERE), "study_1a")
_1B = os.path.join(os.path.dirname(_HERE), "study_1b")
for p in (_1A, _1B):
    sys.path.insert(0, p)
import unified_scenarios as US2                 # study_1a 통일 파이프라인  # noqa: E402
import data_generator as DG                     # study_1b (A0 데이터)  # noqa: E402
from local_trap_landscape import PARAMS         # noqa: E402

US2.N = 100


def gen_a0(sc_cfg, seed):
    n = US2.N; rng = np.random.default_rng(seed)
    z, _ = DG.assign_groups(n, (PARAMS["rho_S"], PARAMS["rho_W"], PARAMS["rho_N"]), rng)
    ts = DG.sample_theta_star(z, PARAMS, rng)
    Yc, _ = DG.sample_Y(ts, PARAMS["alpha"], rng)
    Y, _ = DG.apply_label_conflict(Yc, z, DG.FLIP_RATE_S_TO_0, DG.FLIP_RATE_N_TO_1, rng)
    A = DG.build_sbm_graph(z, DG.P_IN, DG.P_OUT, rng)
    B, u_0, _ = DG.build_B_and_u0(A, DG.DAMPING)
    graph = {"n": n, "A": A, "D": np.diag(A.sum(1)), "group": z}
    a0 = US2.E.alpha_find(u_0, Y, US2.E.GRID)
    scen = dict(name="A0", mu_S=PARAMS["mu_S"], mu_W=PARAMS["mu_W"], mu_N=PARAMS["mu_N"],
                sigma_theta=PARAMS["sigma_theta"], alpha_true=PARAMS["alpha"], n_total=n)
    return graph, Y.astype(float), B, u_0, ts, scen, a0


US2.gen_scenario = gen_a0                         # 파이프라인의 데이터 생성만 A0 로 교체


def main():
    seed = int(sys.argv[1])
    US2.SEED = seed
    rows = US2.run_scenario({"scenario": {"name": "A0"}, "block_probs": None})
    out = os.path.join(_HERE, f"unified_a0_s{seed}.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["seed", "scenario", "method", "rhat_median", "rhat_q95", "rhat_max", "ess",
                    "th_S", "th_W", "th_N", "pi_S", "pi_W", "pi_N", "lowest_U",
                    "spearman", "kendall", "topk", "ndcg50", "mse_all", "T_stop"])
        for row in rows:
            w.writerow([seed] + row)
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
