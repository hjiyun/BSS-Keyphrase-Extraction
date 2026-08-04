"""수렴 진단 (AWSGLD vs SGHMC) — Study 1B 의 다봉 그래프/데이터 위에서.

원본 awsgld_convergence.py 는 study_1a 시나리오로 그래프를 자체 생성하지만, 여기서는
study_1b/data_seed{S}.npz (n=400, SBM + label conflict 로 유도한 다봉 posterior) 를
그대로 불러와 같은 4-panel 진단을 그린다.

  (a) 대표 component(S/W/N 각 3개) θ trace
  (b) ||θ_k - θ̄||₂ trace
  (c) U(x_k) energy trace  ← 다봉 exploration 증거 (bad init 에서 basin 탈출)
  (d) running posterior mean MSE vs θ*

Study 1B 충실 재현: bad init θ⁰=μ_N, AWSGLD σ²_floor=1.0, batch=100, T=5000, burn=500.

실행:  python3 awsgld_convergence_1b.py [seed]   (기본 seed=0)
       python3 awsgld_convergence_1b.py 0 --stdinit   (bad init 대신 BSS 표준 초기화)
"""
import os, sys, time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import awsgld_convergence as C          # 트레이스/플롯 함수 재사용
sys.path.insert(0, os.path.join(_HERE, "..", "..", "code_JOC"))
import keyphrase_functions_awsgld as kfa

# Study 1B 세팅으로 모듈 전역 맞춤
C.T = 5000
C.BURN_IN = 500
C.BATCH_SIZE = 100
C.SGHMC_SIGMA2_FLOOR = 0.5      # 1b sgld_only 와 동일
AWSGLD_SIGMA2_FLOOR = 1.0       # 1b acmh_vs_awsgld 와 동일
MU_N = -0.8                     # 1b PARAMS mu_N (bad init 값)
DATA_DIR = os.path.join(_HERE, "..", "study_1b")


def load_1b(seed):
    d = np.load(os.path.join(DATA_DIR, f"data_seed{seed}.npz"))
    z = np.array([str(x) for x in d["z"]])
    n = int(d["n_total"])
    A = d["A"]
    graph = {"n": n, "A": A, "D": np.diag(A.sum(axis=1)), "group": z}
    return graph, d["theta_star"], d["Y"].astype(float), d["B"], d["u_0"], z


def make_init(B, u_0, Y, n, group, std_init=False):
    grid = C.GRID
    alpha_est = C.alpha_find(u_0, Y, grid)
    if std_init:
        ini = C.base_to_start(np.linalg.solve(B, Y))   # BSS 표준 초기화 (base_to_start)
    else:
        ini = np.full(n, float(MU_N))                   # bad init θ⁰=μ_N (1b 방식)
    return {"B": B, "u_0": u_0, "ini": ini, "alpha_est": alpha_est}


def run_awsgld_1b(graph, Y, init_state):
    """1b 충실: σ²_floor=1.0 로 AWSGLD 호출 (모듈 기본 0.5 override)."""
    np.random.seed(C.SEED)
    res = kfa.gibbs_mh(
        Burn_in=C.BURN_IN, T=C.T, ini=init_state["ini"], n=graph["n"], graph=graph,
        Y=Y, B=init_state["B"], u_0=init_state["u_0"],
        alpha_est=init_state["alpha_est"], grid=C.GRID,
        batch_size=C.BATCH_SIZE, sigma2_floor=AWSGLD_SIGMA2_FLOOR, verbose=False,
    )
    theta_store = res["theta_store"]
    alpha_store = np.array([C.alpha_find(theta_store[t], Y, C.GRID) for t in range(C.T)])
    return {"theta_store": theta_store, "sigma2_store": res["sigma2_store"], "alpha_store": alpha_store}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    std_init = "--stdinit" in sys.argv
    awsgld_only = "--awsgld-only" in sys.argv
    seed = int(args[0]) if args else 0
    suffix = "_awsgld" if awsgld_only else ""
    out_path = os.path.join(_HERE, f"awsgld_convergence_1b_seed{seed}{suffix}.png")

    print("=" * 72)
    print(f"Study 1B 그래프 수렴 진단 | seed={seed} | "
          f"init={'BSS표준' if std_init else 'bad(θ⁰=μ_N)'} | T={C.T} burn={C.BURN_IN}")
    print("=" * 72)

    graph, theta_star, Y, B, u_0, z = load_1b(seed)
    init_state = make_init(B, u_0, Y, graph["n"], z, std_init=std_init)
    print(f"n={graph['n']}  Y=1: {int(Y.sum())}  그룹=(S:{(z=='S').sum()}, W:{(z=='W').sum()}, N:{(z=='N').sum()})")

    t0 = time.perf_counter()
    awsgld_res = run_awsgld_1b(graph, Y, init_state)
    print(f"AWSGLD done {time.perf_counter()-t0:.1f}s")
    sghmc_res = None
    if not awsgld_only:
        t0 = time.perf_counter()
        sghmc_res = C.run_sghmc_with_traces(graph, Y, init_state)
        print(f"SGHMC done {time.perf_counter()-t0:.1f}s")

    scenario = {"name": f"Study1B_multimodal_seed{seed}", "n_total": graph["n"]}
    C.plot_convergence(awsgld_res, theta_star, z, B, Y, u_0, out_path, scenario, sghmc_res=sghmc_res)


if __name__ == "__main__":
    main()
