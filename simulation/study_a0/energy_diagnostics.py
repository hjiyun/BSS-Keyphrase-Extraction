"""Study 1B local trap — 에너지 진단 2종 (AWSGHMC 논문 Fig 2 / Fig 4 유형).

Fig A (min-energy strip): 6 샘플러 × N seed 체인이 도달한 '가장 낮은 공통 에너지 U'.
  baseline 체인이 seed 마다 다른 에너지에 멈춤 = 서로 다른 mode. AWSGLD 는 매 seed 최저.
  공통 에너지: U(θ)=-loglik(Y,α*,θ)+||B(θ-u₀)||²/(2σ*²), 고정 (α*,σ*) → 샘플러 간 비교가능.

Fig B (energy-partition visitation): AWSGLD 가 M=100 energy band 를 방문한 빈도 히스토그램.
  flat-histogram 이 경계에 쌓이지 않고 내부 band 에 퍼지는지 확인.

출력: energy_minU_strip.png / energy_minU.csv / awsgld_partition_visit.png
실행: python3 energy_diagnostics.py [NSEED]   (기본 10)
"""
import os, sys, time
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))
CODE = os.path.join(ROOT, "code_JOC"); ORIG = os.path.join(CODE, "original")
for p in (CODE, ORIG):
    if p not in sys.path: sys.path.insert(0, p)
import keyphrase_functions_awsgld as kfa
from keyphrase_functions import gibbs_mh as acmh_gibbs_mh   # 원본 acMH

_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(_FONT):
    fm.fontManager.addfont(_FONT); plt.rcParams["font.family"] = fm.FontProperties(fname=_FONT).get_name()
plt.rcParams["axes.unicode_minus"] = False

# ── 설정 (study_1b 준수) ──
T = 5000; BURN = 500; BATCH = 100
GRID = (np.arange(10, 43) - 5) / np.arange(10, 43)
MU_N = -0.8
AWSGLD_FLOOR = 1.0; SGLD_FLOOR = 0.5
SGLD_TAU = 1.0; SGLD_LR = 0.02; QSGLD_LR = 0.3; CYCSGLD_LR = 0.01; CYC_CYCLES = 10
SGHMC_LR = 0.01; SGHMC_FRICTION = 0.1; SGHMC_TAU = 1.0
S2_STAR = 1.0                              # 공통 에너지의 σ*² (고정)
METHODS = ["AWSGLD", "SGHMC", "SGLD", "qSGLD", "cycSGLD", "acMH"]
COLORS = {"AWSGLD": "#2456A6", "SGHMC": "#7f7f7f", "SGLD": "#7f7f7f",
          "qSGLD": "#7f7f7f", "cycSGLD": "#7f7f7f", "acMH": "#7f7f7f"}


def inv_logit(x): return kfa.inv_logit(x)
def alpha_lk(theta, Y, a):
    t = np.clip((1 - a) * np.clip(inv_logit(theta), 1e-10, 1 - 1e-10), 1e-10, 1 - 1e-10)
    return np.sum(Y * np.log(t) + (1 - Y) * np.log(1 - t))
def alpha_find(theta, Y, grid): return grid[int(np.argmax([alpha_lk(theta, Y, a) for a in grid]))]


def load():
    d = np.load(os.path.join(_HERE, "data_seed0.npz"))
    n = int(d["n_total"])
    A = d["A"]; graph = {"n": n, "A": A, "D": np.diag(A.sum(1))}
    return graph, d["Y"].astype(float), d["B"], d["u_0"], np.array([str(x) for x in d["z"]])


def energy_trace_common(theta_store, Y, B, u_0, a_star):
    """공통 에너지 U(θ) 벡터화 — 모든 스텝에 대해 (T,) 반환."""
    Dd = theta_store - u_0[None, :]           # T×n
    BD = Dd @ B.T                              # (B(θ-u₀))ᵀ 각 행
    C = np.sum(BD * BD, axis=1)               # T
    temp = np.clip((1 - a_star) * np.clip(inv_logit(theta_store), 1e-10, 1 - 1e-10), 1e-10, 1 - 1e-10)
    loglik = np.sum(Y[None, :] * np.log(temp) + (1 - Y)[None, :] * np.log(1 - temp), axis=1)
    return -loglik + C / (2 * S2_STAR)


# ── 샘플러: 각자 theta_store 반환 ──
def run_awsgld(graph, Y, B, u_0, ini, alpha_est, seed):
    np.random.seed(seed)
    res = kfa.gibbs_mh(Burn_in=BURN, T=T, ini=ini.copy(), n=graph["n"], graph=graph, Y=Y,
                       B=B, u_0=u_0, alpha_est=alpha_est, grid=GRID,
                       batch_size=BATCH, sigma2_floor=AWSGLD_FLOOR, verbose=False)
    return res["theta_store"]

def run_acmh(graph, Y, B, u_0, ini, alpha_est, seed):
    np.random.seed(seed)
    res = acmh_gibbs_mh(Burn_in=BURN, T=T, ini=ini.copy(), n=graph["n"], graph=graph, Y=Y,
                        B=B, u_0=u_0, alpha_est=alpha_est, grid=GRID, verbose=False)
    return res["theta_store"]

def run_sgld_family(method, graph, Y, B, u_0, ini, alpha_est, seed):
    np.random.seed(seed)
    n = graph["n"]; BtB = B.T @ B
    ridge = 1e-6 * np.trace(BtB) / n
    P = np.linalg.solve(BtB + ridge * np.eye(n), np.eye(n)); P = 0.5 * (P + P.T)
    L = np.linalg.cholesky(P + 1e-10 * np.eye(n))
    theta = ini.copy(); theta_store = np.zeros((T, n)); v = np.zeros(n); a = alpha_est
    for t in range(T):
        C = (B @ (theta - u_0)) @ (B @ (theta - u_0))
        s2 = max(1.0 / np.random.gamma(n / 2 + 0.001, 1.0 / (C / 2 + 0.001)), SGLD_FLOOR)
        bidx = np.random.choice(n, size=BATCH, replace=False) if BATCH < n else None
        gU = kfa.grad_posterior_energy(Y, a, theta, u_0, B, s2, batch_idx=bidx, BtB=BtB)
        if method == "SGLD":
            eps = SGLD_LR / ((t + 1) ** 0.6 + 10); theta = theta - eps * gU + np.sqrt(2 * SGLD_TAU * eps) * np.random.randn(n)
        elif method == "qSGLD":
            eps = QSGLD_LR / ((t + 1) ** 0.6 + 10); theta = theta - eps * (P @ gU) + np.sqrt(2 * SGLD_TAU * eps) * (L @ np.random.randn(n))
        elif method == "cycSGLD":
            cl = max(1, T // CYC_CYCLES); beta = (t % cl) / cl
            eps = CYCSGLD_LR / 2 * (np.cos(np.pi * min(beta, 0.8)) + 1); tk = SGLD_TAU if beta >= 0.8 else SGLD_TAU / 1e4
            theta = theta - eps * gU + np.sqrt(2 * tk * eps) * np.random.randn(n)
        elif method == "SGHMC":
            eta = SGHMC_LR / ((t + 1) ** 0.6 + 10); theta = theta + v
            v = (1 - SGHMC_FRICTION) * v - eta * gU + np.sqrt(2 * SGHMC_FRICTION * eta * SGHMC_TAU) * np.random.randn(n)
        theta = np.clip(theta, -700, 700); theta_store[t] = theta; a = alpha_find(theta, Y, GRID)
    return theta_store

RUNNERS = {"AWSGLD": run_awsgld, "acMH": run_acmh,
           "SGLD": lambda *a: run_sgld_family("SGLD", *a), "qSGLD": lambda *a: run_sgld_family("qSGLD", *a),
           "cycSGLD": lambda *a: run_sgld_family("cycSGLD", *a), "SGHMC": lambda *a: run_sgld_family("SGHMC", *a)}


# ── Fig B: AWSGLD partition 방문 (M=100 traced) ──
def awsgld_partition_visit(graph, Y, B, u_0, ini, alpha_est, seed=0, Mr=100, Tn=20000):
    """kfa.gibbs_mh 로직 재현 + band J 기록. flat-histogram 방문 진단."""
    from scipy.stats import invgamma
    np.random.seed(seed); n = graph["n"]
    theta = ini.copy(); a = alpha_est
    BtB = B.T @ B; ridge = 1e-6 * np.trace(BtB) / n
    P = np.linalg.solve(BtB + ridge * np.eye(n), np.eye(n)); P = 0.5 * (P + P.T)
    Lc = np.linalg.cholesky(P + 1e-10 * np.eye(n))
    aw = np.arange(1, Mr + 1, dtype=float) / Mr
    warm = min(100, max(10, Tn // 20)); es = []; emin = du = None; J = Mr - 1
    TAU = kfa.TAU; ZETA = kfa.ZETA; DECAY = kfa.DECAY_LR
    Jhist = []
    for t in range(Tn):
        C = (B @ (theta - u_0)) @ (B @ (theta - u_0))
        s2 = max(invgamma.rvs(n / 2 + 0.001, scale=C / 2 + 0.001), AWSGLD_FLOOR)
        eps = 0.3 / ((t + 1) ** 0.6 + 10)
        Ut = kfa.posterior_energy(Y, a, theta, u_0, B, s2)
        bidx = np.random.choice(n, size=BATCH, replace=False) if BATCH < n else None
        gU = kfa.grad_posterior_energy(Y, a, theta, u_0, B, s2, batch_idx=bidx, BtB=BtB)
        if t < warm:
            es.append(Ut); gm = 1.0
            if t == warm - 1:
                e0, e1 = np.min(es), np.max(es); er = max(e1 - e0, 1.0)
                emin = e0 - 0.5 * er; emax = e1 + 0.5 * er; du = max((emax - emin) / Mr, 1e-8)
        else:
            J = int(np.clip((Ut - emin) / du + 1, 1, Mr - 1)); Jhist.append(J)
            gm = np.clip(1 + (ZETA * TAU / du) * (np.log(aw[J] + 1e-12) - np.log(aw[J - 1] + 1e-12)), 0.1, 10.0)
        theta = theta - eps * gm * (P @ gU) + np.sqrt(2 * TAU * eps) * (Lc @ np.random.randn(n))
        theta = np.clip(theta, -700, 700)
        if t >= warm:
            cw = aw[J]; aw[J:] = aw[J:] + min(1, DECAY / ((t + 1) ** 0.75 + 1000)) * cw * (1 - aw[J:])
            aw[:J] = aw[:J] - min(1, DECAY / ((t + 1) ** 0.75 + 1000)) * cw * aw[:J]; aw = np.clip(aw, 1e-10, 1)
        a = alpha_find(theta, Y, GRID)
    return np.array(Jhist), Mr


def main():
    NSEED = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    graph, Y, B, u_0, z = load()
    ini = np.full(graph["n"], float(MU_N))          # bad init θ⁰=μ_N
    a_star = alpha_find(u_0, Y, GRID)
    print(f"Study1B 에너지 진단 | n={graph['n']} Y=1:{int(Y.sum())} α*={a_star:.3f} σ*²={S2_STAR} | {NSEED}시드", flush=True)

    # ---- Fig B 먼저 (빠름) ----
    print("[Fig B] AWSGLD partition 방문 (M=100, T=20000)...", flush=True)
    t0 = time.time()
    Jhist, Mr = awsgld_partition_visit(graph, Y, B, u_0, ini, a_star, seed=0)
    frac = np.bincount(Jhist, minlength=Mr + 1)[1:Mr + 1] / len(Jhist)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(np.arange(1, Mr + 1), frac, width=1.0, color="#2E9E5B", edgecolor="none")
    ax.set_xlabel("partition band J"); ax.set_ylabel("visit fraction")
    ax.set_title(f"Study 1B AWSGLD energy-partition visitation ({Mr} bands)", fontsize=11)
    ax.set_xlim(0, Mr); ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(_HERE, "awsgld_partition_visit.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  저장 awsgld_partition_visit.png ({int(time.time()-t0)}s, J표본 {len(Jhist)})", flush=True)

    # ---- Fig A: 6 샘플러 × NSEED min-energy ----
    print("[Fig A] 6 샘플러 min-energy strip...", flush=True)
    minU = {m: [] for m in METHODS}; t0 = time.time()
    for m in METHODS:
        for s in range(NSEED):
            ts = RUNNERS[m](graph, Y, B, u_0, ini, a_star, 1000 + s)
            U = energy_trace_common(ts, Y, B, u_0, a_star)
            minU[m].append(float(U[BURN:].min()))
        print(f"  {m:>8}: min U mean={np.mean(minU[m]):.0f} (min {np.min(minU[m]):.0f}, max {np.max(minU[m]):.0f})  {int(time.time()-t0)}s", flush=True)

    import csv
    with open(os.path.join(_HERE, "energy_minU.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["sampler", "seed", "minU"])
        for m in METHODS:
            for s in range(NSEED): w.writerow([m, s, round(minU[m][s], 2)])

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    rng = np.random.RandomState(0)
    for i, m in enumerate(METHODS):
        vals = np.array(minU[m]); xs = i + (rng.rand(len(vals)) - 0.5) * 0.28
        ax.scatter(xs, vals / 1e3, s=42, color=COLORS[m], alpha=0.75, edgecolor="white", linewidth=0.6, zorder=3)
        mean = vals.mean() / 1e3
        ax.plot([i - 0.28, i + 0.28], [mean, mean], color=COLORS[m], lw=3, zorder=4)
        ax.text(i, vals.max() / 1e3 + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.0 + 0.5, f"{mean:.1f}",
                ha="center", va="bottom", fontsize=9, color=COLORS[m], fontweight="bold")
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels([m if m != "AWSGLD" else "AWSGLD" for m in METHODS],
                       fontweight="bold" if False else "normal")
    lbls = ax.get_xticklabels(); lbls[0].set_color(COLORS["AWSGLD"]); lbls[0].set_fontweight("bold")
    ax.set_ylabel(r"Lowest energy $U$ reached  ($\times 10^3$)")
    ax.set_title("Study 1B local trap — lowest energy reached across seeds", fontsize=11)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(_HERE, "energy_minU_strip.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  저장 energy_minU_strip.png, energy_minU.csv ({int(time.time()-t0)}s)", flush=True)


if __name__ == "__main__":
    main()
