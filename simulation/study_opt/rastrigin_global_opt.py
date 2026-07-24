"""전역최소 탐색 데모 (AWSGLD 논문 Fig 2 유형) — Rastrigin 다봉 함수.

목적: BSS 사후분포는 전역최소=prior붕괴라 부적합. 대신 고전 다봉 벤치마크(Rastrigin)에서
'각 샘플러가 도달한 최저 함수값'을 seed별로 찍어, AWSGLD 가 전역최소(0)를 가장 잘/일관되게
찾는지 본다. 참고 논문 global_optimization_of_functions 의 setup 재현.

함수: Rastrigin f(x)=10d+Σ(xᵢ²-10cos(2πxᵢ)), 전역최소 f=0 at x=0, 도메인 [-5.12,5.12].
      수많은 국소최소('groups')에 둘러싸임 → greedy/fixed-temp 는 갇힘.
샘플러: AWSGLD / SGHMC / SGLD / qSGLD / cycSGLD / MH (모두 exp(-f/T) 겨냥, 고온 탐색).
지표: max_iters 동안 도달한 min f. 10 seed strip plot.
"""
import os, sys, time
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIM = 20; LO, HI = -5.12, 5.12
MAX_ITERS = 50000
NSEED = 10
T_LANG = 5.0                       # Langevin 온도 (참고 논문 T=5)
LR = 5e-4                          # 참고 논문 Rastrigin lr
GNOISE = (HI - LO) / 1e3           # stochastic grad 노이즈 (참고 논문과 동일)
# AWSGLD flat-histogram
ZETA = 0.02; PART = 100; DIV = 3.0; DECAY = 200.0
METHODS = ["AWSGLD", "SGHMC", "SGLD", "qSGLD", "cycSGLD", "MH"]

def f(x): return 10 * DIM + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
def gradf(x): return 2 * x + 20 * np.pi * np.sin(2 * np.pi * x)
def sgrad(x, rng): return gradf(x) + rng.normal(size=DIM) * GNOISE
def clip_domain(x): return np.clip(x, LO, HI)

def run_sgld(rng, x, method):
    v = np.zeros(DIM); fmin = f(x); fric = 0.1
    for it in range(1, MAX_ITERS + 1):
        g = sgrad(x, rng)
        if method == "SGLD":
            x = x - LR * g + np.sqrt(2 * LR * T_LANG) * rng.normal(size=DIM)
        elif method == "qSGLD":                     # 그래프 없음 → 대각 precond (곡률 정규화)
            P = 1.0 / (np.abs(gradf(x)) + 5.0)
            x = x - LR * P * g + np.sqrt(2 * LR * T_LANG * P) * rng.normal(size=DIM)
        elif method == "cycSGLD":
            cl = max(1, MAX_ITERS // 10); beta = (it % cl) / cl
            lr_c = LR * (np.cos(np.pi * min(beta, 0.8)) + 1); tk = T_LANG if beta >= 0.8 else T_LANG / 1e2
            x = x - lr_c * g + np.sqrt(2 * lr_c * tk) * rng.normal(size=DIM)
        elif method == "SGHMC":
            x = x + v
            v = (1 - fric) * v - LR * g + np.sqrt(2 * fric * LR * T_LANG) * rng.normal(size=DIM)
        x = clip_domain(x); fx = f(x)
        if fx < fmin: fmin = fx
    return fmin

def run_mh(rng, x):
    fx = f(x); fmin = fx; step = 0.5
    for it in range(1, MAX_ITERS + 1):
        xp = clip_domain(x + step * rng.normal(size=DIM)); fp = f(xp)
        if np.log(rng.uniform() + 1e-300) < -(fp - fx) / T_LANG:
            x, fx = xp, fp
        if fx < fmin: fmin = fx
    return fmin

def run_awsgld(rng, x):
    fmin = f(x); Gcum = np.array([0.1] * PART); J = PART - 1; fstar = 0.0
    def aw_idx(b): return min(max(int((f(b) - fstar) / DIV + 1), 1), PART - 1)
    for it in range(1, MAX_ITERS + 1):
        gm = 1 + ZETA * T_LANG * (np.log(Gcum[J]) - np.log(Gcum[J - 1])) / DIV
        g = sgrad(x, rng)
        x = x - LR * gm * g + np.sqrt(2 * LR * T_LANG) * rng.normal(size=DIM)
        x = clip_domain(x)
        decay = min(1.0, DECAY / (it ** 0.75 + 1000.0))
        J = aw_idx(x)
        Gcum[J:] = Gcum[J:] + decay * Gcum[J] * (1.0 - Gcum[J:])
        Gcum[:J] = Gcum[:J] + decay * (Gcum[J] * (-Gcum[:J]))
        fx = f(x)
        if fx < fmin: fmin = fx
    return fmin

def main():
    print(f"Rastrigin 전역최소 탐색 | dim={DIM} iters={MAX_ITERS} 전역최소=0 | {NSEED}시드", flush=True)
    res = {m: [] for m in METHODS}; t0 = time.time()
    for s in range(NSEED):
        rng = np.random.RandomState(1000 + s)
        x0 = rng.uniform(LO, HI, size=DIM)          # 동일 seed → 동일 시작점 (샘플러 간 공정)
        for m in METHODS:
            r = np.random.RandomState(1000 + s)     # 각 샘플러 동일 난수열
            x0m = r.uniform(LO, HI, size=DIM)
            if m == "AWSGLD": v = run_awsgld(r, x0m.copy())
            elif m == "MH": v = run_mh(r, x0m.copy())
            else: v = run_sgld(r, x0m.copy(), m)
            res[m].append(v)
        print(f"  seed {s}: " + " ".join(f"{m}={res[m][-1]:.1f}" for m in METHODS) + f"  {int(time.time()-t0)}s", flush=True)

    import csv
    with open(os.path.join(os.path.dirname(__file__), "rastrigin_minf.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["sampler", "seed", "min_f"])
        for m in METHODS:
            for s in range(NSEED): w.writerow([m, s, round(res[m][s], 3)])

    print("\n=== 도달 최저 f (전역최소=0, 낮을수록 좋음) ===")
    order = sorted(METHODS, key=lambda m: np.mean(res[m]))
    for m in order:
        v = np.array(res[m]); print(f"  {m:>8}: mean={v.mean():7.2f}  (min {v.min():.2f}, max {v.max():.2f}, std {v.std():.2f})")

    fig, ax = plt.subplots(figsize=(8.5, 5))
    rng = np.random.RandomState(0)
    for i, m in enumerate(METHODS):
        v = np.array(res[m]); xs = i + (rng.rand(len(v)) - 0.5) * 0.28
        col = "#2456A6" if m == "AWSGLD" else "#7f7f7f"
        ax.scatter(xs, v, s=42, color=col, alpha=0.75, edgecolor="white", linewidth=0.6, zorder=3)
        ax.plot([i - 0.28, i + 0.28], [v.mean()] * 2, color=col, lw=3, zorder=4)
        ax.text(i, v.max() + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02, f"{v.mean():.0f}",
                ha="center", va="bottom", fontsize=9, color=col, fontweight="bold")
    ax.set_xticks(range(len(METHODS))); ax.set_xticklabels(METHODS)
    ax.get_xticklabels()[0].set_color("#2456A6"); ax.get_xticklabels()[0].set_fontweight("bold")
    ax.axhline(0, color="#2E9E5B", ls="--", lw=1, alpha=0.7)
    ax.set_ylabel("Lowest $f$ reached  (global min = 0)")
    ax.set_title(f"Rastrigin ({DIM}D) global-optimization — lowest value reached across seeds", fontsize=11)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(os.path.dirname(__file__), "rastrigin_minf_strip.png"), dpi=140, bbox_inches="tight")
    print("\n저장: rastrigin_minf_strip.png, rastrigin_minf.csv")

if __name__ == "__main__":
    main()
