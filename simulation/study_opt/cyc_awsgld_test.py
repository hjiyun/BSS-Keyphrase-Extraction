"""cyc-AWSGLD: AWSGLD 의 flat-histogram 에 cycSGLD 의 순환 학습률/온도를 결합.
아이디어: 탐색 단계(순환 앞부분)=큰 스텝+적응가중으로 장벽 탐색,
          하강 단계(순환 뒷부분)=작은 스텝+저온으로 골짜기 바닥까지.
Rastrigin 20D 에서 기존 6종 + cycAWSGLD 비교. 지표=도달 min U.
"""
import os, time
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIM = 20; LO, HI = -5.12, 5.12
MAX_ITERS = 50000; NSEED = 10
T_LANG = 5.0; LR = 5e-4; GNOISE = (HI - LO) / 1e3
ZETA = 0.02; PART = 100; DIV = 3.0; DECAY = 200.0
CYC_CYCLES = 10
METHODS = ["AWSGLD", "cycAWSGLD", "cycSGLD", "SGHMC", "SGLD", "qSGLD", "MH"]

def f(x): return 10 * DIM + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
def gradf(x): return 2 * x + 20 * np.pi * np.sin(2 * np.pi * x)
def sg(x, r): return gradf(x) + r.normal(size=DIM) * GNOISE
def clip(x): return np.clip(x, LO, HI)
def aw_idx(x): return min(max(int(f(x) / DIV + 1), 1), PART - 1)

def run_awsgld(r, x):
    fmin = f(x); Gc = np.array([0.1] * PART); J = PART - 1
    for it in range(1, MAX_ITERS + 1):
        gm = 1 + ZETA * T_LANG * (np.log(Gc[J]) - np.log(Gc[J - 1])) / DIV
        x = clip(x - LR * gm * sg(x, r) + np.sqrt(2 * LR * T_LANG) * r.normal(size=DIM))
        dec = min(1.0, DECAY / (it**0.75 + 1000.)); J = aw_idx(x)
        Gc[J:] += dec * Gc[J] * (1 - Gc[J:]); Gc[:J] += dec * (Gc[J] * (-Gc[:J]))
        fx = f(x); fmin = min(fmin, fx)
    return fmin

def run_cyc_awsgld(r, x):
    """AWSGLD + 순환 학습률/온도. 탐색 단계는 flat-histogram 강, 하강 단계는 저온·소스텝."""
    fmin = f(x); Gc = np.array([0.1] * PART); J = PART - 1
    cl = max(1, MAX_ITERS // CYC_CYCLES)
    for it in range(1, MAX_ITERS + 1):
        beta = (it % cl) / cl
        # 순환: 앞부분(beta<0.8)=탐색(큰 lr, 정상 온도), 뒷부분=하강(작은 lr, 저온)
        lr_c = LR * (np.cos(np.pi * min(beta, 0.8)) + 1)        # 0~2·LR
        T_c = T_LANG if beta < 0.8 else T_LANG / 1e2            # 탐색 고온 / 하강 저온
        gm = 1 + ZETA * T_c * (np.log(Gc[J]) - np.log(Gc[J - 1])) / DIV
        x = clip(x - lr_c * gm * sg(x, r) + np.sqrt(2 * lr_c * T_c) * r.normal(size=DIM))
        dec = min(1.0, DECAY / (it**0.75 + 1000.)); J = aw_idx(x)
        Gc[J:] += dec * Gc[J] * (1 - Gc[J:]); Gc[:J] += dec * (Gc[J] * (-Gc[:J]))
        fx = f(x); fmin = min(fmin, fx)
    return fmin

def run_sgld(r, x, method):
    fmin = f(x); v = np.zeros(DIM); fric = 0.1
    for it in range(1, MAX_ITERS + 1):
        g = sg(x, r)
        if method == "SGLD":
            eps = LR / ((it)**0.5 + 10); x = clip(x - eps * g + np.sqrt(2 * eps * T_LANG) * r.normal(size=DIM))
        elif method == "qSGLD":
            P = 1.0 / (np.abs(gradf(x)) + 5.0); eps = LR / ((it)**0.5 + 10)
            x = clip(x - eps * P * g + np.sqrt(2 * eps * T_LANG * P) * r.normal(size=DIM))
        elif method == "cycSGLD":
            cl = max(1, MAX_ITERS // CYC_CYCLES); beta = (it % cl) / cl
            lr_c = LR * (np.cos(np.pi * min(beta, 0.8)) + 1); tk = T_LANG if beta >= 0.8 else T_LANG / 1e2
            x = clip(x - lr_c * g + np.sqrt(2 * lr_c * tk) * r.normal(size=DIM))
        elif method == "SGHMC":
            eta = LR / ((it)**0.5 + 10); x = clip(x + v)
            v = (1 - fric) * v - eta * g + np.sqrt(2 * fric * eta * T_LANG) * r.normal(size=DIM)
        fmin = min(fmin, f(x))
    return fmin

def run_mh(r, x):
    fmin = f(x); step = 0.5
    for it in range(1, MAX_ITERS + 1):
        xp = clip(x + step * r.normal(size=DIM)); fp = f(xp)
        if np.log(r.uniform() + 1e-300) < -(fp - f(x)) / T_LANG: x = xp
        fmin = min(fmin, f(x))
    return fmin

def main():
    import csv
    res = {m: [] for m in METHODS}; t0 = time.time()
    for s in range(NSEED):
        r0 = np.random.RandomState(1000 + s); x0 = r0.uniform(LO, HI, DIM)
        for m in METHODS:
            r = np.random.RandomState(1000 + s); x = r.uniform(LO, HI, DIM)
            if m == "AWSGLD": v = run_awsgld(r, x)
            elif m == "cycAWSGLD": v = run_cyc_awsgld(r, x)
            elif m == "MH": v = run_mh(r, x)
            else: v = run_sgld(r, x, m)
            res[m].append(v)
        print(f"  seed {s}: " + " ".join(f"{m}={res[m][-1]:.0f}" for m in METHODS) + f"  {int(time.time()-t0)}s", flush=True)
    print("\n=== Rastrigin 20D 도달 min U (전역최소=0) ===")
    for m in sorted(METHODS, key=lambda m: np.mean(res[m])):
        v = np.array(res[m]); print(f"  {m:>10}: {v.mean():7.2f}  (min {v.min():.2f}, max {v.max():.2f}, std {v.std():.2f})")
    with open(os.path.join(os.path.dirname(__file__), "cyc_awsgld.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["sampler", "min_f_mean", "min_f_std", "min_f_best"])
        for m in METHODS:
            v = np.array(res[m]); w.writerow([m, round(v.mean(), 3), round(v.std(), 3), round(v.min(), 3)])
    fig, ax = plt.subplots(figsize=(9.5, 5)); rng = np.random.RandomState(0)
    for i, m in enumerate(METHODS):
        v = np.array(res[m]); xs = i + (rng.rand(len(v)) - 0.5) * 0.28
        col = "#C0392B" if m == "cycAWSGLD" else ("#2456A6" if m == "AWSGLD" else "#7f7f7f")
        ax.scatter(xs, v, s=40, color=col, alpha=0.8, edgecolor="white", linewidth=0.6, zorder=3)
        ax.plot([i - 0.28, i + 0.28], [v.mean()] * 2, color=col, lw=3, zorder=4)
        ax.text(i, v.max() + 3, f"{v.mean():.0f}", ha="center", va="bottom", fontsize=9, color=col, fontweight="bold")
    ax.axhline(0, color="#2E9E5B", ls="--", lw=1, alpha=0.7)
    ax.set_xticks(range(len(METHODS))); ax.set_xticklabels(METHODS, rotation=25, ha="right", fontsize=9)
    for lbl, m in zip(ax.get_xticklabels(), METHODS):
        if m == "cycAWSGLD": lbl.set_color("#C0392B"); lbl.set_fontweight("bold")
        if m == "AWSGLD": lbl.set_color("#2456A6"); lbl.set_fontweight("bold")
    ax.set_ylabel("Lowest $U$ reached  (global min = 0)")
    ax.set_title("Rastrigin (20D): cyc-AWSGLD vs AWSGLD vs baselines", fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(os.path.dirname(__file__), "cyc_awsgld.png"), dpi=140, bbox_inches="tight")
    print("\n저장: cyc_awsgld.png, cyc_awsgld.csv")

if __name__ == "__main__":
    main()
