"""Study 1B — 실제 U(θ) (400차원 사후 에너지) 위에서 샘플러 탐색 경로 진단.

400차원이라 지형 자체는 못 그리므로 간접 지표:
  (A) band-over-iteration 궤적: x=반복수, y=1000개 에너지 영역(band). AWSGLD 는 여러 band 를
      오가며 탐색(다봉 exploration), 갇힌 샘플러는 저역 band 에 눌러앉음.
  (B) 반복 빈도 히스토그램: 각 band 방문 빈도.

공통 에너지 U(θ) = −loglik(Y|θ,α*) + ‖B(θ−u_0)‖²/(2σ*²) (고정 α*,σ*) 로 두 샘플러를 동일 잣대로
1000개 band 에 매핑한다 (mixture 아님, 실제 사후 에너지 함수).
출력: energy_path_bands.png
"""
import os, sys, time
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import energy_diagnostics as E

_FP = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(_FP):
    fm.fontManager.addfont(_FP); plt.rcParams["font.family"] = fm.FontProperties(fname=_FP).get_name()
plt.rcParams["axes.unicode_minus"] = False

M = 1000            # 에너지 영역(band) 수 — Study 1B AWSGLD 와 동일
Tn = 10000
BURN = 1000

def main():
    graph, Y, B, u_0, z = E.load()
    ini = np.full(graph["n"], float(E.MU_N)); a = E.alpha_find(u_0, Y, E.GRID)
    E.T = Tn                                   # 궤적용으로 길게

    print(f"실제 U(θ) 탐색 경로 | n={graph['n']} M={M} T={Tn}", flush=True)
    t0 = time.time()
    th_aw = E.run_awsgld(graph, Y, B, u_0, ini, a, seed=0)
    print(f"  AWSGLD done {int(time.time()-t0)}s", flush=True); t0 = time.time()
    th_cy = E.run_sgld_family("cycSGLD", graph, Y, B, u_0, ini, a, seed=0)
    print(f"  cycSGLD done {int(time.time()-t0)}s", flush=True)

    U_aw = E.energy_trace_common(th_aw, Y, B, u_0, a)
    U_cy = E.energy_trace_common(th_cy, Y, B, u_0, a)

    # 두 샘플러 post-burn 에너지 범위로 공통 band 보정 (extended anchor)
    allU = np.concatenate([U_aw[BURN:], U_cy[BURN:]])
    lo, hi = allU.min(), allU.max(); rg = max(hi - lo, 1.0)
    emin = lo - 0.5 * rg; du = max((hi + 0.5 * rg - emin) / M, 1e-8)
    band = lambda U: np.clip(((U - emin) / du + 1).astype(int), 1, M)
    J_aw, J_cy = band(U_aw), band(U_cy)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8),
                             gridspec_kw={'width_ratios': [3, 1]})
    it = np.arange(Tn)
    for row, (Jt, Ut, nm, col) in enumerate([
            (J_aw, U_aw, "AWSGLD", "#2456A6"), (J_cy, U_cy, "cycSGLD", "#7f7f7f")]):
        ax = axes[row, 0]
        ax.plot(it, Jt, lw=0.5, color=col, alpha=0.85)
        ax.axvline(BURN, color="k", ls=":", lw=1, alpha=0.5)
        ax.set_ylim(0, M); ax.set_ylabel(f"{nm}\nenergy band J  (1…{M})")
        if row == 1: ax.set_xlabel("iteration")
        ax.set_title(f"({'a' if row==0 else 'b'}) {nm}: energy-band trajectory  "
                     f"(min U = {Ut[BURN:].min():.0f}, bands visited = {len(np.unique(Jt[BURN:]))})",
                     fontsize=10, fontweight="bold")
        ax.grid(alpha=0.15)
        # 오른쪽: 방문 빈도 히스토그램 (band 별)
        axh = axes[row, 1]
        frac = np.bincount(Jt[BURN:], minlength=M + 1)[1:M + 1] / (Tn - BURN)
        axh.barh(np.arange(1, M + 1), frac, height=1.0, color=col, alpha=0.8)
        axh.set_ylim(0, M); axh.set_xlabel("visit fraction")
        axh.set_title("visit frequency", fontsize=9)
        axh.grid(axis="x", alpha=0.2)
    fig.suptitle("Study 1B — sampler exploration path on the true posterior energy $U(\\theta)$ "
                 f"({graph['n']}-dim, {M} energy bands)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "energy_path_bands.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nAWSGLD: min U={U_aw[BURN:].min():.0f}, bands visited={len(np.unique(J_aw[BURN:]))}")
    print(f"cycSGLD: min U={U_cy[BURN:].min():.0f}, bands visited={len(np.unique(J_cy[BURN:]))}")
    print("저장:", out)

if __name__ == "__main__":
    main()
