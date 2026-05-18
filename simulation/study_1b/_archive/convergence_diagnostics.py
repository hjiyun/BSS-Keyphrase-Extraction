"""
Study 1B — 수렴 과정 시각화 (running group mean).

목적
----
acmh_vs_awsgld.py 가 저장한 ava_results.npz 의 chain들을 다시 읽어,
각 step t 까지 누적평균낸 θ̂_g(t) (= group g 의 running 추정치) 를
시간축에 그린다. 'group 별 mode 회복이 언제 일어나는가' 가 한 눈에 보임.

정의
----
- running_theta_hat_i(t) = mean(theta_store[1:t+1, i])  (각 노드별 누적평균)
- group_mean_g(t)        = mean over i in group g of running_theta_hat_i(t)
- 비교 기준: μ_g (S=+2.5, W=+1.0, N=-1.0)

출력
----
- convergence_group_mean.png : 2 row x 2 col 패널
    Row 1: chain 0 (bad init μ_N) — acMH | AWSGLD
    Row 2: 4 chain overlay — acMH | AWSGLD

실행
----
python3 simulation/study_1b/convergence_diagnostics.py
"""
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
from local_trap_landscape import PARAMS  # noqa: E402


_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(_FONT_PATH):
    fm.fontManager.addfont(_FONT_PATH)
    plt.rcParams["font.family"] = fm.FontProperties(fname=_FONT_PATH).get_name()
plt.rcParams["axes.unicode_minus"] = False


C_S = "#2F6DB2"
C_W = "#D85A30"
C_N = "#6B6B6B"
GROUP_COLOR = {"S": C_S, "W": C_W, "N": C_N}
METHODS = ("acMH", "AWSGLD")


def running_group_mean(theta_store, z, group):
    """theta_store: (T, n).  반환: (T,) — group g 내 노드들의 누적평균 평균."""
    mask = z == group
    sub = theta_store[:, mask]                     # (T, n_g)
    csum = np.cumsum(sub, axis=0)                  # (T, n_g)
    counts = np.arange(1, sub.shape[0] + 1)[:, None]
    running_per_node = csum / counts               # (T, n_g)
    return running_per_node.mean(axis=1)           # (T,)


def draw_panel(ax, theta_store, z, mu_map, title, burn_in=None,
               show_legend=True):
    for g in ("N", "W", "S"):
        curve = running_group_mean(theta_store, z, g)
        ax.plot(curve, color=GROUP_COLOR[g], lw=2.0,
                label=f"running mean({g})")
        ax.axhline(mu_map[g], color=GROUP_COLOR[g], ls="--", lw=1.2,
                   alpha=0.8,
                   label=fr"$\mu_{{{g}}}$={mu_map[g]:+.2f}")
    if burn_in is not None:
        ax.axvline(burn_in, color="black", ls=":", lw=1, alpha=0.5,
                   label=f"burn-in={burn_in}")
    ax.set_xlabel("MCMC step  t")
    ax.set_ylabel(r"running $\bar{\hat\theta}_g(t)$")
    ax.set_title(title, fontweight="bold", fontsize=10.5)
    ax.grid(True, alpha=0.15)
    if show_legend:
        ax.legend(fontsize=8, loc="lower right", ncol=2)


def draw_overlay_panel(ax, chains_theta_store, z, mu_map, title, burn_in=None,
                       show_legend=True):
    """모든 chain 을 같이 그림 — 같은 group 은 같은 색, chain 마다 옅은 톤."""
    n_chains = len(chains_theta_store)
    alphas = np.linspace(0.95, 0.45, n_chains)
    lws = [1.8] + [1.0] * (n_chains - 1)
    for c_idx, ts in enumerate(chains_theta_store):
        for g in ("N", "W", "S"):
            curve = running_group_mean(ts, z, g)
            label = (f"running mean({g})"
                     if c_idx == 0 else None)
            ax.plot(curve, color=GROUP_COLOR[g],
                    lw=lws[c_idx], alpha=alphas[c_idx],
                    label=label)
    for g in ("N", "W", "S"):
        ax.axhline(mu_map[g], color=GROUP_COLOR[g], ls="--", lw=1.2,
                   alpha=0.85,
                   label=fr"$\mu_{{{g}}}$={mu_map[g]:+.2f}")
    if burn_in is not None:
        ax.axvline(burn_in, color="black", ls=":", lw=1, alpha=0.5)
    ax.set_xlabel("MCMC step  t")
    ax.set_ylabel(r"running $\bar{\hat\theta}_g(t)$")
    ax.set_title(title, fontweight="bold", fontsize=10.5)
    ax.grid(True, alpha=0.15)
    if show_legend:
        ax.legend(fontsize=8, loc="lower right", ncol=2)


def main():
    npz_path = os.path.join(_THIS_DIR, "ava_results.npz")
    if not os.path.exists(npz_path):
        sys.exit(f"Not found: {npz_path}  → 먼저 acmh_vs_awsgld.py 실행 필요.")

    data = np.load(npz_path)
    z = np.array([str(x) for x in data["z"]])
    burn_in = int(data["BURN_IN"])
    seed = int(data["seed"])
    n = int((z == "S").sum() + (z == "W").sum() + (z == "N").sum())
    mu_map = {"S": PARAMS["mu_S"], "W": PARAMS["mu_W"], "N": PARAMS["mu_N"]}

    chains = {}
    for method in METHODS:
        chs = []
        c_idx = 0
        while f"{method}_chain{c_idx}_theta_store" in data.files:
            chs.append(data[f"{method}_chain{c_idx}_theta_store"])
            c_idx += 1
        chains[method] = chs
        print(f"[{method}] loaded {len(chs)} chains, shape={chs[0].shape}")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Row 1: chain 0 (bad init μ_N)
    for col, method in enumerate(METHODS):
        ax = axes[0, col]
        ts = chains[method][0]
        draw_panel(
            ax, ts, z, mu_map,
            title=(f"{method} | chain 0 (bad init $\\theta^{{(0)}}=\\mu_N=-1.0$)\n"
                   f"running group means — step $t$ 까지 누적평균"),
            burn_in=burn_in,
            show_legend=(col == 0),
        )

    # Row 2: 4 chains overlay
    for col, method in enumerate(METHODS):
        ax = axes[1, col]
        draw_overlay_panel(
            ax, chains[method], z, mu_map,
            title=(f"{method} | 4 chains (dispersed init) overlaid\n"
                   "chain 0 진하게, 나머지 chain 들은 옅게"),
            burn_in=burn_in,
            show_legend=(col == 0),
        )

    fig.suptitle(
        f"Study 1B — running group mean convergence  (seed={seed}, n={n})",
        fontsize=12, fontweight="bold", y=0.995,
    )
    fig.text(
        0.5, 0.005,
        "수렴이 일어나는 시점 = 각 group 의 굵은 선이 자기 색의 점선($\\mu_g$) 에 닿는 step.",
        ha="center", fontsize=9, color="gray",
    )
    fig.tight_layout(rect=[0, 0.015, 1, 0.97])
    out_path = os.path.join(_THIS_DIR, "convergence_group_mean.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")

    # ── 텍스트 요약: 각 chain 의 group 별 "수렴 시점" (μ_g 의 ±0.3 안에 들어와 머문 첫 step)
    print()
    print("Convergence step (running group mean enters μ_g ±0.3 and stays):")
    for method in METHODS:
        ts0 = chains[method][0]
        for g in ("S", "W", "N"):
            curve = running_group_mean(ts0, z, g)
            band = np.abs(curve - mu_map[g]) <= 0.3
            # 끝까지 ±0.3 안에 유지된 가장 빠른 step
            stay = None
            inside = False
            for t_idx in range(len(band)):
                if band[t_idx] and not inside:
                    cand = t_idx
                    if band[t_idx:].all():
                        stay = cand
                        break
                    inside = True
                elif not band[t_idx]:
                    inside = False
            stay_str = f"{stay}" if stay is not None else f"never ({band.mean()*100:.0f}% of time within band)"
            print(f"  [{method}] {g}: converge step = {stay_str}  "
                  f"(curve_end={curve[-1]:+.3f}, target={mu_map[g]:+.2f})")


if __name__ == "__main__":
    main()
