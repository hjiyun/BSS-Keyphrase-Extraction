"""
BSS Study 1A: Posterior Energy Landscape 시각화 (V3)
Mixture posterior: E(θ) = -log[Σ_g ρ_g · exp(-E_g(θ))]

Step 2 설정을 바탕으로, N/W/S basin이 더 잘 분리되도록
시뮬레이션 파일과 동일한 파라미터를 사용 (σ_θ는 시나리오별로 다름):
  Easy:      μ=(2.5, 1.0, -2.5),  σ_θ=0.35, α*=0.20, ρ=(0.20, 0.20, 0.60)
  Moderate:  μ=(2.0, 0.5, -1.8),  σ_θ=0.50, α*=0.35, ρ=(0.18, 0.24, 0.58)
  Difficult: μ=(1.5, 0.0, -1.0),  σ_θ=0.60, α*=0.50, ρ=(0.20, 0.20, 0.60)
  Sparse:    μ=(2.0, 1.0, -1.0),  σ_θ=0.55, α*=0.40, ρ=(0.10, 0.18, 0.72)

실행: python bss_posterior_landscape_v3.py
출력: bss_posterior_landscape_v3.png (같은 디렉토리에 저장)
의존성: numpy, matplotlib
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.signal import argrelmin

# ── 한글 폰트 설정 ──
_font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(_font_path):
    fm.fontManager.addfont(_font_path)
    plt.rcParams["font.family"] = fm.FontProperties(fname=_font_path).get_name()
else:
    print("[WARN] Korean font not found — Hangul may not render correctly")
plt.rcParams["axes.unicode_minus"] = False


# ── 에너지 함수 ──
def sigmoid(t):
    return 1.0 / (1.0 + np.exp(-np.clip(t, -700, 700)))


def energy_Y1(theta, alpha, mu_g, sigma2=1.0):
    """S, W 그룹: Y=1 (키프레이즈 관측됨)"""
    lk = np.log(np.clip((1 - alpha) * sigmoid(theta), 1e-10, None))
    prior = (theta - mu_g) ** 2 / (2 * sigma2)
    return -lk + prior


def energy_Y0(theta, alpha, mu_g, sigma2=1.0):
    """N 그룹: Y=0 (비키프레이즈)"""
    lk = np.log(np.clip(1 - (1 - alpha) * sigmoid(theta), 1e-10, None))
    prior = (theta - mu_g) ** 2 / (2 * sigma2)
    return -lk + prior


def energy_mixture(theta, alpha, mu_S, mu_W, mu_N,
                   rho_S, rho_W, rho_N, sigma2=1.0):
    """
    Mixture posterior의 negative log:
      E(θ) = -log[ρ_S·exp(-E_S) + ρ_W·exp(-E_W) + ρ_N·exp(-E_N)]
    log-sum-exp 트릭으로 수치 안정 계산
    """
    E_S = energy_Y1(theta, alpha, mu_S, sigma2)
    E_W = energy_Y1(theta, alpha, mu_W, sigma2)
    E_N = energy_Y0(theta, alpha, mu_N, sigma2)

    log_p_S = -E_S + np.log(rho_S)
    log_p_W = -E_W + np.log(rho_W)
    log_p_N = -E_N + np.log(rho_N)

    log_stack = np.stack([log_p_S, log_p_W, log_p_N], axis=0)
    log_max = log_stack.max(axis=0)
    log_mixture = log_max + np.log(
        np.exp(log_stack - log_max).sum(axis=0)
    )
    return -log_mixture


def find_local_minimum_near_mu(theta, energy, mu_g, window=2.0):
    """
    mu_g 주변 window 안에서 실제 local minimum을 찾는다.
    여러 개면 mu_g에 가장 가까운 점을 고르고, 없으면 표시를 생략한다.
    """
    mask = (theta >= mu_g - window) & (theta <= mu_g + window)
    if not np.any(mask):
        return None

    theta_window = theta[mask]
    energy_window = energy[mask]

    if theta_window.size < 3:
        return None

    local_idx = argrelmin(energy_window)[0]

    if local_idx.size == 0:
        return None

    closest_idx = local_idx[np.argmin(np.abs(theta_window[local_idx] - mu_g))]
    return float(theta_window[closest_idx]), float(energy_window[closest_idx])


# ── 파라미터 ──
theta_range = np.linspace(-4, 4, 1000)

# ── 시뮬레이션 파일과 동일한 4개 시나리오 (σ_θ는 시나리오별) ──
scenarios = [
    {"name": "Easy",      "mu_S": 2.5, "mu_W": 1.0, "mu_N": -2.5,
     "alpha": 0.20, "rho_S": 0.20, "rho_W": 0.20, "rho_N": 0.60,
     "sigma_theta": 0.35, "color": "#1D9E75"},
    {"name": "Moderate",  "mu_S": 2.0, "mu_W": 0.5, "mu_N": -1.8,
     "alpha": 0.35, "rho_S": 0.18, "rho_W": 0.24, "rho_N": 0.58,
     "sigma_theta": 0.50, "color": "#378ADD"},
    {"name": "Difficult", "mu_S": 1.5, "mu_W": 0.0, "mu_N": -1.0,
     "alpha": 0.50, "rho_S": 0.20, "rho_W": 0.20, "rho_N": 0.60,
     "sigma_theta": 0.60, "color": "#D85A30"},
    {"name": "Sparse",    "mu_S": 2.0, "mu_W": 1.0, "mu_N": -1.0,
     "alpha": 0.40, "rho_S": 0.10, "rho_W": 0.18, "rho_N": 0.72,
     "sigma_theta": 0.55, "color": "#BA7517"},
]

# ── 그룹 색상 ──
C_S = "#534AB7"
C_W = "#1D9E75"
C_N = "#A32D2D"

# ── Figure 구성 ──
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.subplots_adjust(hspace=0.48, wspace=0.28, top=0.93, bottom=0.08,
                    left=0.08, right=0.95)

# ── 사전 계산: 모든 시나리오의 E와 minima를 먼저 모은 뒤 공유 y축 결정 ──
scenario_data = []
for sc in scenarios:
    E = energy_mixture(theta_range, sc["alpha"],
                       sc["mu_S"], sc["mu_W"], sc["mu_N"],
                       sc["rho_S"], sc["rho_W"], sc["rho_N"],
                       sc["sigma_theta"] ** 2)
    E_clipped = np.clip(E, None, 15)

    minima = []
    for glabel, mu_g, color_g in [("N", sc["mu_N"], C_N),
                                  ("W", sc["mu_W"], C_W),
                                  ("S", sc["mu_S"], C_S)]:
        m = find_local_minimum_near_mu(theta_range, E, mu_g, window=2.0)
        if m is None:
            continue
        x_min, y_min = m
        minima.append((glabel, x_min, min(y_min, 15), color_g))

    scenario_data.append({"E_clipped": E_clipped, "minima": minima})

# 모든 서브플롯에 공유될 y축 범위 — Moderate vs Sparse 절대 깊이 차이가 보이게
all_minima_y = [m[2] for d in scenario_data for m in d["minima"]]
global_E_min = min(all_minima_y) if all_minima_y else 0.0
global_E_max = min(max(d["E_clipped"].max() for d in scenario_data), 15)
y_pad = (global_E_max - global_E_min) * 0.12
y_lim_low = max(global_E_min - y_pad, 0)
y_lim_high = global_E_max + y_pad * 0.5
text_offset = (global_E_max - global_E_min) * 0.04
y_top = global_E_max * 0.92

for idx, sc in enumerate(scenarios):
    ax = axes[idx // 2][idx % 2]

    name = sc["name"]
    mu_S, mu_W, mu_N = sc["mu_S"], sc["mu_W"], sc["mu_N"]
    alpha = sc["alpha"]
    data = scenario_data[idx]

    # ── 1. Mixture energy 곡선 ──
    ax.plot(theta_range, data["E_clipped"], color=sc["color"], lw=2.5, zorder=3)

    # ── 2. S/W/N 그룹 배경 음영 ──
    ax.axvspan(mu_N - 0.8, mu_N + 0.8, alpha=0.07, color=C_N, zorder=0)
    ax.axvspan(mu_W - 0.8, mu_W + 0.8, alpha=0.07, color=C_W, zorder=0)
    ax.axvspan(mu_S - 0.8, mu_S + 0.8, alpha=0.07, color=C_S, zorder=0)

    # ── 3. local minimum 표시 ──
    for glabel, x_min, y_min, color_min in data["minima"]:
        ax.plot(x_min, y_min, marker="v", color=color_min,
                markersize=9, zorder=6)
        ax.axvline(x=x_min, color=color_min, ls="--", lw=1.2, alpha=0.6)
        ax.text(x_min, y_min - text_offset,
                f"$\\theta$={x_min:.1f}",
                fontsize=8, ha="center", va="top",
                color=color_min, fontweight="bold")

    # ── 4. 그룹 레이블 (상단) ──
    ax.text(mu_N, y_top, "N", color=C_N, fontsize=11,
            ha="center", fontweight="bold", zorder=5)
    ax.text(mu_W, y_top, "W", color=C_W, fontsize=11,
            ha="center", fontweight="bold", zorder=5)
    ax.text(mu_S, y_top, "S", color=C_S, fontsize=11,
            ha="center", fontweight="bold", zorder=5)

    # ── 축 설정 (모든 서브플롯에 공유) ──
    ax.set_xlim(-4, 4)
    ax.set_ylim(y_lim_low, y_lim_high)
    ax.set_xticks(np.arange(-4, 5, 1))
    ax.set_xlabel(r"$\theta$", fontsize=11)
    ax.set_ylabel(r"$-\log\, p(\theta \mid Y, \alpha)$  [Energy]", fontsize=9)
    ax.grid(True, alpha=0.15)

    # ── 서브플롯 제목 ──
    sigma_theta = sc["sigma_theta"]
    ax.set_title(
        f"{name}  "
        rf"($\alpha^*$={alpha}, $\sigma_\theta$={sigma_theta})" "\n"
        rf"$\mu_S$={mu_S}, $\mu_W$={mu_W}, $\mu_N$={mu_N}",
        fontsize=10, fontweight="bold", pad=8,
    )

# ── 하단 공통 설명 ──
fig.text(
    0.5, 0.01,
    "Per-scenario sigma_theta (shown in each title)  |  "
    "\u25bc = local minimum near each \u03bc_g  |  "
    "shared y-axis across subplots",
    ha="center", fontsize=9, color="gray",
)

# ── 저장 ──
out_dir = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(out_dir, "bss_posterior_landscape_v3.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved → {out_path}")
