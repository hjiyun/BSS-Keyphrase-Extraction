"""
BSS Study 1A: theta recovery evaluation

목표:
  1. theta_hat 이 theta_star 를 잘 회복하는지 MSE로 평가
  2. MSE가 충분히 낮지 않을 때, 중요도 순서가 유지되는지
     Spearman / Kendall / NDCG / Top-k overlap으로 보조 평가

실행:
  python bss_theta_evaluation.py

출력:
  - bss_theta_evaluation_summary.json
  - bss_theta_recovery_summary.png
  - bss_theta_recovery_scatter.png
"""

import json
import os
import sys
from dataclasses import dataclass

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.linalg import solve
from scipy.stats import kendalltau, spearmanr

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_DIR = os.path.join(PROJECT_ROOT, "code_JOC")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from keyphrase_functions import (
    create_fcm_words,
    inv_logit,
    base_to_start,
    alpha_find,
    gibbs_mh,
)


# ── 한글 폰트 설정 ──
_font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(_font_path):
    fm.fontManager.addfont(_font_path)
    plt.rcParams["font.family"] = fm.FontProperties(fname=_font_path).get_name()
else:
    print("[WARN] Korean font not found — Hangul may not render correctly")
plt.rcParams["axes.unicode_minus"] = False


# ── 설정 ──
DATA_DIR = os.path.join(PROJECT_ROOT, "data_JOC")
DOCUMENT = "pre_process/C-42.txt.final"

# 평가 모드:
#   "r_faithful"    — R 원본 재현이 목표. theta 의 scale shrinkage 는 허용하고
#                      Calibrated MSE 를 primary metric 으로 본다.
#   "theta_recovery" — theta 의 scale 까지 맞추는 것이 목표. Raw MSE 가 primary.
MODE = "r_faithful"
assert MODE in ("r_faithful", "theta_recovery")
PRIMARY_MSE_KEY = "mse_calibrated" if MODE == "r_faithful" else "mse_theta"
PRIMARY_MSE_LABEL = "Calibrated MSE" if MODE == "r_faithful" else "Raw MSE"

R = 6
T = 5000
BURN_IN = 1000
SIGMA_THETA = 1.0
THETA_SCALE = 1.0   # theta_star 전체 scale 배율 (1.0 또는 1.2)
BETA_OBS = 1.5      # 관측 모델 p = (1-alpha*) * sigmoid(BETA_OBS * theta*)
SEED_BASE = 20260420

GRID = np.linspace(0.01, 0.95, 60)
DAMPING = 0.85

SCENARIOS = [
    {"name": "Easy",      "mu_S": 2.5, "mu_W": 1.0, "mu_N": -2.5,
     "alpha": 0.05, "rho_S": 0.19, "rho_W": 0.19, "rho_N": 0.62,
     "color": "#1D9E75"},
    # {"name": "Moderate",  "mu_S": 2.2, "mu_W": 0.5, "mu_N": -2.0,
    #  "alpha": 0.30, "rho_S": 0.19, "rho_W": 0.19, "rho_N": 0.62,
    #  "color": "#378ADD"},
    # {"name": "Difficult", "mu_S": 1.5, "mu_W": 0.0, "mu_N": -1.0,
    #  "alpha": 0.50, "rho_S": 0.19, "rho_W": 0.19, "rho_N": 0.62,
    #  "color": "#D85A30"},
    # {"name": "Sparse",    "mu_S": 2.2, "mu_W": 0.5, "mu_N": -2.0,
    #  "alpha": 0.30, "rho_S": 0.125, "rho_W": 0.125, "rho_N": 0.75,
    #  "color": "#BA7517"},
]

GROUP_COLORS = {"S": "#534AB7", "W": "#1D9E75", "N": "#A32D2D"}
GROUP_RELEVANCE = {"S": 2.0, "W": 1.0, "N": 0.0}

SIGMA_ORACLE = 0.5  # oracle prior 에서 theta_star 에 더해질 노이즈 크기

PRIOR_STRATEGIES = [
    {"name": "baseline", "label": "u0 = 0",              "color": "#888888"},
    {"name": "weak",     "label": "u0 = Y-based",        "color": "#378ADD"},
    {"name": "graph",    "label": "u0 = TextRank-logit", "color": "#1D9E75"},
    {"name": "oracle",   "label": "u0 = theta* + noise", "color": "#D85A30"},
]


@dataclass
class TrialResult:
    mse_theta: float
    mse_pi_sigma: float
    mse_calibrated: float
    cal_slope: float
    cal_intercept: float
    spearman: float
    kendall: float
    topk_overlap: float
    ndcg_at_k: float
    alpha_hat: float
    alpha_bias: float
    accept_rate: float
    n_y_obs: int
    theta_star: np.ndarray
    theta_hat: np.ndarray
    group: np.ndarray
    pi_star: np.ndarray


def build_graph():
    """C-42 문서로부터 BSS 그래프를 구성한다 (word 기반)."""
    file_path = os.path.join(DATA_DIR, DOCUMENT)
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    fcm, unique_words, word_to_idx = create_fcm_words(text, window=2)
    A = fcm.copy()
    np.fill_diagonal(A, 0)

    row_sums = A.sum(axis=1)
    row_sums[row_sums == 0] = 1
    D = np.diag(row_sums)

    keep_indices = np.where((A > 0).sum(axis=1) >= 1)[0]
    A_minus = A[np.ix_(keep_indices, keep_indices)]
    rs_minus = A_minus.sum(axis=1)
    rs_minus[rs_minus == 0] = 1
    D_minus = np.diag(rs_minus)

    graph = {
        "n": len(unique_words),
        "A": A,
        "D": D,
        "A_minus": A_minus,
        "D_minus": D_minus,
        "n_minus": len(keep_indices),
        "dictionary_minus": np.array([[unique_words[i], i] for i in keep_indices], dtype=object),
        "keep_indices": keep_indices,
        "unique_words": unique_words,
        "word_to_idx": word_to_idx,
    }
    return graph


def build_B(graph):
    """그래프 라플라시안 기반 B 행렬."""
    G = solve(graph["D_minus"], graph["A_minus"])
    return np.eye(graph["n_minus"]) - DAMPING * G.T


def build_B_star(graph):
    """
    R 원본:
      w <- sqrt(solve(D))
      B_star <- diag(n) - d * w %*% A %*% w
    """
    d_diag = np.diag(graph["D_minus"]).astype(float)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(d_diag))
    return np.eye(graph["n_minus"]) - DAMPING * d_inv_sqrt @ graph["A_minus"] @ d_inv_sqrt


def build_u0(strategy, graph, B, Y, theta_star):
    """u_0 prior mean 을 네 가지 전략에 따라 구성한다."""
    n = graph["n_minus"]
    if strategy == "baseline":
        return np.zeros(n)
    if strategy == "weak":
        # 관측된 positive 에만 logit-scale 의 약한 양의 prior
        return 2.0 * Y
    if strategy == "graph":
        # TextRank 정상분포: r = (I - d G^T)^{-1} (1-d) 1 = solve(B, (1-d)·1)
        tr = solve(B, (1 - DAMPING) * np.ones(n))
        tr_std = tr.std() + 1e-12
        return (tr - tr.mean()) / tr_std * SIGMA_THETA
    if strategy == "oracle":
        return theta_star + np.random.normal(0, SIGMA_ORACLE, n)
    raise ValueError(f"unknown prior strategy: {strategy}")


def sample_theta_star(n, sc):
    """그룹 비율과 평균에 따라 theta_star 와 그룹 라벨을 생성한다."""
    n_s = round(n * sc["rho_S"])
    n_w = round(n * sc["rho_W"])
    n_n = n - n_s - n_w

    group = np.array(["S"] * n_s + ["W"] * n_w + ["N"] * n_n)
    np.random.shuffle(group)

    theta_star = np.zeros(n)
    theta_star[group == "S"] = sc["mu_S"] + np.random.normal(0, SIGMA_THETA, n_s)
    theta_star[group == "W"] = sc["mu_W"] + np.random.normal(0, SIGMA_THETA, n_w)
    theta_star[group == "N"] = sc["mu_N"] + np.random.normal(0, SIGMA_THETA, n_n)
    theta_star = THETA_SCALE * theta_star
    return theta_star, group


def generate_Y(theta_star, alpha_true):
    """Y_i ~ Bernoulli((1-alpha_true) * sigmoid(BETA_OBS * theta_star_i))"""
    pi_star = (1 - alpha_true) * inv_logit(BETA_OBS * theta_star)
    pi_star = np.clip(pi_star, 0, 1)
    Y = np.random.binomial(1, pi_star).astype(float)
    return Y, pi_star


def choose_top_k(group):
    """ranking 평가는 실제 keyphrase 후보 수와 맞추기 위해 S+W 개수를 기본 k로 사용."""
    return int(np.sum(np.isin(group, ["S", "W"])))


def compute_ndcg_at_k(theta_star, theta_hat, group, k):
    """True relevance는 S=2, W=1, N=0으로 두고 NDCG@k를 계산."""
    relevance = np.array([GROUP_RELEVANCE[g] for g in group], dtype=float)
    pred_rank = np.argsort(theta_hat)[::-1][:k]
    ideal_rank = np.argsort(relevance)[::-1][:k]

    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = np.sum((2 ** relevance[pred_rank] - 1) * discounts)
    idcg = np.sum((2 ** relevance[ideal_rank] - 1) * discounts)
    if idcg <= 0:
        return 0.0
    return float(dcg / idcg)


def compute_topk_overlap(theta_star, theta_hat, k):
    """theta 기준 top-k 집합 일치율."""
    true_top = set(np.argsort(theta_star)[::-1][:k].tolist())
    est_top = set(np.argsort(theta_hat)[::-1][:k].tolist())
    return float(len(true_top & est_top) / max(k, 1))


def run_single_trial(graph, B, B_star, sc, prior_strategy="baseline",
                     theta_star=None, Y=None, pi_star=None, group=None):
    n = graph["n_minus"]
    if theta_star is None:
        theta_star, group = sample_theta_star(n, sc)
        Y, pi_star = generate_Y(theta_star, sc["alpha"])

    u_0 = build_u0(prior_strategy, graph, B, Y, theta_star)
    base_line = solve(B_star, Y)
    ini = base_to_start(base_line)
    alpha_est = alpha_find(u_0, Y, GRID)

    result = gibbs_mh(
        Burn_in=BURN_IN, T=T, ini=ini, n=n,
        graph=graph, Y=Y, B=B, u_0=u_0,
        alpha_est=alpha_est, grid=GRID, verbose=False,
    )

    theta_hat = np.mean(result["theta_store"][BURN_IN:, :], axis=0)
    alpha_hat = float(result["alpha_mn"])
    k = choose_top_k(group)

    spearman_val = float(spearmanr(theta_star, theta_hat).statistic)
    kendall_val = float(kendalltau(theta_star, theta_hat).statistic)

    # Linear calibration: theta_star ~ a + b * theta_hat → MSE after best-fit linear transform
    cal_slope, cal_intercept = np.polyfit(theta_hat, theta_star, 1)
    theta_hat_cal = cal_intercept + cal_slope * theta_hat
    mse_calibrated = float(np.mean((theta_hat_cal - theta_star) ** 2))

    return TrialResult(
        mse_theta=float(np.mean((theta_hat - theta_star) ** 2)),
        mse_pi_sigma=float(np.mean((inv_logit(theta_hat) - inv_logit(theta_star)) ** 2)),
        mse_calibrated=mse_calibrated,
        cal_slope=float(cal_slope),
        cal_intercept=float(cal_intercept),
        spearman=spearman_val,
        kendall=kendall_val,
        topk_overlap=compute_topk_overlap(theta_star, theta_hat, k),
        ndcg_at_k=compute_ndcg_at_k(theta_star, theta_hat, group, k),
        alpha_hat=alpha_hat,
        alpha_bias=float(alpha_hat - sc["alpha"]),
        accept_rate=float(result["accept"] / T),
        n_y_obs=int(np.sum(Y)),
        theta_star=theta_star,
        theta_hat=theta_hat,
        group=group,
        pi_star=pi_star,
    )


def summarize_trials(trials):
    metric_names = [
        "mse_theta", "mse_pi_sigma", "mse_calibrated", "cal_slope", "cal_intercept",
        "spearman", "kendall", "topk_overlap", "ndcg_at_k",
        "alpha_hat", "alpha_bias", "accept_rate", "n_y_obs",
    ]
    summary = {}
    for name in metric_names:
        values = np.array([getattr(t, name) for t in trials], dtype=float)
        summary[name] = {
            "mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=0)),
            "se": float(np.std(values, ddof=0) / np.sqrt(len(values))),
            "values": [float(v) for v in values],
        }

    representative_idx = int(np.argmin([getattr(t, PRIMARY_MSE_KEY) for t in trials]))
    summary["representative_idx"] = representative_idx
    return summary


def plot_summary(all_results, out_path):
    scenario_names = list(all_results.keys())
    x = np.arange(len(scenario_names))

    def ms_se(key):
        return (
            [all_results[name]["summary"][key]["mean"] for name in scenario_names],
            [all_results[name]["summary"][key]["se"] for name in scenario_names],
        )

    mse_m, mse_s = ms_se("mse_theta")
    mse_cal_m, mse_cal_s = ms_se("mse_calibrated")
    sp_m, sp_s = ms_se("spearman")
    kd_m, kd_s = ms_se("kendall")
    ndcg_m, ndcg_s = ms_se("ndcg_at_k")

    colors = [all_results[name]["scenario"]["color"] for name in scenario_names]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.subplots_adjust(hspace=0.35, wspace=0.28, top=0.92, bottom=0.08)

    def draw_bar(ax, means, ses, title, ylabel, ylim=None):
        ax.bar(x, means, yerr=ses, capsize=4, color=colors, alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_names)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.15)
        if ylim is not None:
            ax.set_ylim(*ylim)

    if MODE == "r_faithful":
        draw_bar(axes[0, 0], mse_cal_m, mse_cal_s,
                 "Calibrated MSE  [primary]", "MSE(a + b·theta_hat, theta*)")
        draw_bar(axes[0, 1], mse_m, mse_s,
                 "Raw MSE  [reference, biased by shrinkage]", "MSE of theta_hat")
        draw_bar(axes[1, 0], sp_m, sp_s, "Rank Agreement", "Spearman rho", ylim=(0, 1))
        draw_bar(axes[1, 1], ndcg_m, ndcg_s, "Important Phrase Ordering", "NDCG@k", ylim=(0, 1))
        footer = ("R-faithful mode — primary: Calibrated MSE; "
                  "raw MSE shown for reference (scale shrinkage expected)")
    else:  # theta_recovery
        draw_bar(axes[0, 0], mse_m, mse_s,
                 "Raw MSE  [primary]", "MSE of theta_hat")
        draw_bar(axes[0, 1], sp_m, sp_s, "Rank Agreement", "Spearman rho", ylim=(0, 1))
        draw_bar(axes[1, 0], kd_m, kd_s, "Rank Agreement", "Kendall tau", ylim=(0, 1))
        draw_bar(axes[1, 1], ndcg_m, ndcg_s, "Important Phrase Ordering", "NDCG@k", ylim=(0, 1))
        footer = ("Theta-recovery mode — primary: Raw MSE; "
                  "rank correlation and NDCG are backup checks")

    fig.text(0.5, 0.02, footer, ha="center", fontsize=10, color="gray")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_representative_scatter(all_results, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.subplots_adjust(hspace=0.38, wspace=0.28, top=0.92, bottom=0.08)

    for idx, (name, payload) in enumerate(all_results.items()):
        ax = axes[idx // 2][idx % 2]
        rep_idx = payload["summary"]["representative_idx"]
        trial = payload["trials"][rep_idx]

        theta_star = trial.theta_star
        theta_hat = trial.theta_hat
        group = trial.group

        for glabel in ["N", "W", "S"]:
            mask = group == glabel
            ax.scatter(
                theta_star[mask], theta_hat[mask],
                s=18, alpha=0.65, color=GROUP_COLORS[glabel], label=glabel,
            )

        lo = min(theta_star.min(), theta_hat.min()) - 0.2
        hi = max(theta_star.max(), theta_hat.max()) + 0.2
        ax.plot([lo, hi], [lo, hi], color="gray", lw=1.2, ls="--", label="y = x")

        # Regression line: theta_hat ~ theta_star (slope < 1 → shrinkage)
        b_fit, a_fit = np.polyfit(theta_star, theta_hat, 1)
        line_x = np.array([lo, hi])
        ax.plot(line_x, a_fit + b_fit * line_x, color="#D7263D", lw=1.4,
                label=f"fit: slope={b_fit:.2f}")

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(r"True $\theta^*$")
        ax.set_ylabel(r"Estimated $\hat{\theta}$")
        ax.grid(True, alpha=0.15)
        if MODE == "r_faithful":
            title_line = (f"MSE_cal={trial.mse_calibrated:.3f} [primary], "
                          f"MSE={trial.mse_theta:.3f}, "
                          f"Spearman={trial.spearman:.3f}")
        else:
            title_line = (f"MSE={trial.mse_theta:.3f} [primary], "
                          f"MSE_cal={trial.mse_calibrated:.3f}, "
                          f"Spearman={trial.spearman:.3f}")
        ax.set_title(f"{name}\n{title_line}", fontsize=10, fontweight="bold")

        if idx == 0:
            ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    fig.text(
        0.5, 0.02,
        "Representative trial per scenario: the lowest-MSE replicate is shown",
        ha="center", fontsize=10, color="gray",
    )

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_group_boxplot(all_results, out_path):
    """시나리오별 S/W/N 그룹 theta_hat 분포 박스플롯 (모든 trial 집계)."""
    n_scenarios = len(all_results)
    ncols = min(2, n_scenarios)
    nrows = int(np.ceil(n_scenarios / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), squeeze=False)

    for idx, (name, payload) in enumerate(all_results.items()):
        ax = axes[idx // ncols][idx % ncols]

        data_hat = {"S": [], "W": [], "N": []}
        data_star = {"S": [], "W": [], "N": []}
        for trial in payload["trials"]:
            for g in ("S", "W", "N"):
                mask = trial.group == g
                data_hat[g].extend(trial.theta_hat[mask].tolist())
                data_star[g].extend(trial.theta_star[mask].tolist())

        positions = np.arange(3)
        bp = ax.boxplot(
            [data_hat["S"], data_hat["W"], data_hat["N"]],
            positions=positions, widths=0.55,
            patch_artist=True, showfliers=False,
        )
        for patch, g in zip(bp["boxes"], ("S", "W", "N")):
            patch.set_facecolor(GROUP_COLORS[g])
            patch.set_alpha(0.75)
        for med in bp["medians"]:
            med.set_color("black")

        # theta_star 그룹 평균을 참조선으로
        for i, g in enumerate(("S", "W", "N")):
            if data_star[g]:
                ax.hlines(np.mean(data_star[g]), i - 0.35, i + 0.35,
                          colors="black", linestyles=":", lw=1.2)

        means = [np.mean(data_hat[g]) if data_hat[g] else np.nan for g in ("S", "W", "N")]
        stds = [np.std(data_hat[g], ddof=0) if data_hat[g] else np.nan for g in ("S", "W", "N")]

        ax.set_xticks(positions)
        ax.set_xticklabels(["S", "W", "N"])
        ax.set_ylabel(r"$\hat{\theta}$")
        ax.grid(True, axis="y", alpha=0.15)
        ax.set_title(
            f"{name}  "
            f"(mean: S={means[0]:.2f}, W={means[1]:.2f}, N={means[2]:.2f})",
            fontsize=10, fontweight="bold",
        )

    for idx in range(n_scenarios, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle(r"Group-wise $\hat{\theta}$ distribution (dotted = true $\theta^*$ group mean)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def serializable_trial(trial):
    return {
        "mse_theta": trial.mse_theta,
        "mse_pi_sigma": trial.mse_pi_sigma,
        "mse_calibrated": trial.mse_calibrated,
        "cal_slope": trial.cal_slope,
        "cal_intercept": trial.cal_intercept,
        "spearman": trial.spearman,
        "kendall": trial.kendall,
        "topk_overlap": trial.topk_overlap,
        "ndcg_at_k": trial.ndcg_at_k,
        "alpha_hat": trial.alpha_hat,
        "alpha_bias": trial.alpha_bias,
        "accept_rate": trial.accept_rate,
        "n_y_obs": trial.n_y_obs,
    }


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    summary_json = os.path.join(out_dir, f"bss_{MODE}_summary.json")
    summary_png = os.path.join(out_dir, f"bss_{MODE}_main.png")
    scatter_png = os.path.join(out_dir, f"bss_{MODE}_scatter.png")
    boxplot_png = os.path.join(out_dir, f"bss_{MODE}_boxplot.png")

    print("=" * 68)
    print(f"BSS theta recovery evaluation  [mode={MODE}]")
    print(f"Primary metric: {PRIMARY_MSE_LABEL} ({PRIMARY_MSE_KEY})")
    print(f"Document: {DOCUMENT}")
    print(f"R={R}, T={T}, Burn-in={BURN_IN}, sigma_theta={SIGMA_THETA}, "
          f"theta_scale={THETA_SCALE}, beta_obs={BETA_OBS}")
    print("=" * 68)

    graph = build_graph()
    B = build_B(graph)
    B_star = build_B_star(graph)
    print(f"n_minus={graph['n_minus']} words\n")

    all_results = {}

    single_scenario = len(SCENARIOS) == 1
    n = graph["n_minus"]

    for sc_idx, sc in enumerate(SCENARIOS):
        sc_name = sc["name"]
        print(f"[{sc_name}] alpha*={sc['alpha']}, "
              f"mu=({sc['mu_S']}, {sc['mu_W']}, {sc['mu_N']}), "
              f"rho=({sc['rho_S']}, {sc['rho_W']}, {sc['rho_N']})")

        # Paired design: trial r 마다 (theta_star, Y) 를 한 번만 뽑고 4 prior 전략을 모두 실행
        strategy_trials = {ps["name"]: [] for ps in PRIOR_STRATEGIES}

        for r in range(R):
            data_seed = SEED_BASE + sc_idx * 1000 + r
            np.random.seed(data_seed)
            theta_star, group = sample_theta_star(n, sc)
            Y, pi_star = generate_Y(theta_star, sc["alpha"])

            print(f"  Trial {r+1}/{R} (n_obs={int(Y.sum())}):")
            for ps_idx, ps in enumerate(PRIOR_STRATEGIES):
                np.random.seed(data_seed * 7 + ps_idx * 13 + 1)
                trial = run_single_trial(
                    graph, B, B_star, sc, prior_strategy=ps["name"],
                    theta_star=theta_star, Y=Y, pi_star=pi_star, group=group,
                )
                strategy_trials[ps["name"]].append(trial)
                print(
                    f"    [{ps['name']:<8}] "
                    f"MSE={trial.mse_theta:.4f}, "
                    f"MSE_cal={trial.mse_calibrated:.4f} "
                    f"(slope={trial.cal_slope:.2f}), "
                    f"Spearman={trial.spearman:.4f}, "
                    f"NDCG@k={trial.ndcg_at_k:.4f}"
                )

        print(f"\n  [{sc_name}] per-strategy summary")
        for ps in PRIOR_STRATEGIES:
            trials = strategy_trials[ps["name"]]
            summary = summarize_trials(trials)

            sc_copy = dict(sc)
            sc_copy["color"] = ps["color"]
            sc_copy["display_name"] = f"{sc_name} / {ps['label']}"
            key = ps["name"] if single_scenario else f"{sc_name}/{ps['name']}"
            all_results[key] = {
                "scenario": sc_copy,
                "summary": summary,
                "trials": trials,
            }

            prim_mean = summary[PRIMARY_MSE_KEY]["mean"]
            prim_se = summary[PRIMARY_MSE_KEY]["se"]
            other_key = "mse_theta" if PRIMARY_MSE_KEY == "mse_calibrated" else "mse_calibrated"
            print(
                f"    [{ps['name']:<8}] "
                f"{PRIMARY_MSE_LABEL}={prim_mean:.4f} (SE={prim_se:.4f}) [primary] | "
                f"{other_key}={summary[other_key]['mean']:.4f} | "
                f"Spearman={summary['spearman']['mean']:.4f} | "
                f"NDCG@k={summary['ndcg_at_k']['mean']:.4f}"
            )
        print()

    json_payload = {
        "settings": {
            "mode": MODE,
            "primary_metric": PRIMARY_MSE_KEY,
            "document": DOCUMENT,
            "R": R,
            "T": T,
            "BURN_IN": BURN_IN,
            "SIGMA_THETA": SIGMA_THETA,
            "THETA_SCALE": THETA_SCALE,
            "BETA_OBS": BETA_OBS,
            "n_minus": graph["n_minus"],
        },
        "scenarios": {
            name: {
                "scenario": payload["scenario"],
                "summary": payload["summary"],
                "trials": [serializable_trial(t) for t in payload["trials"]],
            }
            for name, payload in all_results.items()
        },
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, ensure_ascii=False, indent=2)

    plot_summary(all_results, summary_png)
    plot_representative_scatter(all_results, scatter_png)
    plot_group_boxplot(all_results, boxplot_png)

    print("-" * 68)
    print(f"Saved JSON   -> {summary_json}")
    print(f"Saved plot   -> {summary_png}")
    print(f"Saved scatter-> {scatter_png}")
    print(f"Saved boxplot-> {boxplot_png}")


if __name__ == "__main__":
    main()
