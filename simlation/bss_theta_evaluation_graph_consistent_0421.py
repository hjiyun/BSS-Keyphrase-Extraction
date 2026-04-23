"""
BSS Study 1A: graph-consistent theta recovery evaluation

핵심 아이디어:
  1. 실제 문서 그래프를 고정한다.
  2. 그래프 위에서 smooth latent score를 먼저 생성한다.
  3. 생성된 theta*의 분위수 구간으로 N/W/S 그룹을 정의한다.
  4. Y ~ Bernoulli((1-alpha*) * sigmoid(theta*))를 생성한다.

즉, 그룹을 랜덤 셔플로 붙이지 않고
theta* 자체가 그래프 구조와 정합적으로 생성되도록 만든 모의실험이다.

실행:
  python bss_theta_evaluation_graph_consistent.py

출력:
  - bss_theta_graph_consistent_summary.json
  - bss_theta_graph_consistent_summary.png
  - bss_theta_graph_consistent_scatter.png
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

R = 3        # 튜닝용: 3. 최종 숫자 뽑을 땐 6 으로 복귀
T = 5000
BURN_IN = 1000
SEED_BASE = 20260421
DAMPING = 0.85
BETA_TRUE = 1.2        # 관측 모델 slope
INTERCEPT_TRUE = 0.0   # 관측 모델 intercept: p_star = sigmoid(BETA_TRUE * theta* - INTERCEPT_TRUE)
GRID = np.linspace(0.01, 0.95, 60)

SCENARIOS = [
    {"name": "Easy",      "alpha": 0.25, "rho_S": 0.19, "rho_W": 0.19, "rho_N": 0.62,
     "theta_loc": 0.0, "theta_scale": 0.8, "smooth_noise": 0.25, "color": "#1D9E75"},
    # {"name": "Moderate",  "alpha": 0.30, "rho_S": 0.19, "rho_W": 0.19, "rho_N": 0.62,
    #  "theta_loc": 0.0, "theta_scale": 1.8, "smooth_noise": 0.65, "color": "#378ADD"},
    # {"name": "Difficult", "alpha": 0.50, "rho_S": 0.19, "rho_W": 0.19, "rho_N": 0.62,
    #  "theta_loc": 0.0, "theta_scale": 1.4, "smooth_noise": 0.90, "color": "#D85A30"},
    # {"name": "Sparse",    "alpha": 0.30, "rho_S": 0.125, "rho_W": 0.125, "rho_N": 0.75,
    #  "theta_loc": 0.0, "theta_scale": 1.8, "smooth_noise": 0.70, "color": "#BA7517"},
]

PRIOR_STRATEGIES = [
    # 튜닝용: baseline 만. 최종 실행 시 graph, weak, oracle 되살리기
    {"name": "baseline", "label": "u0 = stationary prior", "color": "#888888"},
    # {"name": "weak",     "label": "u0 = Y-based",          "color": "#378ADD"},
    # {"name": "graph",    "label": "u0 = graph score",      "color": "#1D9E75"},
    # {"name": "oracle",   "label": "u0 = theta* + noise",   "color": "#D85A30"},
]

GROUP_COLORS = {"S": "#534AB7", "W": "#1D9E75", "N": "#A32D2D"}
GROUP_RELEVANCE = {"S": 2.0, "W": 1.0, "N": 0.0}
SIGMA_ORACLE = 0.5


@dataclass
class TrialResult:
    mse_theta: float
    mse_pi_sigma: float
    mse_calibrated: float
    spearman: float
    kendall: float
    topk_overlap: float
    ndcg_at_k: float
    precision_at_k: float
    alpha_hat: float
    alpha_bias: float
    accept_rate: float
    n_y_obs: int
    theta_star: np.ndarray
    theta_hat: np.ndarray
    group: np.ndarray
    u0_mean: float = 0.0
    u0_std: float = 0.0
    ini_mean: float = 0.0
    ini_std: float = 0.0


def build_graph():
    """C-42 문서로부터 word graph를 구성한다."""
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

    return {
        "A_minus": A_minus,
        "D_minus": D_minus,
        "n_minus": len(keep_indices),
        "dictionary_minus": np.array([[unique_words[i], i] for i in keep_indices], dtype=object),
        "keep_indices": keep_indices,
        "unique_words": unique_words,
        "word_to_idx": word_to_idx,
    }


def build_B(graph):
    G = solve(graph["D_minus"], graph["A_minus"])
    return np.eye(graph["n_minus"]) - DAMPING * G.T


def build_B_star(graph):
    d_diag = np.diag(graph["D_minus"]).astype(float)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(d_diag))
    return np.eye(graph["n_minus"]) - DAMPING * d_inv_sqrt @ graph["A_minus"] @ d_inv_sqrt


def make_stationary_prior(B):
    n = B.shape[0]
    return solve(B, np.full(n, 1.0 - DAMPING))


def sample_graph_consistent_theta(B, sc):
    """
    smooth latent score를 먼저 생성한다.
    white noise를 graph operator로 확산시켜 인접 노드끼리 비슷한 theta*가 되게 한다.
    """
    n = B.shape[0]
    base = make_stationary_prior(B)
    eps = np.random.normal(0, sc["smooth_noise"], n)
    smooth = solve(B, eps)
    theta_raw = base + smooth

    theta_std = theta_raw.std() + 1e-12
    theta_star = sc["theta_loc"] + sc["theta_scale"] * (theta_raw - theta_raw.mean()) / theta_std
    return theta_star


def assign_groups_from_theta(theta_star, sc):
    """
    theta*의 분위수 구간으로 그룹을 정의한다.
    낮은 구간 = N, 중간 = W, 높은 구간 = S.
    S 구간은 (rho_N + rho_W) 분위수 위로 암묵적으로 정의되므로
    rho_S + rho_W + rho_N 이 정확히 1이어야 한다.
    """
    rho_sum = sc["rho_S"] + sc["rho_W"] + sc["rho_N"]
    assert abs(rho_sum - 1.0) < 1e-8, (
        f"rho_S + rho_W + rho_N must equal 1, got {rho_sum} "
        f"(S={sc['rho_S']}, W={sc['rho_W']}, N={sc['rho_N']})"
    )
    q_n = np.quantile(theta_star, sc["rho_N"])
    q_w = np.quantile(theta_star, sc["rho_N"] + sc["rho_W"])

    group = np.full(theta_star.shape[0], "W", dtype="<U1")
    group[theta_star <= q_n] = "N"
    group[theta_star > q_w] = "S"
    return group


def generate_Y(theta_star, alpha_true):
    """
    두 단계 관측 모델 (intercept 포함):
      y_star_i ~ Bernoulli(sigmoid(BETA_TRUE * theta_star_i - INTERCEPT_TRUE))
      Y_i = y_star_i * Bernoulli(1 - alpha_true)
    """
    p_star = inv_logit(BETA_TRUE * theta_star - INTERCEPT_TRUE)
    y_star = np.random.binomial(1, p_star).astype(float)
    Y = y_star * np.random.binomial(1, 1 - alpha_true, size=theta_star.shape[0])
    return Y.astype(float), y_star


def build_u0(strategy, B, Y, theta_star):
    n = B.shape[0]
    if strategy == "baseline":
        return make_stationary_prior(B)
    if strategy == "weak":
        return 2.0 * Y
    if strategy == "graph":
        base = make_stationary_prior(B)
        return (base - base.mean()) / (base.std() + 1e-12)
    if strategy == "oracle":
        return theta_star + np.random.normal(0, SIGMA_ORACLE, n)
    raise ValueError(f"unknown prior strategy: {strategy}")


def choose_top_k(group):
    return int(np.sum(np.isin(group, ["S", "W"])))


def compute_topk_overlap(theta_star, theta_hat, k):
    true_top = set(np.argsort(theta_star)[::-1][:k].tolist())
    est_top = set(np.argsort(theta_hat)[::-1][:k].tolist())
    return float(len(true_top & est_top) / max(k, 1))


def compute_ndcg_at_k(theta_hat, group, k):
    relevance = np.array([GROUP_RELEVANCE[g] for g in group], dtype=float)
    pred_rank = np.argsort(theta_hat)[::-1][:k]
    ideal_rank = np.argsort(relevance)[::-1][:k]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = np.sum((2 ** relevance[pred_rank] - 1) * discounts)
    idcg = np.sum((2 ** relevance[ideal_rank] - 1) * discounts)
    return float(dcg / idcg) if idcg > 0 else 0.0


def compute_precision_at_k(theta_hat, group, k):
    """Top-k 예측 중 S 또는 W 그룹에 속하는 비율."""
    pred_rank = np.argsort(theta_hat)[::-1][:k]
    relevant = np.isin(group[pred_rank], ["S", "W"])
    return float(np.sum(relevant) / max(k, 1))


def run_single_trial(graph, B, B_star, sc, prior_strategy, theta_star, group, Y):
    n = graph["n_minus"]
    u_0 = build_u0(prior_strategy, B, Y, theta_star)
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
    cal_slope, cal_intercept = np.polyfit(theta_hat, theta_star, 1)
    theta_hat_cal = cal_intercept + cal_slope * theta_hat

    return TrialResult(
        mse_theta=float(np.mean((theta_hat - theta_star) ** 2)),
        mse_pi_sigma=float(np.mean((inv_logit(theta_hat) - inv_logit(theta_star)) ** 2)),
        mse_calibrated=float(np.mean((theta_hat_cal - theta_star) ** 2)),
        spearman=float(spearmanr(theta_star, theta_hat).statistic),
        kendall=float(kendalltau(theta_star, theta_hat).statistic),
        topk_overlap=compute_topk_overlap(theta_star, theta_hat, k),
        ndcg_at_k=compute_ndcg_at_k(theta_hat, group, k),
        precision_at_k=compute_precision_at_k(theta_hat, group, k),
        alpha_hat=alpha_hat,
        alpha_bias=float(alpha_hat - sc["alpha"]),
        accept_rate=float(result["accept"] / T),
        n_y_obs=int(np.sum(Y)),
        theta_star=theta_star,
        theta_hat=theta_hat,
        group=group,
        u0_mean=float(u_0.mean()),
        u0_std=float(u_0.std()),
        ini_mean=float(ini.mean()),
        ini_std=float(ini.std()),
    )


def summarize_trials(trials):
    metric_names = [
        # 메인: top-k 계열
        "ndcg_at_k", "topk_overlap", "precision_at_k",
        # 보조
        "spearman", "kendall",
        # 참고
        "mse_theta", "mse_pi_sigma", "mse_calibrated",
        # 기타
        "alpha_hat", "alpha_bias", "accept_rate", "n_y_obs",
    ]
    out = {}
    for name in metric_names:
        vals = np.array([getattr(t, name) for t in trials], dtype=float)
        out[name] = {
            "mean": float(np.mean(vals)),
            "se": float(np.std(vals, ddof=0) / np.sqrt(len(vals))),
            "values": [float(v) for v in vals],
        }
    # 대표 trial 은 NDCG@k 최고 (MSE 대신)
    out["representative_idx"] = int(np.argmax([t.ndcg_at_k for t in trials]))
    return out


def plot_summary(all_results, out_path):
    scenario_names = list(all_results.keys())
    x = np.arange(len(scenario_names))
    colors = [all_results[name]["scenario"]["color"] for name in scenario_names]

    metrics = [
        ("mse_theta", "Theta Recovery Error", "MSE of theta_hat", None),
        ("spearman", "Rank Agreement", "Spearman rho", (0, 1)),
        ("kendall", "Rank Agreement", "Kendall tau", (0, 1)),
        ("ndcg_at_k", "Important Phrase Ordering", "NDCG@k", (0, 1)),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.subplots_adjust(hspace=0.35, wspace=0.28, top=0.92, bottom=0.08)

    for ax, (metric, title, ylabel, ylim) in zip(axes.flat, metrics):
        means = [all_results[name]["summary"][metric]["mean"] for name in scenario_names]
        ses = [all_results[name]["summary"][metric]["se"] for name in scenario_names]
        ax.bar(x, means, yerr=ses, capsize=4, color=colors, alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_names)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.15)
        if ylim is not None:
            ax.set_ylim(*ylim)

    fig.text(
        0.5, 0.02,
        "Graph-consistent simulation: theta* is generated smoothly first, then groups are defined by theta intervals",
        ha="center", fontsize=10, color="gray",
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_representative_scatter(all_results, out_path):
    n_scenarios = len(all_results)
    ncols = min(2, n_scenarios)
    nrows = int(np.ceil(n_scenarios / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 5.2 * nrows), squeeze=False)
    fig.subplots_adjust(hspace=0.40, wspace=0.28, top=0.92, bottom=0.08)

    for idx, (name, payload) in enumerate(all_results.items()):
        ax = axes[idx // ncols][idx % ncols]
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
        slope, intercept = np.polyfit(theta_star, theta_hat, 1)
        line_x = np.array([lo, hi])
        ax.plot(line_x, intercept + slope * line_x, color="#D7263D", lw=1.4,
                label=f"fit: slope={slope:.2f}")

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(r"True $\theta^*$")
        ax.set_ylabel(r"Estimated $\hat{\theta}$")
        ax.grid(True, alpha=0.15)
        ax.set_title(
            f"{name}\n"
            f"MSE={trial.mse_theta:.3f}, "
            f"MSE_cal={trial.mse_calibrated:.3f}, "
            f"Spearman={trial.spearman:.3f}",
            fontsize=10, fontweight="bold",
        )
        if idx == 0:
            ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    for idx in range(n_scenarios, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.text(
        0.5, 0.02,
        "Representative trial per scenario: lowest-MSE replicate under baseline prior",
        ha="center", fontsize=10, color="gray",
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def serializable_trial(trial):
    return {
        "mse_theta": trial.mse_theta,
        "mse_pi_sigma": trial.mse_pi_sigma,
        "mse_calibrated": trial.mse_calibrated,
        "spearman": trial.spearman,
        "kendall": trial.kendall,
        "topk_overlap": trial.topk_overlap,
        "ndcg_at_k": trial.ndcg_at_k,
        "precision_at_k": trial.precision_at_k,
        "alpha_hat": trial.alpha_hat,
        "alpha_bias": trial.alpha_bias,
        "accept_rate": trial.accept_rate,
        "n_y_obs": trial.n_y_obs,
    }


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    summary_json = os.path.join(out_dir, "bss_theta_graph_consistent_summary.json")
    summary_png = os.path.join(out_dir, "bss_theta_graph_consistent_summary.png")
    scatter_png = os.path.join(out_dir, "bss_theta_graph_consistent_scatter.png")

    print("=" * 72)
    print("BSS theta recovery evaluation: graph-consistent simulation")
    print(f"Document: {DOCUMENT}")
    print(f"R={R}, T={T}, Burn-in={BURN_IN}, beta_obs={BETA_TRUE}")
    print("=" * 72)

    graph = build_graph()
    B = build_B(graph)
    B_star = build_B_star(graph)
    n = graph["n_minus"]
    print(f"n_minus={n} words\n")

    all_results = {}
    strategy_results = {}

    for sc_idx, sc in enumerate(SCENARIOS):
        name = sc["name"]
        print(f"[{name}] alpha*={sc['alpha']}, "
              f"rho=({sc['rho_S']}, {sc['rho_W']}, {sc['rho_N']}), "
              f"theta_scale={sc['theta_scale']}, smooth_noise={sc['smooth_noise']}")

        strategy_trials = {ps["name"]: [] for ps in PRIOR_STRATEGIES}

        for r in range(R):
            data_seed = SEED_BASE + sc_idx * 1000 + r
            np.random.seed(data_seed)
            theta_star = sample_graph_consistent_theta(B, sc)
            group = assign_groups_from_theta(theta_star, sc)
            Y, y_star = generate_Y(theta_star, sc["alpha"])

            print(f"  Trial {r+1}/{R} (n_obs={int(Y.sum())}, "
                  f"y_star={int(y_star.sum())}, mask_loss={int(y_star.sum()-Y.sum())}): "
                  f"theta* mean={theta_star.mean():.3f}, std={theta_star.std():.3f}")
            for ps_idx, ps in enumerate(PRIOR_STRATEGIES):
                np.random.seed(data_seed * 11 + ps_idx * 17 + 3)
                trial = run_single_trial(
                    graph, B, B_star, sc, ps["name"], theta_star, group, Y
                )
                strategy_trials[ps["name"]].append(trial)
                print(
                    f"    [{ps['name']:<8}] "
                    f"NDCG@k={trial.ndcg_at_k:.4f}, "
                    f"Top-k={trial.topk_overlap:.4f}, "
                    f"P@k={trial.precision_at_k:.4f} [main] | "
                    f"Spearman={trial.spearman:.4f} [aux] | "
                    f"MSE={trial.mse_theta:.2f}, MSE_cal={trial.mse_calibrated:.3f} [ref] | "
                    f"u0(m={trial.u0_mean:.2f},s={trial.u0_std:.2f}) "
                    f"ini(m={trial.ini_mean:.2f},s={trial.ini_std:.2f}) "
                    f"theta_hat(m={trial.theta_hat.mean():.2f},s={trial.theta_hat.std():.2f})"
                )

        print(f"  [{name}] per-strategy summary")
        scenario_payload = {
            "scenario": sc,
            "strategies": {},
        }
        for ps in PRIOR_STRATEGIES:
            trials = strategy_trials[ps["name"]]
            summary = summarize_trials(trials)
            scenario_payload["strategies"][ps["name"]] = {
                "meta": ps,
                "summary": summary,
                "trials": [serializable_trial(t) for t in trials],
            }
            print(
                f"    [{ps['name']:<8}] "
                f"NDCG@k={summary['ndcg_at_k']['mean']:.4f} "
                f"(SE={summary['ndcg_at_k']['se']:.4f}), "
                f"Top-k={summary['topk_overlap']['mean']:.4f}, "
                f"P@k={summary['precision_at_k']['mean']:.4f} [main] | "
                f"Spearman={summary['spearman']['mean']:.4f} [aux] | "
                f"MSE={summary['mse_theta']['mean']:.2f}, "
                f"MSE_cal={summary['mse_calibrated']['mean']:.3f} [ref]"
            )
        print()
        strategy_results[name] = scenario_payload

        # plotting/summary용 기본 전략은 baseline
        baseline_summary = scenario_payload["strategies"]["baseline"]["summary"]
        baseline_trials = strategy_trials["baseline"]
        all_results[name] = {
            "scenario": sc,
            "summary": baseline_summary,
            "trials": baseline_trials,
        }

    json_payload = {
        "settings": {
            "document": DOCUMENT,
            "R": R,
            "T": T,
            "BURN_IN": BURN_IN,
            "BETA_TRUE": BETA_TRUE,
            "n_minus": n,
            "simulation_type": "graph_consistent_theta_then_group_by_interval",
        },
        "scenarios": {name: payload["scenario"] for name, payload in all_results.items()},
        "full_results": {
            name: {
                "scenario": payload["scenario"],
                "summary": payload["summary"],
                "trials": [serializable_trial(t) for t in payload["trials"]],
            }
            for name, payload in all_results.items()
        },
        "strategy_results": strategy_results,
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, ensure_ascii=False, indent=2)

    plot_summary(all_results, summary_png)
    plot_representative_scatter(all_results, scatter_png)

    print("-" * 72)
    print(f"Saved JSON    -> {summary_json}")
    print(f"Saved summary -> {summary_png}")
    print(f"Saved scatter -> {scatter_png}")


if __name__ == "__main__":
    main()
