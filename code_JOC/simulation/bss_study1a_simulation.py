"""
BSS Study 1A: Semi-Synthetic Simulation
u_0 = 0 (flat prior), θ* = mu_g + noise
α*에 맞게 k 조정, μ와 α 둘 다 시나리오별 변경

실행: python bss_study1a_simulation.py
출력: bss_study1a_results.json
"""

import sys
import os
import json
import time

import numpy as np
from datetime import timedelta
from scipy.linalg import solve

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from keyphrase_functions import (
    create_fcm_phrases,
    inv_logit, alpha_find,
    posterior_gibbstheta, gibbs_mh,
)

# ── 설정 ──
DATA_DIR = "/home/jiyoon/3차/BSS-Keyphrase-Extraction-master/data_JOC"
DOCUMENT = "pre_process/C-42.txt.final"

R = 3
T = 2000
BURN_IN = 200
SIGMA_THETA = 1.0

grid = np.linspace(0.01, 0.95, 60)
d = 0.85

scenarios = [
    {"name": "Easy",      "mu_S": 2.5, "mu_W": 1.0, "mu_N": -2.5,
     "alpha": 0.10, "rho_S": 0.19, "rho_W": 0.19, "rho_N": 0.62},
    {"name": "Moderate",  "mu_S": 2.2, "mu_W": 0.5, "mu_N": -2.0,
     "alpha": 0.30, "rho_S": 0.19, "rho_W": 0.19, "rho_N": 0.62},
    {"name": "Difficult", "mu_S": 1.5, "mu_W": 0.0, "mu_N": -1.0,
     "alpha": 0.50, "rho_S": 0.19, "rho_W": 0.19, "rho_N": 0.62},
    {"name": "Sparse",    "mu_S": 2.2, "mu_W": 0.5, "mu_N": -2.0,
     "alpha": 0.30, "rho_S": 0.125, "rho_W": 0.125, "rho_N": 0.75},
]


def build_graph_matrices(graph):
    """그래프에서 B 행렬 추출"""
    n = graph['n_minus']
    G = solve(graph['D_minus'], graph['A_minus'])
    B = np.eye(n) - d * G.T
    return n, B


def sample_theta_star(n, mu_S, mu_W, mu_N, rho_S, rho_W, rho_N):
    """θ* = mu_g + noise (u_0과 무관)"""
    n_S = round(n * rho_S)
    n_W = round(n * rho_W)
    n_N = n - n_S - n_W

    group = np.array(['S'] * n_S + ['W'] * n_W + ['N'] * n_N)
    np.random.shuffle(group)

    theta_star = np.zeros(n)
    theta_star[group == 'S'] = mu_S + np.random.normal(0, SIGMA_THETA, n_S)
    theta_star[group == 'W'] = mu_W + np.random.normal(0, SIGMA_THETA, n_W)
    theta_star[group == 'N'] = mu_N + np.random.normal(0, SIGMA_THETA, n_N)

    return theta_star, group


def generate_Y(theta_star, alpha_true):
    """
    Y 생성:
    1. π* = sigmoid(θ*)
    2. Y_true ~ Bernoulli(π*)
    3. Y_obs: Y_true=1 중 (1-α*) 비율만 관측
    """
    pi_star = inv_logit(theta_star)
    Y_true = np.random.binomial(1, pi_star)

    true_pos_idx = np.where(Y_true == 1)[0]
    k = max(5, round(len(true_pos_idx) * (1 - alpha_true)))
    seed_idx = np.random.choice(true_pos_idx, min(k, len(true_pos_idx)), replace=False)

    Y = np.zeros(len(theta_star), dtype=float)
    Y[seed_idx] = 1

    return Y, Y_true


def run_single_trial(graph, sc):
    """단일 시행"""
    n, B = build_graph_matrices(graph)

    # 1. θ* = mu_g + noise
    theta_star, group = sample_theta_star(
        n, sc['mu_S'], sc['mu_W'], sc['mu_N'],
        sc['rho_S'], sc['rho_W'], sc['rho_N'],
    )
    pi_star = inv_logit(theta_star)

    # 2. Y 생성
    Y, Y_true = generate_Y(theta_star, sc['alpha'])

    # 3. prior mean = true parameter (sanity check)
    u_0 = theta_star.copy()

    # 4. MCMC 초기값: Y 기반 (C>0 보장, σ² cold start 방지)
    ini = np.where(Y == 1, 1.0, -0.5)
    alpha_est = alpha_find(ini, Y, grid)

    # 4. MCMC 실행
    result = gibbs_mh(
        Burn_in=BURN_IN, T=T, ini=ini, n=n,
        graph=graph, Y=Y, B=B, u_0=u_0,
        alpha_est=alpha_est, grid=grid, verbose=False,
    )

    # 5. 추정값
    theta_hat = np.mean(result['theta_store'][BURN_IN:, :], axis=0)
    pi_hat = result['poster_pi_mn']
    alpha_hat = result['alpha_mn']

    # 6. 메트릭
    mse_theta = float(np.mean((theta_hat - theta_star) ** 2))
    mse_pi = float(np.mean((pi_hat - pi_star) ** 2))
    alpha_bias = float(alpha_hat - sc['alpha'])

    return {
        'mse_theta': mse_theta,
        'mse_pi': mse_pi,
        'alpha_hat': float(alpha_hat),
        'alpha_bias': alpha_bias,
        'accept_rate': float(result['accept'] / T),
        'n_Y_true': int(np.sum(Y_true)),
        'n_Y_obs': int(np.sum(Y)),
    }


def main():
    print("=" * 60)
    print("BSS Study 1A: u_0=0, θ*=mu_g+noise")
    print(f"R={R}, T={T}, Burn-in={BURN_IN}")
    print("=" * 60)

    # ── 그래프 생성 ──
    file_path = os.path.join(DATA_DIR, DOCUMENT)
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()

    fcm, unique_phrases, phrase_to_idx, phrase_counts = create_fcm_phrases(text)
    A = fcm.copy()
    np.fill_diagonal(A, 0)
    row_sums = A.sum(axis=1); row_sums[row_sums == 0] = 1
    D = np.diag(row_sums)
    keep_indices = np.where((A > 0).sum(axis=1) >= 1)[0]
    A_f = A[np.ix_(keep_indices, keep_indices)]
    rs_f = A_f.sum(axis=1); rs_f[rs_f == 0] = 1

    graph = {
        'n': len(unique_phrases), 'A': A, 'D': D,
        'A_minus': A_f, 'D_minus': np.diag(rs_f),
        'n_minus': len(keep_indices),
        'dictionary_minus': np.array([[unique_phrases[i], i] for i in keep_indices], dtype=object),
        'keep_indices': keep_indices,
        'phrase_counts': phrase_counts,
        'unique_phrases': unique_phrases,
        'phrase_to_idx': phrase_to_idx,
    }
    n = graph['n_minus']
    print(f"n={n} phrases\n")

    all_results = {
        'settings': {'R': R, 'T': T, 'BURN_IN': BURN_IN, 'SIGMA_THETA': SIGMA_THETA, 'n': n},
        'scenarios': {},
    }

    total_start = time.time()

    for sc in scenarios:
        name = sc['name']
        print(f"{'=' * 60}")
        print(f"[{name}]  α*={sc['alpha']}, "
              f"μ=({sc['mu_S']}, {sc['mu_W']}, {sc['mu_N']}), "
              f"ρ=({sc['rho_S']}, {sc['rho_W']}, {sc['rho_N']})")
        print("=" * 60)

        trials = []
        for r in range(R):
            np.random.seed(42 + r * 100)
            t0 = time.time()
            trial = run_single_trial(graph, sc)
            elapsed = time.time() - t0
            trials.append(trial)
            print(f"  Trial {r+1}/{R}: "
                  f"MSE_θ={trial['mse_theta']:.4f}, "
                  f"MSE_π={trial['mse_pi']:.6f}, "
                  f"α̂={trial['alpha_hat']:.3f} "
                  f"(bias={trial['alpha_bias']:+.3f}), "
                  f"accept={trial['accept_rate']:.3f}, "
                  f"Y_true={trial['n_Y_true']}, "
                  f"Y_obs={trial['n_Y_obs']}, "
                  f"({elapsed:.1f}s)")

        # 집계
        summary = {
            'alpha_true': sc['alpha'],
            'mu_S': sc['mu_S'], 'mu_W': sc['mu_W'], 'mu_N': sc['mu_N'],
            'rho_S': sc['rho_S'], 'rho_W': sc['rho_W'], 'rho_N': sc['rho_N'],
            'MSE_theta_mean': float(np.mean([t['mse_theta'] for t in trials])),
            'MSE_theta_SE': float(np.std([t['mse_theta'] for t in trials]) / np.sqrt(R)),
            'MSE_pi_mean': float(np.mean([t['mse_pi'] for t in trials])),
            'MSE_pi_SE': float(np.std([t['mse_pi'] for t in trials]) / np.sqrt(R)),
            'alpha_bias_mean': float(np.mean([t['alpha_bias'] for t in trials])),
            'alpha_RMSE': float(np.sqrt(np.mean(np.array([t['alpha_bias'] for t in trials]) ** 2))),
            'alpha_hat_mean': float(np.mean([t['alpha_hat'] for t in trials])),
            'accept_rate_mean': float(np.mean([t['accept_rate'] for t in trials])),
            'trials': trials,
        }
        all_results['scenarios'][name] = summary

        print(f"\n  --- {name} 요약 ---")
        print(f"  MSE_θ  = {summary['MSE_theta_mean']:.4f} ± {summary['MSE_theta_SE']:.4f}")
        print(f"  MSE_π  = {summary['MSE_pi_mean']:.6f} ± {summary['MSE_pi_SE']:.6f}")
        print(f"  α̂ bias = {summary['alpha_bias_mean']:+.4f}, RMSE_α = {summary['alpha_RMSE']:.4f}")
        print(f"  α̂ mean = {summary['alpha_hat_mean']:.3f} (true={sc['alpha']})\n")

    total_elapsed = time.time() - total_start

    # 저장
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bss_study1a_results.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # 최종 요약
    print("=" * 60)
    print(f"완료! {timedelta(seconds=int(total_elapsed))}")
    print(f"저장: {out_path}\n")
    print(f"{'시나리오':<12} {'α*':>5} {'MSE_θ':>10} {'MSE_π':>12} {'α̂':>8} {'α̂ bias':>9}")
    print("-" * 60)
    for name, s in all_results['scenarios'].items():
        print(f"{name:<12} {s['alpha_true']:>5.2f} {s['MSE_theta_mean']:>10.4f} "
              f"{s['MSE_pi_mean']:>12.6f} {s['alpha_hat_mean']:>8.3f} "
              f"{s['alpha_bias_mean']:>+9.4f}")


if __name__ == "__main__":
    main()
