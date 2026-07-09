#!/usr/bin/env python3
"""
AWSGLD with tunable step-size (eps_k) schedule.

원본 keyphrase_functions_awsgld.gibbs_mh 를 그대로 복사하되,
eps_k = eps_scale / ((t+1)^eps_pow + eps_offset) 로 파라미터화한 사본.
원본 파일은 건드리지 않고, 에너지/gradient/유틸/전역값은 import 해서 동기화.

기본값 (eps_scale=0.3, eps_pow=0.6, eps_offset=10) → 원본과 동일.
eps_scale 을 키우면 보폭↑ → heavy-tail 탐색 범위↑ (단 과하면 발산).
"""
import time
from datetime import timedelta
import numpy as np
from scipy.linalg import solve
from scipy.stats import invgamma

import keyphrase_functions_awsgld as M   # 원본 함수/전역값 사용 (수정 안 함)


def gibbs_mh_eps(Burn_in, T, ini, n, graph, Y, B, u_0, alpha_est, grid,
                 eps_scale=0.3, eps_pow=0.6, eps_offset=10.0,
                 batch_size=None, sigma2_floor=0.5, verbose=False):
    """원본 gibbs_mh 와 동일, eps_k 스케줄만 (eps_scale, eps_pow, eps_offset) 로 조정."""
    M_REGIONS = M.M_REGIONS; ZETA = M.ZETA; TAU = M.TAU; DECAY_LR = M.DECAY_LR
    inv_logit = M.inv_logit; alpha_find = M.alpha_find
    posterior_energy = M.posterior_energy; grad_posterior_energy = M.grad_posterior_energy

    theta_store = np.zeros((T, n)); sigma2_store = np.zeros(T)
    alpha_store = np.zeros(T); accept = 0
    theta = ini.copy()
    adaptive_weights = np.arange(1, M_REGIONS + 1, dtype=float) / M_REGIONS
    energy_samples = []; warmup = min(100, max(10, T // 20))
    energy_min = None; delta_u_actual = None; J_tilde = M_REGIONS - 1

    BtB_fixed = B.T @ B
    ridge = 1e-6 * np.trace(BtB_fixed) / n
    P_precond = solve(BtB_fixed + ridge * np.eye(n), np.eye(n))
    P_sym = 0.5 * (P_precond + P_precond.T)
    L_precond = np.linalg.cholesky(P_sym + 1e-10 * np.eye(n))

    for t in range(T):
        C = (B @ (theta - u_0)).T @ (B @ (theta - u_0))
        sigma2 = invgamma.rvs(n / 2 + 0.001, scale=C / 2 + 0.001)
        sigma2 = max(sigma2, sigma2_floor)
        sigma2_store[t] = sigma2

        eps_k = eps_scale / ((t + 1) ** eps_pow + eps_offset)   # ← 파라미터화된 보폭
        decay = min(1.0, DECAY_LR / (((t + 1) ** 0.75) + 1000.0))

        U_tilde = posterior_energy(Y, alpha_est, theta, u_0, B, sigma2)
        batch_idx = None if (batch_size is None or batch_size >= n) else np.random.choice(n, size=batch_size, replace=False)
        grad_U = grad_posterior_energy(Y, alpha_est, theta, u_0, B, sigma2, batch_idx=batch_idx, BtB=BtB_fixed)

        if t < warmup:
            energy_samples.append(U_tilde); grad_mult = 1.0
            if t == warmup - 1:
                e_min = np.min(energy_samples); e_max = np.max(energy_samples)
                e_range = max(e_max - e_min, 1.0)
                energy_min = e_min - 0.5 * e_range; energy_max = e_max + 0.5 * e_range
                delta_u_actual = max((energy_max - energy_min) / M_REGIONS, 1e-8)
                energy_samples = None
        else:
            J_tilde = int(np.clip((U_tilde - energy_min) / delta_u_actual + 1, 1, M_REGIONS - 1))
            grad_mult = 1 + (ZETA * TAU / delta_u_actual) * (
                np.log(adaptive_weights[J_tilde] + 1e-12) - np.log(adaptive_weights[J_tilde - 1] + 1e-12))
            grad_mult = np.clip(grad_mult, 0.1, 10.0)

        noise = np.random.randn(n)
        theta = theta - eps_k * grad_mult * (P_precond @ grad_U) + np.sqrt(2 * TAU * eps_k) * (L_precond @ noise)
        theta = np.clip(theta, -700, 700)

        if t >= warmup:
            cw = adaptive_weights[J_tilde]
            adaptive_weights[J_tilde:] = adaptive_weights[J_tilde:] + decay * cw * (1.0 - adaptive_weights[J_tilde:])
            adaptive_weights[:J_tilde] = adaptive_weights[:J_tilde] - decay * cw * adaptive_weights[:J_tilde]
            adaptive_weights = np.clip(adaptive_weights, 1e-10, 1.0)

        theta_store[t, :] = theta
        alpha_est = alpha_find(theta, Y, grid); alpha_store[t] = alpha_est

    prob = inv_logit(theta_store)
    poster_pi_md = np.median(prob[Burn_in:T, :], axis=0)
    m1 = poster_pi_md == 1; poster_pi_md[m1] = 1 + np.random.normal(0, 0.01, np.sum(m1))
    poster_pi_mn = np.mean(prob[Burn_in:T, :], axis=0)
    return {'poster_pi_md': poster_pi_md, 'poster_pi_mn': poster_pi_mn,
            'theta_store': theta_store, 'sigma2_store': sigma2_store,
            'alpha_mn': np.mean(alpha_store[Burn_in:]), 'alpha_md': np.median(alpha_store[Burn_in:])}
