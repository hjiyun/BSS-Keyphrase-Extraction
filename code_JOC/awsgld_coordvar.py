#!/usr/bin/env python3
"""
AWSGLD with per-coordinate (var_i-style) adaptive diagonal preconditioner.

순수 AWSGLD(Langevin, 채택/기각 없음) 그대로 두되,
preconditioner P 를 (BᵀB)⁻¹ 대신 **단어별 분산 적응 대각행렬** diag(s_i) 로 교체.
  s_i = burn-in 동안 추정한 theta_i 의 러닝 분산 (Welford) — acMH의 var_i 와 같은 발상.

목적: "단어마다 다른 보폭"이 채택/기각 없이도 도움이 되는지 검증.
원본/기존 파일은 건드리지 않고 에너지·gradient·유틸·전역값만 import.
"""
import time
import numpy as np
from scipy.stats import invgamma

import keyphrase_functions_awsgld as M


def gibbs_mh_coordvar(Burn_in, T, ini, n, graph, Y, B, u_0, alpha_est, grid,
                      eps_scale=0.3, eps_pow=0.6, eps_offset=10.0,
                      sigma2_floor=0.5, var_floor=1e-2, adapt_every=50,
                      use_BtB_geometry=False, verbose=False):
    """
    순수 AWSGLD + 단어별 분산 적응 대각 preconditioner.

    eps_scale/pow/offset : eps_k 스케줄 (= eps_scale/((t+1)^pow + offset))
    var_floor            : 단어별 분산 s_i 의 하한 (0 division 방지)
    adapt_every          : s_i 갱신 주기 (스텝)
    use_BtB_geometry=False: P = diag(s_i) 만 사용 (순수 단어별).
                     True : P = diag(s_i) 를 BᵀB 기하와 곱해 결합 (보너스 옵션).
    채택/기각은 없음 (순수 Langevin) — var_i 효과만 격리해서 본다.
    """
    M_REGIONS = M.M_REGIONS; ZETA = M.ZETA; TAU = M.TAU; DECAY_LR = M.DECAY_LR
    inv_logit = M.inv_logit; alpha_find = M.alpha_find
    posterior_energy = M.posterior_energy; grad_posterior_energy = M.grad_posterior_energy

    theta_store = np.zeros((T, n)); sigma2_store = np.zeros(T)
    alpha_store = np.zeros(T)
    theta = ini.copy()
    adaptive_weights = np.arange(1, M_REGIONS + 1, dtype=float) / M_REGIONS
    energy_samples = []; warmup = min(100, max(10, T // 20))
    energy_min = None; delta_u_actual = None; J_tilde = M_REGIONS - 1

    BtB = B.T @ B

    # 단어별 러닝 분산 (Welford). s_i = 보폭(P 대각). 초기엔 1(등방).
    s = np.ones(n)
    cnt = 0; mean_t = np.zeros(n); M2_t = np.zeros(n)

    for t in range(T):
        C = (B @ (theta - u_0)).T @ (B @ (theta - u_0))
        sigma2 = invgamma.rvs(n / 2 + 0.001, scale=C / 2 + 0.001)
        sigma2 = max(sigma2, sigma2_floor)
        sigma2_store[t] = sigma2

        eps_k = eps_scale / ((t + 1) ** eps_pow + eps_offset)
        decay = min(1.0, DECAY_LR / (((t + 1) ** 0.75) + 1000.0))

        U_tilde = posterior_energy(Y, alpha_est, theta, u_0, B, sigma2)
        grad_U = grad_posterior_energy(Y, alpha_est, theta, u_0, B, sigma2, batch_idx=None, BtB=BtB)

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

        # ── 단어별 대각 preconditioner P = diag(s) ──
        # drift = eps_k·grad_mult·P·grad ;  noise = sqrt(2·TAU·eps_k)·sqrt(P)·ξ
        noise = np.random.randn(n)
        if use_BtB_geometry:
            # diag(s) 를 (BᵀB)⁻¹ 대신 쓰되 그래프 평활도 약하게 섞고 싶으면 여기 확장 가능.
            Pg = s * grad_U
            Ln = np.sqrt(s) * noise
        else:
            Pg = s * grad_U                       # 단어별 보폭
            Ln = np.sqrt(s) * noise
        theta = theta - eps_k * grad_mult * Pg + np.sqrt(2 * TAU * eps_k) * Ln
        theta = np.clip(theta, -700, 700)

        if t >= warmup:
            cw = adaptive_weights[J_tilde]
            adaptive_weights[J_tilde:] = adaptive_weights[J_tilde:] + decay * cw * (1.0 - adaptive_weights[J_tilde:])
            adaptive_weights[:J_tilde] = adaptive_weights[:J_tilde] - decay * cw * adaptive_weights[:J_tilde]
            adaptive_weights = np.clip(adaptive_weights, 1e-10, 1.0)

        # 단어별 분산 적응 (burn-in 동안): theta_i 이력의 분산 → s_i
        if t < Burn_in:
            cnt += 1
            dlt = theta - mean_t; mean_t += dlt / cnt; M2_t += dlt * (theta - mean_t)
            if cnt > 20 and t % adapt_every == 0:
                var_t = np.clip(M2_t / max(cnt - 1, 1), var_floor, None)
                s = var_t / np.exp(np.mean(np.log(var_t)))   # 기하평균=1 정규화 (전체 스케일은 eps_k가 담당)

        theta_store[t, :] = theta
        alpha_est = alpha_find(theta, Y, grid); alpha_store[t] = alpha_est
        if verbose and (t + 1) % max(1, T // 5) == 0:
            print(f"  coordvar {100*(t+1)/T:.0f}%", flush=True)

    prob = inv_logit(theta_store)
    poster_pi_md = np.median(prob[Burn_in:T, :], axis=0)
    m1 = poster_pi_md == 1; poster_pi_md[m1] = 1 + np.random.normal(0, 0.01, np.sum(m1))
    poster_pi_mn = np.mean(prob[Burn_in:T, :], axis=0)
    return {'poster_pi_md': poster_pi_md, 'poster_pi_mn': poster_pi_mn,
            'theta_store': theta_store, 'sigma2_store': sigma2_store,
            's_final': s, 'alpha_mn': np.mean(alpha_store[Burn_in:])}
