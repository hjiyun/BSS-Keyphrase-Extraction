#!/usr/bin/env python3
"""
MALA (Metropolis-adjusted Langevin Algorithm) for the BSS keyphrase posterior.

핵심: Langevin gradient 제안 + MH 채택/기각 보정 → exact 샘플러.
  - 제안: θ* = θ − (ε/2)·P·∇U(θ) + sqrt(ε)·L·ξ   (LLᵀ=P)
  - 채택: MH ratio (제안 비대칭성 보정) → 두꺼운 꼬리에서도 편향 없음.
타깃은 σ² 적분형(모델 충실) 에너지 U = posterior_energy_integrated (acMH와 동일).

기존 awsgld/acmh 파일은 건드리지 않고 에너지·gradient만 import.
"""
import os, sys, time
import numpy as np

_CODE = os.path.dirname(os.path.abspath(__file__))
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)
import keyphrase_functions_awsgld as M   # inv_logit/base_to_start/alpha_find/alpha_lk (원본 함수만 사용)


# ── σ² 적분형 에너지/gradient (BSS 모델 충실, acMH와 동일 타깃) ──
# 원본 awsgld 파일을 건드리지 않도록 여기서 자체 정의한다.
def energy_integrated(Y, alpha, theta, u_0, B, eps=0.001):
    n = theta.shape[0]
    temp = np.clip((1 - alpha) * M.inv_logit(theta), 1e-10, 1 - 1e-10)
    C = (B @ (theta - u_0)) @ (B @ (theta - u_0))
    lglk = np.sum(Y * np.log(temp) + (1 - Y) * np.log(1 - temp)) - (n / 2 + eps) * np.log(C / 2 + eps)
    return -lglk


def grad_energy_integrated(Y, alpha, theta, u_0, B, BtB=None, eps=0.001):
    n = theta.shape[0]
    pi = np.clip(M.inv_logit(theta), 1e-10, 1 - 1e-10)
    dpi = pi * (1 - pi)
    temp = np.clip((1 - alpha) * pi, 1e-10, 1 - 1e-10)
    denom = np.clip(1 - temp, 1e-10, None)
    grad_ll = np.zeros_like(theta)
    seed = (Y == 1); unl = ~seed
    grad_ll[seed] = 1 - pi[seed]
    grad_ll[unl] = -(1 - alpha) * dpi[unl] / denom[unl]
    if BtB is None:
        BtB = B.T @ B
    diff = theta - u_0
    C = (B @ diff) @ (B @ diff)
    coef = (n / 2 + eps) / (C / 2 + eps)            # = E[1/σ²|θ]
    grad_prior = -coef * (BtB @ diff)
    return -(grad_ll + grad_prior)


def mala_sampler(T, ini, n, B, Y, u_0, alpha_est, grid,
                 Burn_in=2000, eps0=0.1, preconditioned=True,
                 target_accept=0.574, adapt=True, update_alpha=True, verbose=False):
    """
    반환: dict(theta_store, poster_pi_mn, poster_pi_md, alpha_store, accept_rate, eps)
    eps0: 초기 step size. adapt=True면 burn-in 동안 target_accept(MALA 최적 ~0.574)로 조정.
    preconditioned=True: P=(BᵀB)⁻¹ 기하 사용 (그래프 곡률 보정).
    update_alpha=True: 매 스텝 alpha를 grid 최대우도로 갱신 (다른 샘플러와 동일 패턴).
    """
    BtB = B.T @ B
    if preconditioned:
        ridge = 1e-6 * np.trace(BtB) / n
        Pinv = BtB + ridge * np.eye(n)          # = P⁻¹ (제안 밀도 metric)
        P = np.linalg.inv(Pinv)
        P = 0.5 * (P + P.T)
        L = np.linalg.cholesky(P + 1e-10 * np.eye(n))
    else:
        Pinv = np.eye(n); P = np.eye(n); L = np.eye(n)

    def U(th, a):
        return energy_integrated(Y, a, th, u_0, B)

    def gradU(th, a):
        return grad_energy_integrated(Y, a, th, u_0, B, BtB=BtB)

    theta = ini.copy()
    log_eps = np.log(eps0)
    theta_store = np.zeros((T, n))
    alpha_store = np.zeros(T)
    accept = 0
    t0 = time.time()

    for t in range(T):
        eps = np.exp(log_eps)
        g = gradU(theta, alpha_est)
        Uc = U(theta, alpha_est)
        # forward 제안
        m = theta - 0.5 * eps * (P @ g)
        xi = np.random.randn(n)
        theta_star = np.clip(m + np.sqrt(eps) * (L @ xi), -700, 700)
        g_star = gradU(theta_star, alpha_est)
        U_star = U(theta_star, alpha_est)
        m_star = theta_star - 0.5 * eps * (P @ g_star)   # reverse 제안 평균
        # log q(θ|θ*) − log q(θ*|θ), metric = P⁻¹
        d_fwd = theta_star - m
        d_rev = theta - m_star
        logq_fwd = -0.5 / eps * (d_fwd @ (Pinv @ d_fwd))
        logq_rev = -0.5 / eps * (d_rev @ (Pinv @ d_rev))
        log_alpha = (-U_star + Uc) + (logq_rev - logq_fwd)
        a_prob = min(1.0, np.exp(min(log_alpha, 0.0)))

        if np.log(np.random.rand() + 1e-300) < log_alpha:
            theta = theta_star
            accept += 1

        # step size 적응 (burn-in 동안만): Robbins-Monro
        if adapt and t < Burn_in:
            lr = 0.05 / (1 + t / 200.0)
            log_eps += lr * (a_prob - target_accept)
            log_eps = np.clip(log_eps, np.log(1e-6), np.log(50.0))

        if update_alpha:
            alpha_est = grid[np.argmax([M.alpha_lk(theta, Y, gg) for gg in grid])]
        theta_store[t] = theta
        alpha_store[t] = alpha_est

        if verbose and (t + 1) % max(1, T // 10) == 0:
            print(f"  MALA {100*(t+1)/T:.0f}% | eps={eps:.4f} | acc={accept/(t+1):.3f} "
                  f"| {time.time()-t0:.0f}s", flush=True)

    prob = M.inv_logit(theta_store[Burn_in:T])
    poster_pi_mn = prob.mean(axis=0)
    poster_pi_md = np.median(prob, axis=0)
    md1 = poster_pi_md == 1
    poster_pi_md[md1] = 1 + np.random.normal(0, 0.01, md1.sum())
    return {
        'theta_store': theta_store,
        'poster_pi_mn': poster_pi_mn,
        'poster_pi_md': poster_pi_md,
        'alpha_store': alpha_store,
        'accept_rate': accept / T,
        'eps': float(np.exp(log_eps)),
    }


def mala_v2(T, ini, n, B, Y, u_0, alpha_est, grid,
            Burn_in=2000, eps0=0.1, state_precond=True, coord_step=True,
            precond_reg=0.25, target_accept=0.574, update_alpha=True, verbose=False):
    """
    개선된 MALA — 두 가지를 BᵀB 고유기저에서 통합:
      (1) coord_step : 방향별(좌표별) step 스케일 a_i 를 burn-in 동안 분산에 맞춰 적응
      (2) state_precond : 상태의존 preconditioner P(θ) = (coef(θ)·BᵀB + reg)⁻¹,
                          coef(θ)=E[1/σ²|θ]=(n/2+ε)/(C/2+ε) (sharp 곡률 보정)

    고유기저 BᵀB = V diag(Λ) Vᵀ. 그 기저에서 P 의 고유값:
        d_i = a_i / base_i,   base_i = (coef·Λ_i + reg) if state_precond else 1
                              a_i    = 적응 스케일       if coord_step    else 1
    상태의존이라 MH 비율에 log|P| 보정 포함 (정확성 유지).
    """
    BtB = B.T @ B
    epsc = 0.001
    evals, V = np.linalg.eigh(BtB)
    evals = np.clip(evals, 0.0, None)

    def U(th, a):
        return energy_integrated(Y, a, th, u_0, B)

    def gradU(th, a):
        return grad_energy_integrated(Y, a, th, u_0, B, BtB=BtB)

    def base_prec(th):
        if not state_precond:
            return np.ones(n)
        C = (B @ (th - u_0)) @ (B @ (th - u_0))
        coef = (n / 2 + epsc) / (C / 2 + epsc)
        return coef * evals + precond_reg          # P⁻¹ 고유값(스케일 a 제외)

    # 방향별 적응 스케일 a_i (고유기저). burn-in 동안 φ=Vᵀθ 의 분산으로 적응.
    a_scale = np.ones(n)
    cnt = 0; mean_phi = np.zeros(n); M2_phi = np.zeros(n)

    def pieces(th, a, eps):
        """주어진 θ에서 P 고유값 d, 제안평균 m, 그리고 보조량 반환."""
        g = gradU(th, a)
        bp = base_prec(th)
        d = a_scale / bp                            # P 고유값
        gV = V.T @ g
        # m = θ - (eps/2) P g  = θ - (eps/2) V (d * gV)
        m = th - 0.5 * eps * (V @ (d * gV))
        return d, m, bp

    def logq(x, m, d):
        """log N(x; m, eps·P) 의 θ-의존 부분: -1/2 log|P| - 1/(2eps) (x-m)ᵀP⁻¹(x-m)."""
        r = V.T @ (x - m)
        quad = np.sum((r * r) / (d * eps))          # (x-m)ᵀ P⁻¹ (x-m)/eps,  P⁻¹ 고유값=1/d
        logdetP = np.sum(np.log(d))
        return -0.5 * logdetP - 0.5 * quad

    theta = ini.copy()
    log_eps = np.log(eps0)
    theta_store = np.zeros((T, n)); alpha_store = np.zeros(T)
    accept = 0; t0 = time.time()

    for t in range(T):
        eps = np.exp(log_eps)
        Uc = U(theta, alpha_est)
        d_f, m_f, _ = pieces(theta, alpha_est, eps)
        xi = np.random.randn(n)
        theta_star = np.clip(m_f + V @ (np.sqrt(eps * d_f) * xi), -700, 700)
        U_star = U(theta_star, alpha_est)
        d_r, m_r, _ = pieces(theta_star, alpha_est, eps)
        log_alpha = (-U_star + Uc) + (logq(theta, m_r, d_r) - logq(theta_star, m_f, d_f))
        a_prob = min(1.0, np.exp(min(log_alpha, 0.0)))
        if np.log(np.random.rand() + 1e-300) < log_alpha:
            theta = theta_star; accept += 1

        if t < Burn_in:
            # eps 전역 적응
            lr = 0.05 / (1 + t / 200.0)
            log_eps = np.clip(log_eps + lr * (a_prob - target_accept), np.log(1e-6), np.log(50.0))
            # 방향별 분산 적응 (coord_step)
            if coord_step:
                phi = V.T @ theta
                cnt += 1
                dlt = phi - mean_phi; mean_phi += dlt / cnt; M2_phi += dlt * (phi - mean_phi)
                if cnt > 20 and t % 50 == 0:
                    var_phi = M2_phi / max(cnt - 1, 1)
                    s = np.clip(var_phi, 1e-3, None)
                    a_scale = s / np.exp(np.mean(np.log(s)))   # 기하평균=1 정규화

        if update_alpha:
            alpha_est = grid[np.argmax([M.alpha_lk(theta, Y, gg) for gg in grid])]
        theta_store[t] = theta; alpha_store[t] = alpha_est
        if verbose and (t + 1) % max(1, T // 10) == 0:
            print(f"  MALAv2 {100*(t+1)/T:.0f}% | eps={eps:.4f} | acc={accept/(t+1):.3f} | {time.time()-t0:.0f}s", flush=True)

    prob = M.inv_logit(theta_store[Burn_in:T])
    pmn = prob.mean(axis=0); pmd = np.median(prob, axis=0)
    md1 = pmd == 1; pmd[md1] = 1 + np.random.normal(0, 0.01, md1.sum())
    return {'theta_store': theta_store, 'poster_pi_mn': pmn, 'poster_pi_md': pmd,
            'alpha_store': alpha_store, 'accept_rate': accept / T, 'eps': float(np.exp(log_eps))}
