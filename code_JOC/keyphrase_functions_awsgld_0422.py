"""
다수 문서 Batch 처리 — 단어(Word) 기반 키프레이즈 추출
R 원본 Keyphrase_functions.R의 main.function2() Python 변환
(Componentwise MCMC: Component_MCMC.cpp → Python 변환 포함)

원본 keyphrase_functions.py와 동일한 구조를 유지하되,
gibbs_mh의 theta 업데이트만 AWSGLD로 교체한 버전.
"""

import numpy as np
import os
import re
import time
from datetime import timedelta
from scipy.linalg import solve
from scipy.stats import invgamma

# ========== 설정 ==========
DATA_DIR = "/home/jiyoon/3차/BSS-Keyphrase-Extraction-master/data_JOC"
PREPROCESS_DIR = os.path.join(DATA_DIR, "pre_process")
TRUTH_DIR = os.path.join(DATA_DIR, "pre_process_reader_truth")

# MCMC 파라미터 (R 원본과 동일)
T = 50000
BURN_IN = 2000

# AWSGLD 파라미터
TAU = 1.0
ZETA = 5.0
M_REGIONS = 1000
DECAY_LR = 100.0

np.random.seed(12345)
k = 4
grid = (np.arange(10, 43) - 5) / np.arange(10, 43)
FDR_LEVELS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]


# ========== 기본 함수들 ==========
def inv_logit(x):
    x = np.clip(x, -700, 700)
    return np.exp(x) / (1 + np.exp(x))


def base_to_start(Base_Line):
    """R Base_to_start()"""
    ini_point = Base_Line.copy()
    ini_point[ini_point >= 1] = 0.99
    ini_point[ini_point <= 0] = 0.01
    ini_point = np.log(ini_point / (1 - ini_point))
    return ini_point


def alpha_find(Base_Line, Y, grid):
    alpha_est = grid[np.argmax([alpha_lk(Base_Line, Y, alpha) for alpha in grid])]
    return alpha_est


def alpha_lk(Base_Line, Y, alpha):
    pi = inv_logit(Base_Line)
    pi = np.clip(pi, 1e-10, 1 - 1e-10)
    temp = (1 - alpha) * pi
    temp = np.clip(temp, 1e-10, 1 - 1e-10)
    return np.sum(Y * np.log(temp) + (1 - Y) * np.log(1 - temp))


def posterior_gibbstheta(Y, alpha, theta, u_0, B, sigma2):
    """Posterior of theta, given sigma^2 — Gibbs.MH용"""
    temp = (1 - alpha) * inv_logit(theta)
    temp = np.clip(temp, 1e-10, 1 - 1e-10)
    C = (B @ (theta - u_0)).T @ (B @ (theta - u_0))
    lglk = np.sum(Y * np.log(temp) + (1 - Y) * np.log(1 - temp)) - C / (2 * sigma2)
    return lglk


def posterior_energy(Y, alpha, theta, u_0, B, sigma2):
    """AWSGLD용 energy = -log posterior"""
    return -posterior_gibbstheta(Y, alpha, theta, u_0, B, sigma2)


def grad_posterior_energy(Y, alpha, theta, u_0, B, sigma2):
    """AWSGLD용 gradient of energy = -gradient of log posterior"""
    pi_theta = inv_logit(theta)
    pi_theta = np.clip(pi_theta, 1e-10, 1 - 1e-10)

    dpi_dtheta = pi_theta * (1 - pi_theta)
    temp = (1 - alpha) * pi_theta
    temp = np.clip(temp, 1e-10, 1 - 1e-10)
    denom = np.clip(1 - temp, 1e-10, None)

    grad_ll = np.zeros_like(theta)
    seed_mask = (Y == 1)
    unlabeled_mask = ~seed_mask

    if np.any(seed_mask):
        grad_ll[seed_mask] = 1 - pi_theta[seed_mask]
    if np.any(unlabeled_mask):
        grad_ll[unlabeled_mask] = -(1 - alpha) * dpi_dtheta[unlabeled_mask] / denom[unlabeled_mask]

    BtB = B.T @ B
    grad_prior = -BtB @ (theta - u_0) / sigma2
    return -(grad_ll + grad_prior)


def log_posterior(n, Y, alpha, theta, u_0, B):
    """Posterior of theta (sigma^2 integrated out) — Componentwise MCMC용
    R: log.posterior() / C++ log_posterior()"""
    temp = (1 - alpha) * inv_logit(theta)
    temp = np.clip(temp, 1e-10, 1 - 1e-10)
    C = (B @ (theta - u_0)).T @ (B @ (theta - u_0))
    lglk = np.sum(Y * np.log(temp) + (1 - Y) * np.log(1 - temp)) - (n / 2 + 0.001) * np.log(C / 2 + 0.001)
    return lglk


def force_obs_to_key(Y, poster_pi_mean, k):
    """R force_obs_to_key(): Y==1 위치에 큰 값 부여"""
    poster_pi_mean = poster_pi_mean.copy()
    obs_indices = np.where(Y == 1)[0]
    if len(obs_indices) > 0:
        poster_pi_mean[obs_indices] = 10 + np.random.normal(0.01, 0.01, len(obs_indices))
    return poster_pi_mean


def force_obs_to_key2(Y, poster_pi_mean):
    """R force_obs_to_key2(): Y==1 위치를 1로 설정"""
    poster_pi_mean = poster_pi_mean.copy()
    poster_pi_mean[Y == 1] = 1
    return poster_pi_mean


def FDR_calculate(cutoff, poster_md_adjust):
    set_vals = poster_md_adjust[poster_md_adjust >= cutoff]
    return np.sum(1 - set_vals) / len(set_vals) if len(set_vals) > 0 else 0


def vec_FDR_cal(cutoffs, poster_md_adjust):
    return np.array([FDR_calculate(c, poster_md_adjust) for c in cutoffs])


# ========== FDR Cutoff (R FDR_cutoff — truth 포함 버전) ==========
def FDR_cutoff_full(poster_pi_md, c, Y, truth):
    """
    R FDR_cutoff(): cutoff 계산 + FDR_pos, FDR_tp, Real_FDR 반환
    """
    poster_md_adjust = force_obs_to_key2(Y, poster_pi_md)
    cutoffs = np.unique(poster_md_adjust)[::-1]  # 내림차순
    FDRs = vec_FDR_cal(cutoffs, poster_md_adjust)

    valid = np.where(FDRs < c)[0]
    if len(valid) > 0:
        index = np.max(valid)
        cutoff = cutoffs[index]
    else:
        cutoff = cutoffs[np.argmin(FDRs)]

    # R: FDR_pos <- sum(poster_md_adjust >= cutoff)
    selected = np.where(poster_md_adjust >= cutoff)[0]
    FDR_pos = len(selected)
    # R: FDR_tp <- sum(which(poster_md_adjust >= cutoff) %in% truth)
    FDR_tp = np.sum(np.isin(selected, truth))
    # R: Real_FDR <- (FDR_pos - FDR_tp) / FDR_pos
    Real_FDR = (FDR_pos - FDR_tp) / FDR_pos if FDR_pos > 0 else 0

    return FDR_pos, FDR_tp, Real_FDR


def vec_FDR_cutoff(poster_pi_md, c_levels, Y, truth):
    """R vec_FDR_cutoff: 여러 FDR level에 대해 FDR_cutoff_full 실행"""
    FDR_pos = np.zeros(len(c_levels))
    FDR_tp = np.zeros(len(c_levels))
    Real_FDR = np.zeros(len(c_levels))
    for i, c in enumerate(c_levels):
        FDR_pos[i], FDR_tp[i], Real_FDR[i] = FDR_cutoff_full(poster_pi_md, c, Y, truth)
    return FDR_pos, FDR_tp, Real_FDR


# ========== Geweke 수렴 진단 ==========
def geweke_diag(chain, frac1=0.1, frac2=0.5):
    """
    R coda::geweke.diag() 간략 구현
    앞쪽 frac1 구간과 뒤쪽 frac2 구간의 평균 비교 z-score
    """
    n = len(chain)
    n1 = int(n * frac1)
    n2 = int(n * frac2)
    if n1 < 2 or n2 < 2:
        return 0.0
    chain1 = chain[:n1]
    chain2 = chain[n - n2:]
    mean1, mean2 = np.mean(chain1), np.mean(chain2)
    var1 = np.var(chain1, ddof=1) / n1
    var2 = np.var(chain2, ddof=1) / n2
    denom = np.sqrt(var1 + var2)
    if denom < 1e-15:
        return 0.0
    return (mean1 - mean2) / denom


# ========== 단어(Word) 기반 FCM 생성 (R quanteda::fcm 동일) ==========
def create_fcm_words(text, window=2):
    """R quanteda::fcm(raw_text, context="window", window=2, tri=F)"""
    text = text.lower()
    text = re.sub(r'[\t\r;]', '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    words = text.split()

    seen = {}
    unique_words = []
    for w in words:
        if w not in seen:
            seen[w] = len(unique_words)
            unique_words.append(w)

    word_to_idx = seen
    n = len(unique_words)

    fcm = np.zeros((n, n))
    for i in range(len(words)):
        for j in range(i + 1, min(i + window + 1, len(words))):
            idx1 = word_to_idx[words[i]]
            idx2 = word_to_idx[words[j]]
            if idx1 != idx2:
                fcm[idx1, idx2] += 1
                fcm[idx2, idx1] += 1

    return fcm, unique_words, word_to_idx


# ========== 그래프 생성 (R graph.generate) ==========
def create_graph(article_name):
    """
    R graph.generate() 재현
    article_name: 확장자 없는 이름 (예: "C-42")
    """
    text_path = os.path.join(PREPROCESS_DIR, article_name + ".txt.final")
    if not os.path.exists(text_path):
        text_path = os.path.join(DATA_DIR, article_name + ".txt")
    if not os.path.exists(text_path):
        return None

    with open(text_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()

    fcm, unique_words, word_to_idx = create_fcm_words(text, window=2)
    n = len(unique_words)
    if n == 0:
        return None

    A = fcm.copy()
    np.fill_diagonal(A, 0)
    row_sums = A.sum(axis=1)
    D = np.diag(row_sums)

    # R: units <- length(strsplit(raw_text, " ")[[1]])
    raw_text = text.lower()
    raw_text = re.sub(r'[\t\r;]', '', raw_text)
    raw_text = re.sub(r'\n', ' ', raw_text)
    units = len(raw_text.split())

    # R: degree <- sum((as.matrix(out)!=0) %*% rep(1,n))
    degree = np.sum((A != 0).sum(axis=1))

    # Truth 레이블 — R: key_list, truth
    truth_path = os.path.join(TRUTH_DIR, article_name)
    truth = []
    if os.path.exists(truth_path):
        with open(truth_path, 'r', encoding='utf-8', errors='ignore') as f:
            key_text = f.read().strip()
        key_text = key_text.lower()
        key_text = re.sub(r'[\t\r;]', '', key_text)
        key_text = re.sub(r'\n', ' ', key_text)
        key_words = key_text.replace(',', ' ').split()
        truth = list(set(word_to_idx[w] for w in key_words if w in word_to_idx))

    return {
        'n': n,
        'A': A,
        'D': D,
        'unique_words': unique_words,
        'word_to_idx': word_to_idx,
        'truth': truth,
        'units': units,
        'degree': degree,
    }


# ========== MCMC (theta update: AWSGLD, sigma2 update: Gibbs) ==========
def gibbs_mh(Burn_in, T, ini, n, graph, Y, B, u_0, alpha_est, grid, verbose=True):
    """
    기존 Gibbs.MH와 동일한 인터페이스를 유지하되,
    theta 업데이트만 AWSGLD로 교체한 샘플러.
    """
    theta_store = np.zeros((T, n))
    sigma2_store = np.zeros(T)
    C_store = np.zeros(T)  # DIAGNOSTIC: track C = ||B(theta-u_0)||^2
    alpha_store = np.zeros(T)
    accept = 0
    theta = ini.copy()

    start_time = time.time()
    print_interval = max(1, T // 10)
    adaptive_weights = np.arange(1, M_REGIONS + 1, dtype=float) / M_REGIONS
    energy_samples = []
    warmup = min(100, max(10, T // 20))
    energy_min = None
    delta_u_actual = None
    J_tilde = M_REGIONS - 1

    # DIAGNOSTIC: preconditioner P ≈ (B^T B)^{-1} with ridge.
    # grad_U already contains BtB @ (theta-u_0)/sigma2 in its prior part,
    # so P @ grad_U turns the prior term into (theta-u_0)/sigma2 (well-scaled)
    # and smooths the likelihood term on the graph natural geometry.
    BtB_fixed = B.T @ B
    ridge = 1e-6 * np.trace(BtB_fixed) / n
    P_precond = solve(BtB_fixed + ridge * np.eye(n), np.eye(n))
    # Cholesky of P for isotropic noise in preconditioned geometry: L L^T = P
    P_sym = 0.5 * (P_precond + P_precond.T)
    L_precond = np.linalg.cholesky(P_sym + 1e-10 * np.eye(n))

    for t in range(T):
        C = (B @ (theta - u_0)).T @ (B @ (theta - u_0))
        sigma2 = invgamma.rvs(n / 2 + 0.001, scale=C / 2 + 0.001)
        sigma2 = max(sigma2, 0.5)  # floor to prevent prior-gradient blowup in Langevin
        sigma2_store[t] = sigma2
        C_store[t] = C

        eps_k = 0.3 / ((t + 1) ** 0.6 + 10)
        decay = min(1.0, DECAY_LR / (((t + 1) ** 0.75) + 1000.0))

        U_tilde = posterior_energy(Y, alpha_est, theta, u_0, B, sigma2)
        grad_U = grad_posterior_energy(Y, alpha_est, theta, u_0, B, sigma2)

        if t < warmup:
            energy_samples.append(U_tilde)
            grad_mult = 1.0
            if t == warmup - 1:
                e_min = np.min(energy_samples)
                e_max = np.max(energy_samples)
                e_range = max(e_max - e_min, 1.0)
                energy_min = e_min - 0.5 * e_range
                energy_max = e_max + 0.5 * e_range
                delta_u_actual = max((energy_max - energy_min) / M_REGIONS, 1e-8)
                energy_samples = None
        else:
            J_tilde = int(np.clip((U_tilde - energy_min) / delta_u_actual + 1, 1, M_REGIONS - 1))
            eps_log = 1e-12
            grad_mult = 1 + (ZETA * TAU / delta_u_actual) * (
                np.log(adaptive_weights[J_tilde] + eps_log)
                - np.log(adaptive_weights[J_tilde - 1] + eps_log)
            )
            grad_mult = np.clip(grad_mult, 0.1, 10.0)

        noise = np.random.randn(n)
        theta = (
            theta
            - eps_k * grad_mult * (P_precond @ grad_U)
            + np.sqrt(2 * TAU * eps_k) * (L_precond @ noise)
        )
        theta = np.clip(theta, -700, 700)

        if t >= warmup:
            current_weight = adaptive_weights[J_tilde]
            adaptive_weights[J_tilde:] = (
                adaptive_weights[J_tilde:]
                + decay * current_weight * (1.0 - adaptive_weights[J_tilde:])
            )
            adaptive_weights[:J_tilde] = (
                adaptive_weights[:J_tilde]
                - decay * current_weight * adaptive_weights[:J_tilde]
            )
            adaptive_weights = np.clip(adaptive_weights, 1e-10, 1.0)

        theta_store[t, :] = theta
        alpha_est = alpha_find(theta, Y, grid)
        alpha_store[t] = alpha_est

        if verbose and (t + 1) % print_interval == 0:
            elapsed = time.time() - start_time
            progress = (t + 1) / T * 100
            eta = timedelta(seconds=int(elapsed / (t + 1) * (T - t - 1)))
            print(f"\r  AWSGLD {progress:.0f}% ({t+1}/{T}) | "
                  f"{timedelta(seconds=int(elapsed))} | ETA {eta} | "
                  f"J={J_tilde}, mult={grad_mult:.2f}", end='', flush=True)

    if verbose:
        print()

    prob = inv_logit(theta_store)
    poster_pi_md = np.median(prob[Burn_in:T, :], axis=0)
    mask_one = poster_pi_md == 1
    poster_pi_md[mask_one] = 1 + np.random.normal(0, 0.01, np.sum(mask_one))
    poster_pi_mn = np.mean(prob[Burn_in:T, :], axis=0)

    alpha_mn = np.mean(alpha_store[Burn_in:])
    alpha_md = np.median(alpha_store[Burn_in:])

    return {
        'poster_pi_md': poster_pi_md,
        'poster_pi_mn': poster_pi_mn,
        'sigma2_store': sigma2_store,
        'C_store': C_store,
        'theta_store': theta_store,
        'alpha_mn': alpha_mn,
        'alpha_md': alpha_md,
        'accept': accept,
        'adaptive_weights': adaptive_weights,
    }


# ========== Componentwise MCMC (R/C++ Componetwise_MCMC) ==========
def componentwise_mcmc(T, ini, n, grid, alpha_est, u_0, B, Y, verbose=True):
    """
    Component_MCMC.cpp → Python 변환
    각 theta_i를 개별적으로 MH update (adaptive variance)
    """
    theta_store = np.zeros((T, n))
    alpha_store = np.zeros(T)
    accept = 0

    theta_store[0, :] = ini
    alpha_store[0] = alpha_est

    start_time = time.time()
    print_interval = max(1, T // 10)

    for t in range(1, T):
        theta_current = theta_store[t - 1, :].copy()

        for i in range(n):
            # Adaptive variance — C++: sqrt(2.4 * (var(history) + 0.01))
            if t < 10:
                var_i = 1.0
            else:
                sp_var = theta_store[:t, i]
                var_i = np.sqrt(2.4 * (np.var(sp_var, ddof=1) + 0.01))

            # Propose theta_i
            theta_star = theta_current.copy()
            theta_star[i] = np.random.normal(theta_current[i], var_i)

            # MH acceptance
            lg_MH = (log_posterior(n, Y, alpha_est, theta_star, u_0, B) -
                      log_posterior(n, Y, alpha_est, theta_current, u_0, B))
            MH_rate = np.exp(np.clip(lg_MH, -700, 700))

            if np.random.uniform() < MH_rate:
                theta_current[i] = theta_star[i]
                accept += 1

        theta_store[t, :] = theta_current

        # Update alpha
        alpha_vals = np.array([alpha_lk(theta_current, Y, g) for g in grid])
        alpha_est = grid[np.argmax(alpha_vals)]
        alpha_store[t] = alpha_est

        if verbose and (t + 1) % print_interval == 0:
            elapsed = time.time() - start_time
            progress = (t + 1) / T * 100
            eta = timedelta(seconds=int(elapsed / (t + 1) * (T - t - 1)))
            print(f"\r  Comp MCMC {progress:.0f}% ({t+1}/{T}) | "
                  f"{timedelta(seconds=int(elapsed))} | ETA {eta} | "
                  f"Accept {accept/(t*n):.3f}", end='', flush=True)

    if verbose:
        print()

    alpha_mn = np.mean(alpha_store)
    alpha_md = np.median(alpha_store)

    return {
        'theta': theta_store,
        'accept': accept,
        'alpha_store': alpha_store,
        'alpha_mn': alpha_mn,
        'alpha_md': alpha_md,
    }


# ========== 단일 문서 처리 (R semi.keyphrase2) ==========
def semi_keyphrase(graph, k, grid, T, Burn_in, verbose=True):
    """
    R semi.keyphrase2() — 단일 문서에 대해 두 체인(Gibbs, Componentwise) 실행 및 평가
    """
    n = graph['n']
    truth = graph['truth']
    d = 0.85

    # R: alpha <- (length(graph$truth) - k) / length(graph$truth)
    alpha = (len(truth) - k) / len(truth)

    G = solve(graph['D'], graph['A'])
    B = np.eye(n) - d * G.T
    w = np.diag(1.0 / np.sqrt(np.diag(graph['D'])))
    B_star = np.eye(n) - d * w @ graph['A'] @ w

    # R: Y[sample(truth, k, replace=FALSE)] = 1
    Y = np.zeros(n)
    obs_label = list(np.random.choice(truth, k, replace=False))
    Y[obs_label] = 1

    u_0 = solve(B, np.ones(n) * (1 - d))
    Base_Line = solve(B_star, Y)
    ini = base_to_start(Base_Line)
    alpha_est = alpha_find(u_0, Y, grid)

    # Gibbs.MH
    chain1 = gibbs_mh(Burn_in, T, ini, n, graph, Y, B, u_0, alpha_est, grid, verbose=verbose)
    poster_pi_md = chain1['poster_pi_md']
    poster_pi_mn = chain1['poster_pi_mn']
    sigma_chain = chain1['sigma2_store']

    # R: geweke.diag — 수렴 진단
    z_score = geweke_diag(sigma_chain[Burn_in:T])
    if abs(z_score) > 8:
        print(f"  chain may not converge well (z={z_score:.2f})")
        return {'z_score': z_score, 'converged': False}

    truth_Y = np.zeros(n)
    truth_Y[truth] = 1

    u_0_adjust = force_obs_to_key(Y, u_0, k)
    Base_Line_adjust = force_obs_to_key(Y, Base_Line, k)

    # FDR 계산
    FDR_pos_mn, FDR_tp_mn, Real_FDR_mn = vec_FDR_cutoff(poster_pi_mn, FDR_LEVELS, Y, truth)
    FDR_pos_md, FDR_tp_md, Real_FDR_md = vec_FDR_cutoff(poster_pi_md, FDR_LEVELS, Y, truth)

    return {
        'converged': True,
        'truth': truth,
        'obs_label': obs_label,
        'u_0_adjust': u_0_adjust,
        'Base_Line_adjust': Base_Line_adjust,
        'alpha': alpha,
        'alpha_mn1': chain1['alpha_mn'],
        'alpha_md1': chain1['alpha_md'],
        'poster_pi_md': poster_pi_md,
        'poster_pi_mn': poster_pi_mn,
        'FDR_pos_md': FDR_pos_md, 'FDR_tp_md': FDR_tp_md, 'Real_FDR_md': Real_FDR_md,
        'FDR_pos_mn': FDR_pos_mn, 'FDR_tp_mn': FDR_tp_mn, 'Real_FDR_mn': Real_FDR_mn,
        'z_score': z_score,
    }


# ========== Baseline 비교 (R Baseline_finding) ==========
def baseline_finding(u_0_adjust, Base_Line_adjust, FDR_cut, truth, total_word,
                     split_points, split_key):
    """
    R Baseline_finding(): 각 문서별로 BSS가 찾은 개수만큼 TextRank/Semi에서 상위 k개를 뽑아
    True Positive를 계산
    """
    n_levels = len(FDR_LEVELS)
    n_articles = len(split_points) - 1
    check_tr = np.zeros((n_articles, n_levels))
    check_bl = np.zeros((n_articles, n_levels))

    for i in range(n_articles):
        start = split_points[i]
        end = split_points[i + 1]
        key_start = split_key[i]
        key_end = split_key[i + 1]

        article_len = end - start
        local_truth = truth[key_start:key_end]

        for j in range(n_levels):
            # R: part <- round(FDR_cut[,1] / total_word * (split_points[i+1] - split_points[i]))
            part = int(round(FDR_cut[j, 0] / total_word * article_len))
            if part <= 0:
                continue

            # TextRank top-part에서 truth와 겹치는 개수
            u0_local = u_0_adjust[start:end]
            tr_top = np.argsort(u0_local)[::-1][:part]
            check_tr[i, j] = np.sum(np.isin(tr_top, local_truth))

            # Semi top-part에서 truth와 겹치는 개수
            bl_local = Base_Line_adjust[start:end]
            bl_top = np.argsort(bl_local)[::-1][:part]
            check_bl[i, j] = np.sum(np.isin(bl_top, local_truth))

        # FDR_cut에 TR, BL TP 누적
        for j in range(n_levels):
            FDR_cut[j, 2] += check_tr[i, j]  # Textrank TP
            FDR_cut[j, 3] += check_bl[i, j]  # Semi TP

    # R: FDR_cut[,6] <- 1 - FDR_cut[,3] / FDR_cut[,1]  (Textrank FDR)
    # R: FDR_cut[,7] <- 1 - FDR_cut[,4] / FDR_cut[,1]  (Semi FDR)
    with np.errstate(divide='ignore', invalid='ignore'):
        FDR_cut[:, 5] = np.where(FDR_cut[:, 0] > 0, 1 - FDR_cut[:, 2] / FDR_cut[:, 0], 0)
        FDR_cut[:, 6] = np.where(FDR_cut[:, 0] > 0, 1 - FDR_cut[:, 3] / FDR_cut[:, 0], 0)

    return FDR_cut, check_tr, check_bl
