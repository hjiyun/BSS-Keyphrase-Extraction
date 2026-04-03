"""
다수 문서 Batch 처리 — 단어(Word) 기반 키프레이즈 추출
R 원본 Keyphrase_functions.R의 main.function2() Python 변환
(Componentwise MCMC: Component_MCMC.cpp → Python 변환 포함)
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

np.random.seed(12345)
k = 4
grid = np.linspace(0.05, 0.90, 33)
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


def create_fcm(text, window=2):
    """Backward-compatible alias used by the converted example scripts."""
    return create_fcm_words(text, window=window)


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


# ========== MCMC (R Gibbs.MH) ==========
def gibbs_mh(Burn_in, T, ini, n, graph, Y, B, u_0, alpha_est, grid, verbose=True):
    """R Gibbs.MH()"""
    theta_store = np.zeros((T, n))
    sigma2_store = np.zeros(T)
    alpha_store = np.zeros(T)
    accept = 0
    theta = ini.copy()

    BtB = B.T @ B
    BtB_inv = solve(BtB, np.eye(n))

    start_time = time.time()
    print_interval = max(1, T // 10)

    for t in range(T):
        C = (B @ (theta - u_0)).T @ (B @ (theta - u_0))
        sigma2 = invgamma.rvs(n / 2 + 0.001, scale=C / 2 + 0.001)
        sigma2_store[t] = sigma2

        cov_matrix = BtB_inv * sigma2 * 4 / n
        try:
            theta_star = np.random.multivariate_normal(theta, cov_matrix)
        except np.linalg.LinAlgError:
            theta_star = theta + np.random.normal(0, np.sqrt(np.diag(cov_matrix)))

        theta_star = np.clip(theta_star, -700, 700)

        log_MH_rate = (posterior_gibbstheta(Y, alpha_est, theta_star, u_0, B, sigma2) -
                       posterior_gibbstheta(Y, alpha_est, theta, u_0, B, sigma2))
        MH_rate = np.exp(np.clip(log_MH_rate, -700, 700))

        if np.random.uniform() < MH_rate:
            theta = theta_star
            accept += 1

        theta_store[t, :] = theta
        alpha_est = alpha_find(theta, Y, grid)
        alpha_store[t] = alpha_est

        if verbose and (t + 1) % print_interval == 0:
            elapsed = time.time() - start_time
            progress = (t + 1) / T * 100
            eta = timedelta(seconds=int(elapsed / (t + 1) * (T - t - 1)))
            print(f"\r  MCMC {progress:.0f}% ({t+1}/{T}) | "
                  f"{timedelta(seconds=int(elapsed))} | ETA {eta} | "
                  f"Accept {accept/(t+1):.3f}", end='', flush=True)

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
        'theta_store': theta_store,
        'alpha_mn': alpha_mn,
        'alpha_md': alpha_md,
        'accept': accept,
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

    # R 원본의 Componetwise_MCMC 체인
    chain3 = componentwise_mcmc(T, ini, n, grid, alpha_est, u_0, B, Y, verbose=verbose)
    prob3 = inv_logit(chain3['theta'])
    poster_pi_md3 = np.median(prob3[Burn_in:T, :], axis=0)
    mask_one3 = poster_pi_md3 == 1
    poster_pi_md3[mask_one3] = 1 + np.random.normal(0, 0.01, np.sum(mask_one3))
    poster_pi_mn3 = np.mean(prob3[Burn_in:T, :], axis=0)

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
    FDR_pos_mn3, FDR_tp_mn3, Real_FDR_mn3 = vec_FDR_cutoff(poster_pi_mn3, FDR_LEVELS, Y, truth)
    FDR_pos_md3, FDR_tp_md3, Real_FDR_md3 = vec_FDR_cutoff(poster_pi_md3, FDR_LEVELS, Y, truth)

    return {
        'converged': True,
        'truth': truth,
        'obs_label': obs_label,
        'u_0_adjust': u_0_adjust,
        'Base_Line_adjust': Base_Line_adjust,
        'alpha': alpha,
        'alpha_mn1': chain1['alpha_mn'],
        'alpha_md1': chain1['alpha_md'],
        'alpha_mn3': chain3['alpha_mn'],
        'alpha_md3': chain3['alpha_md'],
        'poster_pi_md': poster_pi_md,
        'poster_pi_mn': poster_pi_mn,
        'poster_pi_md3': poster_pi_md3,
        'poster_pi_mn3': poster_pi_mn3,
        'FDR_pos_md': FDR_pos_md, 'FDR_tp_md': FDR_tp_md, 'Real_FDR_md': Real_FDR_md,
        'FDR_pos_mn': FDR_pos_mn, 'FDR_tp_mn': FDR_tp_mn, 'Real_FDR_mn': Real_FDR_mn,
        'FDR_pos_md3': FDR_pos_md3, 'FDR_tp_md3': FDR_tp_md3, 'Real_FDR_md3': Real_FDR_md3,
        'FDR_pos_mn3': FDR_pos_mn3, 'FDR_tp_mn3': FDR_tp_mn3, 'Real_FDR_mn3': Real_FDR_mn3,
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
    check_tr_tp = np.zeros((n_articles, n_levels))
    check_bl_tp = np.zeros((n_articles, n_levels))
    check_parts = np.zeros((n_articles, n_levels))

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
            check_parts[i, j] = part
            if part <= 0:
                continue

            # TextRank top-part에서 truth와 겹치는 개수
            u0_local = u_0_adjust[start:end]
            tr_top = np.argsort(u0_local)[::-1][:part]
            check_tr_tp[i, j] = np.sum(np.isin(tr_top, local_truth))

            # Semi top-part에서 truth와 겹치는 개수
            bl_local = Base_Line_adjust[start:end]
            bl_top = np.argsort(bl_local)[::-1][:part]
            check_bl_tp[i, j] = np.sum(np.isin(bl_top, local_truth))

        # FDR_cut에 TR, BL TP 누적
        for j in range(n_levels):
            FDR_cut[j, 2] += check_tr_tp[i, j]  # Textrank TP
            FDR_cut[j, 3] += check_bl_tp[i, j]  # Semi TP

    # R: FDR_cut[,6] <- 1 - FDR_cut[,3] / FDR_cut[,1]  (Textrank FDR)
    # R: FDR_cut[,7] <- 1 - FDR_cut[,4] / FDR_cut[,1]  (Semi FDR)
    with np.errstate(divide='ignore', invalid='ignore'):
        FDR_cut[:, 5] = np.where(FDR_cut[:, 0] > 0, 1 - FDR_cut[:, 2] / FDR_cut[:, 0], 0)
        FDR_cut[:, 6] = np.where(FDR_cut[:, 0] > 0, 1 - FDR_cut[:, 3] / FDR_cut[:, 0], 0)

    check_tr = np.hstack([check_tr_tp, check_parts])
    check_bl = np.hstack([check_bl_tp, check_parts])
    return FDR_cut, check_tr, check_bl


def _rankdata_average(x):
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(len(x), dtype=float)
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return ranks


def roc_auc_score_simple(truth_Y, scores):
    """AUC equivalent for binary labels, using average ranks for ties."""
    truth_Y = np.asarray(truth_Y).astype(int)
    scores = np.asarray(scores)
    pos = truth_Y == 1
    n_pos = np.sum(pos)
    n_neg = len(truth_Y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = _rankdata_average(scores)
    return (np.sum(ranks[pos]) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def precision_recall_curve_points(truth_Y, scores):
    truth_Y = np.asarray(truth_Y).astype(int)
    scores = np.asarray(scores)
    order = np.argsort(scores)[::-1]
    y_sorted = truth_Y[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    positives = np.sum(truth_Y == 1)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / positives if positives > 0 else np.zeros_like(tp, dtype=float)
    return precision, recall


def precision_recall_auc(*args, c=None):
    """
    Python equivalent of the two R precision.recall.auc signatures.

    Supported calls:
      precision_recall_auc(Y, truth, truth_Y, poster_pi_md, k, c=...)
      precision_recall_auc(Y, truth_Y, poster_pi_md, per, k)
    """
    if len(args) == 5:
        Y, truth_Y, poster_pi_md, _per, k = args
        truth = np.where(np.asarray(truth_Y) == 1)[0]
        c_values = FDR_LEVELS if c is None else c
    elif len(args) == 6:
        Y, truth, truth_Y, poster_pi_md, k, c_values = args
    else:
        raise TypeError("precision_recall_auc expects 5 or 6 positional arguments")

    Y = np.asarray(Y)
    truth = np.asarray(truth, dtype=int)
    truth_Y = np.asarray(truth_Y).astype(int)
    poster_pi_md = np.asarray(poster_pi_md, dtype=float)
    c_values = np.atleast_1d(FDR_LEVELS if c_values is None else c_values).astype(float)

    auc = roc_auc_score_simple(truth_Y, poster_pi_md)
    precision, recall = precision_recall_curve_points(truth_Y, poster_pi_md)
    sample_idx = np.maximum(np.round(np.linspace(0.1, 1.0, 10) * len(recall)).astype(int) - 1, 0)
    sample_idx = np.minimum(sample_idx, len(recall) - 1)

    poster_md_adjust = force_obs_to_key(Y, poster_pi_md, k)
    auc_l = roc_auc_score_simple(truth_Y, poster_md_adjust)
    precision_l, recall_l = precision_recall_curve_points(truth_Y, poster_md_adjust)
    sample_idx_l = np.maximum(np.round(np.linspace(0.1, 1.0, 10) * len(recall_l)).astype(int) - 1, 0)
    sample_idx_l = np.minimum(sample_idx_l, len(recall_l) - 1)

    FDR_pos, FDR_tp, Real_FDR = vec_FDR_cutoff(poster_pi_md, c_values, Y, truth)
    return {
        'auc': auc,
        'auc_l': auc_l,
        'recall': recall[sample_idx],
        'recall_l': recall_l[sample_idx_l],
        'precision': precision[sample_idx],
        'precision_l': precision_l[sample_idx_l],
        'FDR_pos': FDR_pos,
        'FDR_tp': FDR_tp,
        'Real_FDR': Real_FDR,
    }


def _empty_fdr_table():
    return np.zeros((len(FDR_LEVELS), 7), dtype=float)


def main_function2(k, file_list=None, start_idx=201, end_idx=500, base_dir=None,
                   T_override=None, burn_in_override=None, verbose=False):
    """
    Batch driver corresponding to R main.function2().

    R's loop is 1-based and inclusive. This keeps the same external convention:
    start_idx=201, end_idx=500 processes the 201st through 500th file names.
    """
    global PREPROCESS_DIR
    old_preprocess_dir = PREPROCESS_DIR
    if base_dir is not None:
        PREPROCESS_DIR = base_dir

    if file_list is None:
        if not os.path.isdir(PREPROCESS_DIR):
            PREPROCESS_DIR = old_preprocess_dir
            raise FileNotFoundError(f"preprocess directory not found: {PREPROCESS_DIR}")
        file_list = sorted(os.listdir(PREPROCESS_DIR))

    T_run = T if T_override is None else T_override
    burn_in_run = BURN_IN if burn_in_override is None else burn_in_override

    accum = {
        'u_0_adjust': [], 'Base_Line_adjust': [], 'poster_pi_md': [], 'poster_pi_mn': [],
        'poster_pi_md3': [], 'poster_pi_mn3': [], 'truth': [], 'units': [], 'degree': [],
        'alpha': [], 'alpha_mn1': [], 'alpha_md1': [], 'alpha_mn3': [], 'alpha_md3': [],
        'num_of_key': [], 'words': [], 'files': [], 'dict': [], 'truth_list': [],
        'obs_label': [], 'TP_0_15mn_article': [],
    }
    split_points = [0]
    split_key = [0]
    total_word = 0
    total_keys = 0
    fdr_tables = {
        'md': _empty_fdr_table(),
        'mn': _empty_fdr_table(),
        'md3': _empty_fdr_table(),
        'mn3': _empty_fdr_table(),
    }
    md_tp_pos = []
    mn_tp_pos = []

    for r_index in range(start_idx, end_idx + 1):
        py_index = r_index - 1
        if py_index < 0 or py_index >= len(file_list):
            continue
        filename = file_list[py_index]
        article_name = os.path.splitext(os.path.basename(filename))[0]
        article_name = article_name.replace(".txt", "").replace(".final", "")
        graph = create_graph(article_name)
        if graph is None or len(graph['truth']) <= 10:
            if verbose:
                print(f"{filename} does not have enough keyphrase")
            continue

        ans = semi_keyphrase(graph, k, grid, T_run, burn_in_run, verbose=verbose)
        if not ans.get('converged', False):
            continue

        accum['files'].append(filename)
        accum['dict'].append(graph.get('unique_words', []))
        accum['truth_list'].append(graph['truth'])
        accum['truth'].extend(ans['truth'])
        accum['units'].append(graph['units'])
        accum['degree'].append(graph['degree'])
        accum['alpha'].append(ans['alpha'])
        accum['alpha_mn1'].append(ans['alpha_mn1'])
        accum['alpha_md1'].append(ans['alpha_md1'])
        accum['alpha_mn3'].append(ans['alpha_mn3'])
        accum['alpha_md3'].append(ans['alpha_md3'])
        accum['obs_label'].append(np.asarray(ans['obs_label'], dtype=int))
        accum['u_0_adjust'].extend(ans['u_0_adjust'])
        accum['Base_Line_adjust'].extend(ans['Base_Line_adjust'])
        accum['poster_pi_md'].extend(ans['poster_pi_md'])
        accum['poster_pi_mn'].extend(ans['poster_pi_mn'])
        accum['poster_pi_md3'].extend(ans['poster_pi_md3'])
        accum['poster_pi_mn3'].extend(ans['poster_pi_mn3'])

        for key, suffix in [('md', 'md'), ('mn', 'mn'), ('md3', 'md3'), ('mn3', 'mn3')]:
            fdr_tables[key][:, 0] += ans[f'FDR_pos_{suffix}']
            fdr_tables[key][:, 1] += ans[f'FDR_tp_{suffix}']
            fdr_tables[key][:, 4] += ans[f'Real_FDR_{suffix}']

        md_tp_pos.append(np.r_[ans['FDR_tp_md'], ans['FDR_pos_md']])
        mn_tp_pos.append(np.r_[ans['FDR_tp_mn'], ans['FDR_pos_mn']])
        accum['TP_0_15mn_article'].append([ans['FDR_pos_mn'][2], ans['FDR_tp_mn'][2]])

        total_word += graph['n']
        total_keys += len(graph['truth'])
        accum['words'].append(graph['n'])
        accum['num_of_key'].append(len(graph['truth']))
        split_points.append(total_word)
        split_key.append(total_keys)

    for table in fdr_tables.values():
        with np.errstate(divide='ignore', invalid='ignore'):
            table[:, 4] = np.where(table[:, 0] > 0, 1 - table[:, 1] / table[:, 0], 0)

    truth_arr = np.asarray(accum['truth'], dtype=int)
    u0_arr = np.asarray(accum['u_0_adjust'], dtype=float)
    bl_arr = np.asarray(accum['Base_Line_adjust'], dtype=float)
    mn_comparison = baseline_finding(u0_arr, bl_arr, fdr_tables['mn'].copy(), truth_arr,
                                     total_word, split_points, split_key)
    mn3_comparison = baseline_finding(u0_arr, bl_arr, fdr_tables['mn3'].copy(), truth_arr,
                                      total_word, split_points, split_key)

    output = {
        'u_0_adjust': u0_arr,
        'Base_Line_adjust': bl_arr,
        'poster_pi_md': np.asarray(accum['poster_pi_md'], dtype=float),
        'poster_pi_mn': np.asarray(accum['poster_pi_mn'], dtype=float),
        'poster_pi_md3': np.asarray(accum['poster_pi_md3'], dtype=float),
        'poster_pi_mn3': np.asarray(accum['poster_pi_mn3'], dtype=float),
        'split_points': np.asarray(split_points, dtype=int),
        'truth': truth_arr,
        'split_key': np.asarray(split_key, dtype=int),
        'units': np.asarray(accum['units']),
        'degree': np.asarray(accum['degree']),
        'obs_label': accum['obs_label'],
        'article': len(accum['files']),
        'total_keys': total_keys,
        'total_word': total_word,
        'files': accum['files'],
        'dict': accum['dict'],
        'truth_list': accum['truth_list'],
        'alpha': np.asarray(accum['alpha']),
        'alpha_mn1': np.asarray(accum['alpha_mn1']),
        'alpha_md1': np.asarray(accum['alpha_md1']),
        'alpha_mn3': np.asarray(accum['alpha_mn3']),
        'alpha_md3': np.asarray(accum['alpha_md3']),
        'num_of_key': np.asarray(accum['num_of_key'], dtype=int),
        'words': np.asarray(accum['words'], dtype=int),
        'FDR_cut_md': fdr_tables['md'],
        'FDR_cut_mn': mn_comparison[0],
        'FDR_cut_md3': fdr_tables['md3'],
        'FDR_cut_mn3': mn3_comparison[0],
        'TP_0_15mn_article': np.asarray(accum['TP_0_15mn_article'], dtype=float),
        'mn_comparison': {'FDR_cut_md': mn_comparison[0], 'check_mn_tr': mn_comparison[1], 'check_mn_bl': mn_comparison[2]},
        'mn3_comparison': {'FDR_cut_md': mn3_comparison[0], 'check_mn_tr': mn3_comparison[1], 'check_mn_bl': mn3_comparison[2]},
        'md_TP_Pos': np.asarray(md_tp_pos, dtype=float),
        'mn_TP_Pos': np.asarray(mn_tp_pos, dtype=float),
    }
    PREPROCESS_DIR = old_preprocess_dir
    return output
