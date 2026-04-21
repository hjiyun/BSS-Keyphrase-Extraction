"""
구(Phrase) 기반 키프레이즈 추출 v2
- keyphrase_functions_phrase.py 기반
- AWSGLD v5와 동일한 Non-seed rescue prior 추가:
  lambda_prior/2 * sum w_i(theta_i - b_i)^2  (seed: w=0)
- posterior_gibbstheta, log_posterior, gibbs_mh, componentwise_mcmc 모두 반영
- lambda_prior=0이면 v1과 동일
"""

import numpy as np
import os
import re
import time
import json
import csv
import random
from datetime import datetime, timedelta
from scipy.linalg import solve
from scipy.stats import invgamma
from collections import Counter


# ========== 설정 ==========
DATA_DIR = "/home/jiyoon/3차/BSS-Keyphrase-Extraction-master/data_JOC"
PREPROCESS_DIR = os.path.join(DATA_DIR, "pre_process")
TRUTH_DIR = os.path.join(DATA_DIR, "pre_process_reader_truth")

# MCMC 파라미터 (R 원본과 동일)
T = 50000
BURN_IN = 2000

np.random.seed(12345)
k = 4
grid = (np.arange(10, 43) - 5) / np.arange(10, 43)
FDR_LEVELS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

# Non-seed rescue prior
LAMBDA_PRIOR = 0.5   # AWSGLD v5와 동일 기본값
SIGMA2_FIXED = 3.0   # gibbs_mh에서 sigma2 sampling 대신 고정할 때 사용 (기본은 sampling)


# ========== Phrase 정규화 ==========
def normalize_phrase(phrase):
    phrase = phrase.lower().strip()
    phrase = re.sub(r'-', ' ', phrase)
    phrase = re.sub(r'[^\w\s]', '', phrase)
    phrase = re.sub(r'\s+', ' ', phrase).strip()
    return phrase


# ========== 기본 함수들 ==========
def inv_logit(x):
    x = np.clip(x, -700, 700)
    return np.exp(x) / (1 + np.exp(x))


def base_to_start(Base_Line):
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


def force_obs_to_key(Y, poster_pi_mean, k):
    poster_pi_mean = poster_pi_mean.copy()
    obs_indices = np.where(Y == 1)[0]
    if len(obs_indices) > 0:
        poster_pi_mean[obs_indices] = 10 + np.random.normal(0.01, 0.01, len(obs_indices))
    return poster_pi_mean


def force_obs_to_key2(Y, poster_pi_mean):
    poster_pi_mean = poster_pi_mean.copy()
    poster_pi_mean[Y == 1] = 1
    return poster_pi_mean


def FDR_calculate(cutoff, poster_md_adjust):
    set_vals = poster_md_adjust[poster_md_adjust >= cutoff]
    return np.sum(1 - set_vals) / len(set_vals) if len(set_vals) > 0 else 0


def vec_FDR_cal(cutoffs, poster_md_adjust):
    return np.array([FDR_calculate(c, poster_md_adjust) for c in cutoffs])


# ========== FDR Cutoff ==========
def FDR_cutoff_full(poster_pi_md, c, Y, truth):
    poster_md_adjust = force_obs_to_key2(Y, poster_pi_md)
    cutoffs = np.unique(poster_md_adjust)[::-1]
    FDRs = vec_FDR_cal(cutoffs, poster_md_adjust)
    valid = np.where(FDRs < c)[0]
    if len(valid) > 0:
        cutoff = cutoffs[np.max(valid)]
    else:
        cutoff = cutoffs[np.argmin(FDRs)]
    selected = np.where(poster_md_adjust >= cutoff)[0]
    FDR_pos = len(selected)
    FDR_tp = np.sum(np.isin(selected, truth))
    Real_FDR = (FDR_pos - FDR_tp) / FDR_pos if FDR_pos > 0 else 0
    return FDR_pos, FDR_tp, Real_FDR


def vec_FDR_cutoff(poster_pi_md, c_levels, Y, truth):
    FDR_pos = np.zeros(len(c_levels))
    FDR_tp = np.zeros(len(c_levels))
    Real_FDR = np.zeros(len(c_levels))
    for i, c in enumerate(c_levels):
        FDR_pos[i], FDR_tp[i], Real_FDR[i] = FDR_cutoff_full(poster_pi_md, c, Y, truth)
    return FDR_pos, FDR_tp, Real_FDR


# ========== Geweke 수렴 진단 ==========
def geweke_diag(chain, frac1=0.1, frac2=0.5):
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


# ========== Phrase FCM 설정 ==========
MAX_NGRAM = 4
MIN_FREQ = 2
WINDOW_SIZE = 2

STOPWORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
    'would', 'could', 'should', 'may', 'might', 'must', 'shall',
    'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
    'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'between',
    'under', 'again', 'further', 'then', 'once', 'here', 'there',
    'when', 'where', 'why', 'how', 'all', 'each', 'few', 'more',
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
    'own', 'same', 'so', 'than', 'too', 'very', 'just', 'and',
    'but', 'if', 'or', 'because', 'until', 'while', 'this', 'that',
    'these', 'those', 'it', 'its', 'also', 'which', 'their', 'them',
    'they', 'we', 'our', 'you', 'your', 'he', 'she', 'him', 'her',
    'i', 'me', 'my', 'who', 'whom', 'what', 'any', 'both', 'about',
    'over', 'out', 'up', 'down', 'off', 'every', 's', 't', 'don',
    'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'et', 'al',
    'eg', 'ie', 'etc', 'vs', 'via', 'use',
}


# ========== 구(Phrase) 기반 FCM 생성 ==========
def create_fcm_phrases(text, max_ngram=MAX_NGRAM, window=WINDOW_SIZE, min_freq=MIN_FREQ):
    text = text.lower()
    sentences = re.split(r'[.!?]+', text)

    all_phrases_per_sent = []
    phrase_counts = Counter()

    for sent in sentences:
        sent = re.sub(r'-', ' ', sent)
        sent = re.sub(r'[^\w\s]', ' ', sent)
        sent = re.sub(r'\s+', ' ', sent).strip()

        words = sent.split()
        words_clean = [w for w in words if len(w) > 1]
        if len(words_clean) == 0:
            continue

        sent_phrases = []
        for n in range(1, max_ngram + 1):
            for i in range(len(words_clean) - n + 1):
                phrase_words = words_clean[i:i+n]
                if n == 1 and phrase_words[0] in STOPWORDS:
                    continue
                if n > 1 and (phrase_words[0] in STOPWORDS or phrase_words[-1] in STOPWORDS):
                    continue
                if all(w in STOPWORDS for w in phrase_words):
                    continue
                phrase = ' '.join(phrase_words)
                sent_phrases.append(phrase)
                phrase_counts[phrase] += 1

        all_phrases_per_sent.append(sent_phrases)

    valid_phrases = [p for p, c in phrase_counts.items() if c >= min_freq]
    n = len(valid_phrases)
    if n == 0:
        return np.array([]), [], {}, phrase_counts

    phrase_to_idx = {p: i for i, p in enumerate(valid_phrases)}
    fcm = np.zeros((n, n))

    for sent_phrases in all_phrases_per_sent:
        valid_in_sent = [p for p in sent_phrases if p in phrase_to_idx]
        for i in range(len(valid_in_sent)):
            for j in range(i + 1, min(i + window + 1, len(valid_in_sent))):
                idx1 = phrase_to_idx[valid_in_sent[i]]
                idx2 = phrase_to_idx[valid_in_sent[j]]
                if idx1 != idx2:
                    fcm[idx1, idx2] += 1
                    fcm[idx2, idx1] += 1

    return fcm, valid_phrases, phrase_to_idx, phrase_counts


# ========== 그래프 생성 (Phrase 기반) — AWSGLD v5와 동일 구조 ==========
def create_graph(article_name):
    text_path = os.path.join(PREPROCESS_DIR, article_name + ".txt.final")
    if not os.path.exists(text_path):
        text_path = os.path.join(DATA_DIR, article_name + ".txt")
    if not os.path.exists(text_path):
        return None

    with open(text_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()

    fcm, unique_phrases, phrase_to_idx, phrase_counts = create_fcm_phrases(text)
    n_total = len(unique_phrases)
    if n_total == 0:
        return None

    n = n_total
    A = fcm.copy()
    np.fill_diagonal(A, 0)
    row_sums = A.sum(axis=1)
    row_sums[row_sums == 0] = 1
    D = np.diag(row_sums)

    raw_text = text.lower()
    raw_text = re.sub(r'[\t\r;]', '', raw_text)
    raw_text = re.sub(r'\n', ' ', raw_text)
    units = len(raw_text.split())
    degree = np.sum((A != 0).sum(axis=1))

    # Truth — 정규화된 구 단위 매칭
    truth_path = os.path.join(TRUTH_DIR, article_name)
    truth = []
    truth_phrases = []
    if os.path.exists(truth_path):
        with open(truth_path, 'r', encoding='utf-8', errors='ignore') as f:
            key_text = f.read().strip().lower()
        key_text = re.sub(r'[\t\r;]', '', key_text)
        key_text = re.sub(r'\n', ' ', key_text)
        truth_phrases = [p.strip() for p in key_text.split(',') if p.strip()]

        norm_to_idx = {}
        for i in range(n):
            np_ = normalize_phrase(unique_phrases[i])
            if np_ not in norm_to_idx:
                norm_to_idx[np_] = i
        for tp in truth_phrases:
            ntp = normalize_phrase(tp)
            if ntp in norm_to_idx:
                truth.append(norm_to_idx[ntp])

    ngram_dist = {1: 0, 2: 0, 3: 0, 4: 0}
    for p in unique_phrases:
        wc = len(p.split())
        if wc in ngram_dist:
            ngram_dist[wc] += 1

    return {
        'n': n,
        'A': A,
        'D': D,
        'unique_phrases': unique_phrases,
        'phrase_to_idx': phrase_to_idx,
        'phrase_counts': phrase_counts,
        'truth': truth,
        'truth_phrases': truth_phrases,
        'units': units,
        'degree': degree,
        'ngram_dist': ngram_dist,
    }


# ==========================================================
#  Posterior 함수들 — Non-seed rescue prior 반영
# ==========================================================

def posterior_gibbstheta(Y, alpha, theta, u_0, B, sigma2,
                         baseline_center=None, baseline_weight=None, lambda_prior=0.0):
    """
    Gibbs.MH용 posterior (sigma2 given)
    = R 원본 likelihood + graph prior + non-seed rescue prior
    """
    temp = (1 - alpha) * inv_logit(theta)
    temp = np.clip(temp, 1e-10, 1 - 1e-10)
    C = (B @ (theta - u_0)).T @ (B @ (theta - u_0))
    lglk = np.sum(Y * np.log(temp) + (1 - Y) * np.log(1 - temp)) - C / (2 * sigma2)

    # Non-seed rescue prior: -lambda/2 * sum w_i(theta_i - b_i)^2
    if lambda_prior > 0 and baseline_center is not None and baseline_weight is not None:
        lglk -= 0.5 * lambda_prior * np.sum(baseline_weight * (theta - baseline_center)**2)

    return lglk


def log_posterior(n, Y, alpha, theta, u_0, B,
                  baseline_center=None, baseline_weight=None, lambda_prior=0.0):
    """
    Componentwise MCMC용 posterior (sigma2 적분형)
    = R 원본 likelihood + integrated graph prior + non-seed rescue prior
    """
    temp = (1 - alpha) * inv_logit(theta)
    temp = np.clip(temp, 1e-10, 1 - 1e-10)
    C = (B @ (theta - u_0)).T @ (B @ (theta - u_0))
    lglk = np.sum(Y * np.log(temp) + (1 - Y) * np.log(1 - temp)) - (n / 2 + 0.001) * np.log(C / 2 + 0.001)

    # Non-seed rescue prior
    if lambda_prior > 0 and baseline_center is not None and baseline_weight is not None:
        lglk -= 0.5 * lambda_prior * np.sum(baseline_weight * (theta - baseline_center)**2)

    return lglk


# ==========================================================
#  MCMC 샘플러들 — baseline_center/weight/lambda_prior 전달
# ==========================================================

def gibbs_mh(Burn_in, T, ini, n, graph, Y, B, u_0, alpha_est, grid,
             baseline_center=None, baseline_weight=None, lambda_prior=0.0,
             sigma2_mode='sample', sigma2_fixed=SIGMA2_FIXED,
             verbose=True):
    """
    R Gibbs.MH() + non-seed rescue prior
    sigma2_mode='sample': R 원본처럼 inverse-gamma sampling
    sigma2_mode='fixed':  AWSGLD v5처럼 sigma2 고정
    """
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
        # sigma2: sampling 또는 고정
        C = (B @ (theta - u_0)).T @ (B @ (theta - u_0))
        if sigma2_mode == 'sample':
            sigma2 = invgamma.rvs(n / 2 + 0.001, scale=C / 2 + 0.001)
        else:
            sigma2 = sigma2_fixed
        sigma2_store[t] = sigma2

        # Proposal
        cov_matrix = BtB_inv * sigma2 * 4 / n
        try:
            theta_star = np.random.multivariate_normal(theta, cov_matrix)
        except np.linalg.LinAlgError:
            theta_star = theta + np.random.normal(0, np.sqrt(np.diag(cov_matrix)))
        theta_star = np.clip(theta_star, -700, 700)

        # MH acceptance — rescue prior 포함
        log_MH_rate = (
            posterior_gibbstheta(Y, alpha_est, theta_star, u_0, B, sigma2,
                                baseline_center, baseline_weight, lambda_prior) -
            posterior_gibbstheta(Y, alpha_est, theta, u_0, B, sigma2,
                                baseline_center, baseline_weight, lambda_prior)
        )
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
            mode_str = f"σ²={'fixed='+str(sigma2_fixed) if sigma2_mode=='fixed' else 'sample'}"
            print(f"\r  Gibbs.MH {progress:.0f}% ({t+1}/{T}) | "
                  f"{timedelta(seconds=int(elapsed))} | ETA {eta} | "
                  f"Accept {accept/(t+1):.3f} | {mode_str}", end='', flush=True)

    if verbose:
        print()

    prob = inv_logit(theta_store)
    poster_pi_md = np.median(prob[Burn_in:T, :], axis=0)
    mask_one = poster_pi_md == 1
    poster_pi_md[mask_one] = 1 + np.random.normal(0, 0.01, np.sum(mask_one))
    poster_pi_mn = np.mean(prob[Burn_in:T, :], axis=0)

    return {
        'poster_pi_md': poster_pi_md,
        'poster_pi_mn': poster_pi_mn,
        'sigma2_store': sigma2_store,
        'alpha_mn': np.mean(alpha_store[Burn_in:]),
        'alpha_md': np.median(alpha_store[Burn_in:]),
        'accept': accept,
        'sigma2_mode': sigma2_mode,
    }


def componentwise_mcmc(T, ini, n, grid, alpha_est, u_0, B, Y,
                        baseline_center=None, baseline_weight=None, lambda_prior=0.0,
                        verbose=True):
    """Component_MCMC.cpp → Python + non-seed rescue prior"""
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
            if t < 10:
                var_i = 1.0
            else:
                sp_var = theta_store[:t, i]
                var_i = np.sqrt(2.4 * (np.var(sp_var, ddof=1) + 0.01))

            theta_star = theta_current.copy()
            theta_star[i] = np.random.normal(theta_current[i], var_i)

            # MH acceptance — rescue prior 포함
            lg_MH = (
                log_posterior(n, Y, alpha_est, theta_star, u_0, B,
                              baseline_center, baseline_weight, lambda_prior) -
                log_posterior(n, Y, alpha_est, theta_current, u_0, B,
                              baseline_center, baseline_weight, lambda_prior)
            )
            MH_rate = np.exp(np.clip(lg_MH, -700, 700))

            if np.random.uniform() < MH_rate:
                theta_current[i] = theta_star[i]
                accept += 1

        theta_store[t, :] = theta_current

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

    return {
        'theta': theta_store,
        'accept': accept,
        'alpha_store': alpha_store,
        'alpha_mn': np.mean(alpha_store),
        'alpha_md': np.median(alpha_store),
    }


# ==========================================================
#  run_gibbs_extraction — AWSGLD v5의 run_awsgld_extraction과 동일 구조
# ==========================================================

def run_gibbs_extraction(graph, obs_label, T, burn_in, alpha_fixed=None,
                         lambda_prior=LAMBDA_PRIOR, method='gibbs',
                         sigma2_mode='sample', sigma2_fixed=SIGMA2_FIXED):
    """
    AWSGLD v5와 동일한 그래프/prior/baseline 계산 후 Gibbs.MH 또는 Componentwise MCMC 실행

    method: 'gibbs' (Gibbs.MH) 또는 'comp' (Componentwise MCMC)
    sigma2_mode: 'sample' (R 원본) 또는 'fixed' (AWSGLD v5와 동일)
    lambda_prior=0: v1과 동일 (graph prior만)
    lambda_prior>0: non-seed rescue prior 추가
    """
    n = graph['n']
    d = 0.85

    # AWSGLD v5와 동일한 행렬 계산
    G = solve(graph['D'], graph['A'])
    B = np.eye(n) - d * G.T
    w = np.diag(1.0 / np.sqrt(np.diag(graph['D'])))
    B_star = np.eye(n) - d * w @ graph['A'] @ w

    Y = np.zeros(n)
    for idx in obs_label:
        if idx < n:
            Y[idx] = 1

    u_0 = solve(B, np.ones(n) * (1 - d))
    Base_Line = solve(B_star, Y)

    # Non-seed rescue prior 구성 — AWSGLD v5와 완전히 동일
    baseline_center = base_to_start(Base_Line.copy())
    bl_max = np.max(Base_Line)
    bl_max = bl_max if bl_max > 0 else 1.0
    baseline_weight = Base_Line / bl_max
    for idx in obs_label:
        if idx < n:
            baseline_weight[idx] = 0.0

    ini = base_to_start(Base_Line)

    if alpha_fixed is None:
        alpha_fixed = alpha_find(u_0, Y, grid)

    if method == 'gibbs':
        s2_str = f"σ²={'fixed='+str(sigma2_fixed) if sigma2_mode=='fixed' else 'sample'}"
        print(f"  Gibbs.MH (T={T}, lambda={lambda_prior}, alpha={alpha_fixed:.4f}, {s2_str})")
        chain = gibbs_mh(burn_in, T, ini, n, graph, Y, B, u_0, alpha_fixed, grid,
                         baseline_center=baseline_center,
                         baseline_weight=baseline_weight,
                         lambda_prior=lambda_prior,
                         sigma2_mode=sigma2_mode,
                         sigma2_fixed=sigma2_fixed,
                         verbose=True)
        return {
            'poster_pi_mn': chain['poster_pi_mn'],
            'poster_pi_md': chain['poster_pi_md'],
            'sigma2_store': chain['sigma2_store'],
            'sigma2_mode': sigma2_mode,
            'accept': chain['accept'],
            'alpha_mn': chain['alpha_mn'],
            'alpha_md': chain['alpha_md'],
            'u_0': u_0,
            'Base_Line': Base_Line,
            'baseline_center': baseline_center,
            'baseline_weight': baseline_weight,
            'Y': Y,
            'lambda_prior': lambda_prior,
        }
    elif method == 'comp':
        print(f"  Comp MCMC (T={T}, lambda_prior={lambda_prior}, alpha={alpha_fixed:.4f})")
        chain = componentwise_mcmc(T, ini, n, grid, alpha_fixed, u_0, B, Y,
                                    baseline_center=baseline_center,
                                    baseline_weight=baseline_weight,
                                    lambda_prior=lambda_prior,
                                    verbose=True)
        prob = inv_logit(chain['theta'])
        poster_pi_md = np.median(prob[burn_in:T, :], axis=0)
        mask_one = poster_pi_md == 1
        poster_pi_md[mask_one] = 1 + np.random.normal(0, 0.01, np.sum(mask_one))
        poster_pi_mn = np.mean(prob[burn_in:T, :], axis=0)

        return {
            'poster_pi_mn': poster_pi_mn,
            'poster_pi_md': poster_pi_md,
            'accept': chain['accept'],
            'alpha_mn': chain['alpha_mn'],
            'alpha_md': chain['alpha_md'],
            'u_0': u_0,
            'Base_Line': Base_Line,
            'baseline_center': baseline_center,
            'baseline_weight': baseline_weight,
            'Y': Y,
            'lambda_prior': lambda_prior,
        }


# ========== Baseline 비교 ==========
def baseline_finding(u_0_adjust, Base_Line_adjust, FDR_cut, truth, total_word,
                     split_points, split_key):
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
            part = int(round(FDR_cut[j, 0] / total_word * article_len))
            if part <= 0:
                continue

            u0_local = u_0_adjust[start:end]
            tr_top = np.argsort(u0_local)[::-1][:part]
            check_tr[i, j] = np.sum(np.isin(tr_top, local_truth))

            bl_local = Base_Line_adjust[start:end]
            bl_top = np.argsort(bl_local)[::-1][:part]
            check_bl[i, j] = np.sum(np.isin(bl_top, local_truth))

        for j in range(n_levels):
            FDR_cut[j, 2] += check_tr[i, j]
            FDR_cut[j, 3] += check_bl[i, j]

    with np.errstate(divide='ignore', invalid='ignore'):
        FDR_cut[:, 5] = np.where(FDR_cut[:, 0] > 0, 1 - FDR_cut[:, 2] / FDR_cut[:, 0], 0)
        FDR_cut[:, 6] = np.where(FDR_cut[:, 0] > 0, 1 - FDR_cut[:, 3] / FDR_cut[:, 0], 0)

    return FDR_cut, check_tr, check_bl


# ==========================================================
#  메인: C-84 Gibbs.MH — sigma2 fixed vs sample 비교
#  AWSGLD v5와 동일: T=5000, Burn=500, lambda=0.5, FDR=0.10
# ==========================================================
if __name__ == "__main__":
    DOC = "C-84"
    LAMBDA_RUN = 0.5
    FDR_RUN = 0.10
    T_RUN = 5000
    BURN_RUN = 500

    print("=" * 70)
    print(f"  Gibbs.MH v2 (Phrase) — {DOC}")
    print(f"  sigma2 fixed(3.0) vs sample 비교")
    print(f"  lambda_prior={LAMBDA_RUN}, FDR={FDR_RUN}, T={T_RUN}, Burn={BURN_RUN}")
    print("=" * 70)

    graph = create_graph(DOC)
    if graph is None:
        raise RuntimeError("그래프 생성 실패")

    n = graph['n']
    reader_idx = graph['truth']
    print(f"\n  총 {n}개 구 (1g={graph['ngram_dist'][1]}, 2g={graph['ngram_dist'][2]}, "
          f"3g={graph['ngram_dist'][3]}, 4g={graph['ngram_dist'][4]})")
    print(f"  Truth: {len(graph['truth_phrases'])}개 → 매칭 {len(reader_idx)}개")

    np.random.seed(200)
    random.seed(200)
    obs_label = list(np.random.choice(reader_idx, k, replace=False))
    seed_set = set(obs_label)
    ns_truth = [i for i in reader_idx if i not in seed_set]
    alpha_fixed = (len(reader_idx) - k) / len(reader_idx)

    print(f"  n={n}, truth matched={len(reader_idx)}, non-seed truth={len(ns_truth)}")
    print(f"  Seeds: {[graph['unique_phrases'][i] for i in obs_label]}")
    print(f"  NS-truth: {[graph['unique_phrases'][i] for i in ns_truth]}")
    print(f"  Alpha: {alpha_fixed:.4f}")

    # ---- 두 모드 비교 ----
    modes = [
        ('fixed', 'σ²=3.0 고정 (AWSGLD와 동일)'),
        ('sample', 'σ² inverse-gamma sampling (R 원본)'),
    ]

    all_results = {}

    for sigma2_mode, mode_desc in modes:
        print(f"\n{'='*70}")
        print(f"  [{mode_desc}]")
        print(f"{'='*70}")

        np.random.seed(200)
        random.seed(200)

        result = run_gibbs_extraction(graph, obs_label, T_RUN, BURN_RUN,
                                      alpha_fixed=alpha_fixed,
                                      lambda_prior=LAMBDA_RUN,
                                      method='gibbs',
                                      sigma2_mode=sigma2_mode,
                                      sigma2_fixed=3.0)

        pmn = result['poster_pi_mn']
        poster_adj = force_obs_to_key2(result['Y'], pmn)

        cutoffs = np.unique(poster_adj)[::-1]
        FDRs = vec_FDR_cal(cutoffs, poster_adj)
        valid = np.where(FDRs < FDR_RUN)[0]
        cutoff = cutoffs[np.max(valid)] if len(valid) > 0 else cutoffs[np.argmin(FDRs)]

        identified = np.where(poster_adj >= cutoff)[0]
        tp = sum(1 for i in identified if i in reader_idx)
        ns_tp = sum(1 for i in identified if i in reader_idx and i not in seed_set)
        tot = len(identified)
        prec = tp / tot if tot > 0 else 0
        rec = tp / len(reader_idx) if len(reader_idx) > 0 else 0

        rank_order = np.argsort(pmn)[::-1]
        rank_map = {idx: r + 1 for r, idx in enumerate(rank_order)}
        ns_ranks = {idx: rank_map[idx] for idx in ns_truth}

        all_results[sigma2_mode] = {
            'pmn': pmn, 'cutoff': cutoff, 'tot': tot, 'tp': tp,
            'ns_tp': ns_tp, 'prec': prec, 'rec': rec,
            'accept': result['accept'], 'rank_order': rank_order,
            'rank_map': rank_map, 'ns_ranks': ns_ranks,
            'poster_adj': poster_adj, 'result': result,
        }

        print(f"\n  Cutoff={cutoff:.6f}, 추출={tot}, TP={tp}, NS-TP={ns_tp}")
        print(f"  Precision={prec:.4f}, Recall={rec:.4f}")
        print(f"  Accept rate: {result['accept']/T_RUN:.3f}")

        if sigma2_mode == 'sample':
            z = geweke_diag(result['sigma2_store'][BURN_RUN:])
            s2_mean = np.mean(result['sigma2_store'][BURN_RUN:])
            s2_med = np.median(result['sigma2_store'][BURN_RUN:])
            print(f"  sigma2: mean={s2_mean:.4f}, median={s2_med:.4f}, Geweke z={z:.4f}")

        # Top-20
        print(f"\n  [Top-20]")
        print(f"  {'Rank':>4s}  {'Phrase':40s}  {'Score':>8s}  Tag")
        print(f"  {'-'*65}")
        for i, idx in enumerate(rank_order[:20], 1):
            p = graph['unique_phrases'][idx]
            sc = pmn[idx]
            tag = ""
            if idx in seed_set:
                tag = "[Seed]"
            elif idx in reader_idx:
                tag = "[T]"
            selected = "<<" if poster_adj[idx] >= cutoff else ""
            print(f"  {i:>4d}  {p:40s}  {sc:>8.6f}  {tag:8s} {selected}")

        # Non-seed truth
        print(f"\n  [NS-truth 순위]")
        for idx in ns_truth:
            r = rank_map[idx]
            p = graph['unique_phrases'][idx]
            sc = pmn[idx]
            print(f"    {r:>5d}위  {p:40s}  score={sc:.6f}")

    # ---- 비교 요약 테이블 ----
    print(f"\n{'='*70}")
    print(f"  [비교 요약] C-84, lambda={LAMBDA_RUN}, FDR={FDR_RUN}, T={T_RUN}")
    print(f"  {'Method':25s} | {'추출':>4s} {'TP':>4s} {'NS-TP':>5s} | {'Prec':>7s} {'Rec':>7s} | {'Accept':>7s} | NS-truth 순위")
    print(f"  {'-'*95}")

    for sigma2_mode, mode_desc in modes:
        r = all_results[sigma2_mode]
        ns_str = ', '.join(f"{r['ns_ranks'][i]}" for i in ns_truth)
        acc = r['accept'] / T_RUN
        label = f"Gibbs({sigma2_mode})"
        print(f"  {label:25s} | {r['tot']:>4d} {r['tp']:>4d} {r['ns_tp']:>5d} | "
              f"{r['prec']:>7.4f} {r['rec']:>7.4f} | {acc:>7.3f} | {ns_str}")

    # AWSGLD v5 결과 참고값 (직전 실행)
    print(f"  {'AWSGLD v5 (fixed σ²=3.0)':25s} |    8    5     1 |  0.6250  0.4545 |     n/a | (참고)")

    print(f"\n{'='*70}")
    print("  Done!")
    print(f"{'='*70}")
