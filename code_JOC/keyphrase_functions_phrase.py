"""
다수 문서 Batch 처리 — 구(Phrase) 기반 키프레이즈 추출
R 원본 구조(MCMC, FDR, 평가) + Phrase 단위 FCM
keyphrase_functions.py와 동일 구조, FCM/그래프/truth만 phrase 단위로 변경
"""

import numpy as np
import os
import re
import time
from datetime import timedelta
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


# ========== Phrase 정규화 ==========
def normalize_phrase(phrase):
    """본문 phrase와 truth phrase에 동일하게 적용하는 정규화"""
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


# ========== Phrase FCM 설정 ==========
MAX_NGRAM = 4
MIN_FREQ = 2
WINDOW_SIZE = 2     # R 원본과 동일

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
    """구(phrase) 기반 co-occurrence matrix 생성"""
    text = text.lower()

    # 먼저 문장 분리 (마침표가 살아있을 때)
    sentences = re.split(r'[.!?]+', text)

    all_phrases_per_sent = []
    phrase_counts = Counter()

    for sent in sentences:
        # 각 문장 내에서 정규화 (하이픈 → 공백, normalize_phrase와 동일 기준)
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

    # 빈도 필터링 (min_freq=1이면 사실상 필터 없음)
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


# ========== 그래프 생성 (Phrase 기반) ==========
def create_graph(article_name):
    """
    구(Phrase) 기반 그래프 생성 + truth 구 단위 매칭
    article_name: 확장자 없는 이름 (예: "C-42")
    """
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

    # R 원본과 동일: 노드 필터 없이 전체 사용
    n = n_total
    A = fcm.copy()
    np.fill_diagonal(A, 0)
    row_sums = A.sum(axis=1)
    row_sums[row_sums == 0] = 1  # 0으로 나누기 방지
    D = np.diag(row_sums)

    raw_text = text.lower()
    raw_text = re.sub(r'[\t\r;]', '', raw_text)
    raw_text = re.sub(r'\n', ' ', raw_text)
    units = len(raw_text.split())
    degree = np.sum((A != 0).sum(axis=1))

    # Truth 레이블 — 정규화된 구 단위 매칭
    truth_path = os.path.join(TRUTH_DIR, article_name)
    truth = []
    if os.path.exists(truth_path):
        with open(truth_path, 'r', encoding='utf-8', errors='ignore') as f:
            key_text = f.read().strip()
        key_text = key_text.lower()
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

    return {
        'n': n,
        'A': A,
        'D': D,
        'unique_phrases': unique_phrases,
        'phrase_to_idx': phrase_to_idx,
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


