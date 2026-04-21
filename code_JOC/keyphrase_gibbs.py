"""
Gibbs-MH 기반 키프레이즈 추출 (Phrase 단위)
- AWSGLD와 비교를 위한 Gibbs-MH 버전
- keyphrase_awsgld.py와 동일한 phrase 기반 구조 사용
"""

import numpy as np
import os
import re
import json
import csv
from scipy.linalg import solve
from scipy.stats import invgamma
from collections import Counter
import time
from datetime import datetime, timedelta

# ========== 설정 ==========
DATA_DIR = "/home/jiyoon/3차/BSS-Keyphrase-Extraction-master/data_JOC"
DOCUMENT = "C-42.txt"

# FDR 관련 설정
FDR_CUTOFF = 0.05
TOP_K_RESULTS = 50

# Gibbs-MH 파라미터
T = 2000              # 총 반복 횟수
BURN_IN = 200        # Burn-in 기간

# N-gram 설정
MAX_NGRAM = 3
MIN_FREQ = 2
WINDOW_SIZE = 5

# N-gram 가중치 설정
NGRAM_WEIGHT_PRESET = 'mild'

NGRAM_WEIGHT_OPTIONS = {
    'none':     {1: 1.0,  2: 1.0,  3: 1.0},
    'mild':     {1: 0.95, 2: 1.05, 3: 1.08},
    'moderate': {1: 0.85, 2: 1.10, 3: 1.15},
    'strong':   {1: 0.70, 2: 1.30, 3: 1.50},
}

NGRAM_WEIGHT = NGRAM_WEIGHT_OPTIONS[NGRAM_WEIGHT_PRESET]

# 출력 옵션
PREFER_LONGER_PHRASES = True
MIN_NGRAM_OUTPUT = 1
EXCLUDE_SUBSUMED = True

# 결과 저장 설정
SAVE_RESULTS = True
OUTPUT_DIR = "/home/jiyoon/3차/BSS-Keyphrase-Extraction-master/results"
SAVE_FORMAT = ['csv', 'txt', 'json']

import random

np.random.seed(12345)
random.seed(12345)

# Ground truth 기반 seed 설정
TRUTH_DIR = os.path.join(DATA_DIR, "pre_process_author_truth")
ALPHA_FIXED = 0.25  # AWSGLD와 동일하게 설정


# ========== 기본 수학 함수들 ==========
def inv_logit(x):
    """Inverse logit (sigmoid) function"""
    x = np.clip(x, -700, 700)
    return np.exp(x) / (1 + np.exp(x))


def base_to_start(Base_Line):
    """Semi-supervised score를 초기값으로 변환"""
    ini_point = Base_Line.copy()
    ini_point[ini_point >= 1] = 0.99
    ini_point[ini_point <= 0] = 0.01
    ini_point = np.log(ini_point / (1 - ini_point))
    return ini_point


def alpha_find(Base_Line, Y, grid):
    """현재 반복에서 alpha 찾기"""
    alpha_est = grid[np.argmax([alpha_lk(Base_Line, Y, alpha) for alpha in grid])]
    return alpha_est


def alpha_lk(Base_Line, Y, alpha):
    """Alpha likelihood 계산 (BSS PU learning structure)"""
    pi = inv_logit(Base_Line)
    pi = np.clip(pi, 1e-10, 1 - 1e-10)

    # Seeds (Y=1): log π
    log_lik = 0.0
    if np.sum(Y == 1) > 0:
        log_lik += np.sum(np.log(pi[Y == 1]))

    # Unlabeled (Y=0): log[(1-α)(1-π) + α·π]
    unlabeled_mask = (Y == 0)
    if np.sum(unlabeled_mask) > 0:
        prob_unlabeled = (1 - alpha) * (1 - pi[unlabeled_mask]) + alpha * pi[unlabeled_mask]
        prob_unlabeled = np.clip(prob_unlabeled, 1e-10, 1 - 1e-10)
        log_lik += np.sum(np.log(prob_unlabeled))

    return log_lik


def posterior_gibbstheta(Y, alpha, theta, u_0, B, sigma2):
    """Gibbs-MH용 log posterior"""
    pi_theta = inv_logit(theta)
    pi_theta = np.clip(pi_theta, 1e-10, 1 - 1e-10)

    # Log-likelihood
    log_lik_seeds = 0.0
    if np.sum(Y == 1) > 0:
        log_lik_seeds = np.sum(np.log(pi_theta[Y == 1]))

    log_lik_unlabeled = 0.0
    unlabeled_mask = (Y == 0)
    if np.sum(unlabeled_mask) > 0:
        prob_unlabeled = (1 - alpha) * (1 - pi_theta[unlabeled_mask]) + alpha * pi_theta[unlabeled_mask]
        prob_unlabeled = np.clip(prob_unlabeled, 1e-10, 1 - 1e-10)
        log_lik_unlabeled = np.sum(np.log(prob_unlabeled))

    log_likelihood = log_lik_seeds + log_lik_unlabeled

    # Log-prior
    C = (B @ (theta - u_0)).T @ (B @ (theta - u_0))
    log_prior = -C / (2 * sigma2)

    return log_likelihood + log_prior


def force_obs_to_key(Y, poster_pi_mean, k):
    """관찰된 레이블을 양성 키프레이즈로 강제"""
    poster_pi_mean = poster_pi_mean.copy()
    obs_indices = np.where(Y == 1)[0]
    if len(obs_indices) > 0:
        poster_pi_mean[obs_indices] = 10 + np.random.normal(0, 0.01, len(obs_indices))
    return poster_pi_mean


def force_obs_to_key2(Y, poster_pi_mean):
    """관찰된 레이블을 양성 키프레이즈로 강제 (버전 2)"""
    poster_pi_mean = poster_pi_mean.copy()
    poster_pi_mean[Y == 1] = 0.99
    return poster_pi_mean


def FDR_calculate(cutoff, poster_md_adjust):
    """주어진 cutoff에 대한 FDR 계산"""
    set_vals = poster_md_adjust[poster_md_adjust >= cutoff]
    if len(set_vals) == 0:
        return 1.0
    FDR = np.sum(1 - set_vals) / len(set_vals)
    return FDR


def vec_FDR_cal(cutoffs, poster_md_adjust):
    """벡터화된 FDR 계산"""
    return np.array([FDR_calculate(c, poster_md_adjust) for c in cutoffs])


# ========== 구(Phrase) 기반 FCM 생성 ==========
def get_stopwords():
    """영어 불용어 집합 반환"""
    return {
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
        'such', 'into', 'over', 'after', 'out', 'up', 'down', 'off',
        'under', 'again', 'once', 'here', 'there', 'when', 'where',
        'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
        'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
        'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will',
        'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're',
        've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn',
        'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn',
        'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn', 'et', 'al',
        'eg', 'ie', 'etc', 'vs', 'via', 'use'
    }


def create_fcm_phrases(text, max_ngram=3, window=5, min_freq=2):
    """구(phrase) 기반 co-occurrence matrix 생성"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    sentences = re.split(r'[.!?]', text)
    stopwords = get_stopwords()

    all_phrases_per_sent = []
    phrase_counts = Counter()

    for sent in sentences:
        words = sent.split()
        words_clean = [w for w in words if len(w) > 1]

        if len(words_clean) == 0:
            continue

        sent_phrases = []
        for n in range(1, max_ngram + 1):
            for i in range(len(words_clean) - n + 1):
                phrase_words = words_clean[i:i+n]

                if n == 1 and phrase_words[0] in stopwords:
                    continue
                if n > 1 and (phrase_words[0] in stopwords or phrase_words[-1] in stopwords):
                    continue
                if all(w in stopwords for w in phrase_words):
                    continue

                phrase = ' '.join(phrase_words)
                sent_phrases.append(phrase)
                phrase_counts[phrase] += 1

        all_phrases_per_sent.append(sent_phrases)

    valid_phrases = [p for p, c in phrase_counts.items() if c >= min_freq]
    valid_phrases = [p for p in valid_phrases if not p.replace(' ', '').isdigit()]
    valid_phrases = [p for p in valid_phrases if len(p) > 2 or ' ' in p]

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


# ========== 그래프 생성 함수 ==========
def create_graph(article_name, data_dir=DATA_DIR):
    """구(phrase) 기반 그래프 생성"""
    file_path = os.path.join(data_dir, article_name)

    if not os.path.exists(file_path):
        print(f"파일 없음: {file_path}")
        return None

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()

    fcm, unique_phrases, phrase_to_idx, phrase_counts = create_fcm_phrases(
        text, max_ngram=MAX_NGRAM, window=WINDOW_SIZE, min_freq=MIN_FREQ
    )
    n = len(unique_phrases)

    if n == 0:
        print("유효한 구가 없습니다.")
        return None

    print(f"  총 {n}개의 구(phrase) 생성됨")

    ngram_dist = {1: 0, 2: 0, 3: 0}
    for p in unique_phrases:
        word_count = len(p.split())
        if word_count in ngram_dist:
            ngram_dist[word_count] += 1
    print(f"  N-gram 분포: 1-gram={ngram_dist[1]}, 2-gram={ngram_dist[2]}, 3-gram={ngram_dist[3]}")

    d = 0.85
    A = fcm.copy()
    np.fill_diagonal(A, 0)

    row_sums = A.sum(axis=1)
    row_sums[row_sums == 0] = 1
    D = np.diag(row_sums)

    connection_counts = (A > 0).sum(axis=1)
    keep_mask = connection_counts >= 1
    keep_indices = np.where(keep_mask)[0]

    if len(keep_indices) == 0:
        print("필터링 후 유효한 노드가 없습니다.")
        return None

    A_filtered = A[np.ix_(keep_indices, keep_indices)]
    row_sums_filtered = A_filtered.sum(axis=1)
    row_sums_filtered[row_sums_filtered == 0] = 1
    D_filtered = np.diag(row_sums_filtered)

    dictionary_minus = np.array([[unique_phrases[i], i] for i in keep_indices], dtype=object)

    print(f"  필터링 후 {len(keep_indices)}개의 구 유지됨")

    return {
        'n': n,
        'A': A,
        'D': D,
        'A_minus': A_filtered,
        'D_minus': D_filtered,
        'n_minus': len(keep_indices),
        'dictionary_minus': dictionary_minus,
        'keep_indices': keep_indices,
        'phrase_counts': phrase_counts,
        'unique_phrases': unique_phrases,
        'phrase_to_idx': phrase_to_idx
    }


# ========== Gibbs-MH 샘플링 함수 ==========
def gibbs_mh_sampler(Burn_in, T, ini, n, graph, Y, B, u_0, alpha_est, grid, verbose=True):
    """
    Gibbs-MH Sampler
    """
    d = 0.85
    theta_store = np.zeros((T, n))
    sigma2_store = np.zeros(T)
    alpha_store = np.zeros(T)
    accept = 0
    theta = ini.copy()
    
    start_time = time.time()
    print_interval = max(1, T // 20)
    
    for t in range(T):
        # Sample from sigma
        C = (B @ (theta - u_0)).T @ (B @ (theta - u_0))
        sigma2 = invgamma.rvs(n / 2 + 0.001, scale=C / 2 + 0.001)
        sigma2_store[t] = sigma2
        
        # Sample from theta
        try:
            BtB = B.T @ B
            BtB_inv = solve(BtB + np.eye(n) * 1e-6, np.eye(n))
            cov_matrix = BtB_inv * sigma2 * 4 / n
            cov_diag = np.diag(cov_matrix)
            cov_diag = np.clip(cov_diag, 1e-10, 1e10)
            theta_star = theta + np.random.normal(0, np.sqrt(cov_diag))
        except:
            theta_star = theta + np.random.normal(0, np.sqrt(sigma2 * 4 / n), n)
        
        theta_star = np.clip(theta_star, -700, 700)
        
        log_MH_rate = (posterior_gibbstheta(Y, alpha_est, theta_star, u_0, B, sigma2) - 
                       posterior_gibbstheta(Y, alpha_est, theta, u_0, B, sigma2))
        MH_rate = np.exp(np.clip(log_MH_rate, -700, 700))
        u = np.random.uniform()
        
        if u < MH_rate:
            theta = theta_star
            accept += 1
        
        theta_store[t, :] = theta
        alpha_est = alpha_find(theta, Y, grid)
        alpha_store[t] = alpha_est
        
        # 진행률 표시
        if verbose and (t + 1) % print_interval == 0:
            elapsed = time.time() - start_time
            progress = (t + 1) / T * 100
            eta_seconds = elapsed / (t + 1) * (T - t - 1)
            eta = timedelta(seconds=int(eta_seconds))
            print(f"\r  Gibbs-MH 진행률: {progress:.1f}% ({t+1}/{T}) | "
                  f"경과: {timedelta(seconds=int(elapsed))} | "
                  f"남은 시간: {eta} | "
                  f"Accept: {accept/(t+1):.3f}", end='', flush=True)
    
    if verbose:
        print()
    
    total_time = time.time() - start_time
    print(f"  Gibbs-MH 완료! 소요 시간: {timedelta(seconds=int(total_time))}, Accept rate: {accept / T:.4f}")
    
    prob = inv_logit(theta_store)
    poster_pi_md = np.median(prob[Burn_in:T, :], axis=0)
    poster_pi_md[poster_pi_md == 1] = 1 - np.random.uniform(0, 0.01, np.sum(poster_pi_md == 1))
    poster_pi_mn = np.mean(prob[Burn_in:T, :], axis=0)
    
    alpha_mn = np.mean(alpha_store[Burn_in:])
    alpha_md = np.median(alpha_store[Burn_in:])
    
    return {
        'poster_pi_md': poster_pi_md,
        'poster_pi_mn': poster_pi_mn,
        'theta_store': theta_store,
        'sigma2_store': sigma2_store,
        'alpha_mn': alpha_mn,
        'alpha_md': alpha_md,
        'accept': accept
    }


def run_keyphrase_extraction(graph, obs_label, T, burn_in):
    """키프레이즈 추출 실행 (Gibbs-MH 사용)"""
    n_minus = graph['n_minus']
    d = 0.85

    G_minus = solve(graph['D_minus'], graph['A_minus'])
    B_minus = np.eye(n_minus) - d * G_minus.T
    u_0_minus = solve(B_minus, np.ones(n_minus) * (1 - d))

    w_minus = np.diag(1.0 / np.sqrt(np.diag(graph['D_minus'])))
    B_star_minus = np.eye(n_minus) - d * w_minus @ graph['A_minus'] @ w_minus

    Y_minus = np.zeros(n_minus)
    for idx in obs_label:
        if idx < n_minus:
            Y_minus[idx] = 1

    Base_Line_minus = solve(B_star_minus, Y_minus)

    ini = base_to_start(u_0_minus)
    
    # Alpha grid (AWSGLD와 동일)
    grid = np.linspace(0.05, 0.5, 30)
    alpha_est = ALPHA_FIXED

    print(f"  Gibbs-MH 시작 (T={T}, Burn_in={burn_in})...")
    print(f"  Gibbs-MH 파라미터: alpha={ALPHA_FIXED}")

    # Gibbs-MH 샘플러 사용
    test_chain = gibbs_mh_sampler(burn_in, T, ini, n_minus, graph, Y_minus,
                                   B_minus, u_0_minus, alpha_est, grid)

    return {
        'poster_pi_mn': test_chain['poster_pi_mn'],
        'poster_pi_md': test_chain['poster_pi_md'],
        'dictionary_minus': graph['dictionary_minus'],
        'u_0_minus': u_0_minus,
        'Base_Line_minus': Base_Line_minus,
        'Y_minus': Y_minus,
        'accept_rate': test_chain['accept'] / T
    }


# ========== FDR Cutoff 함수 ==========
def calculate_cutoff(poster_pi_mn, c, Y):
    """FDR cutoff 계산"""
    poster_md_adjust = force_obs_to_key2(Y, poster_pi_mn)
    cutoffs = np.unique(np.sort(poster_md_adjust))
    FDRs = vec_FDR_cal(cutoffs, poster_md_adjust)

    print(f"\n  [Cutoff 계산 디버깅]")
    print(f"    Cutoff 범위: {cutoffs.min():.4f} ~ {cutoffs.max():.4f}")
    print(f"    FDR 범위: {FDRs.min():.4f} ~ {FDRs.max():.4f}")

    valid_mask = FDRs <= c

    if np.any(valid_mask):
        valid_cutoffs = cutoffs[valid_mask]
        selected_cutoff = valid_cutoffs.min()
        selected_fdr = FDRs[cutoffs == selected_cutoff][0]
        print(f"    선택된 Cutoff: {selected_cutoff:.4f} (FDR: {selected_fdr:.4f})")
        return selected_cutoff
    else:
        min_fdr_idx = np.argmin(FDRs)
        selected_cutoff = cutoffs[min_fdr_idx]
        print(f"    FDR 조건 불만족, 최소 FDR cutoff 사용: {selected_cutoff:.4f} (FDR: {FDRs[min_fdr_idx]:.4f})")
        return selected_cutoff


# ========== 필터링 함수들 ==========
def filter_subsumed_phrases(phrases_with_scores, dictionary):
    """더 긴 구에 포함된 짧은 구 제거"""
    sorted_items = sorted(phrases_with_scores, key=lambda x: x[1], reverse=True)

    kept = []
    kept_phrases = set()

    for idx, score in sorted_items:
        phrase = dictionary[idx, 0]
        words = set(phrase.split())

        is_subsumed = False
        for kept_phrase in kept_phrases:
            kept_words = set(kept_phrase.split())
            if words.issubset(kept_words) and len(words) < len(kept_words):
                is_subsumed = True
                break

        if not is_subsumed:
            kept.append((idx, score))
            kept_phrases.add(phrase)

    return kept


def apply_ngram_weights(scores, dictionary):
    """N-gram 길이에 따른 가중치 적용"""
    weighted_scores = scores.copy()

    for idx in range(len(weighted_scores)):
        phrase = dictionary[idx, 0]
        n = len(phrase.split())
        weight = NGRAM_WEIGHT.get(n, 1.0)
        weighted_scores[idx] *= weight

    return weighted_scores


def filter_by_ngram_length(phrases_with_scores, dictionary, min_ngram=1):
    """최소 n-gram 길이로 필터링"""
    filtered = []
    for idx, score in phrases_with_scores:
        phrase = dictionary[idx, 0]
        n = len(phrase.split())
        if n >= min_ngram:
            filtered.append((idx, score))
    return filtered


# ========== 결과 출력 함수 ==========
def print_results(result, cutoff, fdr_cutoff=FDR_CUTOFF, top_k=TOP_K_RESULTS):
    """결과 출력"""
    print(f"\n{'='*70}")
    print(f"Gibbs-MH 추출 결과")
    print(f"{'='*70}")
    print(f"FDR Cutoff: {cutoff:.4f}")
    print(f"N-gram 가중치 ({NGRAM_WEIGHT_PRESET}): 1-gram={NGRAM_WEIGHT[1]}, 2-gram={NGRAM_WEIGHT[2]}, 3-gram={NGRAM_WEIGHT[3]}")
    print()

    results_dict = {
        'document': DOCUMENT,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'method': 'Gibbs-MH',
        'accept_rate': result.get('accept_rate', 0.0)
    }

    # BSS-Gibbs 방법
    weighted_scores = apply_ngram_weights(result['poster_pi_mn'], result['dictionary_minus'])
    avg_weight = np.mean(list(NGRAM_WEIGHT.values()))
    adjusted_cutoff = cutoff * avg_weight * 0.8

    identified_idx = np.where(weighted_scores >= adjusted_cutoff)[0]
    phrases_with_scores = [(idx, weighted_scores[idx]) for idx in identified_idx]

    if MIN_NGRAM_OUTPUT > 1:
        phrases_with_scores = filter_by_ngram_length(phrases_with_scores, result['dictionary_minus'], MIN_NGRAM_OUTPUT)
    if EXCLUDE_SUBSUMED:
        phrases_with_scores = filter_subsumed_phrases(phrases_with_scores, result['dictionary_minus'])

    sorted_phrases = sorted(phrases_with_scores, key=lambda x: x[1], reverse=True)

    print(f"[BSS-Gibbs 방법] - 총 {len(sorted_phrases)}개 키프레이즈 (상위 {top_k}개 표시)")
    print("-" * 70)

    bss_results = []

    for i, (idx, score) in enumerate(sorted_phrases, 1):
        phrase = result['dictionary_minus'][idx, 0]
        word_count = len(phrase.split())
        original_score = result['poster_pi_mn'][idx]

        bss_results.append({
            'rank': i,
            'phrase': phrase,
            'weighted_score': float(score),
            'original_score': float(original_score),
            'ngram': word_count
        })

        if i <= top_k:
            print(f"  {i:2d}. {phrase:40s} (가중:{score:.4f}, 원본:{original_score:.4f}) [{word_count}-gram]")

    if len(sorted_phrases) > top_k:
        print(f"  ... 외 {len(sorted_phrases) - top_k}개 더")

    results_dict['keyphrases'] = bss_results

    return results_dict


# ========== 메인 실행 ==========
if __name__ == "__main__":
    print(f"{'='*70}")
    print(f"Gibbs-MH 기반 키프레이즈 추출 (Phrase 단위)")
    print(f"{'='*70}")
    print(f"문서: {DOCUMENT}")
    print(f"Gibbs-MH 설정: T={T}, Burn_in={BURN_IN}")
    print(f"N-gram 설정: max_ngram={MAX_NGRAM}, min_freq={MIN_FREQ}, window={WINDOW_SIZE}")
    print(f"N-gram 가중치 ({NGRAM_WEIGHT_PRESET}): {NGRAM_WEIGHT}")
    print(f"FDR Cutoff: {FDR_CUTOFF}")
    print(f"{'='*70}\n")

    # 1. 그래프 생성
    print("1. 그래프 생성 중...")
    graph = create_graph(DOCUMENT)

    if graph is None:
        print("그래프 생성 실패!")
        exit(1)

    print(f"  그래프 생성 완료\n")

    # 2. Ground truth 기반 seed 선택
    print("2. 관찰 레이블(Seed) 선택 중...")
    doc_name = os.path.splitext(DOCUMENT)[0]
    truth_path = os.path.join(TRUTH_DIR, doc_name)
    with open(truth_path, 'r', encoding='utf-8') as f:
        ground_truth = [w.strip() for w in f.read().strip().split(',')]
    print(f"  Ground truth ({len(ground_truth)}개): {ground_truth}")

    # Ground truth에서 dictionary에 있는 것만 index 찾기
    truth_indices = []
    for i, row in enumerate(graph['dictionary_minus']):
        if row[0] in ground_truth:
            truth_indices.append(i)
    print(f"  Dictionary에서 매칭된 ground truth: {len(truth_indices)}개")

    # Seed 3개 (AWSGLD와 동일)
    k_total = len(truth_indices)
    k_seed = 3
    OBS_LABEL = random.sample(truth_indices, k_seed)

    print(f"  Alpha = {ALPHA_FIXED:.4f} (seed {k_seed}/{k_total})")
    print(f"  선택된 Seed 구 (총 {len(OBS_LABEL)}개):")
    for idx in OBS_LABEL:
        phrase = graph['dictionary_minus'][idx, 0]
        word_count = len(phrase.split())
        print(f"    - [{idx}] {phrase} ({word_count}-gram)")
    print()

    # 3. Gibbs-MH로 키프레이즈 추출
    print("3. Gibbs-MH 키프레이즈 추출 중...")
    result = run_keyphrase_extraction(graph, OBS_LABEL, T, BURN_IN)
    print("  추출 완료\n")

    # 4. 통계 정보
    print("4. 통계 정보")
    print("="*50)
    print(f"  poster_pi_mn 통계:")
    print(f"    Min: {np.min(result['poster_pi_mn']):.6f}")
    print(f"    Max: {np.max(result['poster_pi_mn']):.6f}")
    print(f"    Mean: {np.mean(result['poster_pi_mn']):.6f}")
    print(f"    Std: {np.std(result['poster_pi_mn']):.6f}")

    poster_adjusted = force_obs_to_key2(result['Y_minus'], result['poster_pi_mn'])
    cutoffs = np.unique(np.sort(poster_adjusted))
    FDRs = vec_FDR_cal(cutoffs, poster_adjusted)
    print(f"\n  FDR 분포:")
    print(f"    Min FDR: {np.min(FDRs):.6f}")
    print(f"    Max FDR: {np.max(FDRs):.6f}")
    print(f"    FDR < {FDR_CUTOFF}인 cutoff 개수: {np.sum(FDRs < FDR_CUTOFF)}")
    print("="*50)

    # 5. 결과 출력
    cutoff = calculate_cutoff(result['poster_pi_mn'], FDR_CUTOFF, result['Y_minus'])
    results_dict = print_results(result, cutoff)

    # 6. 평가
    from evaluate import quick_evaluate

    predictions = [item['phrase'] for item in results_dict['keyphrases']]

    print(f"\n{'='*70}")
    print("평가 결과")
    print(f"{'='*70}")

    metrics = quick_evaluate(predictions, ground_truth, match_type='partial')

    print(f"{'='*70}")

    print(f"\n{'='*70}")
    print("  모든 작업 완료!")
    print(f"{'='*70}")
