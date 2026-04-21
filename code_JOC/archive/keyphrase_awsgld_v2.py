"""
AWSGLD 기반 키프레이즈 추출 v2
- 구(Phrase) 단위 FCM 사용
- keyphrase_functions.py에서 기본 수학 함수만 import
- v1 대비 수정: seed 선택, FDR cutoff 버그 수정, force_obs_to_key2 통일
"""

import numpy as np
import os
import re
import json
import csv
import random
import time
from datetime import datetime, timedelta
from scipy.linalg import solve
from collections import Counter

from keyphrase_functions import (
    inv_logit, base_to_start, alpha_find, alpha_lk,
    force_obs_to_key,
    FDR_calculate, vec_FDR_cal,
    DATA_DIR, TRUTH_DIR, grid,
)

# ========== 설정 ==========
DOCUMENT = "C-42"
PREPROCESS_DIR = os.path.join(DATA_DIR, "pre_process")

# FDR
FDR_CUTOFF = 0.05
TOP_K_RESULTS = 50

# AWSGLD 파라미터
T = 5000
BURN_IN = 500
TAU = 1.0
ZETA = 5.0
SIGMA2_FIXED = 3.0
M_REGIONS = 1000

# Phrase FCM 설정
MAX_NGRAM = 4
MIN_FREQ = 2
WINDOW_SIZE = 2     # R 원본과 동일

k = 4

np.random.seed(12345)
random.seed(12345)

# 결과 저장
SAVE_RESULTS = True
OUTPUT_DIR = "/home/jiyoon/3차/BSS-Keyphrase-Extraction-master/results"
SAVE_FORMAT = ['csv', 'txt', 'json']


# ========== Phrase 정규화 ==========
def normalize_phrase(phrase):
    """본문 phrase와 truth phrase에 동일하게 적용하는 정규화"""
    phrase = phrase.lower().strip()
    phrase = re.sub(r'-', ' ', phrase)         # 하이픈 → 공백 ("markov-chain" → "markov chain")
    phrase = re.sub(r'[^\w\s]', '', phrase)    # 나머지 구두점 제거
    phrase = re.sub(r'\s+', ' ', phrase).strip()
    return phrase


# ========== force_obs_to_key2 (R 원본과 동일: Y==1 → 1) ==========
def force_obs_to_key2(Y, poster_pi_mean):
    poster_pi_mean = poster_pi_mean.copy()
    poster_pi_mean[Y == 1] = 1
    return poster_pi_mean


# ========== 불용어 ==========
def get_stopwords():
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
        'over', 'out', 'up', 'down', 'off', 'every', 's', 't', 'don',
        'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'et', 'al',
        'eg', 'ie', 'etc', 'vs', 'via', 'use',
    }


# ========== 구(Phrase) 기반 FCM 생성 ==========
def create_fcm_phrases(text, max_ngram=MAX_NGRAM, window=WINDOW_SIZE, min_freq=MIN_FREQ):
    """
    Phrase 기반 co-occurrence matrix 생성
    R 원본과 최대한 가깝게: sentence split → sentence별 cleaning → n-gram → sentence 내 co-occurrence
    """
    text = text.lower()

    # 1. 먼저 문장 분리 (구두점이 살아있을 때)
    sentences = re.split(r'[.!?]+', text)

    stopwords = get_stopwords()
    all_phrases_per_sent = []
    phrase_counts = Counter()

    for sent in sentences:
        # 2. 각 문장 내에서 정규화 (하이픈 → 공백, normalize_phrase와 동일 기준)
        sent = re.sub(r'-', ' ', sent)
        sent = re.sub(r'[^\w\s]', ' ', sent)
        sent = re.sub(r'\s+', ' ', sent).strip()

        words = sent.split()
        words_clean = [w for w in words if len(w) > 1]
        if len(words_clean) == 0:
            continue

        # 3. N-gram 생성 (문장 단위)
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

    # 4. 빈도 필터링 (min_freq=1이면 사실상 필터 없음)
    valid_phrases = [p for p, c in phrase_counts.items() if c >= min_freq]

    n = len(valid_phrases)
    if n == 0:
        return np.array([]), [], {}, phrase_counts

    phrase_to_idx = {p: i for i, p in enumerate(valid_phrases)}

    # 5. Co-occurrence (문장 내 window)
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
def create_phrase_graph(article_name):
    """구(phrase) 기반 그래프 생성 + truth 로드"""
    text_path = os.path.join(PREPROCESS_DIR, article_name + ".txt.final")
    if not os.path.exists(text_path):
        text_path = os.path.join(DATA_DIR, article_name + ".txt")
    if not os.path.exists(text_path):
        print(f"파일 없음: {text_path}")
        return None

    with open(text_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()

    fcm, unique_phrases, phrase_to_idx, phrase_counts = create_fcm_phrases(
        text, max_ngram=MAX_NGRAM, window=WINDOW_SIZE, min_freq=MIN_FREQ)
    n_total = len(unique_phrases)
    if n_total == 0:
        print("유효한 구가 없습니다.")
        return None

    n = n_total
    ngram_dist = {1: 0, 2: 0, 3: 0, 4: 0}
    for p in unique_phrases:
        wc = len(p.split())
        if wc in ngram_dist:
            ngram_dist[wc] += 1
    print(f"  총 {n}개 구 (1g={ngram_dist[1]}, 2g={ngram_dist[2]}, 3g={ngram_dist[3]}, 4g={ngram_dist[4]})")

    # R 원본과 동일: 노드 필터 없이 전체 사용
    A = fcm.copy()
    np.fill_diagonal(A, 0)
    row_sums = A.sum(axis=1)
    row_sums[row_sums == 0] = 1  # 0으로 나누기 방지
    D = np.diag(row_sums)

    dictionary = np.array([[unique_phrases[i], i] for i in range(n)], dtype=object)

    # Truth 로드 (정규화된 구 단위 매칭)
    truth_path = os.path.join(TRUTH_DIR, article_name)
    truth_indices = []
    truth_phrases = []
    if os.path.exists(truth_path):
        with open(truth_path, 'r', encoding='utf-8', errors='ignore') as f:
            key_text = f.read().strip().lower()
        key_text = re.sub(r'[\t\r;]', '', key_text)
        key_text = re.sub(r'\n', ' ', key_text)
        truth_phrases = [p.strip() for p in key_text.split(',') if p.strip()]

        # 정규화: dictionary와 truth 양쪽 모두 normalize_phrase 적용
        norm_to_idx = {}
        for i in range(n):
            np_ = normalize_phrase(unique_phrases[i])
            if np_ not in norm_to_idx:
                norm_to_idx[np_] = i

        for tp in truth_phrases:
            ntp = normalize_phrase(tp)
            if ntp in norm_to_idx:
                truth_indices.append(norm_to_idx[ntp])

        print(f"  Truth: {len(truth_phrases)}개 keyphrase → dictionary 매칭 {len(truth_indices)}개")
    else:
        print(f"  Truth 파일 없음: {truth_path}")

    return {
        'n': n,
        'A': A,
        'D': D,
        'dictionary': dictionary,
        'unique_phrases': unique_phrases,
        'phrase_to_idx': phrase_to_idx,
        'phrase_counts': phrase_counts,
        'truth_indices': truth_indices,
        'truth_phrases': truth_phrases,
    }


# ========== AWSGLD 전용 함수 ==========
def posterior_energy(Y, alpha, theta, u_0, B, sigma2):
    """
    AWSGLD용 energy = -log posterior
    R 원본 likelihood:
      temp <- (1-alpha)*inv_logit(theta)
      lglk <- sum(Y*log(temp) + (1-Y)*log(1-temp))
    """
    pi_theta = inv_logit(theta)
    pi_theta = np.clip(pi_theta, 1e-10, 1 - 1e-10)

    temp = (1 - alpha) * pi_theta
    temp = np.clip(temp, 1e-10, 1 - 1e-10)

    log_likelihood = np.sum(Y * np.log(temp) + (1 - Y) * np.log(1 - temp))

    C = (B @ (theta - u_0)).T @ (B @ (theta - u_0))
    log_prior = -C / (2 * sigma2)

    return -(log_likelihood + log_prior)


def grad_posterior_energy(Y, alpha, theta, u_0, B, sigma2):
    """
    AWSGLD용 gradient of energy = -gradient of log posterior
    R 원본 likelihood: log L = Y*log[(1-α)π] + (1-Y)*log[1-(1-α)π]

    ∂logL/∂θ:
      Y=1: ∂/∂θ log[(1-α)π] = π'(θ)/π(θ) = (1-π)
      Y=0: ∂/∂θ log[1-(1-α)π] = -(1-α)π'(θ) / [1-(1-α)π]
           where π'(θ) = π(1-π)
    """
    n = len(theta)
    pi_theta = inv_logit(theta)
    pi_theta = np.clip(pi_theta, 1e-10, 1 - 1e-10)

    dpi_dtheta = pi_theta * (1 - pi_theta)  # π'(θ)

    grad_ll = np.zeros(n)

    # Y=1: ∂/∂θ log[(1-α)π] = (1-π)
    seed_mask = (Y == 1)
    if np.sum(seed_mask) > 0:
        grad_ll[seed_mask] = 1 - pi_theta[seed_mask]

    # Y=0: ∂/∂θ log[1-(1-α)π] = -(1-α)π(1-π) / [1-(1-α)π]
    unlabeled_mask = (Y == 0)
    if np.sum(unlabeled_mask) > 0:
        temp_u = (1 - alpha) * pi_theta[unlabeled_mask]
        temp_u = np.clip(temp_u, 1e-10, 1 - 1e-10)
        denom = 1 - temp_u
        denom = np.clip(denom, 1e-10, None)
        grad_ll[unlabeled_mask] = -(1 - alpha) * dpi_dtheta[unlabeled_mask] / denom

    # Gradient of log prior: -B^TB(θ-u₀)/σ²
    BtB = B.T @ B
    grad_prior = -BtB @ (theta - u_0) / sigma2

    # Energy = -log posterior → gradient = -(grad_ll + grad_prior)
    return -(grad_ll + grad_prior)


def awsgld_sampler(Burn_in, T, ini, n, Y, B, u_0, alpha_est,
                   tau=TAU, zeta=ZETA, m=M_REGIONS, verbose=True):
    theta_store = np.zeros((T, n))
    alpha_store = np.zeros(T)
    adaptive_weights = np.arange(1, m + 1) / m
    theta = ini.copy()

    start_time = time.time()
    print_interval = max(1, T // 20)

    energy_samples = []
    warmup = 100
    energy_min = energy_max = delta_u_actual = None

    for t in range(T):
        sigma2 = SIGMA2_FIXED
        eps_k = 0.3 / ((t + 1)**0.6 + 10)
        omega_k = 0.02 / ((t + 1)**0.6 + 100)

        U_tilde = posterior_energy(Y, alpha_est, theta, u_0, B, sigma2)
        grad_U = grad_posterior_energy(Y, alpha_est, theta, u_0, B, sigma2)

        if t < warmup:
            energy_samples.append(U_tilde)
            J_tilde = m // 2
            grad_mult = 1.0
            if t == warmup - 1:
                e_min = np.min(energy_samples)
                e_max = np.max(energy_samples)
                e_range = max(e_max - e_min, 1.0)
                energy_min = e_min - 0.5 * e_range
                energy_max = e_max + 0.5 * e_range
                delta_u_actual = (energy_max - energy_min) / m
                if verbose:
                    print(f"\n  Energy range: [{energy_min:.2f}, {energy_max:.2f}]")
                energy_samples = None
        else:
            J_tilde = int(np.clip(np.floor((U_tilde - energy_min) / delta_u_actual), 0, m - 1))
            eps_log = 1e-12
            grad_mult = 1 + (zeta * tau / delta_u_actual) * (
                np.log(adaptive_weights[J_tilde] + eps_log)
                - np.log(adaptive_weights[max(J_tilde - 1, 0)] + eps_log))
            grad_mult = np.clip(grad_mult, 0.5, 5.0)

        e_k = np.random.randn(n)
        theta = theta - eps_k * grad_mult * grad_U + np.sqrt(2 * tau * eps_k) * e_k
        theta = np.clip(theta, -700, 700)

        for i in range(m):
            indicator = 1 if i >= J_tilde else 0
            adaptive_weights[i] += omega_k * adaptive_weights[J_tilde] * (indicator - adaptive_weights[i])
        adaptive_weights = np.clip(adaptive_weights, 1e-10, 1.0)

        theta_store[t, :] = theta
        alpha_est = alpha_find(theta, Y, grid)  # Alpha MLE 재추정
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
    print(f"  AWSGLD 완료! ({timedelta(seconds=int(time.time() - start_time))})")

    prob = inv_logit(theta_store)
    poster_pi_md = np.median(prob[Burn_in:T, :], axis=0)
    mask = poster_pi_md == 1
    poster_pi_md[mask] = 1 + np.random.normal(0, 0.01, np.sum(mask))
    poster_pi_mn = np.mean(prob[Burn_in:T, :], axis=0)

    return {
        'poster_pi_md': poster_pi_md,
        'poster_pi_mn': poster_pi_mn,
        'theta_store': theta_store,
        'alpha_mn': np.mean(alpha_store[Burn_in:]),
        'alpha_md': np.median(alpha_store[Burn_in:]),
        'adaptive_weights': adaptive_weights,
    }


# ========== 키프레이즈 추출 ==========
def run_awsgld_extraction(graph, obs_label, T, burn_in, alpha_fixed=0.5):
    n = graph['n']
    d = 0.85

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

    ini = base_to_start(Base_Line)

    print(f"  AWSGLD (T={T}, Burn_in={burn_in}, tau={TAU}, zeta={ZETA}, alpha={alpha_fixed})")
    chain = awsgld_sampler(burn_in, T, ini, n, Y, B, u_0, alpha_fixed)

    return {
        'poster_pi_mn': chain['poster_pi_mn'],
        'poster_pi_md': chain['poster_pi_md'],
        'u_0': u_0,
        'Base_Line': Base_Line,
        'Y': Y,
        'adaptive_weights': chain['adaptive_weights'],
    }


# ========== FDR Cutoff (내림차순 — 수정됨) ==========
def calculate_cutoff(poster_pi_mn, c, Y):
    poster_md_adjust = force_obs_to_key2(Y, poster_pi_mn)
    cutoffs = np.unique(poster_md_adjust)[::-1]  # 내림차순
    FDRs = vec_FDR_cal(cutoffs, poster_md_adjust)
    valid = np.where(FDRs < c)[0]
    if len(valid) > 0:
        return cutoffs[np.max(valid)]
    return cutoffs[np.argmin(FDRs)]


# ========== 결과 저장 ==========
def save_results_to_csv(results_dict, filepath):
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['# AWSGLD v2 키프레이즈 추출 결과'])
        writer.writerow(['Document', results_dict['document']])
        writer.writerow(['Timestamp', results_dict['timestamp']])
        writer.writerow([])
        writer.writerow(['# 설정'])
        for key, value in results_dict['settings'].items():
            writer.writerow([key, value])
        writer.writerow([])
        for method_name, keyphrases in results_dict['methods'].items():
            writer.writerow([f'# {method_name}'])
            writer.writerow(['Rank', 'Phrase', 'Score', 'N-gram'])
            for item in keyphrases:
                writer.writerow([item['rank'], item['phrase'],
                                 f"{item['score']:.6f}", item['ngram']])
            writer.writerow([])


def save_results_to_txt(results_dict, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("AWSGLD v2 키프레이즈 추출 결과 (Phrase 기반)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"문서: {results_dict['document']}\n")
        f.write(f"시간: {results_dict['timestamp']}\n\n")
        for key, value in results_dict['settings'].items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        for method_name, keyphrases in results_dict['methods'].items():
            f.write("=" * 70 + "\n")
            f.write(f"[{method_name}] - {len(keyphrases)}개\n")
            f.write("-" * 70 + "\n")
            for item in keyphrases:
                f.write(f"  {item['rank']:3d}. {item['phrase']:40s} "
                        f"({item['score']:.4f}) [{item['ngram']}-gram]\n")
            f.write("\n")


def save_results_to_json(results_dict, filepath):
    def convert(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, dict): return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, list): return [convert(i) for i in obj]
        return obj
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(convert(results_dict), f, ensure_ascii=False, indent=2)


def save_all_results(results_dict, output_dir, document_name, formats):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved = []
    for fmt in formats:
        p = os.path.join(output_dir, f"{document_name}_awsgld_v2_{ts}.{fmt}")
        if fmt == 'csv': save_results_to_csv(results_dict, p)
        elif fmt == 'txt': save_results_to_txt(results_dict, p)
        elif fmt == 'json': save_results_to_json(results_dict, p)
        saved.append(p)
        print(f"  {fmt.upper()}: {p}")
    return saved


# ========== 메인 실행 ==========
if __name__ == "__main__":
    print(f"{'='*70}")
    print(f"AWSGLD v2 키프레이즈 추출 (Phrase 기반)")
    print(f"{'='*70}")
    print(f"문서: {DOCUMENT}")
    print(f"Phrase: max_ngram={MAX_NGRAM}, min_freq={MIN_FREQ}, window={WINDOW_SIZE}")
    print(f"AWSGLD: T={T}, Burn_in={BURN_IN}, tau={TAU}, zeta={ZETA}, sigma2={SIGMA2_FIXED}")
    print(f"FDR Cutoff: {FDR_CUTOFF}")
    print(f"{'='*70}\n")

    # 1. 그래프 생성 (Phrase 기반)
    print("1. 그래프 생성...")
    graph = create_phrase_graph(DOCUMENT)
    if graph is None:
        print("그래프 생성 실패!")
        exit(1)
    print()

    # 2. Seed 선택 (truth keyphrase에서 구 단위 매칭)
    print("2. Seed 선택...")
    truth_idx = graph['truth_indices']
    truth_phrases = graph['truth_phrases']
    print(f"  Truth keyphrases: {len(truth_phrases)}개")
    print(f"  Dictionary 매칭: {len(truth_idx)}개")

    if len(truth_idx) < k:
        print(f"  매칭된 truth({len(truth_idx)}) < k({k}), TextRank fallback")
        d = 0.85
        n = graph['n']
        G = solve(graph['D'], graph['A'])
        B = np.eye(n) - d * G.T
        u_0 = solve(B, np.ones(n) * (1 - d))
        OBS_LABEL = list(np.argsort(u_0)[::-1][:k])
        ALPHA_FIXED = 0.5
    else:
        OBS_LABEL = list(np.random.choice(truth_idx, k, replace=False))
        ALPHA_FIXED = (len(truth_idx) - k) / len(truth_idx)

    print(f"  Alpha = {ALPHA_FIXED:.4f}")
    print(f"  선택된 Seed ({len(OBS_LABEL)}개):")
    for idx in OBS_LABEL:
        phrase = graph['dictionary'][idx, 0]
        ngram = len(phrase.split())
        print(f"    - [{idx}] {phrase} ({ngram}-gram)")
    print()

    # 3. AWSGLD 추출
    print("3. AWSGLD 추출...")
    result = run_awsgld_extraction(graph, OBS_LABEL, T, BURN_IN, alpha_fixed=ALPHA_FIXED)
    print()

    # 4. 통계
    print("4. 통계")
    print(f"  poster_pi_mn: min={np.min(result['poster_pi_mn']):.4f}, "
          f"max={np.max(result['poster_pi_mn']):.4f}, "
          f"mean={np.mean(result['poster_pi_mn']):.4f}")

    # 5. FDR cutoff & 결과
    cutoff = calculate_cutoff(result['poster_pi_mn'], FDR_CUTOFF, result['Y'])
    print(f"  FDR cutoff: {cutoff:.4f}\n")

    poster_md_adjust = force_obs_to_key2(result['Y'], result['poster_pi_mn'])
    identified_idx = np.where(poster_md_adjust >= cutoff)[0]
    identified_idx = identified_idx[np.argsort(poster_md_adjust[identified_idx])[::-1]]

    results_dict = {
        'document': DOCUMENT,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'settings': {
            'Method': 'AWSGLD_v2_phrase',
            'T': T, 'BURN_IN': BURN_IN,
            'TAU': TAU, 'ZETA': ZETA, 'SIGMA2': SIGMA2_FIXED,
            'MAX_NGRAM': MAX_NGRAM, 'MIN_FREQ': MIN_FREQ, 'WINDOW_SIZE': WINDOW_SIZE,
            'FDR_CUTOFF': FDR_CUTOFF,
            'k': k, 'alpha': ALPHA_FIXED,
        },
        'methods': {},
        'ngram_distribution': {},
    }

    # Seed 인덱스 set
    seed_set = set(OBS_LABEL)

    # BSS-AWSGLD
    print(f"{'='*70}")
    print(f"[BSS-AWSGLD] - {len(identified_idx)}개 키프레이즈")
    print("-" * 70)
    bss_list = []
    bss_ngram = {1: 0, 2: 0, 3: 0}
    for i, idx in enumerate(identified_idx[:TOP_K_RESULTS], 1):
        phrase = graph['dictionary'][idx, 0]
        score_adj = poster_md_adjust[idx]
        score_raw = result['poster_pi_mn'][idx]
        ngram = len(phrase.split())
        is_truth = idx in truth_idx
        is_seed = idx in seed_set
        bss_ngram[ngram] = bss_ngram.get(ngram, 0) + 1
        tag = '[T]' if is_truth else ''
        if is_seed:
            print(f"  {i:2d}. {phrase:40s} ({score_raw:.4f}*) [{ngram}g] {tag}")
        else:
            print(f"  {i:2d}. {phrase:40s} ({score_raw:.4f})  [{ngram}g] {tag}")
        bss_list.append({'rank': i, 'phrase': phrase, 'score': float(score_raw),
                         'ngram': ngram, 'is_seed': is_seed, 'is_truth': is_truth})
    if len(identified_idx) > TOP_K_RESULTS:
        print(f"  ... 외 {len(identified_idx) - TOP_K_RESULTS}개")
    results_dict['methods']['BSS-AWSGLD'] = bss_list
    results_dict['ngram_distribution']['BSS-AWSGLD'] = bss_ngram

    # TextRank
    u_0_adj = force_obs_to_key(result['Y'], result['u_0'], k)
    tr_idx = np.argsort(u_0_adj)[::-1][:len(identified_idx)]
    print(f"\n[TextRank] - {len(tr_idx)}개")
    print("-" * 70)
    tr_list = []
    for i, idx in enumerate(tr_idx[:TOP_K_RESULTS], 1):
        phrase = graph['dictionary'][idx, 0]
        score_adj = u_0_adj[idx]
        score_raw = result['u_0'][idx]
        ngram = len(phrase.split())
        is_truth = idx in truth_idx
        is_seed = idx in seed_set
        tag = '[T]' if is_truth else ''
        if is_seed:
            print(f"  {i:2d}. {phrase:40s} ({score_raw:.4f}*) [{ngram}g] {tag}")
        else:
            print(f"  {i:2d}. {phrase:40s} ({score_raw:.4f})  [{ngram}g] {tag}")
        tr_list.append({'rank': i, 'phrase': phrase, 'score': float(score_raw),
                        'ngram': ngram, 'is_seed': is_seed, 'is_truth': is_truth})
    results_dict['methods']['TextRank'] = tr_list

    # Semi-supervised
    bl_adj = force_obs_to_key(result['Y'], result['Base_Line'], k)
    semi_idx = np.argsort(bl_adj)[::-1][:len(identified_idx)]
    print(f"\n[Semi-supervised] - {len(semi_idx)}개")
    print("-" * 70)
    semi_list = []
    for i, idx in enumerate(semi_idx[:TOP_K_RESULTS], 1):
        phrase = graph['dictionary'][idx, 0]
        score_adj = bl_adj[idx]
        score_raw = result['Base_Line'][idx]
        ngram = len(phrase.split())
        is_truth = idx in truth_idx
        is_seed = idx in seed_set
        tag = '[T]' if is_truth else ''
        if is_seed:
            print(f"  {i:2d}. {phrase:40s} ({score_raw:.4f}*) [{ngram}g] {tag}")
        else:
            print(f"  {i:2d}. {phrase:40s} ({score_raw:.4f})  [{ngram}g] {tag}")
        semi_list.append({'rank': i, 'phrase': phrase, 'score': float(score_raw),
                          'ngram': ngram, 'is_seed': is_seed, 'is_truth': is_truth})
    results_dict['methods']['Semi-supervised'] = semi_list

    # N-gram 분포
    print(f"\n[N-gram 분포 — BSS-AWSGLD]")
    for ng in [1, 2, 3]:
        print(f"  {ng}-gram: {bss_ngram.get(ng, 0)}개")

    # 6. 저장
    if SAVE_RESULTS:
        print(f"\n결과 저장...")
        save_all_results(results_dict, OUTPUT_DIR, DOCUMENT, SAVE_FORMAT)

    print(f"\n{'='*70}")
    print("완료!")
    print(f"{'='*70}")
