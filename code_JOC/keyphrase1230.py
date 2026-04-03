"""
구(Phrase) 기반 BSS 키프레이즈 추출 - 결과 저장 기능 포함
- 단어가 아닌 n-gram 구(phrase)를 추출
- TextRank, Semi-supervised, BSS 방법 비교
- CSV, TXT, JSON 형식으로 결과 저장
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
# 데이터 경로 설정
DATA_DIR = "/home/jiyoon/3차/BSS-Keyphrase-Extraction-master/data_JOC"
DOCUMENT = "C-42.txt"

# FDR 관련 설정
FDR_CUTOFF = 0.1          # FDR cutoff (낮을수록 더 엄격, 0.1~0.3 권장)
TOP_K_RESULTS = 50        # 상위 몇 개만 출력할지

# MCMC 파라미터
T = 2000          # 총 반복 횟수
BURN_IN = 200     # Burn-in 기간

# N-gram 설정
MAX_NGRAM = 3     # 최대 n-gram 크기 (1, 2, 3-gram)
MIN_FREQ = 2      # 최소 빈도수
WINDOW_SIZE = 5   # Co-occurrence 윈도우 크기

# N-gram 가중치 설정
NGRAM_WEIGHT_PRESET = 'mild'  # 'none', 'mild', 'moderate', 'strong' 중 선택

NGRAM_WEIGHT_OPTIONS = {
    'none':     {1: 1.0,  2: 1.0,  3: 1.0},   # 가중치 없음
    'mild':     {1: 0.95, 2: 1.05, 3: 1.08},  # 약한 조정
    'moderate': {1: 0.85, 2: 1.10, 3: 1.15},  # 중간 조정
    'strong':   {1: 0.70, 2: 1.30, 3: 1.50},  # 강한 조정 (3-gram 위주)
}

NGRAM_WEIGHT = NGRAM_WEIGHT_OPTIONS[NGRAM_WEIGHT_PRESET]

# 출력 옵션
PREFER_LONGER_PHRASES = True   # True면 더 긴 구를 선호 (중복 제거)
MIN_NGRAM_OUTPUT = 1           # 출력할 최소 n-gram (2로 설정하면 1-gram 제외)
EXCLUDE_SUBSUMED = True        # 더 긴 구에 포함된 짧은 구 제외

# 결과 저장 설정
SAVE_RESULTS = True            # 결과 저장 여부
OUTPUT_DIR = "/home/jiyoon/3차/BSS-Keyphrase-Extraction-master/results"  # 결과 저장 경로
SAVE_FORMAT = ['csv', 'txt', 'json']   # 저장 형식

np.random.seed(12345)
k = 4  # seed 키프레이즈 수
grid = (np.arange(10, 43) - 5) / np.arange(10, 43)


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
    """Alpha likelihood 계산"""
    pi = inv_logit(Base_Line)
    pi = np.clip(pi, 1e-10, 1 - 1e-10)
    temp = (1 - alpha) * pi
    temp = np.clip(temp, 1e-10, 1 - 1e-10)
    return np.sum(Y * np.log(temp) + (1 - Y) * np.log(1 - temp))


def posterior_gibbstheta(Y, alpha, theta, u_0, B, sigma2):
    """theta의 posterior 계산 (given sigma^2)"""
    temp = (1 - alpha) * inv_logit(theta)
    temp = np.clip(temp, 1e-10, 1 - 1e-10)
    C = (B @ (theta - u_0)).T @ (B @ (theta - u_0))
    lglk = np.sum(Y * np.log(temp) + (1 - Y) * np.log(1 - temp)) - C / (2 * sigma2)
    return lglk


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
    poster_pi_mean[Y == 1] = 1
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


# ========== 관찰 레이블 자동 선택 ==========
def select_obs_labels(graph, k=4, method='textrank'):
    """관찰 레이블(seed keywords) 자동 선택"""
    n_minus = graph['n_minus']
    d = 0.85
    
    if method == 'textrank':
        try:
            G_minus = solve(graph['D_minus'], graph['A_minus'])
            B_minus = np.eye(n_minus) - d * G_minus.T
            u_0_minus = solve(B_minus, np.ones(n_minus) * (1 - d))
            top_indices = np.argsort(u_0_minus)[::-1][:k]
            return list(top_indices)
        except:
            method = 'frequency'
    
    if method == 'frequency':
        phrase_counts = graph['phrase_counts']
        dictionary = graph['dictionary_minus']
        
        scores = []
        for i, row in enumerate(dictionary):
            phrase = row[0]
            count = phrase_counts.get(phrase, 0)
            word_count = len(phrase.split())
            weighted_score = count * (1 + 0.3 * (word_count - 1))
            scores.append(weighted_score)
        
        top_indices = np.argsort(scores)[::-1][:k]
        return list(top_indices)
    
    if method == 'random':
        return list(np.random.choice(n_minus, min(k, n_minus), replace=False))
    
    return list(range(min(k, n_minus)))


# ========== MCMC 실행 함수 ==========
def gibbs_mh(Burn_in, T, ini, n, graph, Y, B, u_0, alpha_est, grid, verbose=True):
    """Gibbs and Metropolis-Hastings sampling"""
    d = 0.85
    theta_store = np.zeros((T, n))
    sigma2_store = np.zeros(T)
    alpha_store = np.zeros(T)
    accept = 0
    theta = ini.copy()
    
    start_time = time.time()
    print_interval = max(1, T // 20)
    
    for t in range(T):
        C = (B @ (theta - u_0)).T @ (B @ (theta - u_0))
        sigma2 = invgamma.rvs(n / 2 + 0.001, scale=C / 2 + 0.001)
        sigma2_store[t] = sigma2
        
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
        
        if verbose and (t + 1) % print_interval == 0:
            elapsed = time.time() - start_time
            progress = (t + 1) / T * 100
            eta_seconds = elapsed / (t + 1) * (T - t - 1)
            eta = timedelta(seconds=int(eta_seconds))
            print(f"\r  MCMC 진행률: {progress:.1f}% ({t+1}/{T}) | "
                  f"경과: {timedelta(seconds=int(elapsed))} | "
                  f"남은 시간: {eta} | "
                  f"Accept: {accept/(t+1):.3f}", end='', flush=True)
    
    if verbose:
        print()
    
    total_time = time.time() - start_time
    print(f"  MCMC 완료! 소요 시간: {timedelta(seconds=int(total_time))}, Accept rate: {accept / T:.4f}")
    
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
    """키프레이즈 추출 실행"""
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
    alpha_est = alpha_find(u_0_minus, Y_minus, grid)
    
    print(f"  MCMC 시작 (T={T}, Burn_in={burn_in})...")
    test_chain = gibbs_mh(burn_in, T, ini, n_minus, graph, Y_minus, B_minus, u_0_minus, alpha_est, grid)
    
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


# ========== 결과 저장 함수들 ==========
def save_results_to_csv(results_dict, filepath):
    """결과를 CSV 파일로 저장"""
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # 메타 정보
        writer.writerow(['# 키프레이즈 추출 결과'])
        writer.writerow(['Document', results_dict['document']])
        writer.writerow(['Timestamp', results_dict['timestamp']])
        writer.writerow([])
        
        # 설정 정보
        writer.writerow(['# 설정'])
        for key, value in results_dict['settings'].items():
            writer.writerow([key, value])
        writer.writerow([])
        
        # 각 방법별 결과
        for method_name, keyphrases in results_dict['methods'].items():
            writer.writerow([f'# {method_name}'])
            writer.writerow(['Rank', 'Keyphrase', 'Weighted_Score', 'Original_Score', 'N-gram'])
            
            for item in keyphrases:
                writer.writerow([
                    item['rank'],
                    item['phrase'],
                    f"{item['weighted_score']:.6f}",
                    f"{item['original_score']:.6f}",
                    item['ngram']
                ])
            writer.writerow([])
        
        # N-gram 분포
        writer.writerow(['# N-gram 분포'])
        writer.writerow(['Method', '1-gram', '2-gram', '3-gram'])
        for method, dist in results_dict['ngram_distribution'].items():
            writer.writerow([method, dist.get(1, 0), dist.get(2, 0), dist.get(3, 0)])


def save_results_to_txt(results_dict, filepath):
    """결과를 TXT 파일로 저장"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("키프레이즈 추출 결과\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"문서: {results_dict['document']}\n")
        f.write(f"생성 시간: {results_dict['timestamp']}\n\n")
        
        f.write("[설정]\n")
        for key, value in results_dict['settings'].items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        # 각 방법별 결과
        for method_name, keyphrases in results_dict['methods'].items():
            f.write("="*70 + "\n")
            f.write(f"[{method_name}] - {len(keyphrases)}개 키프레이즈\n")
            f.write("-"*70 + "\n")
            
            for item in keyphrases:
                f.write(f"  {item['rank']:3d}. {item['phrase']:40s} ")
                f.write(f"(가중:{item['weighted_score']:.4f}, 원본:{item['original_score']:.4f}) ")
                f.write(f"[{item['ngram']}-gram]\n")
            f.write("\n")
        
        # N-gram 분포
        f.write("="*70 + "\n")
        f.write("[N-gram 분포]\n")
        f.write("-"*70 + "\n")
        for method, dist in results_dict['ngram_distribution'].items():
            total = sum(dist.values())
            f.write(f"  {method}:\n")
            for n in [1, 2, 3]:
                count = dist.get(n, 0)
                pct = count / total * 100 if total > 0 else 0
                f.write(f"    {n}-gram: {count:3d}개 ({pct:.1f}%)\n")


def save_results_to_json(results_dict, filepath):
    """결과를 JSON 파일로 저장"""
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {str(k): convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    serializable_dict = convert_to_serializable(results_dict)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(serializable_dict, f, ensure_ascii=False, indent=2)


def save_all_results(results_dict, output_dir, document_name, formats=['csv', 'txt', 'json']):
    """모든 형식으로 결과 저장"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(document_name)[0]
    
    saved_files = []
    
    if 'csv' in formats:
        csv_path = os.path.join(output_dir, f"{base_name}_results_{timestamp}.csv")
        save_results_to_csv(results_dict, csv_path)
        saved_files.append(csv_path)
        print(f"  CSV 저장: {csv_path}")
    
    if 'txt' in formats:
        txt_path = os.path.join(output_dir, f"{base_name}_results_{timestamp}.txt")
        save_results_to_txt(results_dict, txt_path)
        saved_files.append(txt_path)
        print(f"  TXT 저장: {txt_path}")
    
    if 'json' in formats:
        json_path = os.path.join(output_dir, f"{base_name}_results_{timestamp}.json")
        save_results_to_json(results_dict, json_path)
        saved_files.append(json_path)
        print(f"  JSON 저장: {json_path}")
    
    return saved_files


# ========== 결과 출력 및 저장 함수 ==========
def print_and_save_results(result, cutoff, fdr_cutoff=FDR_CUTOFF, top_k=TOP_K_RESULTS):
    """결과 출력 및 저장"""
    print(f"\n{'='*70}")
    print(f"추출 결과")
    print(f"{'='*70}")
    print(f"FDR Cutoff: {cutoff:.4f}")
    print(f"N-gram 가중치 ({NGRAM_WEIGHT_PRESET}): 1-gram={NGRAM_WEIGHT[1]}, 2-gram={NGRAM_WEIGHT[2]}, 3-gram={NGRAM_WEIGHT[3]}")
    print()
    
    # 결과 저장용 딕셔너리
    results_dict = {
        'document': DOCUMENT,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'settings': {
            'FDR_CUTOFF': fdr_cutoff,
            'MCMC_T': T,
            'MCMC_BURN_IN': BURN_IN,
            'MAX_NGRAM': MAX_NGRAM,
            'MIN_FREQ': MIN_FREQ,
            'WINDOW_SIZE': WINDOW_SIZE,
            'NGRAM_WEIGHT_PRESET': NGRAM_WEIGHT_PRESET,
            'NGRAM_WEIGHT_1gram': NGRAM_WEIGHT[1],
            'NGRAM_WEIGHT_2gram': NGRAM_WEIGHT[2],
            'NGRAM_WEIGHT_3gram': NGRAM_WEIGHT[3],
            'EXCLUDE_SUBSUMED': EXCLUDE_SUBSUMED,
            'MIN_NGRAM_OUTPUT': MIN_NGRAM_OUTPUT
        },
        'methods': {},
        'ngram_distribution': {}
    }
    
    # ===== BSS 방법 =====
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
    
    print(f"[BSS 방법] - 총 {len(sorted_phrases)}개 키프레이즈 (상위 {top_k}개 표시)")
    print("-" * 70)
    
    bss_results = []
    bss_ngram_dist = {1: 0, 2: 0, 3: 0}
    
    for i, (idx, score) in enumerate(sorted_phrases, 1):
        phrase = result['dictionary_minus'][idx, 0]
        word_count = len(phrase.split())
        original_score = result['poster_pi_mn'][idx]
        
        bss_ngram_dist[word_count] = bss_ngram_dist.get(word_count, 0) + 1
        
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
    
    results_dict['methods']['BSS'] = bss_results
    results_dict['ngram_distribution']['BSS'] = bss_ngram_dist
    
    # ===== TextRank 방법 =====
    u_0_adjusted = force_obs_to_key(result['Y_minus'], result['u_0_minus'], k)
    weighted_textrank = apply_ngram_weights(u_0_adjusted, result['dictionary_minus'])
    
    textrank_scores = [(idx, weighted_textrank[idx]) for idx in range(len(weighted_textrank))]
    
    if MIN_NGRAM_OUTPUT > 1:
        textrank_scores = filter_by_ngram_length(textrank_scores, result['dictionary_minus'], MIN_NGRAM_OUTPUT)
    if EXCLUDE_SUBSUMED:
        textrank_scores = filter_subsumed_phrases(textrank_scores, result['dictionary_minus'])
    
    textrank_sorted = sorted(textrank_scores, key=lambda x: x[1], reverse=True)
    
    print(f"\n[TextRank 방법] - 총 {len(textrank_sorted)}개 (상위 {top_k}개 표시)")
    print("-" * 70)
    
    textrank_results = []
    textrank_ngram_dist = {1: 0, 2: 0, 3: 0}
    
    for i, (idx, score) in enumerate(textrank_sorted, 1):
        phrase = result['dictionary_minus'][idx, 0]
        word_count = len(phrase.split())
        original_score = u_0_adjusted[idx]
        
        textrank_ngram_dist[word_count] = textrank_ngram_dist.get(word_count, 0) + 1
        
        textrank_results.append({
            'rank': i,
            'phrase': phrase,
            'weighted_score': float(score),
            'original_score': float(original_score),
            'ngram': word_count
        })
        
        if i <= top_k:
            print(f"  {i:2d}. {phrase:40s} (가중:{score:.4f}, 원본:{original_score:.4f}) [{word_count}-gram]")
    
    results_dict['methods']['TextRank'] = textrank_results
    results_dict['ngram_distribution']['TextRank'] = textrank_ngram_dist
    
    # ===== Semi-supervised 방법 =====
    weighted_semi = apply_ngram_weights(result['Base_Line_minus'], result['dictionary_minus'])
    
    semi_scores = [(idx, weighted_semi[idx]) for idx in range(len(weighted_semi))]
    
    if MIN_NGRAM_OUTPUT > 1:
        semi_scores = filter_by_ngram_length(semi_scores, result['dictionary_minus'], MIN_NGRAM_OUTPUT)
    if EXCLUDE_SUBSUMED:
        semi_scores = filter_subsumed_phrases(semi_scores, result['dictionary_minus'])
    
    semi_sorted = sorted(semi_scores, key=lambda x: x[1], reverse=True)
    
    print(f"\n[Semi-supervised 방법] - 총 {len(semi_sorted)}개 (상위 {top_k}개 표시)")
    print("-" * 70)
    
    semi_results = []
    semi_ngram_dist = {1: 0, 2: 0, 3: 0}
    
    for i, (idx, score) in enumerate(semi_sorted, 1):
        phrase = result['dictionary_minus'][idx, 0]
        word_count = len(phrase.split())
        original_score = result['Base_Line_minus'][idx]
        
        semi_ngram_dist[word_count] = semi_ngram_dist.get(word_count, 0) + 1
        
        semi_results.append({
            'rank': i,
            'phrase': phrase,
            'weighted_score': float(score),
            'original_score': float(original_score),
            'ngram': word_count
        })
        
        if i <= top_k:
            print(f"  {i:2d}. {phrase:40s} (가중:{score:.4f}, 원본:{original_score:.4f}) [{word_count}-gram]")
    
    results_dict['methods']['Semi-supervised'] = semi_results
    results_dict['ngram_distribution']['Semi-supervised'] = semi_ngram_dist
    
    # N-gram 분포 출력
    print(f"\n{'='*70}")
    print("N-gram 분포")
    print("="*70)
    for method, dist in results_dict['ngram_distribution'].items():
        total = sum(dist.values())
        print(f"  [{method}]")
        for n in [1, 2, 3]:
            count = dist.get(n, 0)
            pct = count / total * 100 if total > 0 else 0
            print(f"    {n}-gram: {count:3d}개 ({pct:.1f}%)")
    
    return results_dict


# ========== 메인 실행 ==========
if __name__ == "__main__":
    print(f"{'='*70}")
    print(f"구(Phrase) 기반 BSS 키프레이즈 추출")
    print(f"{'='*70}")
    print(f"문서: {DOCUMENT}")
    print(f"MCMC 설정: T={T}, Burn_in={BURN_IN}")
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

    print(f"✅ 그래프 생성 완료\n")

    # 2. 관찰 레이블 자동 선택
    print("2. 관찰 레이블(Seed) 선택 중...")
    OBS_LABEL = select_obs_labels(graph, k=k, method='textrank')
    print(f"  선택된 Seed 구 (총 {len(OBS_LABEL)}개):")
    for idx in OBS_LABEL:
        phrase = graph['dictionary_minus'][idx, 0]
        word_count = len(phrase.split())
        print(f"    - [{idx}] {phrase} ({word_count}-gram)")
    print()

    # 3. MCMC 실행
    print("3. 키프레이즈 추출 중...")
    result = run_keyphrase_extraction(graph, OBS_LABEL, T, BURN_IN)
    print("✅ 추출 완료\n")

    # 4. 통계 정보
    print("4. 통계 정보")
    print("="*50)
    print(f"  poster_pi_mn 통계:")
    print(f"    Min: {np.min(result['poster_pi_mn']):.6f}")
    print(f"    Max: {np.max(result['poster_pi_mn']):.6f}")
    print(f"    Mean: {np.mean(result['poster_pi_mn']):.6f}")
    print(f"    Std: {np.std(result['poster_pi_mn']):.6f}")
    print(f"  Accept rate: {result['accept_rate']:.4f}")

    poster_adjusted = force_obs_to_key2(result['Y_minus'], result['poster_pi_mn'])
    cutoffs = np.unique(np.sort(poster_adjusted))
    FDRs = vec_FDR_cal(cutoffs, poster_adjusted)
    print(f"\n  FDR 분포:")
    print(f"    Min FDR: {np.min(FDRs):.6f}")
    print(f"    Max FDR: {np.max(FDRs):.6f}")
    print(f"    FDR < {FDR_CUTOFF}인 cutoff 개수: {np.sum(FDRs < FDR_CUTOFF)}")
    print("="*50)

    # 5. 결과 출력 및 저장
    cutoff = calculate_cutoff(result['poster_pi_mn'], FDR_CUTOFF, result['Y_minus'])
    results_dict = print_and_save_results(result, cutoff)

    # 6. 결과 저장
    if SAVE_RESULTS:
        print(f"\n{'='*70}")
        print("결과 저장 중...")
        saved_files = save_all_results(results_dict, OUTPUT_DIR, DOCUMENT, SAVE_FORMAT)
        print(f"✅ {len(saved_files)}개 파일 저장 완료!")

    print(f"\n{'='*70}")
    print("✅ 모든 작업 완료!")
    print(f"{'='*70}")