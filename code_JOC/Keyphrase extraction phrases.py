"""
구(Phrase) 기반 BSS 키프레이즈 추출 - 통합 버전
- 단어가 아닌 n-gram 구(phrase)를 추출
- TextRank, Semi-supervised, BSS 방법 비교
"""

import numpy as np
import os
import re
from scipy.linalg import solve
from scipy.stats import invgamma
from collections import Counter
import time
from datetime import timedelta

# ========== 설정 ==========
# 아래 경로를 실제 데이터 경로로 수정하세요
DATA_DIR = "/home/jiyoon/3차/BSS-Keyphrase-Extraction-master/data_JOC"
DOCUMENT = "C-42.txt"
FDR_CUTOFF = 0.3

# MCMC 파라미터
T = 2000          # 총 반복 횟수
BURN_IN = 200     # Burn-in 기간

# N-gram 설정
MAX_NGRAM = 3     # 최대 n-gram 크기 (1, 2, 3-gram)
MIN_FREQ = 2      # 최소 빈도수
WINDOW_SIZE = 5   # Co-occurrence 윈도우 크기

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
    """
    주어진 cutoff에 대한 FDR 계산
    
    FDR = E[# of false positives] / # of selected
        = sum(1 - p_i) / count(p_i >= cutoff)
        
    여기서 p_i는 키프레이즈일 확률
    """
    # cutoff 이상인 것들 선택
    set_vals = poster_md_adjust[poster_md_adjust >= cutoff]
    if len(set_vals) == 0:
        return 1.0  # 아무것도 선택 안 되면 FDR = 1
    
    # FDR = 평균 (1 - 확률) = 평균 false positive 비율
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
        'eg', 'ie', 'etc', 'vs', 'via'
    }


def create_fcm_phrases(text, max_ngram=3, window=5, min_freq=2):
    """
    구(phrase) 기반 co-occurrence matrix 생성
    
    Parameters:
    - text: 입력 텍스트
    - max_ngram: 최대 n-gram 크기 (기본값: 3)
    - window: co-occurrence 윈도우 크기
    - min_freq: 최소 빈도수
    
    Returns:
    - fcm: co-occurrence matrix
    - valid_phrases: 유효한 구 리스트
    - phrase_to_idx: 구 → 인덱스 매핑
    - phrase_counts: 구별 빈도수
    """
    # 전처리
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 문장 분리
    sentences = re.split(r'[.!?]', text)
    
    stopwords = get_stopwords()
    
    all_phrases_per_sent = []
    phrase_counts = Counter()
    
    for sent in sentences:
        words = sent.split()
        # 길이가 2 이상인 단어만 유지
        words_clean = [w for w in words if len(w) > 1]
        
        if len(words_clean) == 0:
            continue
        
        # 1-gram, 2-gram, 3-gram 생성
        sent_phrases = []
        for n in range(1, max_ngram + 1):
            for i in range(len(words_clean) - n + 1):
                phrase_words = words_clean[i:i+n]
                
                # n-gram 유효성 검사
                # 1-gram: 불용어 제외
                if n == 1 and phrase_words[0] in stopwords:
                    continue
                # 2-gram 이상: 첫/끝 단어가 불용어면 제외
                if n > 1 and (phrase_words[0] in stopwords or phrase_words[-1] in stopwords):
                    continue
                
                # 모든 단어가 불용어인 경우 제외
                if all(w in stopwords for w in phrase_words):
                    continue
                
                phrase = ' '.join(phrase_words)
                sent_phrases.append(phrase)
                phrase_counts[phrase] += 1
        
        all_phrases_per_sent.append(sent_phrases)
    
    # 빈도 기반 필터링
    valid_phrases = [p for p, c in phrase_counts.items() if c >= min_freq]
    
    # 숫자만 있는 구 제거
    valid_phrases = [p for p in valid_phrases if not p.replace(' ', '').isdigit()]
    
    # 너무 짧은 구 제거 (1글자 또는 2글자 단일 단어)
    valid_phrases = [p for p in valid_phrases if len(p) > 2 or ' ' in p]
    
    n = len(valid_phrases)
    if n == 0:
        return np.array([]), [], {}, phrase_counts
    
    phrase_to_idx = {p: i for i, p in enumerate(valid_phrases)}
    
    # Co-occurrence matrix 생성
    fcm = np.zeros((n, n))
    
    for sent_phrases in all_phrases_per_sent:
        # 해당 문장에서 유효한 구만 필터링
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
    
    # 구 기반 FCM 생성
    fcm, unique_phrases, phrase_to_idx, phrase_counts = create_fcm_phrases(
        text, max_ngram=MAX_NGRAM, window=WINDOW_SIZE, min_freq=MIN_FREQ
    )
    n = len(unique_phrases)
    
    if n == 0:
        print("유효한 구가 없습니다.")
        return None
    
    print(f"  총 {n}개의 구(phrase) 생성됨")
    
    # N-gram 분포 출력
    ngram_dist = {1: 0, 2: 0, 3: 0}
    for p in unique_phrases:
        word_count = len(p.split())
        if word_count in ngram_dist:
            ngram_dist[word_count] += 1
    print(f"  N-gram 분포: 1-gram={ngram_dist[1]}, 2-gram={ngram_dist[2]}, 3-gram={ngram_dist[3]}")
    
    d = 0.85
    A = fcm.copy()
    np.fill_diagonal(A, 0)
    
    # D 행렬 생성 (0인 행 처리)
    row_sums = A.sum(axis=1)
    row_sums[row_sums == 0] = 1  # 0으로 나누기 방지
    D = np.diag(row_sums)
    
    # 필터링: 연결이 있는 노드만 유지
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
    
    # dictionary_minus 생성
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
    """
    관찰 레이블(seed keywords) 자동 선택
    
    Parameters:
    - graph: 그래프 딕셔너리
    - k: 선택할 레이블 수
    - method: 'textrank', 'frequency', 'random', 'hybrid' 중 선택
    
    Returns:
    - obs_label: 선택된 인덱스 리스트
    """
    n_minus = graph['n_minus']
    d = 0.85
    
    if method == 'textrank':
        # TextRank 점수로 상위 k개 선택
        try:
            G_minus = solve(graph['D_minus'], graph['A_minus'])
            B_minus = np.eye(n_minus) - d * G_minus.T
            u_0_minus = solve(B_minus, np.ones(n_minus) * (1 - d))
            top_indices = np.argsort(u_0_minus)[::-1][:k]
            return list(top_indices)
        except:
            method = 'frequency'
    
    if method == 'frequency':
        # 빈도수 기반 선택
        phrase_counts = graph['phrase_counts']
        dictionary = graph['dictionary_minus']
        
        scores = []
        for i, row in enumerate(dictionary):
            phrase = row[0]
            count = phrase_counts.get(phrase, 0)
            # 2-gram, 3-gram에 가중치 부여
            word_count = len(phrase.split())
            weighted_score = count * (1 + 0.3 * (word_count - 1))
            scores.append(weighted_score)
        
        top_indices = np.argsort(scores)[::-1][:k]
        return list(top_indices)
    
    if method == 'hybrid':
        # TextRank + Frequency 조합
        try:
            G_minus = solve(graph['D_minus'], graph['A_minus'])
            B_minus = np.eye(n_minus) - d * G_minus.T
            u_0_minus = solve(B_minus, np.ones(n_minus) * (1 - d))
            
            phrase_counts = graph['phrase_counts']
            dictionary = graph['dictionary_minus']
            
            scores = []
            for i, row in enumerate(dictionary):
                phrase = row[0]
                count = phrase_counts.get(phrase, 0)
                word_count = len(phrase.split())
                # TextRank 점수 + 빈도 점수 + n-gram 보너스
                hybrid_score = u_0_minus[i] + 0.1 * np.log1p(count) + 0.2 * (word_count - 1)
                scores.append(hybrid_score)
            
            top_indices = np.argsort(scores)[::-1][:k]
            return list(top_indices)
        except:
            return list(range(min(k, n_minus)))
    
    if method == 'random':
        return list(np.random.choice(n_minus, min(k, n_minus), replace=False))
    
    return list(range(min(k, n_minus)))


# ========== MCMC 실행 함수 ==========
def gibbs_mh(Burn_in, T, ini, n, graph, Y, B, u_0, alpha_est, grid, verbose=True):
    """
    Gibbs and Metropolis-Hastings sampling
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
        # Sample from sigma^2 (Inverse Gamma)
        C = (B @ (theta - u_0)).T @ (B @ (theta - u_0))
        sigma2 = invgamma.rvs(n / 2 + 0.001, scale=C / 2 + 0.001)
        sigma2_store[t] = sigma2
        
        # Sample from theta (Metropolis-Hastings)
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
        
        # MH acceptance
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
            print(f"\r  MCMC 진행률: {progress:.1f}% ({t+1}/{T}) | "
                  f"경과: {timedelta(seconds=int(elapsed))} | "
                  f"남은 시간: {eta} | "
                  f"Accept: {accept/(t+1):.3f}", end='', flush=True)
    
    if verbose:
        print()
    
    total_time = time.time() - start_time
    print(f"  MCMC 완료! 소요 시간: {timedelta(seconds=int(total_time))}, Accept rate: {accept / T:.4f}")
    
    # 결과 계산
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
    
    # 그래프 행렬 계산
    G_minus = solve(graph['D_minus'], graph['A_minus'])
    B_minus = np.eye(n_minus) - d * G_minus.T
    u_0_minus = solve(B_minus, np.ones(n_minus) * (1 - d))
    
    w_minus = np.diag(1.0 / np.sqrt(np.diag(graph['D_minus'])))
    B_star_minus = np.eye(n_minus) - d * w_minus @ graph['A_minus'] @ w_minus
    
    # 관찰 레이블 설정
    Y_minus = np.zeros(n_minus)
    for idx in obs_label:
        if idx < n_minus:
            Y_minus[idx] = 1
    
    Base_Line_minus = solve(B_star_minus, Y_minus)
    
    # MCMC 초기값
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
    """
    FDR cutoff 계산
    
    FDR < c를 만족하는 가장 낮은 cutoff를 찾음
    (즉, 가장 많은 키프레이즈를 추출하면서도 FDR 조건 만족)
    """
    poster_md_adjust = force_obs_to_key2(Y, poster_pi_mn)
    
    # cutoff 후보들을 오름차순으로 정렬 (낮은 것부터)
    cutoffs = np.unique(np.sort(poster_md_adjust))
    
    # 각 cutoff에 대한 FDR 계산
    FDRs = vec_FDR_cal(cutoffs, poster_md_adjust)
    
    # 디버깅: FDR 분포 확인
    print(f"\n  [Cutoff 계산 디버깅]")
    print(f"    Cutoff 범위: {cutoffs.min():.4f} ~ {cutoffs.max():.4f}")
    print(f"    FDR 범위: {FDRs.min():.4f} ~ {FDRs.max():.4f}")
    
    # FDR < c를 만족하는 cutoff 찾기
    valid_mask = FDRs <= c
    
    if np.any(valid_mask):
        # FDR 조건을 만족하는 cutoff 중 가장 낮은 것 선택
        # (가장 많은 키프레이즈 추출)
        valid_cutoffs = cutoffs[valid_mask]
        selected_cutoff = valid_cutoffs.min()
        
        # 선택된 cutoff의 FDR 확인
        selected_fdr = FDRs[cutoffs == selected_cutoff][0]
        print(f"    선택된 Cutoff: {selected_cutoff:.4f} (FDR: {selected_fdr:.4f})")
        
        return selected_cutoff
    else:
        # FDR 조건을 만족하는 게 없으면 가장 낮은 FDR의 cutoff 반환
        min_fdr_idx = np.argmin(FDRs)
        selected_cutoff = cutoffs[min_fdr_idx]
        print(f"    FDR 조건 불만족, 최소 FDR cutoff 사용: {selected_cutoff:.4f} (FDR: {FDRs[min_fdr_idx]:.4f})")
        return selected_cutoff


# ========== 결과 출력 함수 ==========
def print_results(result, cutoff, fdr_cutoff=FDR_CUTOFF):
    """결과 출력"""
    print(f"\n{'='*70}")
    print(f"추출 결과")
    print(f"{'='*70}")
    print(f"FDR Cutoff: {cutoff:.4f}\n")
    
    # BSS 방법 - cutoff 이상인 것 추출 (>= 사용)
    identified_idx = np.where(result['poster_pi_mn'] >= cutoff)[0]
    print(f"[BSS 방법] - {len(identified_idx)}개 키프레이즈")
    print("-" * 70)
    
    # 점수 기준으로 정렬
    sorted_idx = sorted(identified_idx, key=lambda x: result['poster_pi_mn'][x], reverse=True)
    
    for i, idx in enumerate(sorted_idx[:30], 1):
        phrase = result['dictionary_minus'][idx, 0]
        score = result['poster_pi_mn'][idx]
        word_count = len(phrase.split())
        ngram_type = f"{word_count}-gram"
        print(f"  {i:2d}. {phrase:45s} ({score:.4f}) [{ngram_type}]")
    
    if len(sorted_idx) > 30:
        print(f"  ... 외 {len(sorted_idx) - 30}개 더")
    
    # TextRank 방법
    if len(identified_idx) > 0:
        u_0_adjusted = force_obs_to_key(result['Y_minus'], result['u_0_minus'], k)
        textrank_idx = np.argsort(u_0_adjusted)[::-1][:len(identified_idx)]
        
        print(f"\n[TextRank 방법] - {len(textrank_idx)}개")
        print("-" * 70)
        for i, idx in enumerate(textrank_idx[:30], 1):
            phrase = result['dictionary_minus'][idx, 0]
            score = u_0_adjusted[idx]
            word_count = len(phrase.split())
            ngram_type = f"{word_count}-gram"
            print(f"  {i:2d}. {phrase:45s} ({score:.4f}) [{ngram_type}]")
    
    # Semi-supervised 방법
    if len(identified_idx) > 0:
        semi_idx = np.argsort(result['Base_Line_minus'])[::-1][:len(identified_idx)]
        
        print(f"\n[Semi-supervised 방법] - {len(semi_idx)}개")
        print("-" * 70)
        for i, idx in enumerate(semi_idx[:30], 1):
            phrase = result['dictionary_minus'][idx, 0]
            score = result['Base_Line_minus'][idx]
            word_count = len(phrase.split())
            ngram_type = f"{word_count}-gram"
            print(f"  {i:2d}. {phrase:45s} ({score:.4f}) [{ngram_type}]")
    
    # N-gram 분포 통계
    print(f"\n{'='*70}")
    print("N-gram 분포 (BSS 결과)")
    print("="*70)
    ngram_counts = {1: 0, 2: 0, 3: 0}
    for idx in identified_idx:
        phrase = result['dictionary_minus'][idx, 0]
        n = len(phrase.split())
        if n in ngram_counts:
            ngram_counts[n] += 1
        elif n > 3:
            ngram_counts[3] += 1  # 3-gram 이상은 3-gram으로 분류
    
    total = sum(ngram_counts.values())
    for n, count in ngram_counts.items():
        pct = count / total * 100 if total > 0 else 0
        print(f"  {n}-gram: {count:3d}개 ({pct:.1f}%)")


# ========== 메인 실행 ==========
if __name__ == "__main__":
    print(f"{'='*70}")
    print(f"구(Phrase) 기반 BSS 키프레이즈 추출")
    print(f"{'='*70}")
    print(f"문서: {DOCUMENT}")
    print(f"MCMC 설정: T={T}, Burn_in={BURN_IN}")
    print(f"N-gram 설정: max_ngram={MAX_NGRAM}, min_freq={MIN_FREQ}, window={WINDOW_SIZE}")
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

    # 4. 디버깅 정보
    print("4. 통계 정보")
    print("="*50)
    print(f"  poster_pi_mn 통계:")
    print(f"    Min: {np.min(result['poster_pi_mn']):.6f}")
    print(f"    Max: {np.max(result['poster_pi_mn']):.6f}")
    print(f"    Mean: {np.mean(result['poster_pi_mn']):.6f}")
    print(f"    Std: {np.std(result['poster_pi_mn']):.6f}")
    print(f"  Accept rate: {result['accept_rate']:.4f}")

    # FDR 분포 확인
    poster_adjusted = force_obs_to_key2(result['Y_minus'], result['poster_pi_mn'])
    cutoffs = np.unique(np.sort(poster_adjusted)[::-1])
    FDRs = vec_FDR_cal(cutoffs, poster_adjusted)
    print(f"\n  FDR 분포:")
    print(f"    Min FDR: {np.min(FDRs):.6f}")
    print(f"    Max FDR: {np.max(FDRs):.6f}")
    print(f"    FDR < {FDR_CUTOFF}인 cutoff 개수: {np.sum(FDRs < FDR_CUTOFF)}")
    print("="*50)

    # 5. 결과 출력
    cutoff = calculate_cutoff(result['poster_pi_mn'], FDR_CUTOFF, result['Y_minus'])
    print_results(result, cutoff)

    print(f"\n{'='*70}")
    print("✅ 완료!")
    print(f"{'='*70}")