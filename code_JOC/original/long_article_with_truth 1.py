"""
Long Article 1개만 처리 - 독립 실행 버전
"""

import numpy as np
import os
import re
from scipy.linalg import solve
from keyphrase_functions import (
    base_to_start, gibbs_mh, force_obs_to_key, force_obs_to_key2,
    alpha_find, vec_FDR_cal, create_fcm
)

# ========== 설정 ==========
DATA_DIR = "/home/jiyoon/3차/BSS-Keyphrase-Extraction-master/data_JOC"
DOCUMENT = "C-42.txt"
OBS_LABEL = [2, 18, 35, 75]
FDR_CUTOFF = 0.3

# MCMC 파라미터 (여기서 직접 제어!)
T = 200         
BURN_IN = 20     

np.random.seed(12345)
k = 4
grid = np.linspace(0.05, 0.90, 33)

print(f"{'='*60}")
print(f"문서 처리: {DOCUMENT}")
print(f"MCMC 설정: T={T}, Burn_in={BURN_IN}")
print(f"{'='*60}\n")

# ========== 그래프 생성 함수 ==========
def create_graph(article_name):
    """그래프 생성 - 구(phrase) 기반"""
    file_path = os.path.join(DATA_DIR, article_name)
    
    if not os.path.exists(file_path):
        print(f"파일 없음: {file_path}")
        return None
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read().lower()
    
    text = re.sub(r'[/=]', ' ', text)
    
    # 구 기반 FCM 생성 (수정된 함수 사용)
    fcm, unique_phrases, phrase_to_idx, phrase_counts = create_fcm_phrases(text)
    n = len(unique_phrases)
    
    if n == 0:
        print("유효한 구가 없습니다.")
        return None
    
    d = 0.85
    A = fcm.copy()
    np.fill_diagonal(A, 0)
    
    # D 행렬 생성 (0인 행 처리)
    row_sums = A.sum(axis=1)
    row_sums[row_sums == 0] = 1  # 0으로 나누기 방지
    D = np.diag(row_sums)
    
    # 필터링: 연결이 있는 노드만 유지
    keep_mask = A.sum(axis=1) >= 1
    keep_indices = np.where(keep_mask)[0]
    
    if len(keep_indices) == 0:
        print("필터링 후 유효한 노드가 없습니다.")
        return None
    
    A_filtered = A[np.ix_(keep_indices, keep_indices)]
    row_sums_filtered = A_filtered.sum(axis=1)
    row_sums_filtered[row_sums_filtered == 0] = 1
    D_filtered = np.diag(row_sums_filtered)
    
    return {
        'n': n,
        'A': A,
        'D': D,
        'A_minus': A_filtered,
        'D_minus': D_filtered,
        'n_minus': len(keep_indices),
        'dictionary_minus': np.array([[unique_phrases[i], i] for i in keep_indices], dtype=object),
        'keep_indices': keep_indices,
        'phrase_counts': phrase_counts
    }

# ========== MCMC 실행 함수 ==========
def run_keyphrase_extraction(graph, obs_label, T, burn_in):
    """키프레이즈 추출"""
    n_minus = graph['n_minus']
    d = 0.85
    
    # 그래프 계산
    G_minus = solve(graph['D_minus'], graph['A_minus'])
    B_minus = np.eye(n_minus) - d * G_minus.T
    u_0_minus = solve(B_minus, np.ones(n_minus) * (1 - d))
    w_minus = np.diag(1.0 / np.sqrt(np.diag(graph['D_minus'])))
    B_star_minus = np.eye(n_minus) - d * w_minus @ graph['A_minus'] @ w_minus
    
    # 관찰 레이블
    Y_minus = np.zeros(n_minus)
    Y_minus[obs_label] = 1
    Base_Line_minus = solve(B_star_minus, Y_minus)
    
    # MCMC
    ini = base_to_start(u_0_minus)
    alpha_est = alpha_find(u_0_minus, Y_minus, grid)
    
    print(f"MCMC 시작 (T={T}, Burn_in={burn_in})...")
    test_chain = gibbs_mh(burn_in, T, ini, n_minus, graph, Y_minus, B_minus, u_0_minus, alpha_est, grid)
    
    return {
        'poster_pi_mn': test_chain['poster_pi_mn'],
        'dictionary_minus': graph['dictionary_minus'],
        'u_0_minus': u_0_minus,
        'Base_Line_minus': Base_Line_minus,
        'Y_minus': Y_minus
    }

# ========== FDR Cutoff 함수 ==========
def calculate_cutoff(poster_pi_mn, c, Y):
    """FDR cutoff 계산"""
    poster_md_adjust = force_obs_to_key2(Y, poster_pi_mn)
    cutoffs = np.unique(np.sort(poster_md_adjust)[::-1])
    FDRs = vec_FDR_cal(cutoffs, poster_md_adjust)
    
    if np.any(FDRs < c):
        index = np.max(np.where(FDRs < c)[0])
        return cutoffs[index]
    return cutoffs[0] if len(cutoffs) > 0 else 1.0

# ========== 실행 ==========
print("1. 그래프 생성 중...")
graph = create_graph(DOCUMENT)

if graph is None:
    exit(1)

print(f"✅ 생성 완료 (노드: {graph['n_minus']})\n")

print("2. MCMC 실행 중...")
result = run_keyphrase_extraction(graph, OBS_LABEL, T, BURN_IN)
print("✅ MCMC 완료\n")

print("3. 키워드 추출 중...")
cutoff = calculate_cutoff(result['poster_pi_mn'], FDR_CUTOFF, result['Y_minus'])

# ===== 디버깅 코드 추가 =====
print("\n" + "="*50)
print("[디버깅 정보]")
print("="*50)

print(f"\nposter_pi_mn 통계:")
print(f"  Shape: {result['poster_pi_mn'].shape}")
print(f"  Min: {np.min(result['poster_pi_mn']):.6f}")
print(f"  Max: {np.max(result['poster_pi_mn']):.6f}")
print(f"  Mean: {np.mean(result['poster_pi_mn']):.6f}")
print(f"  Std: {np.std(result['poster_pi_mn']):.6f}")

# OBS_LABEL 확인
print(f"\nOBS_LABEL에 해당하는 단어들:")
for idx in OBS_LABEL:
    if idx < len(result['dictionary_minus']):
        word = result['dictionary_minus'][idx, 0]
        prob = result['poster_pi_mn'][idx]
        print(f"  인덱스 {idx}: {word} (prob: {prob:.4f})")

# FDR 계산 과정 확인
poster_adjusted = force_obs_to_key2(result['Y_minus'], result['poster_pi_mn'])
cutoffs = np.unique(np.sort(poster_adjusted)[::-1])
FDRs = vec_FDR_cal(cutoffs, poster_adjusted)

print(f"\nFDR 분포:")
print(f"  Cutoff 개수: {len(cutoffs)}")
print(f"  Min FDR: {np.min(FDRs):.6f}")
print(f"  Max FDR: {np.max(FDRs):.6f}")
print(f"  FDR < 0.3인 cutoff 개수: {np.sum(FDRs < FDR_CUTOFF)}")

# 상위 10개 노드 확인
top_idx = np.argsort(result['poster_pi_mn'])[::-1][:10]
print(f"\n상위 10개 노드 (poster_pi_mn 기준):")
for i, idx in enumerate(top_idx, 1):
    word = result['dictionary_minus'][idx, 0]
    prob = result['poster_pi_mn'][idx]
    print(f"  {i:2d}. {word:20s} (prob: {prob:.6f})")

print("="*50 + "\n")
# ===== 디버깅 코드 끝 =====

cutoff = calculate_cutoff(result['poster_pi_mn'], FDR_CUTOFF, result['Y_minus'])

print(f"\n{'='*60}")
print(f"결과")
print(f"{'='*60}")
print(f"FDR Cutoff: {cutoff:.4f}\n")

print(f"\n{'='*60}")
print(f"결과")
print(f"{'='*60}")
print(f"FDR Cutoff: {cutoff:.4f}\n")

# BSS 방법
identified_idx = np.where(result['poster_pi_mn'] > cutoff)[0]
print(f"[BSS 방법] - {len(identified_idx)}개")
for i, idx in enumerate(identified_idx[:20], 1):
    word = result['dictionary_minus'][idx, 0]
    score = result['poster_pi_mn'][idx]
    print(f"  {i:2d}. {word:20s} ({score:.4f})")

# TextRank 방법
if len(identified_idx) > 0:
    u_0_adjusted = force_obs_to_key(result['Y_minus'], result['u_0_minus'], k)
    textrank_idx = np.argsort(u_0_adjusted)[::-1][:len(identified_idx)]
    
    print(f"\n[TextRank 방법] - {len(textrank_idx)}개")
    for i, idx in enumerate(textrank_idx[:20], 1):
        word = result['dictionary_minus'][idx, 0]
        print(f"  {i:2d}. {word}")

# Semi-supervised 방법
if len(identified_idx) > 0:
    semi_idx = np.argsort(result['Base_Line_minus'])[::-1][:len(identified_idx)]
    
    print(f"\n[Semi-supervised 방법] - {len(semi_idx)}개")
    for i, idx in enumerate(semi_idx[:20], 1):
        word = result['dictionary_minus'][idx, 0]
        print(f"  {i:2d}. {word}")

print(f"\n{'='*60}")
print("✅ 완료!")
print(f"{'='*60}")
