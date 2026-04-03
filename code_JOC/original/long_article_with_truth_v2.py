"""
Long Article with Truth Script - Python Version
Converted from Long_article_with_truth.R
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import multivariate_normal, invgamma
from scipy.linalg import solve
from numpy import diag
import os
import re
import time
from datetime import datetime, timedelta
from keyphrase_functions_v2 import (
    inv_logit, posterior_gibbstheta, base_to_start, 
    gibbs_mh, force_obs_to_key, force_obs_to_key2,
    alpha_find, alpha_lk, FDR_calculate, vec_FDR_cal,
    precision_recall_auc, create_fcm
)


def big_graph_generate(article, base_dir="/home/jiyoon/3차/BSS-Keyphrase-Extraction-master/data_JOC"):
    """Generate graph from article file"""
    file_path = os.path.join(base_dir, article)
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        long_text = f.read().lower()
    
    long_text = re.sub(r'[/=]', ' ', long_text)
    
    # Create FCM
    fcm, unique_words, word_to_idx = create_fcm(long_text)
    n = len(unique_words)
    
    d = 0.85
    long_test = np.column_stack([fcm.sum(axis=1), np.arange(n)])
    A = fcm.copy()
    np.fill_diagonal(A, 0)
    D = np.diag(fcm.sum(axis=1))
    
    G = solve(D, A)
    B = np.eye(n) - d * G.T
    u_0 = solve(B, np.ones(n) * (1 - d))
    
    # Filter
    minus_list = np.where((long_test[:, 0] < 12) | (u_0 < np.sort(u_0)[::-1][199]))[0]
    # Additional manual removals
    additional_removals = [87, 145, 152, 153, 171, 172, 181, 232, 328, 333, 334, 369, 411, 425, 438, 523, 563, 589, 591, 824]
    minus_list = np.unique(np.concatenate([minus_list, np.array([x for x in additional_removals if x < n])]))
    
    A_minus = np.delete(np.delete(A, minus_list, axis=0), minus_list, axis=1)
    n_minus = n - len(minus_list)
    D_minus = np.diag(A_minus.sum(axis=1))
    
    dictionary_minus = np.column_stack([long_test[~np.isin(np.arange(n), minus_list), 1], 
                                       np.arange(n_minus)])
    
    return {
        'n': n,
        'A': A,
        'D': D,
        'dictionary': long_test,
        'A_minus': A_minus,
        'D_minus': D_minus,
        'n_minus': n_minus,
        'minus_list': minus_list,
        'dictionary_minus': dictionary_minus
    }


# Generate graph
print(f"\n{'='*60}")
print(f"그래프 생성 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*60}\n")

graph_start = time.time()
graph = big_graph_generate("EM_Preprocessed.txt")
graph_time = time.time() - graph_start
print(f"그래프 생성 완료 (소요 시간: {timedelta(seconds=int(graph_time))})")
print(f"노드 수: {graph['n_minus'] if graph else 0}\n")


def FDR_cutoff_notruth(poster_pi_md, c, Y):
    """Find FDR cutoff without ground truth"""
    poster_md_adjust = force_obs_to_key2(Y, poster_pi_md)
    cutoffs = np.unique(np.sort(poster_md_adjust)[::-1])
    FDRs = vec_FDR_cal(cutoffs, poster_md_adjust)
    index = np.max(np.where(FDRs < c)[0]) if np.any(FDRs < c) else 0
    cutoff = cutoffs[index]
    return cutoff


def semi_keyphrase2(graph, obs_label, grid):
    """Run semi-supervised keyphrase extraction"""
    if graph is None:
        return None
    
    n_minus = graph['n_minus']
    d = 0.85
    
    # obs_label 유효성 검사 및 필터링
    obs_label = np.array(obs_label)
    valid_mask = (obs_label >= 0) & (obs_label < n_minus)
    valid_obs_label = obs_label[valid_mask]
    
    if len(valid_obs_label) < len(obs_label):
        invalid = obs_label[~valid_mask]
        print(f"  ⚠ 경고: 인덱스 {invalid}는 범위를 벗어났습니다 (0-{n_minus-1}). 무시됩니다.")
        print(f"  유효한 관찰 인덱스: {valid_obs_label}")
    
    if len(valid_obs_label) == 0:
        print("  ✗ 오류: 유효한 관찰 인덱스가 없습니다.")
        return None
    
    G_minus = solve(graph['D_minus'], graph['A_minus'])
    B_minus = np.eye(n_minus) - d * G_minus.T
    u_0_minus = solve(B_minus, np.ones(n_minus) * (1 - d))
    w_minus = np.diag(1.0 / np.sqrt(np.diag(graph['D_minus'])))
    B_star_minus = np.eye(n_minus) - d * w_minus @ graph['A_minus'] @ w_minus
    
    Y_minus = np.zeros(n_minus)
    Y_minus[valid_obs_label] = 1
    Base_Line_minus = solve(B_star_minus, Y_minus)
    
    T = 50000
    Burn_in = 2000
    ini = base_to_start(u_0_minus)
    alpha_est = alpha_find(u_0_minus, Y_minus, grid)
    
    print(f"\n{'='*60}")
    print(f"MCMC 샘플링 시작")
    print(f"총 반복 횟수: {T:,} (Burn-in: {Burn_in:,})")
    print(f"노드 수: {n_minus}")
    print(f"{'='*60}\n")
    
    test_chain = gibbs_mh(Burn_in, T, ini, n_minus, graph, Y_minus, B_minus, u_0_minus, alpha_est, grid)
    
    return {
        'poster_pi_md': test_chain['poster_pi_md'],
        'poster_pi_mn': test_chain['poster_pi_mn'],
        'poster_pi_mnV2': inv_logit(np.mean(test_chain.get('theta_store', []), axis=0)) if 'theta_store' in test_chain else test_chain['poster_pi_mn'],
        'obs_label': valid_obs_label,
        'dictionary_minus': graph['dictionary_minus'],
        'u_0_minus': u_0_minus,
        'Base_Line_minus': Base_Line_minus,
        'Y_minus': Y_minus,
        'alpha_est': test_chain['alpha_mn'],
        'sigma2_store': test_chain.get('sigma2_store', []),
        'accept_rate': test_chain.get('accept', 0) / T
    }


k = 4
grid = np.linspace(0.05, 0.90, 33)
np.random.seed(12345)

total_start = time.time()
Non_para_result = semi_keyphrase2(graph, np.array([2, 18, 35, 75]), grid)
total_time = time.time() - total_start

if Non_para_result is not None:
    print(f"\n{'='*60}")
    print(f"전체 처리 완료!")
    print(f"총 소요 시간: {timedelta(seconds=int(total_time))}")
    print(f"완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    cutoff = FDR_cutoff_notruth(Non_para_result['poster_pi_mn'], 0.3, Non_para_result['Y_minus'])
    print(f"FDR Cutoff: {cutoff}")
    
    # Identified keywords
    identified = Non_para_result['dictionary_minus'][Non_para_result['poster_pi_mn'] > cutoff]
    print(f"\nIdentified keywords by BSS method:\n{identified}")
    
    num_of_positive = np.sum(Non_para_result['poster_pi_mn'] > cutoff)
    
    # TextRank method
    u_0_adjusted = force_obs_to_key(Non_para_result['Y_minus'], Non_para_result['u_0_minus'], k)
    textrank_top = Non_para_result['dictionary_minus'][np.argsort(u_0_adjusted)[::-1][:num_of_positive]]
    print(f"\nTop keywords by TextRank:\n{textrank_top}")
    
    # Semi-supervised method
    semi_top = Non_para_result['dictionary_minus'][np.argsort(Non_para_result['Base_Line_minus'])[::-1][:num_of_positive]]
    print(f"\nTop keywords by Semi-supervised:\n{semi_top}")


# Amazon review example
def amazon_graph_generate(article, base_dir="/home/jiyoon/3차/BSS-Keyphrase-Extraction-master/data_JOC"):
    """Generate graph from Amazon review"""
    file_path = os.path.join(base_dir, article)
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        long_text = f.read().lower()
    
    long_text = re.sub(r'[/=]', ' ', long_text)
    
    # Create FCM
    fcm, unique_words, word_to_idx = create_fcm(long_text)
    n = len(unique_words)
    
    d = 0.85
    long_test = np.column_stack([fcm.sum(axis=1), np.arange(n)])
    A = fcm.copy()
    np.fill_diagonal(A, 0)
    D = np.diag(fcm.sum(axis=1))
    
    return {
        'n': n,
        'A': A,
        'D': D,
        'dictionary': long_test,
        'A_minus': A,
        'D_minus': D,
        'n_minus': n
    }


amazon_graph = amazon_graph_generate("amazon_pre.txt")

if amazon_graph is not None:
    print(f"\nAmazon 그래프 생성 완료")
    print(f"노드 수: {amazon_graph['n_minus']}")
    print(f"유효한 인덱스 범위: 0-{amazon_graph['n_minus']-1}")
    
    np.random.seed(12345)
    # 인덱스가 범위를 벗어나지 않도록 조정
    max_idx = amazon_graph['n_minus'] - 1
    obs_indices = np.array([3, min(83, max_idx)])  # 83이 범위를 벗어나면 최대값 사용
    if 83 > max_idx:
        print(f"  ⚠ 인덱스 83이 범위를 벗어나 {max_idx}로 조정됩니다.")
    
    Amazon_result = semi_keyphrase2(amazon_graph, obs_indices, grid)
    
    if Amazon_result is not None:
        cutoff2 = FDR_cutoff_notruth(Amazon_result['poster_pi_mn'], 0.25, Amazon_result['Y_minus'])
        print(f"\nAmazon review cutoff: {cutoff2}")
        
        # Identified keywords
        identified2 = amazon_graph['dictionary'][Amazon_result['poster_pi_mn'] >= cutoff2]
        print(f"\nIdentified keywords by BSS method:\n{identified2}")
        
        num_of_positive2 = np.sum(Amazon_result['poster_pi_mn'] >= cutoff2)
        
        # TextRank method
        u_0_adjusted2 = force_obs_to_key(Amazon_result['Y_minus'], Amazon_result['u_0_minus'], 2)
        textrank_top2 = amazon_graph['dictionary'][np.argsort(u_0_adjusted2)[::-1][:num_of_positive2]]
        print(f"\nTop keywords by TextRank:\n{textrank_top2}")
        
        # Semi-supervised method
        semi_top2 = amazon_graph['dictionary'][np.argsort(Amazon_result['Base_Line_minus'])[::-1][:num_of_positive2]]
        print(f"\nTop keywords by Semi-supervised:\n{semi_top2}")
