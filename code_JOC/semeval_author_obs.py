"""
SemEval Author Observations Script - Python Version
Converted from semeval_author_obs.R
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import multivariate_normal, invgamma
from scipy.linalg import solve
import os
import re
import sys

_ORIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "original")
if _ORIG_DIR not in sys.path:
    sys.path.insert(0, _ORIG_DIR)

from keyphrase_functions_v2 import (
    inv_logit, posterior_gibbstheta, base_to_start,
    gibbs_mh, force_obs_to_key, force_obs_to_key2,
    alpha_find, alpha_lk, FDR_calculate, vec_FDR_cal,
    precision_recall_auc, create_fcm
)

# Get command line argument
if len(sys.argv) > 1:
    i = int(sys.argv[1])
else:
    i = 1

# File paths (adjust as needed)
DATA_DIR = "/home/jiyoon/BSS-Keyphrase-Extraction/data_JOC"
preprocess_dir = os.path.join(DATA_DIR, "pre_process")
author_truth_dir = os.path.join(DATA_DIR, "pre_process_author_truth")
reader_truth_dir = os.path.join(DATA_DIR, "pre_process_reader_truth")

file_list = os.listdir(preprocess_dir) if os.path.exists(preprocess_dir) else []
author_key_list = os.listdir(author_truth_dir) if os.path.exists(author_truth_dir) else []
reader_key_list = os.listdir(reader_truth_dir) if os.path.exists(reader_truth_dir) else []


def big_graph_generate(i):
    """Generate graph from SemEval data"""
    if i >= len(file_list):
        return None
    
    file_path = os.path.join(preprocess_dir, file_list[i])
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        long_text = f.read().lower()
    
    long_text = re.sub(r'[[:punct:]]', '', long_text)
    
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
    
    # Filter by frequency and textrank
    minus_list = np.where((long_test[:, 0] < 12) | (u_0 < np.sort(u_0)[::-1][149]))[0]
    n_minus1 = n - len(minus_list)
    
    A_minus = np.delete(np.delete(A, minus_list, axis=0), minus_list, axis=1)
    n_minus = n - len(minus_list)
    D_minus = np.diag(A_minus.sum(axis=1))
    
    # Read truth keyphrases
    key_file = file_list[i].replace('.txt.final', '')
    key_directory = os.path.join(preprocess_dir.replace('pre_process', 'pre_process_truth'), key_file)
    
    if os.path.exists(key_directory):
        with open(key_directory, 'r', encoding='utf-8', errors='ignore') as f:
            long_key = f.read().lower()
    else:
        long_key = ""
    
    long_key = re.sub(r'[\t\r;]', '', long_key)
    long_key = re.sub(r',', ' ', long_key)
    longkey_list = long_key.split()
    
    # Find truth indices
    long_truth = []
    for word in longkey_list:
        if word in word_to_idx:
            idx = word_to_idx[word]
            if int(long_test[idx, 1]) not in long_truth:
                long_truth.append(int(long_test[idx, 1]))
    
    dictionary_minus = np.column_stack([long_test[~np.isin(np.arange(n), minus_list), 1], 
                                       np.arange(n_minus)])
    
    truth_minus = []
    for word in longkey_list:
        if word in word_to_idx:
            idx = word_to_idx[word]
            if idx not in minus_list:
                new_idx = np.where(dictionary_minus[:, 0] == long_test[idx, 1])[0]
                if len(new_idx) > 0 and new_idx[0] not in truth_minus:
                    truth_minus.append(int(new_idx[0]))
    
    # Read observed keyphrases (author)
    obs_key_file = author_key_list[i] if i < len(author_key_list) else ""
    obs_key_directory = os.path.join(author_truth_dir, obs_key_file)
    
    if os.path.exists(obs_key_directory):
        with open(obs_key_directory, 'r', encoding='utf-8', errors='ignore') as f:
            obs_key = f.read().lower()
    else:
        obs_key = ""
    
    obs_key = re.sub(r'[\t\r;]', '', obs_key)
    obs_key = re.sub(r',', ' ', obs_key)
    obskey_list = obs_key.split()
    
    obs = []
    for word in obskey_list:
        if word in word_to_idx:
            idx = word_to_idx[word]
            if idx not in minus_list:
                new_idx = np.where(dictionary_minus[:, 0] == long_test[idx, 1])[0]
                if len(new_idx) > 0 and new_idx[0] not in obs:
                    obs.append(int(new_idx[0]))
    
    return {
        'file': file_list[i],
        'n': n,
        'A': A,
        'D': D,
        'dictionary': long_test,
        'truth': np.array(long_truth),
        'A_minus': A_minus,
        'D_minus': D_minus,
        'n_minus': n_minus1,
        'minus_list': minus_list,
        'dictionary_minus': dictionary_minus,
        'truth_minus': np.array(truth_minus),
        'obs': np.array(obs)
    }


def semi_keyphrase2(graph, grid):
    """Run semi-supervised keyphrase extraction"""
    if graph is None:
        return None
    
    file = graph['file']
    n_minus = graph['n_minus']
    truth_minus = graph['truth_minus']
    d = 0.85
    
    G_minus = solve(graph['D_minus'], graph['A_minus'])
    B_minus = np.eye(n_minus) - d * G_minus.T
    u_0_minus = solve(B_minus, np.ones(n_minus) * (1 - d))
    w_minus = np.diag(1.0 / np.sqrt(np.diag(graph['D_minus'])))
    B_star_minus = np.eye(n_minus) - d * w_minus @ graph['A_minus'] @ w_minus
    
    obs_label = np.array(graph['obs'])
    # obs_label 유효성 검사 및 필터링
    valid_mask = (obs_label >= 0) & (obs_label < n_minus)
    valid_obs_label = obs_label[valid_mask]
    
    if len(valid_obs_label) < len(obs_label):
        invalid = obs_label[~valid_mask]
        print(f"  ⚠ 경고: 인덱스 {invalid}는 범위를 벗어났습니다 (0-{n_minus-1}). 무시됩니다.")
    
    if len(valid_obs_label) == 0:
        print("  ✗ 오류: 유효한 관찰 인덱스가 없습니다.")
        return None
    
    k = len(valid_obs_label)
    True_Y_minus = np.zeros(n_minus)
    True_Y_minus[truth_minus] = 1
    Y_minus = np.zeros(n_minus)
    Y_minus[valid_obs_label] = 1
    
    Base_Line_minus = solve(B_star_minus, Y_minus)
    T = 50000
    Burn_in = 2000
    
    ini = Base_Line_minus.copy()
    alpha_est = alpha_find(u_0_minus, Y_minus, grid)
    
    test_chain = gibbs_mh(Burn_in, T, ini, n_minus, graph, Y_minus, B_minus, u_0_minus, alpha_est, grid)
    
    return {
        'file': file,
        'poster_pi_md': test_chain['poster_pi_md'],
        'poster_pi_mn': test_chain['poster_pi_mn'],
        'poster_pi_mnV2': inv_logit(np.mean(test_chain.get('theta_store', []), axis=0)) if 'theta_store' in test_chain else test_chain['poster_pi_mn'],
        'truth_minus': truth_minus,
        'obs_label': valid_obs_label,
        'dictionary_minus': graph['dictionary_minus'],
        'u_0_minus': u_0_minus,
        'Base_Line_minus': Base_Line_minus,
        'Y_minus': Y_minus,
        'alpha_est': test_chain['alpha_mn'],
        'n': graph['n'],
        'accept_rate': test_chain.get('accept', 0) / T
    }


# Set random seed
np.random.seed(12345)

alpha_min = (10 - 5) / 10
alpha_max = (150 - 5) / 150
grid = (np.arange(10, 151) - 5) / np.arange(10, 151)

start_time = pd.Timestamp.now()


def group_article(j):
    """Process a group of articles"""
    total_ans = []
    for idx in range((j - 1) * 10 + 1, (j - 1) * 10 + 11):
        if idx <= 144 and idx < len(file_list):
            try:
                graph = big_graph_generate(idx - 1)  # 0-indexed
                if graph is not None:
                    Ans1 = semi_keyphrase2(graph, grid)
                    if Ans1 is not None:
                        total_ans.append(Ans1)
            except Exception as e:
                print(f"An error occurred for article {idx} {file_list[idx-1] if idx-1 < len(file_list) else 'unknown'}:\n{e}")
    return total_ans


total_ans = group_article(i)

end_time = pd.Timestamp.now()
print(f"Time elapsed: {end_time - start_time}")

save_path = f"Semeval_obs_author_{i}.npz"
np.savez(save_path, total_ans=total_ans)
print(f"Results saved to {save_path}")

