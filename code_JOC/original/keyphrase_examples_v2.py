"""
Keyphrase Examples Script - Python Version
Converted from keyphrase_examples.R
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import multivariate_normal, invgamma
from scipy.linalg import solve
import os
import re
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
    minus_list = np.where((long_test[:, 0] < 12) | (u_0 < np.sort(u_0)[::-1][149]))[0]
    
    A_minus = np.delete(np.delete(A, minus_list, axis=0), minus_list, axis=1)
    n_minus = n - len(minus_list)
    D_minus = np.diag(A_minus.sum(axis=1))
    
    # Known keyphrases (hardcoded for this example)
    longkeyphrase = 'ensembl kalman filter,data assimil methodolog,hydrocarbon reservoir simul,energi explor,tigr grid comput environ,grid comput,cyberinfrastructur develop project,high perform comput,tigr grid middlewar,strateg applic area,gridway metaschedul,pool licens,grid-en,reservoir model,enkf,tigr'
    long_key = re.sub(r'[\t\r]', '', longkeyphrase.lower())
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
    
    print("Dictionary minus:")
    print(dictionary_minus)
    print(f"\nTruth minus: {truth_minus}")
    
    return {
        'n': n,
        'A': A,
        'D': D,
        'dictionary': long_test,
        'truth': np.array(long_truth),
        'A_minus': A_minus,
        'D_minus': D_minus,
        'n_minus': n_minus,
        'minus_list': minus_list,
        'dictionary_minus': dictionary_minus,
        'truth_minus': np.array(truth_minus)
    }


# Generate graph
big_graphH41 = big_graph_generate("C-42(2).txt")


def semi_keyphrase2(graph, obs_label, grid):
    """Run semi-supervised keyphrase extraction"""
    if graph is None:
        return None
    
    k = len(obs_label)
    n_minus = graph['n_minus']
    truth_minus = graph['truth_minus']
    d = 0.85
    
    G_minus = solve(graph['D_minus'], graph['A_minus'])
    B_minus = np.eye(n_minus) - d * G_minus.T
    u_0_minus = solve(B_minus, np.ones(n_minus) * (1 - d))
    w_minus = np.diag(1.0 / np.sqrt(np.diag(graph['D_minus'])))
    B_star_minus = np.eye(n_minus) - d * w_minus @ graph['A_minus'] @ w_minus
    
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
    
    True_Y_minus = np.zeros(n_minus)
    True_Y_minus[truth_minus] = 1
    Y_minus = np.zeros(n_minus)
    Y_minus[valid_obs_label] = 1
    
    Base_Line_minus = solve(B_star_minus, Y_minus)
    T = 50000
    Burn_in = 2000
    
    ini = base_to_start(u_0_minus)
    alpha_est = alpha_find(u_0_minus, Y_minus, grid)
    
    test_chain = gibbs_mh(Burn_in, T, ini, n_minus, graph, Y_minus, B_minus, u_0_minus, alpha_est, grid)
    
    per = np.arange(0.05, 1.05, 0.05)
    pre_rec_auc_md = precision_recall_auc(Y_minus, True_Y_minus, test_chain['poster_pi_md'], per, k)
    pre_rec_auc_mn = precision_recall_auc(Y_minus, True_Y_minus, test_chain['poster_pi_mn'], per, k)
    pre_rec_auc_mnV2 = precision_recall_auc(Y_minus, True_Y_minus, test_chain.get('poster_pi_mnV2', test_chain['poster_pi_mn']), per, k)
    pre_rec_auc_u_0 = precision_recall_auc(Y_minus, True_Y_minus, u_0_minus, per, k)
    pre_rec_auc_bl = precision_recall_auc(Y_minus, True_Y_minus, Base_Line_minus, per, k)
    
    return {
        'poster_pi_md': test_chain['poster_pi_md'],
        'poster_pi_mn': test_chain['poster_pi_mn'],
        'poster_pi_mnV2': test_chain.get('poster_pi_mnV2', test_chain['poster_pi_mn']),
        'truth_minus': truth_minus,
        'obs_label': valid_obs_label,
        'dictionary_minus': graph['dictionary_minus'],
        'u_0_minus': u_0_minus,
        'Base_Line_minus': Base_Line_minus,
        'Y_minus': Y_minus,
        'alpha_est': test_chain['alpha_mn'],
        'sigma2_store': test_chain.get('sigma2_store', []),
        'theta_store': test_chain.get('theta_store', []),
        'MH_rate_store': test_chain.get('MH_rate_store', []),
        'pos_md': pre_rec_auc_md,
        'pos_mn': pre_rec_auc_mn,
        'pos_mnV2': pre_rec_auc_mnV2,
        'textrank': pre_rec_auc_u_0,
        'semi': pre_rec_auc_bl,
        'accept_rate': test_chain.get('accept', 0) / T
    }


def FDR_cutoff(poster_pi_md, c, Y, truth):
    """Given the FDR, find out where the cutoff is"""
    poster_md_adjust = force_obs_to_key2(Y, poster_pi_md)
    cutoffs = np.unique(np.sort(poster_md_adjust)[::-1])
    FDRs = vec_FDR_cal(cutoffs, poster_md_adjust)
    index = np.max(np.where(FDRs < c)[0]) if np.any(FDRs < c) else 0
    cutoff = cutoffs[index]
    selected = np.where(poster_md_adjust >= cutoff)[0]
    FDR_pos = len(selected)
    FDR_tp = np.sum(np.isin(selected, truth)) if len(truth) > 0 else 0
    Real_FDR = (FDR_pos - FDR_tp) / FDR_pos if FDR_pos > 0 else 0
    return {'FDR_pos': FDR_pos, 'FDR_tp': FDR_tp, 'Real_FDR': Real_FDR}


def vec_FDR_cutoff(poster_pi_md, c_values, Y, truth):
    """Vectorized FDR cutoff"""
    results = []
    for c in c_values:
        results.append(FDR_cutoff(poster_pi_md, c, Y, truth))
    return {
        'FDR_pos': np.array([r['FDR_pos'] for r in results]),
        'FDR_tp': np.array([r['FDR_tp'] for r in results]),
        'Real_FDR': np.array([r['Real_FDR'] for r in results])
    }


grid = np.linspace(0.05, 0.90, 33)
np.random.seed(541)

obs_label1 = np.array([8, 13, 10, 11, 69, 29])
k1 = len(obs_label1)

Ans1 = semi_keyphrase2(big_graphH41, obs_label1, grid)

if Ans1 is not None:
    # Save result
    save_path = "/home/jiyoon/3차/BSS-Keyphrase-Extraction-master/code_JOC/original/biggraphC42_sub_title_v2.npz"
    np.savez(save_path, **Ans1)
    
    # FDR cutoff at 0.25
    positive1 = vec_FDR_cutoff(Ans1['poster_pi_mn'], [0.25], Ans1['Y_minus'], Ans1['truth_minus'])
    print(f"Positive identified by BSS method: {positive1['FDR_pos'][0]}")
    
    # Check observed keywords
    print(f"\nObserved keywords:\n{Ans1['dictionary_minus'][Ans1['obs_label']]}")
    
    # True positives from each method
    num_pos = positive1['FDR_pos'][0]
    
    # TextRank method
    u_0_adjusted = force_obs_to_key(Ans1['Y_minus'], Ans1['u_0_minus'], k1)
    tr_top = np.argsort(u_0_adjusted)[::-1][:num_pos]
    tr_tp = np.intersect1d(tr_top, Ans1['truth_minus'])
    print(f"\nTextRank True Positives:\n{Ans1['dictionary_minus'][tr_tp]}")
    
    # Semi-supervised baseline
    ss_top = np.argsort(Ans1['Base_Line_minus'])[::-1][:num_pos]
    ss_tp = np.intersect1d(ss_top, Ans1['truth_minus'])
    print(f"\nSemi-supervised True Positives:\n{Ans1['dictionary_minus'][ss_tp]}")
    
    # BSS method
    bss_top = np.argsort(Ans1['poster_pi_mn'])[::-1][:num_pos]
    bss_tp = np.intersect1d(bss_top, Ans1['truth_minus'])
    print(f"\nBSS True Positives:\n{Ans1['dictionary_minus'][bss_tp]}")
