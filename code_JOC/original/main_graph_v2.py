"""
Main Graph Analysis Script - Python Version
Converted from main_graph.R
Result includes: FDR comparisons, precision, recall and F-1 measure comparison.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keyphrase_functions_v2 import main_function2, precision_recall_auc
import os

# Set random seed
np.random.seed(541)
k = 5

# 경로 설정
DATA_DIR = "/home/jiyoon/3차/BSS-Keyphrase-Extraction-master/data_JOC"
PREPROCESS_DIR = os.path.join(DATA_DIR, "short_articles", "pre_process")
RESULTS_DIR = "/home/jiyoon/3차/BSS-Keyphrase-Extraction-master/results"

# 결과 디렉토리 생성
os.makedirs(RESULTS_DIR, exist_ok=True)

# Get file list (adjust path as needed)
file_list_path = "/home/jiyoon/3차/BSS-Keyphrase-Extraction-master/data_JOC/file_list.npy"
if os.path.exists(file_list_path):
    file_list = np.load(file_list_path, allow_pickle=True).tolist()
else:
    # Fallback: create from directory
    preprocess_dir = "/home/jiyoon/3차/BSS-Keyphrase-Extraction-master/data_JOC/short_articles/pre_process"
    if os.path.exists(preprocess_dir):
        file_list = os.listdir(preprocess_dir)
    else:
        print("Warning: Could not find file list or directory")
        file_list = []

# 경로 설정
if os.path.exists(PREPROCESS_DIR):
    base_dir = PREPROCESS_DIR
else:
    # Fallback 경로
    base_dir = "/home/jiyoon/3차/BSS-Keyphrase-Extraction-master/data_JOC/short_articles/pre_process"

result = main_function2(k, file_list, start_idx=201, end_idx=500, base_dir=base_dir)

print("FDR cut mean:")
print(result['FDR_cut_mn'])

# Save result
save_path = "/home/jiyoon/3차/BSS-Keyphrase-Extraction-master/results/result201_500.npz"
np.savez(save_path, **result)

# Calculate metrics for each article
Real_FDR_TR = []
Real_FDR_BL = []
mn_TP_Pos = []
Total_pos_mn = np.zeros(6)

for i in range(len(result['split_points']) - 1):
    u_0_adjust = result['u_0_adjust'][result['split_points'][i]:result['split_points'][i + 1]]
    Base_Line_adjust = result['Base_Line_adjust'][result['split_points'][i]:result['split_points'][i + 1]]
    poster_pi_mn = result['poster_pi_mn'][result['split_points'][i]:result['split_points'][i + 1]]
    
    Y = np.zeros(result['words'][i])
    truth_Y = Y.copy()
    if i < len(result['obs_label']):
        Y[result['obs_label'][i]] = 1
    
    truth = result['truth'][result['split_key'][i]:result['split_key'][i + 1]]
    truth_Y[truth] = 1
    
    c_values = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    pre_rec_auc_mn = precision_recall_auc(Y, truth, truth_Y, poster_pi_mn, k, c_values)
    FDR_tp = pre_rec_auc_mn['FDR_tp']
    FDR_pos = pre_rec_auc_mn['FDR_pos']
    
    mn_TP_Pos.append(np.column_stack([FDR_tp, FDR_pos]))
    Total_pos_mn += FDR_pos

# Create barplot data (Fig 1)
mn_0_15_article = pd.DataFrame({
    'mean pos': result['TP_0_15mn_article'][:, 0] if len(result['TP_0_15mn_article']) > 0 else [],
    'mean TP': result['TP_0_15mn_article'][:, 1] if len(result['TP_0_15mn_article']) > 0 else [],
    'textrank pos': result['mn_comparison']['check_mn_tr'][:, 8] if len(result['mn_comparison']['check_mn_tr']) > 0 else [],
    'textrank TP': result['mn_comparison']['check_mn_tr'][:, 2] if len(result['mn_comparison']['check_mn_tr']) > 0 else [],
    'semi pos': result['mn_comparison']['check_mn_bl'][:, 8] if len(result['mn_comparison']['check_mn_bl']) > 0 else [],
    'semi TP': result['mn_comparison']['check_mn_bl'][:, 2] if len(result['mn_comparison']['check_mn_bl']) > 0 else [],
    'num of key': result['num_of_key'],
    'num of word': result['words']
})

if len(mn_0_15_article) > 0:
    # Create key number groups
    mn_0_15_article['key_num_group'] = pd.cut(
        mn_0_15_article['num of key'],
        bins=[0, 14, 19, 24, np.inf],
        labels=[1, 2, 3, 4]
    ).astype(int)
    
    # Create key proportion groups
    key_prop = mn_0_15_article['num of key'] / mn_0_15_article['num of word']
    pro_group_quantile = np.quantile(key_prop, [0.25, 0.5, 0.75])
    mn_0_15_article['key_pro_group'] = pd.cut(
        key_prop,
        bins=[0, pro_group_quantile[0], pro_group_quantile[1], pro_group_quantile[2], np.inf],
        labels=[1, 2, 3, 4]
    ).astype(int)
    
    # Aggregate by key number group
    mn_0_15_key_group = mn_0_15_article.groupby('key_num_group').agg({
        'mean pos': 'sum',
        'mean TP': 'sum',
        'textrank pos': 'sum',
        'textrank TP': 'sum',
        'semi pos': 'sum',
        'semi TP': 'sum',
        'num of key': 'sum',
        'num of word': 'sum'
    }).reset_index()
    
    mn_0_15_key_group['pos_pro'] = mn_0_15_key_group['mean pos'] / mn_0_15_key_group['num of word']
    mn_0_15_key_group['total_key'] = mn_0_15_article.groupby('key_num_group')['num of key'].sum().values
    
    mn_0_15_key_group['mean_precision'] = mn_0_15_key_group['mean TP'] / mn_0_15_key_group['mean pos']
    mn_0_15_key_group['textrank_precision'] = mn_0_15_key_group['textrank TP'] / mn_0_15_key_group['textrank pos']
    mn_0_15_key_group['semi_precision'] = mn_0_15_key_group['semi TP'] / mn_0_15_key_group['semi pos']
    
    mn_0_15_key_group['mean_recall'] = mn_0_15_key_group['mean TP'] / mn_0_15_key_group['total_key']
    mn_0_15_key_group['textrank_recall'] = mn_0_15_key_group['textrank TP'] / mn_0_15_key_group['total_key']
    mn_0_15_key_group['semi_recall'] = mn_0_15_key_group['semi TP'] / mn_0_15_key_group['total_key']
    
    mn_0_15_key_group['mean_f'] = 2 * (mn_0_15_key_group['mean_precision'] * mn_0_15_key_group['mean_recall']) / \
                                   (mn_0_15_key_group['mean_precision'] + mn_0_15_key_group['mean_recall'])
    mn_0_15_key_group['textrank_f'] = 2 * (mn_0_15_key_group['textrank_precision'] * mn_0_15_key_group['textrank_recall']) / \
                                       (mn_0_15_key_group['textrank_precision'] + mn_0_15_key_group['textrank_recall'])
    mn_0_15_key_group['semi_f'] = 2 * (mn_0_15_key_group['semi_precision'] * mn_0_15_key_group['semi_recall']) / \
                                  (mn_0_15_key_group['semi_precision'] + mn_0_15_key_group['semi_recall'])
    
    # Create barplot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(mn_0_15_key_group))
    width = 0.25
    
    ax.bar(x - width, mn_0_15_key_group['mean_f'], width, label='BSS', alpha=0.8)
    ax.bar(x, mn_0_15_key_group['textrank_f'], width, label='TR', alpha=0.8)
    ax.bar(x + width, mn_0_15_key_group['semi_f'], width, label='SS', alpha=0.8)
    
    ax.set_xlabel('Number of keyphrases')
    ax.set_ylabel('Overall F-measure')
    ax.set_title('F-measure Comparison by Keyphrase Group')
    ax.set_xticks(x)
    ax.set_xticklabels(['A', 'B', 'C', 'D'][:len(mn_0_15_key_group)])
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('f_measure_comparison.png', dpi=300)
    print("Barplot saved as f_measure_comparison.png")
    
    # Make table of keyphrase proportions (Table 4)
    mn_0_15_key_pro = mn_0_15_article.groupby('key_pro_group').agg({
        'mean pos': 'sum',
        'mean TP': 'sum',
        'textrank pos': 'sum',
        'textrank TP': 'sum',
        'semi pos': 'sum',
        'semi TP': 'sum',
        'num of word': 'sum'
    }).reset_index()
    
    mn_0_15_key_pro['total_key'] = mn_0_15_article.groupby('key_pro_group')['num of key'].sum().values
    mn_0_15_key_pro['pos_pro'] = mn_0_15_key_pro['mean pos'] / mn_0_15_key_pro['num of word']
    
    mn_0_15_key_pro['mean_precision'] = mn_0_15_key_pro['mean TP'] / mn_0_15_key_pro['mean pos']
    mn_0_15_key_pro['textrank_precision'] = mn_0_15_key_pro['textrank TP'] / mn_0_15_key_pro['textrank pos']
    mn_0_15_key_pro['semi_precision'] = mn_0_15_key_pro['semi TP'] / mn_0_15_key_pro['semi pos']
    
    mn_0_15_key_pro['mean_recall'] = mn_0_15_key_pro['mean TP'] / mn_0_15_key_pro['total_key']
    mn_0_15_key_pro['textrank_recall'] = mn_0_15_key_pro['textrank TP'] / mn_0_15_key_pro['total_key']
    mn_0_15_key_pro['semi_recall'] = mn_0_15_key_pro['semi TP'] / mn_0_15_key_pro['total_key']
    
    mn_0_15_key_pro['mean_f'] = 2 * (mn_0_15_key_pro['mean_precision'] * mn_0_15_key_pro['mean_recall']) / \
                                 (mn_0_15_key_pro['mean_precision'] + mn_0_15_key_pro['mean_recall'])
    mn_0_15_key_pro['textrank_f'] = 2 * (mn_0_15_key_pro['textrank_precision'] * mn_0_15_key_pro['textrank_recall']) / \
                                    (mn_0_15_key_pro['textrank_precision'] + mn_0_15_key_pro['textrank_recall'])
    mn_0_15_key_pro['semi_f'] = 2 * (mn_0_15_key_pro['semi_precision'] * mn_0_15_key_pro['semi_recall']) / \
                                (mn_0_15_key_pro['semi_precision'] + mn_0_15_key_pro['semi_recall'])
    
    print("\nKeyphrase Proportions Table:")
    print(mn_0_15_key_pro)
