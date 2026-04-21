"""
Monte Carlo 시뮬레이션 — 단일 문서 반복
keyphrase_functions.py의 함수들을 import하여 사용

방식:
  - C-42 문서의 그래프 구조(A) 고정
  - 매 반복: truth에서 Y를 다르게 샘플링 (different random seeds)
  - N_SIM회 반복 → Monte Carlo 평균 성능 측정
"""

import numpy as np
import os
import time
from datetime import timedelta

from keyphrase_functions import (
    create_graph, semi_keyphrase, FDR_LEVELS,
    grid, inv_logit, force_obs_to_key, force_obs_to_key2,
    DATA_DIR,
)

# ========== 시뮬레이션 설정 ==========
DOCUMENT = "C-42"
N_SIM = 100
T = 10000
BURN_IN = 1000
k = 4


def simulation(article_name, k, N_SIM, T, Burn_in):
    """
    단일 문서에 대해 N_SIM회 반복 시뮬레이션
    """
    # 1. 그래프 생성 (1회만 — 고정)
    print(f"그래프 생성: {article_name}")
    graph = create_graph(article_name)
    if graph is None:
        print("그래프 생성 실패!")
        return None

    n = graph['n']
    truth = graph['truth']
    print(f"  n={n}, truth={len(truth)}개 단어\n")

    if len(truth) <= k:
        print(f"truth({len(truth)})가 k({k})보다 작거나 같아 시뮬레이션 불가")
        return None

    # 누적 변수
    n_levels = len(FDR_LEVELS)
    FDR_cut_mn = np.zeros((n_levels, 7))
    FDR_cut_md = np.zeros((n_levels, 7))

    # 시뮬레이션별 결과 저장
    sim_results = {
        'FDR_pos_mn': [], 'FDR_tp_mn': [], 'Real_FDR_mn': [],
        'FDR_pos_md': [], 'FDR_tp_md': [], 'Real_FDR_md': [],
        'alpha_mn': [], 'alpha_md': [],
        'z_score': [],
        'time_elapsed': [],
    }

    converged_count = 0
    total_start = time.time()

    for sim in range(N_SIM):
        sim_start = time.time()
        print(f"[Sim {sim+1}/{N_SIM}]", end=' ')

        # 2. MCMC 실행 (매번 다른 Y — truth에서 랜덤 샘플링)
        ans = semi_keyphrase(graph, k, grid, T, Burn_in, verbose=False)
        sim_elapsed = time.time() - sim_start

        if not ans['converged']:
            print(f"수렴 실패 (z={ans['z_score']:.2f}), 건너뜀")
            continue

        converged_count += 1

        # 3. 결과 누적
        FDR_cut_mn[:, 0] += ans['FDR_pos_mn']
        FDR_cut_mn[:, 1] += ans['FDR_tp_mn']
        FDR_cut_md[:, 0] += ans['FDR_pos_md']
        FDR_cut_md[:, 1] += ans['FDR_tp_md']

        sim_results['FDR_pos_mn'].append(ans['FDR_pos_mn'])
        sim_results['FDR_tp_mn'].append(ans['FDR_tp_mn'])
        sim_results['Real_FDR_mn'].append(ans['Real_FDR_mn'])
        sim_results['FDR_pos_md'].append(ans['FDR_pos_md'])
        sim_results['FDR_tp_md'].append(ans['FDR_tp_md'])
        sim_results['Real_FDR_md'].append(ans['Real_FDR_md'])
        sim_results['alpha_mn'].append(ans['alpha_mn1'])
        sim_results['alpha_md'].append(ans['alpha_md1'])
        sim_results['z_score'].append(ans['z_score'])
        sim_results['time_elapsed'].append(sim_elapsed)

        elapsed = time.time() - total_start
        avg_per_sim = elapsed / (sim + 1)
        eta = timedelta(seconds=int(avg_per_sim * (N_SIM - sim - 1)))
        print(f"완료 ({sim_elapsed:.0f}s) | z={ans['z_score']:.2f} | "
              f"수렴 {converged_count}/{sim+1} | ETA {eta}")

    if converged_count == 0:
        print("수렴된 시뮬레이션이 없습니다!")
        return None

    # ========== 전체 집계 ==========
    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"시뮬레이션 결과: {article_name}")
    print(f"{'='*60}")
    print(f"총 {N_SIM}회 중 수렴 {converged_count}회")
    print(f"총 소요: {timedelta(seconds=int(total_elapsed))}")
    print(f"평균/회: {total_elapsed/N_SIM:.1f}초\n")

    # Overall Real FDR
    total_keys = len(truth)
    with np.errstate(divide='ignore', invalid='ignore'):
        for tbl in [FDR_cut_mn, FDR_cut_md]:
            tbl[:, 4] = np.where(tbl[:, 0] > 0, 1 - tbl[:, 1] / tbl[:, 0], 0)

    # 시뮬레이션 결과를 numpy array로 변환
    for key in sim_results:
        sim_results[key] = np.array(sim_results[key])

    # FDR 테이블 출력
    col_names = ["Avg Pos", "Avg TP", "Avg FDR", "Real FDR", "Precision", "Recall", "F1"]
    row_names = [f"FDR {c}" for c in FDR_LEVELS]

    def print_sim_table(title, pos_arr, tp_arr, fdr_arr):
        print(f"\n[{title}]")
        print(f"{'':>10s}", end='')
        for cn in col_names:
            print(f"{cn:>12s}", end='')
        print()

        for i, rn in enumerate(row_names):
            avg_pos = np.mean(pos_arr[:, i])
            avg_tp = np.mean(tp_arr[:, i])
            avg_fdr = np.mean(fdr_arr[:, i])
            total_pos = np.sum(pos_arr[:, i])
            total_tp = np.sum(tp_arr[:, i])
            real_fdr = 1 - total_tp / total_pos if total_pos > 0 else 0
            prec = total_tp / total_pos if total_pos > 0 else 0
            rec = total_tp / (total_keys * converged_count) if total_keys > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

            print(f"{rn:>10s}"
                  f"{avg_pos:>12.1f}"
                  f"{avg_tp:>12.1f}"
                  f"{avg_fdr:>12.4f}"
                  f"{real_fdr:>12.4f}"
                  f"{prec:>12.4f}"
                  f"{rec:>12.4f}"
                  f"{f1:>12.4f}")

    print_sim_table("Gibbs MH — Mean (poster_pi_mn)",
                    sim_results['FDR_pos_mn'], sim_results['FDR_tp_mn'], sim_results['Real_FDR_mn'])
    print_sim_table("Gibbs MH — Median (poster_pi_md)",
                    sim_results['FDR_pos_md'], sim_results['FDR_tp_md'], sim_results['Real_FDR_md'])

    # 추가 통계
    print(f"\n[시뮬레이션 통계]")
    print(f"  alpha (mean):   {np.mean(sim_results['alpha_mn']):.4f} +/- {np.std(sim_results['alpha_mn']):.4f}")
    print(f"  alpha (median): {np.mean(sim_results['alpha_md']):.4f} +/- {np.std(sim_results['alpha_md']):.4f}")
    print(f"  z-score:        {np.mean(sim_results['z_score']):.4f} +/- {np.std(sim_results['z_score']):.4f}")
    print(f"  시간/회:        {np.mean(sim_results['time_elapsed']):.1f}s +/- {np.std(sim_results['time_elapsed']):.1f}s")

    print(f"\n{'='*60}")
    print("완료!")
    print(f"{'='*60}")

    return {
        'article': article_name,
        'n': n,
        'n_truth': len(truth),
        'N_SIM': N_SIM,
        'converged': converged_count,
        'FDR_cut_mn': FDR_cut_mn,
        'FDR_cut_md': FDR_cut_md,
        'sim_results': sim_results,
    }


if __name__ == "__main__":
    print(f"{'='*60}")
    print(f"BSS Keyphrase Extraction — Monte Carlo 시뮬레이션")
    print(f"문서: {DOCUMENT}")
    print(f"반복: {N_SIM}회, k={k}")
    print(f"MCMC: T={T}, Burn_in={BURN_IN}")
    print(f"FDR levels: {FDR_LEVELS}")
    print(f"{'='*60}\n")

    result = simulation(DOCUMENT, k, N_SIM, T, BURN_IN)

    if result is not None:
        save_path = os.path.join(DATA_DIR, f"sim_result_{DOCUMENT}_k{k}_T{T}_N{N_SIM}.npz")
        np.savez(save_path,
                 FDR_cut_mn=result['FDR_cut_mn'],
                 FDR_cut_md=result['FDR_cut_md'],
                 **{k_: v for k_, v in result['sim_results'].items()})
        print(f"\n결과 저장: {save_path}")
