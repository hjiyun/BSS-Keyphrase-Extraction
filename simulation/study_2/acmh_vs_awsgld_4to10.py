#!/usr/bin/env python3
"""
실데이터 실험: Hulth baseline 중 정답 4~10개 문서(126개)에서
acMH(componentwise MCMC) vs AWSGLD(gibbs_mh) 성능 비교.

설계 (원논문 방식 유지):
  - 그래프: baseline_preprocessed (POS 명사/형용사, 윈도우2)
  - observed 시드 k = floor(truth/2)  (k<truth 보장, 약 50% 숨김)
  - 두 샘플러는 매 (doc, sim)마다 동일한 Y/ini/그래프 입력을 공유 (공정 비교)
  - 평가: FDR cutoff별 precision/recall/F (force_obs_to_key2 — 원본과 동일)
  - T=10000, burn=1000, N_SIM=30

사용:
  python acmh_vs_awsgld_4to10.py            # 전체 실행
  python acmh_vs_awsgld_4to10.py --probe    # 1문서×2sim 타이밍 측정만
"""
import os, sys, csv, json, time, argparse
import numpy as np
from datetime import timedelta

ROOT = "/home/jiyoon/BSS-Keyphrase-Extraction"
CODE = os.path.join(ROOT, "code_JOC")
sys.path.insert(0, CODE)
import keyphrase_functions_awsgld as M
from numpy.linalg import solve

BASE = os.path.join(ROOT, "data_JOC/baseline_preprocessed")
PRE = os.path.join(BASE, "pre_process")
TRU = os.path.join(BASE, "truth")
OUTDIR = os.path.join(ROOT, "simulation/study_2")
os.makedirs(OUTDIR, exist_ok=True)

T, BURN_IN, N_SIM = 10000, 1000, 30
FDR_LEVELS = M.FDR_LEVELS
grid = M.grid
GAMMA_MAIN = 0.15
d = 0.85


def build_graph(doc_id):
    """baseline_preprocessed 텍스트로 그래프 dict 생성 (create_graph와 동일 로직)."""
    with open(os.path.join(PRE, f"{doc_id}.abstr")) as f:
        text = f.read()
    fcm, words, w2i = M.create_fcm_words(text, window=2)
    n = len(words)
    A = fcm.copy(); np.fill_diagonal(A, 0)
    D = np.diag(A.sum(axis=1))
    with open(os.path.join(TRU, f"{doc_id}.uncontr")) as f:
        kw = f.read().split()
    truth = sorted(set(w2i[w] for w in kw if w in w2i))
    return {'n': n, 'A': A, 'D': D, 'truth': truth, 'words': words}


def componentwise_mcmc_fast(T, ini, n, grid, alpha_est, u_0, B, Y, verbose=False):
    """
    M.componentwise_mcmc 와 수학적으로 동일한 acMH, 단 C=||B(theta-u_0)||^2 를
    매 단어 제안마다 O(n^2) 재계산하지 않고 O(n) 증분 갱신.

    theta[i] 만 delta 만큼 바뀌므로:
      v = B(theta-u_0) 유지
      C_new = C + 2*delta*(v . B[:,i]) + delta^2 * ||B[:,i]||^2
      채택 시: v += B[:,i]*delta,  C = C_new      (둘 다 O(n))
    likelihood 도 성분 i 만 변하므로 O(1).
    RNG 소비 순서를 원본과 동일하게 유지(normal 1회 + uniform 1회/단어).
    """
    theta_store = np.zeros((T, n))
    alpha_store = np.zeros(T)
    theta_store[0, :] = ini
    alpha_store[0] = alpha_est
    bb = np.sum(B * B, axis=0)               # bb[i] = ||B[:,i]||^2 (사전계산)
    accept = 0
    eps = 0.001
    coef = (n / 2.0 + eps)

    def lk_i(ti, yi):
        temp = (1 - alpha_est) * M.inv_logit(ti)
        temp = min(max(temp, 1e-10), 1 - 1e-10)
        return yi * np.log(temp) + (1 - yi) * np.log(1 - temp)

    for t in range(1, T):
        theta = theta_store[t - 1, :].copy()
        v = B @ (theta - u_0)                # t당 1회 O(n^2)
        C = float(v @ v)
        for i in range(n):
            if t < 10:
                var_i = 1.0
            else:
                var_i = np.sqrt(2.4 * (np.var(theta_store[:t, i], ddof=1) + 0.01))
            theta_cur = theta[i]
            theta_star = np.random.normal(theta_cur, var_i)   # 원본과 동일 RNG 소비
            delta = theta_star - theta_cur
            Bcol = B[:, i]
            vB = float(v @ Bcol)             # O(n)
            C_new = C + 2.0 * delta * vB + delta * delta * bb[i]
            dlik = lk_i(theta_star, Y[i]) - lk_i(theta_cur, Y[i])
            dprior = -coef * (np.log(C_new / 2.0 + eps) - np.log(C / 2.0 + eps))
            lg_MH = dlik + dprior
            MH_rate = np.exp(np.clip(lg_MH, -700, 700))
            if np.random.uniform() < MH_rate:
                theta[i] = theta_star
                v = v + Bcol * delta         # O(n)
                C = C_new
                accept += 1
        theta_store[t, :] = theta
        alpha_vals = np.array([M.alpha_lk(theta, Y, g) for g in grid])
        alpha_est = grid[np.argmax(alpha_vals)]
        alpha_store[t] = alpha_est
    return {'theta': theta_store, 'accept': accept,
            'alpha_store': alpha_store,
            'alpha_mn': np.mean(alpha_store), 'alpha_md': np.median(alpha_store)}


from numba import njit


@njit(fastmath=False)
def _acmh_numba(T, ini, n, grid, alpha_est, u_0, B, Y, seed):
    """componentwise_mcmc_fast 와 동일 알고리즘을 JIT 컴파일.
    증분 C 갱신 O(n) + Welford 러닝분산 O(1). numba 자체 RNG(seed 지정)."""
    np.random.seed(seed)
    theta_store = np.zeros((T, n))
    for j in range(n):
        theta_store[0, j] = ini[j]
    bb = np.zeros(n)
    for j in range(n):
        s = 0.0
        for r in range(n):
            s += B[r, j] * B[r, j]
        bb[j] = s
    eps = 0.001
    coef = n / 2.0 + eps
    ng = len(grid)
    # Welford 러닝 평균/분산 (저장된 행들 0..t-1 기준)
    cnt = 0
    mean = np.zeros(n)
    M2 = np.zeros(n)
    # row 0 반영
    cnt = 1
    for j in range(n):
        mean[j] = theta_store[0, j]
    accept = 0
    alpha_store = np.zeros(T)
    alpha_store[0] = alpha_est

    for t in range(1, T):
        theta = theta_store[t - 1].copy()
        # v = B(theta-u_0), C = ||v||^2  (t당 1회)
        v = np.zeros(n)
        for r in range(n):
            acc = 0.0
            for j in range(n):
                acc += B[r, j] * (theta[j] - u_0[j])
            v[r] = acc
        C = 0.0
        for r in range(n):
            C += v[r] * v[r]

        for i in range(n):
            if t < 10:
                var_i = 1.0
            else:
                var_i = np.sqrt(2.4 * (M2[i] / (cnt - 1) + 0.01))
            theta_cur = theta[i]
            theta_star = np.random.normal(theta_cur, var_i)
            delta = theta_star - theta_cur
            vB = 0.0
            for r in range(n):
                vB += v[r] * B[r, i]
            C_new = C + 2.0 * delta * vB + delta * delta * bb[i]
            # likelihood 성분 i
            pi_c = 1.0 / (1.0 + np.exp(-theta_cur))
            pi_s = 1.0 / (1.0 + np.exp(-theta_star))
            tc = (1.0 - alpha_est) * pi_c
            ts = (1.0 - alpha_est) * pi_s
            if tc < 1e-10: tc = 1e-10
            if tc > 1 - 1e-10: tc = 1 - 1e-10
            if ts < 1e-10: ts = 1e-10
            if ts > 1 - 1e-10: ts = 1 - 1e-10
            lk_c = Y[i] * np.log(tc) + (1 - Y[i]) * np.log(1 - tc)
            lk_s = Y[i] * np.log(ts) + (1 - Y[i]) * np.log(1 - ts)
            dprior = -coef * (np.log(C_new / 2.0 + eps) - np.log(C / 2.0 + eps))
            lg_MH = (lk_s - lk_c) + dprior
            if lg_MH > 700.0: lg_MH = 700.0
            if lg_MH < -700.0: lg_MH = -700.0
            if np.random.random() < np.exp(lg_MH):
                theta[i] = theta_star
                for r in range(n):
                    v[r] += B[r, i] * delta
                C = C_new
                accept += 1
        for j in range(n):
            theta_store[t, j] = theta[j]
        # Welford: 방금 저장한 행 반영 (다음 t의 분산용)
        cnt += 1
        for j in range(n):
            d0 = theta[j] - mean[j]
            mean[j] += d0 / cnt
            M2[j] += d0 * (theta[j] - mean[j])
        # alpha 업데이트
        best_g = grid[0]
        best_lk = -1e300
        for gi in range(ng):
            g = grid[gi]
            s = 0.0
            for j in range(n):
                pi = 1.0 / (1.0 + np.exp(-theta[j]))
                tmp = (1.0 - g) * pi
                if tmp < 1e-10: tmp = 1e-10
                if tmp > 1 - 1e-10: tmp = 1 - 1e-10
                s += Y[j] * np.log(tmp) + (1 - Y[j]) * np.log(1 - tmp)
            if s > best_lk:
                best_lk = s
                best_g = g
        alpha_est = best_g
        alpha_store[t] = alpha_est
    return theta_store, accept, alpha_store


def componentwise_mcmc_numba(T, ini, n, grid, alpha_est, u_0, B, Y, seed):
    theta, accept, alpha_store = _acmh_numba(
        T, np.ascontiguousarray(ini, dtype=np.float64), n,
        np.ascontiguousarray(grid, dtype=np.float64), float(alpha_est),
        np.ascontiguousarray(u_0, dtype=np.float64),
        np.ascontiguousarray(B, dtype=np.float64),
        np.ascontiguousarray(Y, dtype=np.float64), int(seed))
    return {'theta': theta, 'accept': accept, 'alpha_store': alpha_store,
            'alpha_mn': np.mean(alpha_store), 'alpha_md': np.median(alpha_store)}


def eval_metrics(poster_pi, Y, truth):
    """FDR cutoff별 (pos, tp, realFDR, precision, recall, F) — γ별 dict."""
    pos, tp, realFDR = M.vec_FDR_cutoff(poster_pi, FDR_LEVELS, Y, np.array(truth))
    ntruth = len(truth)
    out = {}
    for j, g in enumerate(FDR_LEVELS):
        P = tp[j] / pos[j] if pos[j] > 0 else 0.0
        R = tp[j] / ntruth if ntruth > 0 else 0.0
        F = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
        out[g] = dict(pos=float(pos[j]), tp=float(tp[j]),
                      realFDR=float(realFDR[j]), precision=P, recall=R, F=F)
    return out


def run_doc(graph, sim_seed):
    """한 문서, 한 sim: 동일 Y로 두 샘플러 실행 + 평가."""
    n, truth = graph['n'], graph['truth']
    k = max(1, len(truth) // 2)           # 적응 k = floor(truth/2)

    rng = np.random.RandomState(sim_seed)
    # prior/그래프 행렬
    G = solve(graph['D'], graph['A'])
    B = np.eye(n) - d * G.T
    w = np.diag(1.0 / np.sqrt(np.diag(graph['D'])))
    B_star = np.eye(n) - d * w @ graph['A'] @ w
    # observed Y (동일 시드로 한 번만 샘플 → 두 샘플러 공유)
    Y = np.zeros(n)
    obs = list(rng.choice(truth, k, replace=False))
    Y[obs] = 1
    u_0 = solve(B, np.ones(n) * (1 - d))
    Base_Line = solve(B_star, Y)
    ini = M.base_to_start(Base_Line)
    alpha_est = M.alpha_find(u_0, Y, grid)

    # 두 샘플러 모두 동일 전역 RNG 상태에서 시작 (재현성)
    np.random.seed(sim_seed + 100000)
    aw = M.gibbs_mh(BURN_IN, T, ini, n, graph, Y, B, u_0, alpha_est, grid, verbose=False)
    pi_aw = aw['poster_pi_mn']

    cm = componentwise_mcmc_numba(T, ini, n, grid, alpha_est, u_0, B, Y, seed=sim_seed + 200000)
    prob = M.inv_logit(cm['theta'])
    pi_ac = np.mean(prob[BURN_IN:T, :], axis=0)

    return {
        'k': k, 'n': n, 'ntruth': len(truth),
        'AWSGLD': eval_metrics(pi_aw, Y, truth),
        'acMH': eval_metrics(pi_ac, Y, truth),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--probe', action='store_true')
    ap.add_argument('--verify', action='store_true')
    args = ap.parse_args()

    rows = [r for r in csv.reader(open(os.path.join(BASE, "doc_stats.csv")))][1:]
    ids = [r[0] for r in rows if 4 <= int(r[2]) <= 10]
    ids.sort(key=int)

    if args.verify:
        # 원본 componentwise_mcmc vs fast — 동일 시드/입력으로 수치 일치 검증
        g = build_graph(ids[len(ids)//2])
        n, truth = g['n'], g['truth']; k = max(1, len(truth)//2)
        rng = np.random.RandomState(7)
        G = solve(g['D'], g['A']); B = np.eye(n) - d*G.T
        w = np.diag(1.0/np.sqrt(np.diag(g['D']))); B_star = np.eye(n) - d*w@g['A']@w
        Y = np.zeros(n); Y[list(rng.choice(truth, k, replace=False))] = 1
        u_0 = solve(B, np.ones(n)*(1-d)); ini = M.base_to_start(solve(B_star, Y))
        ae = M.alpha_find(u_0, Y, grid)
        Tv = 300
        np.random.seed(999); slow = M.componentwise_mcmc(Tv, ini, n, grid, ae, u_0, B, Y, verbose=False)
        np.random.seed(999); fast = componentwise_mcmc_fast(Tv, ini, n, grid, ae, u_0, B, Y, verbose=False)
        d_theta = np.max(np.abs(slow['theta'] - fast['theta']))
        pis = M.inv_logit(slow['theta']); pif = M.inv_logit(fast['theta'])
        d_pi = np.max(np.abs(pis.mean(0) - pif.mean(0)))
        t0=time.time(); M.componentwise_mcmc(Tv, ini, n, grid, ae, u_0, B, Y, verbose=False); ts=time.time()-t0
        t0=time.time(); componentwise_mcmc_fast(Tv, ini, n, grid, ae, u_0, B, Y, verbose=False); tf=time.time()-t0
        print(f"[verify] n={n}, T={Tv}")
        print(f"  theta 최대 절대차: {d_theta:.2e}")
        print(f"  poster_pi 최대 절대차: {d_pi:.2e}")
        print(f"  accept slow={slow['accept']} fast={fast['accept']}")
        print(f"  속도: slow={ts:.2f}s  fast={tf:.2f}s  ({ts/tf:.1f}x 빠름)")
        return

    if args.probe:
        gid = build_graph(ids[len(ids)//2])  # 중앙값 크기 문서
        t0 = time.time()
        run_doc(gid, 42); run_doc(gid, 43)
        dt = (time.time() - t0) / 2
        total = dt * len(ids) * N_SIM
        print(f"[probe] n={gid['n']} 1-sim(두 샘플러) = {dt:.1f}s")
        print(f"[probe] 추정 총시간: {len(ids)}docs × {N_SIM}sim × {dt:.1f}s "
              f"= {timedelta(seconds=int(total))}")
        return

    results_path = os.path.join(OUTDIR, "acmh_vs_awsgld_4to10_results.jsonl")
    done = set()
    if os.path.exists(results_path):
        for line in open(results_path):
            try: done.add(json.loads(line)['id'])
            except: pass
    fout = open(results_path, "a")

    t_start = time.time()
    for di, doc_id in enumerate(ids):
        if doc_id in done:
            continue
        graph = build_graph(doc_id)
        sims = []
        for s in range(N_SIM):
            seed = hash((doc_id, s)) % (2**31)
            sims.append(run_doc(graph, seed))
        rec = {'id': doc_id, 'n': graph['n'], 'ntruth': len(graph['truth']), 'sims': sims}
        fout.write(json.dumps(rec) + "\n"); fout.flush()
        elapsed = time.time() - t_start
        eta = timedelta(seconds=int(elapsed / (di + 1) * (len(ids) - di - 1)))
        print(f"[{di+1}/{len(ids)}] doc {doc_id} (n={graph['n']}, truth={len(graph['truth'])}) "
              f"| elapsed {timedelta(seconds=int(elapsed))} | ETA {eta}", flush=True)
    fout.close()
    summarize(results_path)


def summarize(results_path):
    """샘플러별 γ=0.15 평균 성능 집계."""
    agg = {'AWSGLD': [], 'acMH': []}
    for line in open(results_path):
        rec = json.loads(line)
        for sim in rec['sims']:
            for m in ('AWSGLD', 'acMH'):
                agg[m].append(sim[m][str(GAMMA_MAIN)] if str(GAMMA_MAIN) in sim[m] else sim[m][GAMMA_MAIN])
    summary = {}
    for m in ('AWSGLD', 'acMH'):
        a = agg[m]
        keys = ['precision', 'recall', 'F', 'realFDR', 'pos', 'tp']
        summary[m] = {k: float(np.mean([x[k] for x in a])) for k in keys}
        summary[m]['n_obs'] = len(a)
    with open(os.path.join(OUTDIR, "acmh_vs_awsgld_4to10_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== 요약 (γ=0.15, doc×sim 평균) ===")
    print(f"{'metric':<10} {'AWSGLD':>10} {'acMH':>10}")
    for k in ['precision', 'recall', 'F', 'realFDR']:
        print(f"{k:<10} {summary['AWSGLD'][k]:>10.4f} {summary['acMH'][k]:>10.4f}")


if __name__ == "__main__":
    main()
