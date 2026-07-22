"""논문 §1.5.2 확장: SemEval-2010 장문 논문 5편(C-42 포함, 나머지 랜덤) acMH vs AWSGLD.

논문 §1.5.2 설정 그대로:
  - 전처리: 스테밍 완료 텍스트 + 빈도필터(2회 이하 단어 제거)
  - 관측 Y=1: 제목에서 유래한 키워드 (title ∩ truth)  ← 랜덤 아님, 결정적
  - 정답: reader 키워드
  - 그래프: window=2 동시출현
평가: γ별 FDR-cutoff(P/R/F/actualFDR) + top-k(k=정답수) + ROC AUC. 시드=MCMC 무작위성만.
"""
import os, sys, json, csv, time, collections
import numpy as np
sys.path.insert(0, "/home/jiyoon/BSS-Keyphrase-Extraction/code_JOC")
import keyphrase_functions_awsgld as M
import importlib.util
spec = importlib.util.spec_from_file_location("exp", "acmh_vs_awsgld_4to10.py")
exp = importlib.util.module_from_spec(spec); spec.loader.exec_module(exp)
from numpy.linalg import solve

M.ZETA = 5.0          # AWSGLD 기본값(튜닝 없음)
d = exp.d; grid = exp.grid; T, BURN = exp.T, exp.BURN_IN
LEVELS = exp.FDR_LEVELS; NSEED = 10; MINCOUNT = 3   # 논문: 2회 이하 제거 → 3회 이상 유지
DATA = "/home/jiyoon/BSS-Keyphrase-Extraction/data_JOC"
TITLES = json.load(open(os.path.join(DATA, "semeval_titles.json")))

def load_doc(pid):
    """논문 §1.5.2 전처리로 그래프 + 정답 + 관측 구성."""
    txt = open(os.path.join(DATA, "pre_process", f"{pid}.txt.final")).read()
    cnt = collections.Counter(txt.split())
    kept = {w for w, c in cnt.items() if c >= MINCOUNT}
    ftxt = " ".join(w for w in txt.split() if w in kept)     # 빈도필터
    fcm, words, w2i = M.create_fcm_words(ftxt, window=2)
    n = len(words)
    A = fcm.copy(); np.fill_diagonal(A, 0)
    D = np.diag(A.sum(axis=1))
    kw = set()
    for ph in open(os.path.join(DATA, "pre_process_reader_truth", pid)).read().strip().split(','):
        kw.update(ph.strip().split())
    truth = sorted({w2i[w] for w in kw if w in w2i})                       # 정답(그래프 내)
    obs = sorted({w2i[w] for w in TITLES[pid]['title_stems'] if w in w2i and w2i[w] in truth})  # 제목 유래 관측
    return dict(n=n, A=A, D=D, truth=truth, obs=obs, words=words,
                kw_total=len(kw), lost=len(kw) - len(truth))

# ---- 문서 선정: C-42 + 랜덤 4편 (관측>=2, 정답>=5) ----
cands = []
for pid in sorted(TITLES):
    tf = os.path.join(DATA, "pre_process_reader_truth", pid)
    if not os.path.exists(tf) or pid == "C-42": continue
    cands.append(pid)
rng = np.random.RandomState(20260711)
picked = ["C-42"]
for pid in rng.permutation(cands):
    if len(picked) >= 5: break
    try:
        g = load_doc(pid)
        if len(g['obs']) >= 2 and len(g['truth']) >= 5 and g['n'] <= 400: picked.append(pid)
    except Exception:
        continue
print(f"선정 5편: {picked}\n", flush=True)

graphs = {}
for pid in picked:
    g = load_doc(pid); graphs[pid] = g
    print(f"  {pid}: n={g['n']:>3}, 정답={len(g['truth']):>2}(원본 {g['kw_total']}, 그래프밖 {g['lost']}), "
          f"관측={len(g['obs'])} ({len(g['obs'])/max(len(g['truth']),1)*100:.0f}%)", flush=True)

def auc_roc(pi, truth, n):
    Tt = set(truth); order = np.argsort(-pi); pos = len(Tt); neg = n - pos
    tp = fp = 0; tpr = [0.]; fpr = [0.]
    for i in order:
        if i in Tt: tp += 1
        else: fp += 1
        tpr.append(tp / pos); fpr.append(fp / neg)
    return float(np.trapezoid(tpr, fpr))

FM = ('precision', 'recall', 'F', 'realFDR'); SAMP = ('acMH', 'AWSGLD')
per = {p: {s: {g: {m: [] for m in FM} for g in LEVELS} for s in SAMP} for p in picked}
ind = {p: {s: {'topk': [], 'AUC': []} for s in SAMP} for p in picked}
allg = {s: {g: {m: [] for m in FM} for g in LEVELS} for s in SAMP}
alli = {s: {'topk': [], 'AUC': []} for s in SAMP}
t0 = time.time()
for pid in picked:
    g = graphs[pid]; n = g['n']; truth = g['truth']
    G = solve(g['D'], g['A']); B = np.eye(n) - d * G.T
    w = np.diag(1.0 / np.sqrt(np.diag(g['D']))); Bs = np.eye(n) - d * w @ g['A'] @ w
    Y = np.zeros(n); Y[g['obs']] = 1                     # 제목 유래 관측 (고정)
    u_0 = solve(B, np.ones(n) * (1 - d))
    ini = M.base_to_start(solve(Bs, Y)); a = M.alpha_find(u_0, Y, grid)
    for s in range(NSEED):
        cm = exp.componentwise_mcmc_numba(T, ini, n, grid, a, u_0, B, Y, seed=s + 200000)
        pi = {'acMH': np.mean(M.inv_logit(cm['theta'])[BURN:T, :], axis=0)}
        np.random.seed(s + 100000)
        pi['AWSGLD'] = M.gibbs_mh(BURN, T, ini, n, g, Y, B, u_0, a, grid, verbose=False)['poster_pi_mn']
        for nm in SAMP:
            ev = exp.eval_metrics(pi[nm], Y, truth)
            for gg in LEVELS:
                for m in FM:
                    per[pid][nm][gg][m].append(ev[gg][m]); allg[nm][gg][m].append(ev[gg][m])
            nk = len(truth)
            tk = float(np.isin(np.argsort(-pi[nm])[:nk], truth).sum() / nk)
            au = auc_roc(pi[nm], truth, n)
            ind[pid][nm]['topk'].append(tk); ind[pid][nm]['AUC'].append(au)
            alli[nm]['topk'].append(tk); alli[nm]['AUC'].append(au)
    print(f"  {pid} 완료 | {int(time.time()-t0)}s", flush=True)

ms = lambda v: f"{np.mean(v):.3f}({np.std(v):.3f})"
print(f"\n=== SemEval 장문 5편 (논문 §1.5.2 설정, 제목 유래 관측) γ별 FDR-cutoff — {NSEED}시드 pooled ===")
print(f"{'γ':>5} {'샘플러':>8} | {'Precision':>13} {'Recall':>13} {'F':>13} {'actualFDR':>13}")
print("-" * 76)
for gg in LEVELS:
    for s in SAMP:
        r = allg[s][gg]
        print(f"{gg:>5} {s:>8} | {ms(r['precision']):>13} {ms(r['recall']):>13} {ms(r['F']):>13} {ms(r['realFDR']):>13}")
    print("-" * 76)
print(f"\n=== γ무관 ===")
print(f"{'샘플러':>8} | {'top-k':>13} {'ROC AUC':>13}")
for s in SAMP:
    print(f"{s:>8} | {ms(alli[s]['topk']):>13} {ms(alli[s]['AUC']):>13}")
print(f"\n=== 문서별 F(γ=0.20) / AUC ===")
print(f"{'문서':>6} {'n':>4} {'kw':>3} {'obs':>4} | {'acMH F':>13} {'AWS F':>13} | {'acMH AUC':>13} {'AWS AUC':>13}")
for pid in picked:
    g = graphs[pid]
    print(f"{pid:>6} {g['n']:>4} {len(g['truth']):>3} {len(g['obs']):>4} | "
          f"{ms(per[pid]['acMH'][0.2]['F']):>13} {ms(per[pid]['AWSGLD'][0.2]['F']):>13} | "
          f"{ms(ind[pid]['acMH']['AUC']):>13} {ms(ind[pid]['AWSGLD']['AUC']):>13}")

with open("semeval_long5.csv", "w", newline="") as f:
    wr = csv.writer(f); wr.writerow(["scope", "gamma", "sampler", "precision", "recall", "F", "realFDR"])
    for gg in LEVELS:
        for s in SAMP:
            wr.writerow(["ALL", gg, s] + [round(np.mean(allg[s][gg][m]), 4) for m in FM])
    for s in SAMP:
        wr.writerow(["ALL", "topk/auc", s, round(np.mean(alli[s]['topk']), 4), round(np.mean(alli[s]['AUC']), 4), "", ""])
    for pid in picked:
        for gg in LEVELS:
            for s in SAMP:
                wr.writerow([pid, gg, s] + [round(np.mean(per[pid][s][gg][m]), 4) for m in FM])
print("\n저장: semeval_long5.csv")
