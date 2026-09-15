"""재설계 시뮬레이션 — 표준 BSS(단봉) 위 full-protocol 샘플러 비교.
- 생성기: 일관 그룹-블록 SBM, 난이도(μ격차·α)·n 노브. 표준 에너지(혼합 없음).
- 샘플러: gradient 5종(SGLD/qSGLD/cycSGLD/SGHMC/AWSGLD) 전부 full(θ·σ²·α Langevin) + acMH(논문 MH-within-Gibbs baseline).
- 지표: 논문식 FDR@γ·P/R/F1·AUC·NDCG (gold=θ*>0 진짜키프레이즈) + 진단 R̂·ESS·σ/gold·cov90·Spearman·MSE.
사용: python3 redesign.py <n_seed> <n> <mu_gap> <alpha> [zeta]   → redesign_n{n}_mu{mu}_a{alpha}.csv
"""
import os, sys, csv
import numpy as np
from numpy.linalg import solve, inv, cholesky
from scipy.stats import invgamma, spearmanr, kendalltau
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/home/jiyoon/BSS-Keyphrase-Extraction/code_JOC")
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "study_a0"))
import keyphrase_functions_awsgld as kfa
import energy_diagnostics as E
import fav_conditions as F

NSEED = int(sys.argv[1]) if len(sys.argv) > 1 else 3
N     = int(sys.argv[2]) if len(sys.argv) > 2 else 100
MUGAP = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5
ALPHA = float(sys.argv[4]) if len(sys.argv) > 4 else 0.2
ZETA  = float(sys.argv[5]) if len(sys.argv) > 5 else 2.0
KDOC  = int(sys.argv[6]) if len(sys.argv) > 6 else 1          # >1이면 문서 KDOC개 데이터-병합(표준 에너지 유지)
SPARSE = int(sys.argv[7]) if len(sys.argv) > 7 else 0    # 1이면 관측 positive 75%% 제거(Sparse)
PCROSS = 0.004                                                # 문서 간 희소 연결
SIG = 0.4; PIN, POUT = 0.30, 0.02
GRAD = ["SGLD", "qSGLD", "cycSGLD", "SGHMC", "AWSGLD"]; METH = GRAD + ["acMH"]
GAMMAS = [0.05, 0.1, 0.15, 0.2, 0.3]
sigm = lambda x: 1 / (1 + np.exp(-np.clip(x, -700, 700)))
M_REG = kfa.M_REGIONS; DECAY = kfa.DECAY_LR
E.T = 5000; E.BURN = 500; E.BATCH = N  # acMH baseline용


def gen(seed):
    rng = np.random.default_rng(seed); z = rng.choice([0, 1, 2], N, p=[0.4, 0.3, 0.3])
    mu = np.array([MUGAP, 0.0, -MUGAP]); ts = mu[z] + SIG * rng.standard_normal(N)
    doc = rng.integers(0, KDOC, N) if KDOC > 1 else np.zeros(N, int)   # 문서 배정(데이터-병합)
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            if doc[i] == doc[j]:
                p = PIN if z[i] == z[j] else POUT       # 같은 문서: 그룹 SBM
            else:
                p = PCROSS                               # 다른 문서: 희소 연결(데이터 병합)
            if rng.random() < p: A[i, j] = A[j, i] = 1
    deg = A.sum(1); keep = deg > 0; A = A[np.ix_(keep, keep)]; ts = ts[keep]; z = z[keep]; n = len(ts); deg = A.sum(1)
    B = np.eye(n) - 0.85 * solve(np.diag(deg), A).T; u_0 = solve(B, np.ones(n) * 0.15)
    Y = (rng.random(n) < (1 - ALPHA) * sigm(ts)).astype(float)
    if SPARSE:
        pos = np.where(Y == 1)[0]; Y[pos[rng.random(len(pos)) < 0.75]] = 0.0   # 관측 positive 대량 누락
    a0 = kfa.alpha_find(u_0, Y, kfa.grid)
    graph = {"n": n, "A": A, "D": np.diag(deg)}
    truth = np.where(ts > 0)[0]                 # gold 키프레이즈 = 진짜 π*>0.5 (θ*>0)
    return n, graph, B, u_0, ts, z, Y, a0, truth


def sample_s2(theta, B, u_0, n):
    C = (B @ (theta - u_0)) @ (B @ (theta - u_0))
    return float(np.clip(invgamma.rvs(n / 2 + 0.001, scale=C / 2 + 0.001), 0.05, 100.0))


def half(s): return s[s.shape[0] // 2:]
def split_rhat(chs):
    seg = [s for c in chs for s in np.array_split(c, 2)]; m = np.array([s.mean(0) for s in seg])
    v = np.array([s.var(0, ddof=1) for s in seg]); L = min(s.shape[0] for s in seg)
    return np.sqrt(((L - 1) / L * v.mean(0) + m.var(0, ddof=1)) / np.maximum(v.mean(0), 1e-12))


def full_sampler(method, graph, Y, B, u_0, a0, ini, seed, P, Lc, BtB, n, Tm=5000):
    """θ(method별) + a=logit α + s=log σ² 전부 Langevin (full)."""
    np.random.seed(seed); theta = ini.copy(); aa = np.log(a0 / (1 - a0)); ss = 0.0; v = np.zeros(n)
    aw = np.arange(1, M_REG + 1, dtype=float) / M_REG; warm = 300; es = []; emin = du = None; J = M_REG - 1
    store = np.zeros((Tm, n)); wst = np.zeros(Tm)
    for t in range(Tm):
        alpha = sigm(aa); s2 = np.exp(np.clip(ss, -6, 6))
        gth = kfa.grad_posterior_energy(Y, alpha, theta, u_0, B, s2, BtB=BtB)
        # θ 업데이트 (method별)
        if method == "SGLD":
            ek = 0.02 / ((t + 1) ** 0.6 + 10); theta = theta - ek * gth + np.sqrt(2 * ek) * np.random.randn(n)
        elif method == "qSGLD":
            ek = 0.3 / ((t + 1) ** 0.6 + 10); theta = theta - ek * (P @ gth) + np.sqrt(2 * ek) * (Lc @ np.random.randn(n))
        elif method == "cycSGLD":
            cl = max(1, Tm // 10); be = (t % cl) / cl; ek = 0.01 / 2 * (np.cos(np.pi * min(be, 0.8)) + 1); tk = 1.0 if be >= 0.8 else 1e-4
            theta = theta - ek * gth + np.sqrt(2 * tk * ek) * np.random.randn(n)
        elif method == "SGHMC":
            eta = 0.01 / ((t + 1) ** 0.6 + 10); theta = theta + v; v = 0.9 * v - eta * gth + np.sqrt(2 * 0.1 * eta) * np.random.randn(n)
        elif method == "AWSGLD":
            Uv = kfa.posterior_energy(Y, alpha, theta, u_0, B, s2); gm = 1.0
            if t < warm:
                es.append(Uv)
                if t == warm - 1: lo, hi = min(es), max(es); rg = max(hi - lo, 1.0); emin = lo - 0.5 * rg; du = max((hi + 0.5 * rg - emin) / M_REG, 1e-8)
            else:
                J = int(np.clip((Uv - emin) / du + 1, 1, M_REG - 1)); gm = float(np.clip(1 + (ZETA / du) * (np.log(aw[J] + 1e-12) - np.log(aw[J - 1] + 1e-12)), 0.1, 10.0))
            ek = 0.3 / ((t + 1) ** 0.6 + 10)
            theta = theta - ek * gm * (P @ gth) + np.sqrt(2 * ek * gm) * (Lc @ np.random.randn(n))
            if t >= warm: dec = min(1.0, DECAY / ((t + 1) ** 0.75 + 1000)); cw = aw[J]; aw[J:] += dec * cw * (1 - aw[J:]); aw[:J] -= dec * cw * aw[:J]; aw = np.clip(aw, 1e-10, 1)
        theta = np.clip(theta, -700, 700)
        # a=logit α, s=log σ² Langevin (재매개변수화 + Jacobian)
        pi = sigm(theta); tt = np.clip((1 - alpha) * pi, 1e-10, 1 - 1e-10)
        gA = (np.sum(Y / (1 - alpha) - (1 - Y) * pi / (1 - tt)) - 1 / alpha + 1 / (1 - alpha)) * alpha * (1 - alpha)
        C = (B @ (theta - u_0)) @ (B @ (theta - u_0)); gS = -(C / 2 + 0.001) / s2 + (n / 2 + 0.001)
        eA = 0.2 / ((t + 1) ** 0.6 + 10); eS = 0.2 / ((t + 1) ** 0.6 + 10)
        aa = aa - eA * gA + np.sqrt(2 * eA) * np.random.randn(); ss = np.clip(ss - eS * gS + np.sqrt(2 * eS) * np.random.randn(), -6, 6)
        store[t] = theta; wst[t] = aw[J] if method == "AWSGLD" else 1.0
    return store, wst


def fdr_cutoff(pi, gamma):
    order = np.argsort(-pi); n = len(pi)
    for k in range(1, n + 1):
        sel = order[:k]; est = np.mean(1 - pi[sel])
        if est > gamma: k -= 1; break
    return order[:max(k, 1)]


def metrics(mh, ts, truth, n):
    Tset = set(truth.tolist()); nt = len(truth)
    def prf(sel):
        tp = len(set(sel) & Tset); P = tp / max(len(sel), 1); R = tp / max(nt, 1); Fv = 2 * P * R / (P + R) if P + R > 0 else 0
        return P, R, Fv
    pi = sigm(mh)
    fdr = {}
    for g in GAMMAS:
        sel = fdr_cutoff(pi, g); P, R, Fv = prf(sel); rf = np.mean([1 for i in sel if i not in Tset]) if len(sel) else 0
        fdr[g] = (P, R, Fv, len(set(sel) - Tset) / max(len(sel), 1), len(sel))
    order = np.argsort(-pi); pos = nt; neg = n - nt; tp = fp = 0; tpr = [0.]; fpr = [0.]
    for i in order:
        if i in Tset: tp += 1
        else: fp += 1
        tpr.append(tp / pos); fpr.append(fp / neg)
    auc = float(np.trapezoid(tpr, fpr))
    topk = len(set(order[:nt].tolist()) & Tset) / max(nt, 1)      # P@nt (=Top-k)
    rel = np.array([1.0 if i in Tset else 0.0 for i in order]); k = min(nt, n)
    disc = 1 / np.log2(np.arange(2, 2 + k)); dcg = (rel[:k] * disc).sum(); idcg = disc[:min(nt, k)].sum()
    ndcg = dcg / idcg if idcg > 0 else 0.0
    mse = float(np.mean((mh - ts) ** 2)); kd = kendalltau(mh, ts).correlation
    return fdr, auc, spearmanr(mh, ts).correlation, kd, topk, ndcg, mse


def run_seed(seed):
    n, graph, B, u_0, ts, z, Y, a0, truth = gen(seed); BtB = B.T @ B; ridge = 1e-6 * np.trace(BtB) / n
    P = solve(BtB + ridge * np.eye(n), np.eye(n)); P = 0.5 * (P + P.T); Lc = cholesky(P + 1e-10 * np.eye(n))
    # gold pMALA 제거: 3표 지표(FDR·순위·σ̂/Jaccard/cov90)는 θ*·raw 출력만으로 계산
    uz = sorted(np.unique(z), key=lambda gv: ts[z == gv].mean(), reverse=True)   # S/W/N
    grp = {i: (z == uz[i]) for i in range(3)} if len(uz) >= 3 else {i: (z == uz[min(i, len(uz) - 1)]) for i in range(3)}
    inits = [np.random.RandomState(7000 + seed * 10 + j).randn(n) * 1.5 for j in range(4)]
    rows = []
    for m in METH:
        chains = []; Ws = []
        for ci, ini in enumerate(inits):
            if m == "acMH":
                st = E.RUNNERS["acMH"](graph, Y, B, u_0, ini, a0, ci + seed * 10); chains.append(half(st)); Ws.append(None)
            else:
                st, w = full_sampler(m, graph, Y, B, u_0, a0, ini, ci + seed * 10, P, Lc, BtB, n); chains.append(half(st)); Ws.append(half(w))
        alls = np.concatenate(chains, 0)
        if m == "AWSGLD":
            w = np.concatenate(Ws, 0); wn = w / w.sum(); mh = (wn[:, None] * alls).sum(0); sd = np.median(np.sqrt((wn[:, None] * (alls - mh) ** 2).sum(0)))
        else:
            mh = alls.mean(0); sd = np.median(alls.std(0))
        Rarr = split_rhat(chains); ess = np.median([F.ess_1d(chains[0][:, j]) for j in range(0, n, max(1, n // 40))])
        pih = sigm(mh); thg = [float(mh[grp[g]].mean()) for g in range(3)]; pig = [float(pih[grp[g]].mean()) for g in range(3)]
        lowU = float(np.mean([E.energy_trace_common(c, Y, B, u_0, a0).min() for c in chains]))
        lo = np.quantile(alls, 0.05, 0); hi = np.quantile(alls, 0.95, 0); cov90 = float(np.mean((ts >= lo) & (ts <= hi))); ciw = float(np.mean(hi - lo))
        # 재현성: 4연쇄 각자 top-k(=truth수) 선택 → 쌍별 Jaccard 평균 (같은 데이터, 난수만 다름)
        import itertools as _it
        chsets = []
        for ci2, c in enumerate(chains):
            if m == "AWSGLD":
                wc = Ws[ci2]; wnc = wc / wc.sum(); mc = (wnc[:, None] * c).sum(0)
            else:
                mc = c.mean(0)
            chsets.append(set(np.argsort(mc)[::-1][:len(truth)].tolist()))
        jpairs = [len(a & b) / len(a | b) for a, b in _it.combinations(chsets, 2)]
        jac = float(np.mean(jpairs)) if jpairs else 1.0
        fdr, auc, sp, kd, topk, ndcg, mse = metrics(mh, ts, truth, n)
        base = [seed, m, float(np.median(Rarr)), float(np.quantile(Rarr, 0.95)), float(np.nanmax(Rarr)), ess,
                thg[0], thg[1], thg[2], pig[0], pig[1], pig[2], lowU,
                sp, kd, topk, ndcg, mse, auc, sd, cov90, jac]
        for g in GAMMAS: base += list(fdr[g])   # P,R,F,realFDR,nsel per gamma
        rows.append(base)
    return rows, len(truth)


def main():
    hdr = ["seed", "method", "rmed", "rq95", "rmax", "ess", "thS", "thW", "thN", "piS", "piW", "piN", "lowU",
           "spearman", "kendall", "topk", "ndcg", "mse", "auc", "sigma_raw", "cov90", "jaccard"]
    for g in GAMMAS: hdr += [f"P{g}", f"R{g}", f"F{g}", f"rFDR{g}", f"nsel{g}"]
    rows = []; nt = 0
    for s in range(NSEED):
        r, nt = run_seed(s); rows += r; print(f"[seed {s}] done (truth={nt})", flush=True)
    fn = os.path.join(_HERE, f"redesign_n{N}_mu{MUGAP}_a{ALPHA}_z{ZETA}.csv")
    with open(fn, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["#truth", nt, "n", N, "mu", MUGAP, "alpha", ALPHA, "zeta", ZETA]); w.writerow(hdr); w.writerows(rows)
    print(f"저장: {os.path.basename(fn)} ({len(rows)}행)")


if __name__ == "__main__":
    main()
