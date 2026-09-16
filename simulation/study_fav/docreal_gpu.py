"""문서 수(K) 스윕 — 실제 Hulth 문서 1/3/10개를 표준(단봉) BSS 에너지로 병합.

Study 3과의 차이: Study 3은 log-sum-exp mixture(다봉, 봉우리=문서별 중심)로 trap을 만든다.
여기서는 **병합 그래프 하나의 표준 단봉 에너지**를 쓴다 → 볼록·단봉, θ*(MAP) 존재.
문서 소스는 실제(합성 라벨 배정 아님). 최적화 예산은 AWSGLD Tε.

구성:
  Hulth 문서 K개 → 합집합 어휘로 통합(공유 stem=같은 노드), A=Σ_k A_k
  B = I − d·(D⁻¹A)ᵀ,  u0 = solve(B,(1−d)1)  ← 병합 그래프 TextRank(단일 중심 = 단봉)
  truth = 문서별 키워드 합집합,  Y = truth 의 floor(|truth|/2) 관측(PU)
  α = alpha_find(u0, Y),  σ²=1.0 고정,  U(θ)=−loglik+‖B(θ−u0)‖²/(2σ²)

지표:
  최적화(annealing T→0): min U / opt.gap(U−U*) / ‖θ−θ*‖ / Tε   (θ*=GD MAP)
  샘플링(T=1): γ-FDR(P/R/F1/realFDR) / Top-k / NDCG / AUC  (키워드 truth)
             + σ̂(사후폭) / Jaccard(재현성)                  (gold-free)
  ※ cov90·Spearman·Kendall·MSE 는 실데이터에 데이터생성 θ가 없어 정의 불가 → 제외(Study 3과 동일).

사용: python3 docreal_gpu.py <K> [nseed=5]
"""
import os, sys, csv, itertools, importlib.util
import numpy as np
import torch
from numpy.linalg import solve

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/home/jiyoon/BSS-Keyphrase-Extraction/code_JOC")
import keyphrase_functions_awsgld as kfa
_S2 = "/home/jiyoon/BSS-Keyphrase-Extraction/simulation/study_2"
spec = importlib.util.spec_from_file_location("exp", os.path.join(_S2, "acmh_vs_awsgld_4to10.py"))
exp = importlib.util.module_from_spec(spec); spec.loader.exec_module(exp)

DEV = torch.device("cuda"); DT = torch.float64; torch.set_default_dtype(DT)
K      = int(sys.argv[1]) if len(sys.argv) > 1 else 3
NSEED  = int(sys.argv[2]) if len(sys.argv) > 2 else 5
D_DAMP = 0.85; SIGMA2 = 1.0; ZETA = 1.0
M_REG = kfa.M_REGIONS; DECAY = kfa.DECAY_LR
METH = ["tMH", "SGLD", "qSGLD", "cycSGLD", "SGHMC", "AWSGLD"]
GAMMAS = [0.05, 0.1, 0.15, 0.2, 0.3]
T = 5000; BURN = 2500; THIN = 5; MAXB = 3000; EPS = 1.0
# Study 3 문서 조합(실데이터) — 비교 가능성 위해 동일 ID 사용
DOCSETS = {1: ["1994"], 3: ["1994", "212", "227"],
           10: ["1994", "212", "227", "233", "260", "1982", "2117", "206", "374", "403"]}
sigm = lambda x: 1 / (1 + np.exp(-np.clip(x, -700, 700)))

# 실문서 그래프 로드
G = {}
for dc in DOCSETS[K]:
    G[dc] = exp.build_graph(dc)


def build_merge(docs):
    """K개 문서 → 표준 단봉 병합 에너지 재료 (u0=병합 TextRank, 단일 중심)."""
    vocab = sorted(set().union(*[set(G[x]['words']) for x in docs]))
    idx = {w: i for i, w in enumerate(vocab)}; n = len(vocab)
    A = np.zeros((n, n))
    for x in docs:
        g = G[x]; wm = [idx[w] for w in g['words']]
        A[np.ix_(wm, wm)] += g['A']
    np.fill_diagonal(A, 0)
    keep = A.sum(1) > 0
    if not keep.all():
        A = A[np.ix_(keep, keep)]; vocab = [w for w, k in zip(vocab, keep) if k]
        idx = {w: i for i, w in enumerate(vocab)}; n = len(vocab)
    D = np.diag(A.sum(1)); B = np.eye(n) - D_DAMP * solve(D, A).T
    u0 = solve(B, np.ones(n) * (1 - D_DAMP))              # 병합 그래프 TextRank = 단일 중심(단봉)
    truth = sorted({idx[G[x]['words'][t]] for x in docs for t in G[x]['truth'] if G[x]['words'][t] in idx})
    return n, B, u0, truth, vocab


def make_data(docs, seed):
    n, B, u0, truth, vocab = build_merge(docs)
    rng = np.random.RandomState(20260916 + seed)
    Y = np.zeros(n)
    if truth:
        k = max(1, len(truth) // 2)                        # PU 관측 = truth 절반(study_2 규약)
        Y[list(rng.choice(truth, min(k, len(truth)), replace=False))] = 1.0
    a0 = float(kfa.alpha_find(u0, Y, kfa.grid))
    return n, B, u0, Y, a0, np.array(truth)


def make_torch(n, Bn, u0n, Yn):
    B = torch.tensor(Bn, device=DEV); u_0 = torch.tensor(u0n, device=DEV); Y = torch.tensor(Yn, device=DEV)
    BtB = B.T @ B; ridge = 1e-6 * torch.trace(BtB) / n
    P = torch.linalg.solve(BtB + ridge * torch.eye(n, device=DEV), torch.eye(n, device=DEV)); P = 0.5 * (P + P.T)
    Lc = torch.linalg.cholesky(P + 1e-10 * torch.eye(n, device=DEV))
    return B, u_0, Y, BtB, P, Lc


def energy_fns(B, u_0, Y, BtB, alpha, s2):
    def sig(x): return torch.clamp(torch.sigmoid(x), 1e-10, 1 - 1e-10)
    def gradU(th):
        pi = sig(th); tp = torch.clamp((1 - alpha) * pi, 1e-10, 1 - 1e-10); dn = torch.clamp(1 - tp, min=1e-10)
        gll = torch.where(Y == 1, 1 - pi, -(1 - alpha) * pi * (1 - pi) / dn)
        return -gll + (BtB @ (th - u_0)) / s2
    def U(th):
        pi = sig(th); tp = torch.clamp((1 - alpha) * pi, 1e-10, 1 - 1e-10)
        ll = (Y * torch.log(tp) + (1 - Y) * torch.log(1 - tp)).sum(); Bd = B @ (th - u_0)
        return -ll + (Bd @ Bd) / (2 * s2)
    return sig, gradU, U


# ---------------- 최적화 (annealing T→0, α·σ² 고정) ----------------
def opt_run(method, budget, u_0, P, Lc, BtB, gradU, U, n, Umin, ths):
    g = torch.Generator(device=DEV); g.manual_seed(0)
    theta = torch.randn(n, generator=g, device=DEV) * 1.5; v = torch.zeros(n, device=DEV)
    aw = torch.arange(1, M_REG + 1, device=DEV, dtype=DT) / M_REG; warm = 300; es = []; emin = du = None; J = M_REG - 1
    bU = float("inf"); bth = None; reach = None
    for t in range(budget):
        Tt = max(1e-4, np.exp(-t / 400))
        if method == "tMH":
            prop = theta + 0.25 * np.sqrt(Tt) * (Lc @ torch.randn(n, generator=g, device=DEV))
            dU = float(U(prop)) - float(U(theta))
            if np.log(float(torch.rand((), generator=g, device=DEV)) + 1e-300) < -dU / Tt: theta = prop
        else:
            gr = gradU(theta)
            if method == "SGLD":
                ek = 0.02 / ((t + 1) ** 0.6 + 10); theta = theta - ek * gr + np.sqrt(2 * ek * Tt) * torch.randn(n, generator=g, device=DEV)
            elif method == "qSGLD":
                ek = 0.3 / ((t + 1) ** 0.6 + 10); theta = theta - ek * (P @ gr) + np.sqrt(2 * ek * Tt) * (Lc @ torch.randn(n, generator=g, device=DEV))
            elif method == "SGHMC":
                eta = 0.01 / ((t + 1) ** 0.6 + 10); theta = theta + v; v = 0.9 * v - eta * gr + np.sqrt(2 * 0.1 * eta * Tt) * torch.randn(n, generator=g, device=DEV)
            elif method == "cycSGLD":
                cl = 500; be = (t % cl) / cl; ek = 0.005 * (np.cos(np.pi * min(be, 0.8)) + 1)
                theta = theta - ek * gr + np.sqrt(2 * Tt * ek) * torch.randn(n, generator=g, device=DEV)
            elif method == "AWSGLD":
                Uv = float(U(theta)); gm = 1.0
                if not np.isfinite(Uv): Uv = emin if emin is not None else 0.0
                if t < warm:
                    es.append(Uv)
                    if t == warm - 1:
                        lo, hi = min(es), max(es); rg = max(hi - lo, 1.0); emin = lo - 0.5 * rg; du = max((hi + 0.5 * rg - emin) / M_REG, 1e-8)
                else:
                    J = int(np.clip((Uv - emin) / du + 1, 1, M_REG - 1))
                    gm = float(np.clip(1 + (ZETA / du) * (np.log(float(aw[J]) + 1e-12) - np.log(float(aw[J - 1]) + 1e-12)), 0.1, 10.0))
                ek = 0.3 / ((t + 1) ** 0.6 + 10); theta = theta - ek * gm * (P @ gr) + np.sqrt(2 * ek * gm * Tt) * (Lc @ torch.randn(n, generator=g, device=DEV))
                if t >= warm:
                    dec = min(1.0, DECAY / ((t + 1) ** 0.75 + 1000)); cw = float(aw[J]); aw[J:] += dec * cw * (1 - aw[J:]); aw[:J] -= dec * cw * aw[:J]; aw = torch.clamp(aw, 1e-10, 1)
        theta = torch.clamp(theta, -700, 700)
        u = float(U(theta))
        if u < bU: bU = u; bth = theta.clone()
        if reach is None and u - Umin < EPS: reach = t
    dist = float(torch.linalg.norm(bth - ths))
    return bU, bU - Umin, dist, (reach if reach is not None else np.nan)


# ---------------- 샘플링 (T=1, 전 프로토콜; θ·α·σ² Langevin) ----------------
def samp_chain(method, seed_ci, n, B, u_0, Y, BtB, P, Lc, a0):
    g = torch.Generator(device=DEV); g.manual_seed(seed_ci)
    theta = torch.randn(n, generator=g, device=DEV) * 1.5
    aa = torch.log(torch.tensor(a0 / (1 - a0), device=DEV)); ss = torch.zeros((), device=DEV); v = torch.zeros(n, device=DEV)
    aw = torch.arange(1, M_REG + 1, device=DEV, dtype=DT) / M_REG; warm = 300; es = []; emin = du = None; J = M_REG - 1
    grid = torch.tensor(kfa.grid, device=DEV)
    def sig(x): return torch.clamp(torch.sigmoid(x), 1e-10, 1 - 1e-10)
    def energyU(th, alpha, s2):
        pi = sig(th); tp = torch.clamp((1 - alpha) * pi, 1e-10, 1 - 1e-10)
        ll = (Y * torch.log(tp) + (1 - Y) * torch.log(1 - tp)).sum(); Bd = B @ (th - u_0)
        return -ll + (Bd @ Bd) / (2 * s2)
    def gradU(th, alpha, s2):
        pi = sig(th); tp = torch.clamp((1 - alpha) * pi, 1e-10, 1 - 1e-10); dn = torch.clamp(1 - tp, min=1e-10)
        gll = torch.where(Y == 1, 1 - pi, -(1 - alpha) * pi * (1 - pi) / dn)
        return -gll + (BtB @ (th - u_0)) / s2
    def s2_gibbs(th):
        C = float(((B @ (th - u_0)) ** 2).sum())
        return float(np.clip(1.0 / np.random.gamma(n / 2 + 0.001, 1.0 / (C / 2 + 0.001)), 0.05, 100.0))
    def alpha_eb(th):
        pi = sig(th); tp = torch.clamp((1 - grid[:, None]) * pi[None, :], 1e-10, 1 - 1e-10)
        lk = (Y[None, :] * torch.log(tp) + (1 - Y)[None, :] * torch.log(1 - tp)).sum(1)
        return grid[torch.argmax(lk)]
    keep = []; wts = []
    for t in range(T):
        alpha = torch.sigmoid(aa); s2 = torch.exp(torch.clamp(ss, -6, 6))
        if method == "tMH":
            s2v = s2_gibbs(theta); a_eb = alpha_eb(theta)
            prop = theta + np.sqrt(4.0 / n) * torch.sqrt(torch.tensor(s2v)) * (Lc @ torch.randn(n, generator=g, device=DEV))
            dE = -energyU(prop, a_eb, s2v) + energyU(theta, a_eb, s2v)
            if torch.log(torch.rand((), generator=g, device=DEV)) < dE: theta = prop
            if t >= BURN and t % THIN == 0: keep.append(theta.clone()); wts.append(1.0)
            continue
        gth = gradU(theta, alpha, s2)
        if method == "SGLD":
            ek = 0.02 / ((t + 1) ** 0.6 + 10); theta = theta - ek * gth + np.sqrt(2 * ek) * torch.randn(n, generator=g, device=DEV)
        elif method == "qSGLD":
            ek = 0.3 / ((t + 1) ** 0.6 + 10); theta = theta - ek * (P @ gth) + np.sqrt(2 * ek) * (Lc @ torch.randn(n, generator=g, device=DEV))
        elif method == "cycSGLD":
            cl = max(1, T // 10); be = (t % cl) / cl; ek = 0.01 / 2 * (np.cos(np.pi * min(be, 0.8)) + 1); tk = 1.0 if be >= 0.8 else 1e-4
            theta = theta - ek * gth + np.sqrt(2 * tk * ek) * torch.randn(n, generator=g, device=DEV)
        elif method == "SGHMC":
            eta = 0.01 / ((t + 1) ** 0.6 + 10); theta = theta + v; v = 0.9 * v - eta * gth + np.sqrt(2 * 0.1 * eta) * torch.randn(n, generator=g, device=DEV)
        elif method == "AWSGLD":
            Uv = float(energyU(theta, alpha, s2)); gm = 1.0
            if not np.isfinite(Uv): Uv = float(emin) if emin is not None else 0.0
            if t < warm:
                es.append(Uv)
                if t == warm - 1: lo, hi = min(es), max(es); rg = max(hi - lo, 1.0); emin = lo - 0.5 * rg; du = max((hi + 0.5 * rg - emin) / M_REG, 1e-8)
            else:
                J = int(np.clip((Uv - emin) / du + 1, 1, M_REG - 1)); gm = float(np.clip(1 + (ZETA / du) * (np.log(float(aw[J]) + 1e-12) - np.log(float(aw[J - 1]) + 1e-12)), 0.1, 10.0))
            ek = 0.3 / ((t + 1) ** 0.6 + 10); theta = theta - ek * gm * (P @ gth) + np.sqrt(2 * ek * gm) * (Lc @ torch.randn(n, generator=g, device=DEV))
            if t >= warm: dec = min(1.0, DECAY / ((t + 1) ** 0.75 + 1000)); cw = float(aw[J]); aw[J:] += dec * cw * (1 - aw[J:]); aw[:J] -= dec * cw * aw[:J]; aw = torch.clamp(aw, 1e-10, 1)
        theta = torch.clamp(theta, -700, 700)
        pi = sig(theta); tt = torch.clamp((1 - alpha) * pi, 1e-10, 1 - 1e-10)
        gA = ((Y / (1 - alpha) - (1 - Y) * pi / (1 - tt)).sum() - 1 / alpha + 1 / (1 - alpha)) * alpha * (1 - alpha)
        C = ((B @ (theta - u_0)) ** 2).sum(); gS = -(C / 2 + 0.001) / s2 + (n / 2 + 0.001)
        eA = 0.2 * (100.0 / n) / ((t + 1) ** 0.6 + 10); eS = eA
        aa = torch.clamp(aa - eA * gA + np.sqrt(2 * eA) * torch.randn((), generator=g, device=DEV), -6, 6)
        ss = torch.clamp(ss - eS * gS + np.sqrt(2 * eS) * torch.randn((), generator=g, device=DEV), -6, 6)
        if t >= BURN and t % THIN == 0: keep.append(theta.clone()); wts.append(float(aw[J]) if method == "AWSGLD" else 1.0)
    return torch.stack(keep), torch.tensor(wts, device=DEV)


def samp_metrics(method, seed, n, B, u_0, Y, BtB, P, Lc, a0, truth):
    Ss = []; Ws = []
    for ci in range(4):
        S, w = samp_chain(method, 1000 * seed + ci, n, B, u_0, Y, BtB, P, Lc, a0); Ss.append(S); Ws.append(w)
    chmeans = []
    for ci in range(4):
        wc = Ws[ci]; mc = (wc[:, None] * Ss[ci]).sum(0) / wc.sum() if method == "AWSGLD" else Ss[ci].mean(0)
        chmeans.append(mc)
    allS = torch.cat(Ss, 0); allw = torch.cat(Ws, 0)
    if method == "AWSGLD":
        wn = allw / allw.sum(); mh = (wn[:, None] * allS).sum(0); sd = torch.median(torch.sqrt((wn[:, None] * (allS - mh) ** 2).sum(0)))
    else:
        mh = allS.mean(0); sd = torch.median(allS.std(0))
    mh_c = mh.cpu().numpy(); sd = float(sd)
    nt = len(truth); Tset = set(truth.tolist())
    chsets = [set(np.argsort(c.cpu().numpy())[::-1][:nt].tolist()) for c in chmeans]
    jp = [len(a & b) / len(a | b) for a, b in itertools.combinations(chsets, 2)]; jac = float(np.mean(jp)) if jp else 1.0
    pih = sigm(mh_c); order = np.argsort(-pih)
    def prf(sel):
        tp = len(set(sel.tolist()) & Tset); Pp = tp / max(len(sel), 1); Rr = tp / max(nt, 1)
        return Pp, Rr, (2 * Pp * Rr / (Pp + Rr) if Pp + Rr > 0 else 0)
    fdr = {}
    for gmm in GAMMAS:
        k = 1
        for kk in range(1, n + 1):
            if np.mean(1 - pih[order[:kk]]) > gmm: k = kk - 1; break
            k = kk
        sel = order[:max(k, 1)]; Pp, Rr, Fv = prf(sel)
        fdr[gmm] = (Pp, Rr, Fv, len(set(sel.tolist()) - Tset) / max(len(sel), 1), len(sel))
    pos = nt; neg = n - nt; tp = fp = 0; tpr = [0.]; fpr = [0.]
    for i in order:
        if i in Tset: tp += 1
        else: fp += 1
        tpr.append(tp / pos); fpr.append(fp / max(neg, 1))
    auc = float(np.trapezoid(tpr, fpr))
    topk = len(set(order[:nt].tolist()) & Tset) / max(nt, 1)
    rel = np.zeros(n); rel[truth] = 1; disc = 1 / np.log2(np.arange(2, 2 + nt)); ndcg = float((rel[order[:nt]] * disc).sum() / disc.sum())
    del Ss, Ws, allS; torch.cuda.empty_cache()
    return dict(sigma=sd, jaccard=jac, topk=topk, ndcg=ndcg, auc=auc, fdr=fdr)


def main():
    docs = DOCSETS[K]
    n0, _, _, truth0, _ = build_merge(docs)
    print(f"\n{'#'*70}\n실문서 문서수 K={K}  docs={docs}  n={n0}  truth={len(truth0)}  σ²={SIGMA2}  ({NSEED}시드)\n{'#'*70}")

    # ===== 최적화 (annealing) =====
    opt = {m: [] for m in METH}
    for s in range(NSEED):
        n, Bn, u0n, Yn, a0, truth = make_data(docs, s)
        B, u_0, Y, BtB, P, Lc = make_torch(n, Bn, u0n, Yn)
        _, gradU_o, U_o = energy_fns(B, u_0, Y, BtB, torch.tensor(a0, device=DEV), torch.tensor(SIGMA2, device=DEV))
        ths = u_0.clone()
        for _ in range(8000): ths = ths - 0.05 * gradU_o(ths)
        Umin = float(U_o(ths))
        awT = opt_run("AWSGLD", MAXB, u_0, P, Lc, BtB, gradU_o, U_o, n, Umin, ths)[3]
        budget = int(awT) if not np.isnan(awT) else MAXB - 1
        for m in METH:
            opt[m].append(opt_run(m, budget, u_0, P, Lc, BtB, gradU_o, U_o, n, Umin, ths))
        print(f"[opt seed {s}] budget={budget}", flush=True)
    print(f"\n[최적화 annealing T→0]  예산=AWSGLD Tε (시드별)")
    print(f"  {'method':>8} | {'min U':>16} | {'opt.gap':>14} | {'||θ-θ*||':>13}")
    opt_rows = []
    for m in sorted(METH, key=lambda mm: np.mean([r[1] for r in opt[mm]])):
        a = np.array([r[:3] for r in opt[m]])
        print(f"  {m:>8} | {a[:,0].mean():>8.2f}±{a[:,0].std():>5.2f} | {a[:,1].mean():>6.2f}±{a[:,1].std():>5.2f} | {a[:,2].mean():>6.2f}±{a[:,2].std():>4.2f}")
        opt_rows.append([K, n0, m, a[:,0].mean(), a[:,0].std(), a[:,1].mean(), a[:,1].std(), a[:,2].mean(), a[:,2].std()])

    # ===== 샘플링 (T=1) =====
    samp = {m: [] for m in METH}
    for s in range(NSEED):
        n, Bn, u0n, Yn, a0, truth = make_data(docs, s)
        B, u_0, Y, BtB, P, Lc = make_torch(n, Bn, u0n, Yn)
        for m in METH:
            samp[m].append(samp_metrics(m, s, n, B, u_0, Y, BtB, P, Lc, a0, truth))
        del B, u_0, Y, BtB, P, Lc; torch.cuda.empty_cache()
        print(f"[samp seed {s}] done", flush=True)

    def ms(vals, f="{:.3f}"):
        return f.format(np.mean(vals)) + "±" + f.format(np.std(vals))
    print(f"\n[샘플링 T=1]  Top-k / NDCG / AUC / σ̂ / Jaccard  (키워드 truth·gold-free)")
    print(f"  {'method':>8} | {'Top-k':>13} {'NDCG':>13} {'AUC':>13} {'sigma':>13} {'Jaccard':>13}")
    for m in METH:
        rs = samp[m]
        print(f"  {m:>8} | {ms([r['topk'] for r in rs]):>13} {ms([r['ndcg'] for r in rs]):>13} {ms([r['auc'] for r in rs]):>13} {ms([r['sigma'] for r in rs],'{:.2f}'):>13} {ms([r['jaccard'] for r in rs]):>13}")
    print(f"\n[샘플링 T=1]  γ별 FDR 제어  precision / recall / F1 / realFDR")
    samp_rows = []
    for gmm in GAMMAS:
        print(f"  γ={gmm}")
        print(f"  {'method':>8} | {'precision':>13} {'recall':>13} {'F1':>13} {'realFDR':>13}")
        for m in METH:
            P_ = [r['fdr'][gmm][0] for r in samp[m]]; R_ = [r['fdr'][gmm][1] for r in samp[m]]
            F_ = [r['fdr'][gmm][2] for r in samp[m]]; D_ = [r['fdr'][gmm][3] for r in samp[m]]
            print(f"  {m:>8} | {ms(P_):>13} {ms(R_):>13} {ms(F_):>13} {ms(D_):>13}")
            samp_rows.append([K, n0, gmm, m, np.mean(P_), np.mean(R_), np.mean(F_), np.mean(D_)])

    # CSV 저장
    with open(os.path.join(_HERE, f"docreal_opt_k{K}.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["K", "n", "method", "minU", "minU_std", "gap", "gap_std", "dist", "dist_std"]); w.writerows(opt_rows)
    with open(os.path.join(_HERE, f"docreal_samp_k{K}.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["K", "n", "gamma", "method", "precision", "recall", "F1", "realFDR"]); w.writerows(samp_rows)
    print(f"\n저장: docreal_opt_k{K}.csv, docreal_samp_k{K}.csv")


if __name__ == "__main__":
    main()
