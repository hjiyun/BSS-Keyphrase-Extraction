"""최적화(annealing) 축 공정비교 — annealed acMH vs annealed SG-MCMC 5종. (tMH 미사용)

annealing 표(min U/opt.gap/π-RMS)에 acMH 를 공정하게 넣으려면 acMH 도 annealing 해야 한다.
→ 논문 componentwise MH 를 온도 T→0 로 어닐링(=시뮬레이티드 어닐링)하고 최저 U(θ) 추적.
SG-MCMC 5종도 동일 annealing·동일 예산. 모두 α·σ² 고정(=a0,1), θ만 최적화.
지표: min U/opt.gap/‖π−π*‖/π-RMS + 정확도(Spearman/Top-k/NDCG/MSE/AUC) + time. 예산=AWSGLD Tε.

사용: python3 opt_acmh_fair.py [n=1000] [nseed=3]
"""
import os, sys, time, numpy as np
from numpy.linalg import solve, inv, cholesky
from scipy.stats import spearmanr
sys.path.insert(0, "/home/jiyoon/BSS-Keyphrase-Extraction/code_JOC")
import keyphrase_functions_awsgld as kfa

N     = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
NSEED = int(sys.argv[2]) if len(sys.argv) > 2 else 3
MU    = float(sys.argv[3]) if len(sys.argv) > 3 else 1.2
ALPHA = float(sys.argv[4]) if len(sys.argv) > 4 else 0.35
SIG, PIN, POUT, S2 = 0.4, 0.30, 0.02, 1.0
M_REG = kfa.M_REGIONS; DECAY = kfa.DECAY_LR; ZETA = float(os.environ.get('ZETA',1.0)); MAXB = int(os.environ.get('MAXB', 3000)); EPS = 1.0
TAU = float(os.environ.get('TAU', 400.0))  # annealing 냉각 시간상수(전방법 동일 적용)
TMIN = float(os.environ.get('TMIN', 1e-4))  # annealing 온도 하한(잔여노이즈 바닥)
# 스텝사이즈(prefactor) — 환경변수 LR_* 로 오버라이드(튜닝). 기본값=기존 미튜닝 상수(하위호환)
LR_SGLD   = float(os.environ.get("LR_SGLD", 0.02))
LR_QSGLD  = float(os.environ.get("LR_QSGLD", 0.3))
LR_SGHMC  = float(os.environ.get("LR_SGHMC", 0.01))
LR_CYC    = float(os.environ.get("LR_CYC", 0.005))
LR_AWSGLD = float(os.environ.get("LR_AWSGLD", 0.3))
METH = ["SGLD", "qSGLD", "cycSGLD", "SGHMC", "AWSGLD", "acMH"]   # acMH 마지막(느림) → 앞의 빠른 것부터 실시간 출력
if os.environ.get("SGONLY"):                                     # SGONLY=1 → MH(acMH) 제외, SG-MCMC 5종만
    METH = [m for m in METH if m != "acMH"]
SP_KP  = bool(os.environ.get("SPARSEKP"))                        # 희소 키프레이즈: positive 그룹 비율 축소
SP_LAB = bool(os.environ.get("SPARSELAB"))                       # 희소 라벨: 관측 Y=1 의 75% 드롭(PU)
SPTAG  = os.environ.get("SPTAG", "")                             # 파일명 태그(변형 구분)
ZP = [0.15, 0.25, 0.60] if SP_KP else [0.4, 0.3, 0.3]
sigm = lambda x: 1 / (1 + np.exp(-np.clip(x, -700, 700)))


def gen(seed):
    rng = np.random.default_rng(seed); z = rng.choice([0, 1, 2], N, p=ZP)
    mu = np.array([MU, 0.0, -MU]); ts = mu[z] + SIG * rng.standard_normal(N)
    A = np.zeros((N, N))
    for i in range(N):
        pr = np.where(z == z[i], PIN, POUT); r = rng.random(N); r[:i + 1] = 1.0; A[i, r < pr] = 1.0
    A = np.maximum(A, A.T); np.fill_diagonal(A, 0); deg = A.sum(1); keep = deg > 0
    A = A[np.ix_(keep, keep)]; ts = ts[keep]; n = len(ts); deg = A.sum(1)
    B = np.eye(n) - 0.85 * solve(np.diag(deg), A).T; u0 = solve(B, np.ones(n) * 0.15)
    Y = (rng.random(n) < (1 - ALPHA) * sigm(ts)).astype(np.float64)
    if SP_LAB:                                                   # 관측 positive 75% 제거
        pos = np.where(Y == 1)[0]; Y[pos[rng.random(len(pos)) < 0.75]] = 0.0
    a0 = ALPHA if os.environ.get("TRUEALPHA") else float(kfa.alpha_find(u0, Y, kfa.grid))
    truth = np.where(ts > 0)[0]
    return n, B, u0, Y, a0, ts, truth


def make(n, B, u0, Y, a0):
    BtB = B.T @ B
    def U(t):
        pi = np.clip(sigm(t), 1e-10, 1 - 1e-10); tp = np.clip((1 - a0) * pi, 1e-10, 1 - 1e-10)
        return -(Y * np.log(tp) + (1 - Y) * np.log(1 - tp)).sum() + ((B @ (t - u0)) @ (B @ (t - u0))) / (2 * S2)
    def gradU(t):
        pi = np.clip(sigm(t), 1e-10, 1 - 1e-10); tp = np.clip((1 - a0) * pi, 1e-10, 1 - 1e-10); dn = np.clip(1 - tp, 1e-10, None)
        return -np.where(Y == 1, 1 - pi, -(1 - a0) * pi * (1 - pi) / dn) + (BtB @ (t - u0)) / S2
    return U, gradU, BtB


def sg_run(method, budget, n, B, u0, Y, a0, U, gradU, P, Lc, rng):
    theta = rng.standard_normal(n) * 1.5; v = np.zeros(n)
    aw = np.arange(1, M_REG + 1, dtype=float) / M_REG; warm = 300; es = []; emin = du = None; J = M_REG - 1
    bU = np.inf; bth = None; reach = None; Umin_ref = getattr(sg_run, "_umin", None)
    for t in range(budget):
        Tt = max(TMIN, np.exp(-t / TAU)); g = gradU(theta)
        if method == "SGLD":
            ek = LR_SGLD / ((t + 1) ** 0.6 + 10); theta = theta - ek * g + np.sqrt(2 * ek * Tt) * rng.standard_normal(n)
        elif method == "qSGLD":
            ek = LR_QSGLD / ((t + 1) ** 0.6 + 10); theta = theta - ek * (P @ g) + np.sqrt(2 * ek * Tt) * (Lc @ rng.standard_normal(n))
        elif method == "SGHMC":
            eta = LR_SGHMC / ((t + 1) ** 0.6 + 10); theta = theta + v; v = 0.9 * v - eta * g + np.sqrt(2 * 0.1 * eta * Tt) * rng.standard_normal(n)
        elif method == "cycSGLD":
            cl = 500; be = (t % cl) / cl; ek = LR_CYC * (np.cos(np.pi * min(be, 0.8)) + 1); theta = theta - ek * g + np.sqrt(2 * Tt * ek) * rng.standard_normal(n)
        elif method == "AWSGLD":
            Uv = U(theta); gm = 1.0
            if not np.isfinite(Uv): Uv = emin if emin is not None else 0.0
            if t < warm:
                es.append(Uv)
                if t == warm - 1: lo, hi = min(es), max(es); rg = max(hi - lo, 1.0); emin = lo - 0.5 * rg; du = max((hi + 0.5 * rg - emin) / M_REG, 1e-8)
            else:
                J = int(np.clip((Uv - emin) / du + 1, 1, M_REG - 1)); gm = float(np.clip(1 + (ZETA / du) * (np.log(aw[J] + 1e-12) - np.log(aw[J - 1] + 1e-12)), 0.1, 10.0))
            ek = LR_AWSGLD / ((t + 1) ** 0.6 + 10); theta = theta - ek * gm * (P @ g) + np.sqrt(2 * ek * gm * Tt) * (Lc @ rng.standard_normal(n))
            if t >= warm: dec = min(1.0, DECAY / ((t + 1) ** 0.75 + 1000)); cw = aw[J]; aw[J:] += dec * cw * (1 - aw[J:]); aw[:J] -= dec * cw * aw[:J]; aw = np.clip(aw, 1e-10, 1)
        theta = np.clip(theta, -700, 700); u = U(theta)
        if u < bU: bU = u; bth = theta.copy()
        if reach is None and Umin_ref is not None and u - Umin_ref < EPS: reach = t
    return bU, bth, reach


def acmh_run(budget, n, B, u0, Y, a0, U):
    """annealed componentwise MH (논문 componentwise + 온도 T→0 + 최저 U 추적)."""
    rng = np.random
    theta = rng.standard_normal(n) * 1.5; hist = [theta.copy()]
    bU = np.inf; bth = None; reach = None; Umin_ref = getattr(sg_run, "_umin", None)
    lp_cur = kfa.posterior_gibbstheta(Y, a0, theta, u0, B, 1.0)
    for t in range(budget):
        Tt = max(1e-4, np.exp(-t / 400))
        if t < 10:
            vars_i = np.ones(n)
        else:
            vars_i = np.sqrt(2.4 * (np.array(hist).var(0, ddof=1) + 0.01))   # sweep당 1회 벡터화
        for i in range(n):
            old = theta[i]; theta[i] = old + rng.normal(0, vars_i[i])
            lp_star = kfa.posterior_gibbstheta(Y, a0, theta, u0, B, 1.0)
            if np.log(rng.uniform() + 1e-300) < (lp_star - lp_cur) / Tt:
                lp_cur = lp_star
            else:
                theta[i] = old
        hist.append(theta.copy())
        u = U(theta)
        if u < bU: bU = u; bth = theta.copy()
        if reach is None and Umin_ref is not None and u - Umin_ref < EPS: reach = t
    return bU, bth, reach


def metric(bth, bU, Umin, ts, truth, n, pistar):
    gap = bU - Umin; pihat = sigm(bth)
    pidist = float(np.linalg.norm(pihat - pistar)); pirms = float(np.sqrt(np.mean((pihat - pistar) ** 2)))
    order = np.argsort(-pihat); nt = len(truth); Tset = set(truth.tolist())
    topk = len(set(order[:nt].tolist()) & Tset) / max(nt, 1)
    rel = np.zeros(n); rel[truth] = 1; disc = 1 / np.log2(np.arange(2, 2 + nt)); ndcg = float((rel[order[:nt]] * disc).sum() / disc.sum())
    spear = float(spearmanr(bth, ts).correlation); mse = float(np.mean((bth - ts) ** 2))
    pos = nt; neg = n - nt; tp = fp = 0; tpr = [0.]; fpr = [0.]
    for i in order:
        if i in Tset: tp += 1
        else: fp += 1
        tpr.append(tp / pos); fpr.append(fp / max(neg, 1))
    auc = float(np.trapezoid(tpr, fpr))
    return dict(minU=bU, gap=gap, pidist=pidist, pirms=pirms, spear=spear, topk=topk, ndcg=ndcg, mse=mse, auc=auc)


def main():
    R = {m: [] for m in METH}
    import csv as _csv
    csvfn = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"optacmh_n{N}_mu{MU}_a{ALPHA}" + (f"_{SPTAG}" if SPTAG else "") + "_seeds.csv")
    _new = not os.path.exists(csvfn)
    _cf = open(csvfn, "a", newline=""); _cw = _csv.writer(_cf)
    if _new: _cw.writerow(["seed", "method", "minU", "gap", "pidist", "pirms", "spear", "topk", "ndcg", "mse", "auc", "Te", "time"])
    for s in range(NSEED):
        n, B, u0, Y, a0, ts, truth = gen(s)
        U, gradU, BtB = make(n, B, u0, Y, a0)
        P = inv(BtB + 1e-6 * np.trace(BtB) / n * np.eye(n)); P = 0.5 * (P + P.T); Lc = cholesky(P + 1e-10 * np.eye(n))
        th = u0.copy()
        for _ in range(8000): th = th - 0.05 * gradU(th)
        Umin = U(th); pistar = sigm(th); sg_run._umin = Umin
        for m in METH:                                   # 예산=고정 MAXB(공정), Tε=EPS 도달 스텝(속도) 별도 기록
            t0 = time.time()
            if m == "acMH":
                np.random.seed(0); bU, bth, reach = acmh_run(MAXB, n, B, u0, Y, a0, U)
            else:
                bU, bth, reach = sg_run(m, MAXB, n, B, u0, Y, a0, U, gradU, P, Lc, np.random.default_rng(0))
            dt = time.time() - t0
            mm = metric(bth, bU, Umin, ts, truth, n, pistar); mm['time'] = dt
            mm['Te'] = reach if reach is not None else MAXB; R[m].append(mm)
            _cw.writerow([s, m, mm['minU'], mm['gap'], mm['pidist'], mm['pirms'], mm['spear'], mm['topk'], mm['ndcg'], mm['mse'], mm['auc'], mm['Te'], mm['time']]); _cf.flush()
            print(f"  [seed {s} {m:>7}] minU={mm['minU']:.1f} gap={mm['gap']:.2f} π-RMS={mm['pirms']:.4f} "
                  f"Spear={mm['spear']:.3f} Top-k={mm['topk']:.3f} AUC={mm['auc']:.3f} Tε={mm['Te']} t={mm['time']:.1f}s", flush=True)
        print(f"[seed {s}] done (n={n}, truth={len(truth)})", flush=True)

    def ms(m, k, f="{:.3f}"):
        v = [R[m][s][k] for s in range(NSEED)]; return f.format(np.mean(v)) + "±" + f.format(np.std(v))
    print(f"\n최적화 공정비교 (튜닝 LR, 고정예산 MAXB={MAXB})  n={N} μ±{MU} α={ALPHA}  {NSEED}시드"
          f"  [LR: SGLD={LR_SGLD} qSGLD={LR_QSGLD} cyc={LR_CYC} SGHMC={LR_SGHMC} AWSGLD={LR_AWSGLD}]")
    print(f"  {'method':>8} | {'min U':>13} {'opt.gap':>11} {'π-RMS':>10} {'Tε(속도)':>13} | {'Spearman':>11} {'Top-k':>11} {'NDCG':>11} {'AUC':>11} | {'time(s)':>11}")
    for m in sorted(METH, key=lambda mm: np.mean([R[mm][s]['gap'] for s in range(NSEED)])):
        print(f"  {m:>8} | {ms(m,'minU','{:.2f}'):>13} {ms(m,'gap','{:.2f}'):>11} {ms(m,'pirms','{:.4f}'):>10} {ms(m,'Te','{:.0f}'):>13} | {ms(m,'spear'):>11} {ms(m,'topk'):>11} {ms(m,'ndcg'):>11} {ms(m,'auc'):>11} | {ms(m,'time','{:.1f}'):>11}")


if __name__ == "__main__":
    main()
