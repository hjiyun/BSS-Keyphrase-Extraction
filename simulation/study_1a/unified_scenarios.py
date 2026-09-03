"""Study 1A 난이도별 통합표 — study_a0/1b/mixture 와 동일한 지표 세트 + R̂ 조기종료.

각 시나리오(Easy/Moderate/Difficult/Sparse)마다 4연쇄×seed0 단일 setup.
지표: Spearman / MSE_all / NDCG@50 | split-R̂ (median/q95/max) | ESS(4연쇄 mean)
     | Lowest U / Reached (cutoff = AWSGLD 정상상태 U).  AWSGLD 는 R̂·복원에 π-가중.
조기종료: BURN 이후 CHUNK 마다 split-R̂ 재계산 → R̂max < STOP_RHAT 이면 그 지점에서 정지
         (수렴하면 반복 낭비 안 함). 미수렴 샘플러는 T_MAX 까지.
데이터 생성=1A(build_block_graph 등), 에너지=단일 BSS(kfa), 샘플러는 continuable 인라인.

출력: unified_scenarios.csv + 콘솔.
"""
import os, sys, time, csv
import numpy as np
from scipy.stats import spearmanr, kendalltau, invgamma
from scipy.linalg import solve

_HERE = os.path.dirname(os.path.abspath(__file__))
_A0 = os.path.join(os.path.dirname(_HERE), "study_a0")
for p in (_HERE, _A0, os.path.join(_A0, "_archive")):
    sys.path.insert(0, p)
import langevin_methods_comparison as LMC     # noqa: E402
import energy_diagnostics as E                # alpha_find/GRID/energy_trace_common  # noqa: E402
import keyphrase_functions_awsgld as kfa      # noqa: E402
from extra_metrics import ess_per_node        # noqa: E402

# ── 설정 ──
N = 100; BURN = 1000; BATCH = 50; FLOOR = 1.0
MIN_T = 3000; CHUNK = 1000; T_MAX = 20000; STOP_RHAT = 1.05   # 조기종료: AWSGLD R̂median<1.05 까지 (T_MAX 상한)
SEED = 0
M_REG = kfa.M_REGIONS; ZETA = kfa.ZETA; TAU = kfa.TAU; DECAY = kfa.DECAY_LR
SGLD_LR = 0.02; QSGLD_LR = 0.3; CYCSGLD_LR = 0.01; CYC_CYCLES = 10
SGHMC_LR = 0.01; SGHMC_FRICTION = 0.1
METHODS = ["acMH", "SGLD", "qSGLD", "cycSGLD", "SGHMC", "AWSGLD"]


def gm_ts(a, z, g):
    return float(np.mean(a[z == g]))


def gen_scenario(sc_cfg, seed):
    LMC.BLOCK_PROBS = sc_cfg["block_probs"]
    scen = dict(sc_cfg["scenario"]); scen["n_total"] = N
    rng = np.random.default_rng(seed)
    graph = LMC.build_block_graph(scen, rng)
    ts = LMC.sample_theta_star(graph["group"], scen, rng)
    Y, _ = LMC.generate_labels(ts, scen["alpha_true"], rng)
    B = LMC.build_B(graph)
    u_0 = solve(B, np.full(N, 1.0 - LMC.DAMPING))
    a0 = E.alpha_find(u_0, Y, E.GRID)
    return graph, Y.astype(float), B, u_0, ts, scen, a0


# ── continuable 샘플러: state 딕셔너리 + advance(k) ──
def new_state(method, ini, ctx, seed):
    np.random.seed(seed)
    st = dict(method=method, theta=ini.copy(), t=0, alpha=ctx["a0"],
              ths=np.zeros((T_MAX, N)), Js=np.full(T_MAX, -1), v=np.zeros(N))
    if method == "AWSGLD":
        st["aw"] = np.arange(1, M_REG + 1, dtype=float) / M_REG
        st["warm"] = 300; st["es"] = []; st["emin"] = None; st["du"] = None; st["J"] = M_REG - 1
    return st


def advance(st, k, ctx):
    Y, B, u_0, BtB, P, Lc = ctx["Y"], ctx["B"], ctx["u_0"], ctx["BtB"], ctx["P"], ctx["Lc"]
    m = st["method"]; n = N
    for _ in range(k):
        if st["t"] >= T_MAX:
            break
        t = st["t"]; theta = st["theta"]; a = st["alpha"]
        C = (B @ (theta - u_0)) @ (B @ (theta - u_0))
        s2 = max(invgamma.rvs(n / 2 + 0.001, scale=C / 2 + 0.001), FLOOR)
        if m == "acMH":                                   # 블록 MH (BtB⁻¹ 전제곱)
            star = theta + ctx["McholScaled"](s2) @ np.random.randn(n)
            Uc = kfa.posterior_energy(Y, a, theta, u_0, B, s2)
            Us = kfa.posterior_energy(Y, a, star, u_0, B, s2)
            if np.log(np.random.rand() + 1e-300) < (-Us + Uc):
                theta = star
        else:
            bidx = np.random.choice(n, BATCH, replace=False) if BATCH < n else None
            gU = kfa.grad_posterior_energy(Y, a, theta, u_0, B, s2, batch_idx=bidx, BtB=BtB)
            if m == "SGLD":
                eps = SGLD_LR / ((t + 1) ** 0.6 + 10)
                theta = theta - eps * gU + np.sqrt(2 * TAU * eps) * np.random.randn(n)
            elif m == "qSGLD":
                eps = QSGLD_LR / ((t + 1) ** 0.6 + 10)
                theta = theta - eps * (P @ gU) + np.sqrt(2 * TAU * eps) * (Lc @ np.random.randn(n))
            elif m == "cycSGLD":
                cl = max(1, T_MAX // CYC_CYCLES); beta = (t % cl) / cl
                eps = CYCSGLD_LR / 2 * (np.cos(np.pi * min(beta, 0.8)) + 1)
                tk = TAU if beta >= 0.8 else TAU / 1e4
                theta = theta - eps * gU + np.sqrt(2 * tk * eps) * np.random.randn(n)
            elif m == "SGHMC":
                eta = SGHMC_LR / ((t + 1) ** 0.6 + 10); theta = theta + st["v"]
                st["v"] = (1 - SGHMC_FRICTION) * st["v"] - eta * gU + np.sqrt(2 * SGHMC_FRICTION * eta * TAU) * np.random.randn(n)
            elif m == "AWSGLD":
                U = kfa.posterior_energy(Y, a, theta, u_0, B, s2); gm = 1.0
                if t < st["warm"]:
                    st["es"].append(U)
                    if t == st["warm"] - 1:
                        lo, hi = min(st["es"]), max(st["es"]); rg = max(hi - lo, 1.0)
                        st["emin"] = lo - 0.5 * rg; st["du"] = max((hi + 0.5 * rg - st["emin"]) / M_REG, 1e-8); st["es"] = None
                else:
                    st["J"] = int(np.clip((U - st["emin"]) / st["du"] + 1, 1, M_REG - 1))
                    aw = st["aw"]; J = st["J"]
                    gm = float(np.clip(1 + (ZETA * TAU / st["du"]) * (np.log(aw[J] + 1e-12) - np.log(aw[J - 1] + 1e-12)), 0.1, 10.0))
                theta = theta - eps_awsgld(t) * gm * (P @ gU) + np.sqrt(2 * TAU * eps_awsgld(t)) * (Lc @ np.random.randn(n))
                if t >= st["warm"]:
                    dec = min(1.0, DECAY / ((t + 1) ** 0.75 + 1000)); aw = st["aw"]; J = st["J"]; cw = aw[J]
                    aw[J:] = aw[J:] + dec * cw * (1 - aw[J:]); aw[:J] = aw[:J] - dec * cw * aw[:J]; st["aw"] = np.clip(aw, 1e-10, 1)
                st["Js"][t] = st["J"]
        theta = np.clip(theta, -700, 700)
        st["theta"] = theta; st["ths"][t] = theta
        st["alpha"] = E.alpha_find(theta, Y, E.GRID); st["t"] = t + 1


def eps_awsgld(t):
    return 0.3 / ((t + 1) ** 0.6 + 10)


def ndcg_at_k(ts, th, k):
    rel = np.argsort(np.argsort(ts)).astype(float) / max(len(ts) - 1, 1)
    pred = np.argsort(th)[::-1][:k]; ideal = np.argsort(ts)[::-1][:k]
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    d = np.sum((2 ** rel[pred] - 1) * disc); i = np.sum((2 ** rel[ideal] - 1) * disc)
    return float(d / i) if i > 0 else 0.0


def split_rhat_coords(posts, wts):
    hm = []; hv = []; hL = None
    for ci, post in enumerate(posts):
        L = post.shape[0]; h = L // 2; hL = h
        for sl in (slice(0, h), slice(h, 2 * h)):
            seg = post[sl]
            if wts[ci] is not None:
                w = wts[ci][sl]; sw = w.sum()
                m = (w[:, None] * seg).sum(0) / sw; v = (w[:, None] * (seg - m) ** 2).sum(0) / sw
            else:
                m = seg.mean(0); v = seg.var(0, ddof=1)
            hm.append(m); hv.append(v)
    hm = np.array(hm); hv = np.array(hv); M2 = len(hm)
    gm = hm.mean(0); Bo = ((hm - gm) ** 2).sum(0) / (M2 - 1); W = hv.mean(0)
    return np.sqrt(np.clip(((hL - 1) / hL * W + Bo) / np.maximum(W, 1e-12), 0, None))


def collect(states):
    """현재까지 post-burn 궤적 + (AWSGLD면) π-가중치 리스트."""
    posts = []; wts = []
    for st in states:
        nt = st["t"]; post = st["ths"][BURN:nt]
        posts.append(post)
        if st["method"] == "AWSGLD":
            Jp = st["Js"][BURN:nt]; w = st["aw"][Jp].copy(); w[Jp >= M_REG - 1] = 0.0; wts.append(w)
        else:
            wts.append(None)
    return posts, wts


def run_method(method, ctx, target=None):
    """target=None → R̂max<STOP_RHAT 조기종료(반환 states, Tstop).
       target=정수 → 그 스텝까지만 실행(반환 states, target). 전역 정지점 공유용."""
    INITS = ctx["INITS"]
    states = [new_state(method, ctx["mkini"](ci, val), ctx, seed=100 * SEED + ci)
              for ci, val in enumerate(INITS)]
    if target is not None:
        for st in states:
            advance(st, target, ctx)
        return states, target
    for st in states:
        advance(st, MIN_T, ctx)
    while True:
        posts, wts = collect(states)
        R = split_rhat_coords(posts, wts); rmed = float(np.median(R))
        if rmed < STOP_RHAT or states[0]["t"] >= T_MAX:
            return states, states[0]["t"]
        for st in states:
            advance(st, CHUNK, ctx)


def run_scenario(sc_cfg):
    graph, Y, B, u_0, ts, scen, a0 = gen_scenario(sc_cfg, SEED); n = N
    z = np.array([str(g) for g in graph["group"]])
    BtB = B.T @ B; ridge = 1e-6 * np.trace(BtB) / n
    P = np.linalg.solve(BtB + ridge * np.eye(n), np.eye(n)); P = 0.5 * (P + P.T)
    Lc = np.linalg.cholesky(P + 1e-10 * np.eye(n))
    BtB_inv = np.linalg.solve(BtB + 1e-8 * np.eye(n), np.eye(n))

    def McholScaled(s2):
        return np.linalg.cholesky(BtB_inv * s2 * 4.0 / n + 1e-10 * np.eye(n))
    INITS = [scen["mu_N"], scen["mu_W"], scen["mu_S"], "rand"]

    def mkini(ci, val):
        return (np.random.RandomState(7000 + 100 * SEED + ci).randn(n) * 1.5 if val == "rand" else np.full(n, float(val)))
    ctx = dict(Y=Y, B=B, u_0=u_0, BtB=BtB, P=P, Lc=Lc, a0=a0, BtB_inv=BtB_inv,
               McholScaled=McholScaled, INITS=INITS, mkini=mkini)

    name = scen["name"].replace("Controlled", "").replace("_v2_OptB", "").replace("_v2", "")
    print(f"\n[{name}]  μ(S/W/N)={scen['mu_S']}/{scen['mu_W']}/{scen['mu_N']} σ={scen['sigma_theta']} α={scen['alpha_true']}", flush=True)
    t0 = time.time(); D = {}
    # AWSGLD 를 먼저 조기종료로 돌려 전역 정지점 T_conv 결정 → 나머지는 그 지점까지만
    aw_states, T_conv = run_method("AWSGLD", ctx, target=None)
    print(f"    AWSGLD 수렴점 T_conv={T_conv} → 나머지도 여기서 정지", flush=True)
    states_by = {"AWSGLD": aw_states}
    for m in METHODS:
        if m == "AWSGLD":
            continue
        if m == "acMH":            # 원본 componentwise acMH (프로젝트 표준, E.run_acmh) 를 T_conv 까지
            E.T = T_conv; E.BURN = BURN; E.BATCH = BATCH
            sts = []
            for ci, val in enumerate(INITS):
                np.random.seed(100 * SEED + ci)
                ths = E.run_acmh(graph, Y, B, u_0, mkini(ci, val), a0, 100 * SEED + ci)
                sts.append(dict(method="acMH", ths=ths, t=T_conv))
            states_by[m] = sts
        else:
            states_by[m], _ = run_method(m, ctx, target=T_conv)
    for m in METHODS:
        states = states_by[m]; Tstop = T_conv
        posts, wts = collect(states)
        pn = np.zeros(n); pd = 0.0; ess_list = []; minU = []; statU = []
        for ci, st in enumerate(states):
            post = posts[ci]
            if wts[ci] is not None:
                sw = wts[ci].sum(); pn += (wts[ci][:, None] * post).sum(0); pd += sw
            else:
                pn += post.sum(0); pd += post.shape[0]
            ess_list.append(float(np.nanmedian(ess_per_node(post))))
            Utr = E.energy_trace_common(post, Y, B, u_0, a0)
            minU.append(float(Utr.min())); statU.append(float(np.median(Utr)))
        th = pn / pd; R = split_rhat_coords(posts, wts)
        pi_hat = 1.0 / (1.0 + np.exp(-np.clip(th, -700, 700)))
        k = max(1, int(np.sum(ts > 0)))
        gm = lambda a, g: float(np.mean(a[z == g]))          # 그룹 평균
        D[m] = dict(sp=spearmanr(ts, th).statistic, kend=kendalltau(ts, th).statistic,
                    topk=float(LMC.topk_overlap(ts, th, k)), ndcg=ndcg_at_k(ts, th, 50),
                    mse=float(np.mean((th - ts) ** 2)),
                    thS=gm(th, "S"), thW=gm(th, "W"), thN=gm(th, "N"),
                    piS=gm(pi_hat, "S"), piW=gm(pi_hat, "W"), piN=gm(pi_hat, "N"),
                    rmed=float(np.median(R)), rq95=float(np.quantile(R, 0.95)), rmax=float(np.nanmax(R)),
                    ess=float(np.mean(ess_list)), minU=minU, statU=statU, Tstop=Tstop)
        print(f"    {m:>8}: T_stop={Tstop:>5} R̂med={D[m]['rmed']:.3f} ({int(time.time()-t0)}s)", flush=True)

    CUT = float(np.round(np.median(D["AWSGLD"]["statU"])))    # cutoff = AWSGLD 정상상태 에너지(4연쇄 median)
    for m in METHODS:
        D[m]["low"] = float(np.mean([max(CUT, v) for v in D[m]["minU"]]))

    # ── 표 1: 수렴 + θ̂·π̂ 추정값 + Lowest U ──  (정답: θ*(S/W/N)=%s, cutoff U=%.0f)
    tsg = f"{gm_ts(ts,z,'S'):.2f}/{gm_ts(ts,z,'W'):.2f}/{gm_ts(ts,z,'N'):.2f}"
    h1 = f"{'Sampler':>8} | {'R̂med':>6} {'R̂q95':>6} {'R̂max':>6} | {'ESS':>5} | {'θ̂(S/W/N)':>18} {'π̂(S/W/N)':>18} | {'LowU':>6}"
    print(f"  [표1 수렴·추정]  T_conv={T_conv}  θ*(S/W/N)={tsg}  cutoff U={CUT:.0f}\n  " + h1); print("  " + "-" * len(h1))
    for m in METHODS:
        d = D[m]
        thc = f"{d['thS']:.2f}/{d['thW']:.2f}/{d['thN']:.2f}"; pic = f"{d['piS']:.2f}/{d['piW']:.2f}/{d['piN']:.2f}"
        print(f"  {m:>8} | {d['rmed']:>6.3f} {d['rq95']:>6.3f} {d['rmax']:>6.3f} | {d['ess']:>5.0f} | {thc:>18} {pic:>18} | {d['low']:>6.0f}")
    # ── 표 2: 순위 ──
    h2 = f"{'Sampler':>8} | {'Spear':>6} {'Kendall':>7} {'Top-k':>6} {'NDCG50':>7} {'MSE_all':>7}"
    print(f"  [표2 순위]\n  " + h2); print("  " + "-" * len(h2))
    rows = []
    for m in METHODS:
        d = D[m]
        print(f"  {m:>8} | {d['sp']:>6.2f} {d['kend']:>7.2f} {d['topk']:>6.3f} {d['ndcg']:>7.3f} {d['mse']:>7.2f}")
        rows.append([name, m, round(d["rmed"], 4), round(d["rq95"], 4), round(d["rmax"], 4), round(d["ess"], 2),
                     round(d["thS"], 4), round(d["thW"], 4), round(d["thN"], 4),
                     round(d["piS"], 4), round(d["piW"], 4), round(d["piN"], 4), round(d["low"], 2),
                     round(d["sp"], 4), round(d["kend"], 4), round(d["topk"], 4), round(d["ndcg"], 4),
                     round(d["mse"], 4), d["Tstop"]])
    return rows


def main():
    global SEED
    args = sys.argv[1:]
    seeds = [int(x) for x in args[0].split(",")] if args else [0]
    tag = args[1] if len(args) > 1 else "s" + "_".join(map(str, seeds))
    t0 = time.time()
    print(f"Study 1A 통합표 | n={N} T_MAX={T_MAX} STOP R̂median<{STOP_RHAT} | seeds={seeds}", flush=True)
    allrows = []
    for s in seeds:
        SEED = s
        print(f"\n===== SEED {s} =====", flush=True)
        for sc in LMC.SCENARIOS:
            for row in run_scenario(sc):
                allrows.append([s] + row)
    out = os.path.join(_HERE, f"unified_scenarios_{tag}.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["seed", "scenario", "method", "rhat_median", "rhat_q95", "rhat_max", "ess",
                    "th_S", "th_W", "th_N", "pi_S", "pi_W", "pi_N", "lowest_U",
                    "spearman", "kendall", "topk", "ndcg50", "mse_all", "T_stop"])
        w.writerows(allrows)
    print(f"\n저장: {out} ({int(time.time()-t0)}s)")


if __name__ == "__main__":
    main()
