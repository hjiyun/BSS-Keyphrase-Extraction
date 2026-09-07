"""Study 1C n-scaling 통합표 — study_1a/a0 와 동일한 2표 지표 (acMH 제외 5 샘플러).

데이터 크기 n ∈ {200, 1500, 10000} 별로 5 샘플러(SGLD/qSGLD/cycSGLD/SGHMC/AWSGLD)를
4 연쇄로 비교. AWSGLD split-R̂ median<1.05 조기종료(전역 정지점 T_conv 공유), T_MAX 상한.
표1(수렴·추정): R̂med/q95/max · ESS · θ̂(S/W/N) · π̂(S/W/N) · Lowest U
표2(순위): Spearman · Kendall · Top-k · NDCG@50 · MSE_all
데이터는 기존 data_n{N}_seed{S}.npz 재사용. 대규모 n 은 메서드별 계산 후 즉시 메모리 해제.

사용: python3 unified_scaling.py <n> <seed> [tag]   # (n,seed) 하나씩 → unified_scaling_{tag}.csv
출력 행: [n, seed, method, rhat_med,q95,max, ess, thS,thW,thN, piS,piW,piN, lowU,
          spearman, kendall, topk, ndcg50, mse_all, T_stop]
"""
import os, sys, time, csv
import numpy as np
from scipy.stats import spearmanr, kendalltau, invgamma

_HERE = os.path.dirname(os.path.abspath(__file__))
_A0 = os.path.join(os.path.dirname(_HERE), "study_a0")
ROOT = os.path.dirname(os.path.dirname(_HERE)); CODE = os.path.join(ROOT, "code_JOC")
for p in (_A0, os.path.join(_A0, "_archive"), CODE):
    sys.path.insert(0, p)
import energy_diagnostics as E                 # alpha_find/GRID/energy_trace_common  # noqa: E402
import keyphrase_functions_awsgld as kfa       # noqa: E402
from extra_metrics import ess_per_node         # noqa: E402

# ── 설정 ──
BURN = 1000; MIN_T = 3000; CHUNK = 1000; STOP_RHAT = 1.05; FLOOR = 1.0
SIZE_CFG = {200: dict(BATCH=50, T_MAX=8000), 1500: dict(BATCH=200, T_MAX=12000),
            10000: dict(BATCH=1000, T_MAX=6000)}   # n=10000 은 O(n²)/step 비용으로 T_MAX 하향
METHODS = ["SGLD", "qSGLD", "cycSGLD", "SGHMC", "AWSGLD"]
M_REG = kfa.M_REGIONS; ZETA = kfa.ZETA; TAU = kfa.TAU; DECAY = kfa.DECAY_LR
SGLD_LR = 0.02; QSGLD_LR = 0.3; CYCSGLD_LR = 0.01; CYC_CYCLES = 10
SGHMC_LR = 0.01; SGHMC_FRICTION = 0.1
BATCH = None; T_MAX = None; N = None            # per-size 로 설정


def load(n, seed):
    d = np.load(os.path.join(_HERE, f"data_n{n}_seed{seed}.npz"))
    z = np.array([str(x) for x in d["z"]])
    graph = {"n": int(d["n_total"]), "A": d["A"], "D": np.diag(d["A"].sum(1))}
    mu = (float(d["param_mu_S"]), float(d["param_mu_W"]), float(d["param_mu_N"]))
    return graph, d["Y"].astype(float), d["B"], d["u_0"], d["theta_star"], z, mu


def eps_awsgld(t):
    return 0.3 / ((t + 1) ** 0.6 + 10)


def new_state(method, ini, ctx):
    np.random.seed(ctx["seed_base"] + ctx["ci_of"][id(ini)])
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
            ek = eps_awsgld(t)
            theta = theta - ek * gm * (P @ gU) + np.sqrt(2 * TAU * ek) * (Lc @ np.random.randn(n))
            if t >= st["warm"]:
                dec = min(1.0, DECAY / ((t + 1) ** 0.75 + 1000)); aw = st["aw"]; J = st["J"]; cw = aw[J]
                aw[J:] = aw[J:] + dec * cw * (1 - aw[J:]); aw[:J] = aw[:J] - dec * cw * aw[:J]; st["aw"] = np.clip(aw, 1e-10, 1)
            st["Js"][t] = st["J"]
        theta = np.clip(theta, -700, 700)
        st["theta"] = theta; st["ths"][t] = theta
        st["alpha"] = E.alpha_find(theta, Y, E.GRID); st["t"] = t + 1


def collect(states):
    posts = []; wts = []
    for st in states:
        nt = st["t"]; post = st["ths"][BURN:nt]; posts.append(post)
        if st["method"] == "AWSGLD":
            Jp = st["Js"][BURN:nt]; w = st["aw"][Jp].copy(); w[Jp >= M_REG - 1] = 0.0; wts.append(w)
        else:
            wts.append(None)
    return posts, wts


def split_rhat_coords(posts, wts):
    hm = []; hv = []; hL = None
    for ci, post in enumerate(posts):
        L = post.shape[0]; h = L // 2; hL = h
        for sl in (slice(0, h), slice(h, 2 * h)):
            seg = post[sl]
            if wts[ci] is not None:
                w = wts[ci][sl]; sw = w.sum()
                mn = (w[:, None] * seg).sum(0) / sw; v = (w[:, None] * (seg - mn) ** 2).sum(0) / sw
            else:
                mn = seg.mean(0); v = seg.var(0, ddof=1)
            hm.append(mn); hv.append(v)
    hm = np.array(hm); hv = np.array(hv); M2 = len(hm)
    gm = hm.mean(0); Bo = ((hm - gm) ** 2).sum(0) / (M2 - 1); W = hv.mean(0)
    return np.sqrt(np.clip(((hL - 1) / hL * W + Bo) / np.maximum(W, 1e-12), 0, None))


def ndcg_at_k(ts, th, k):
    rel = np.argsort(np.argsort(ts)).astype(float) / max(len(ts) - 1, 1)
    pred = np.argsort(th)[::-1][:k]; ideal = np.argsort(ts)[::-1][:k]
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    d = np.sum((2 ** rel[pred] - 1) * disc); i = np.sum((2 ** rel[ideal] - 1) * disc)
    return float(d / i) if i > 0 else 0.0


def topk_overlap(ts, th, k):
    return len(set(np.argsort(ts)[::-1][:k].tolist()) & set(np.argsort(th)[::-1][:k].tolist())) / max(k, 1)


def make_states(method, ctx):
    inits = ctx["inits"]; states = []
    ctx["ci_of"] = {}
    for ci, ini in enumerate(inits):
        ctx["ci_of"][id(ini)] = ci
        st = new_state(method, ini, ctx); states.append(st)
    return states


def run_method(method, ctx, target=None):
    states = make_states(method, ctx)
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


def metrics(method, states, ts, z, Y, B, u_0, a0, cutoff, T_conv):
    posts, wts = collect(states); n = N
    pn = np.zeros(n); pd = 0.0; ess_list = []; minU = []
    for ci in range(len(states)):
        post = posts[ci]
        if wts[ci] is not None:
            sw = wts[ci].sum(); pn += (wts[ci][:, None] * post).sum(0); pd += sw
        else:
            pn += post.sum(0); pd += post.shape[0]
        ess_list.append(float(np.nanmedian(ess_per_node(post))))
        minU.append(float(E.energy_trace_common(post, Y, B, u_0, a0).min()))
    th = pn / pd; R = split_rhat_coords(posts, wts)
    pih = 1.0 / (1.0 + np.exp(-np.clip(th, -700, 700)))
    k = max(1, int(np.sum(ts > 0)))
    gm = lambda a, g: float(np.mean(a[z == g]))
    low = float(np.mean([max(cutoff, v) for v in minU])) if cutoff is not None else float("nan")
    return dict(rmed=float(np.median(R)), rq95=float(np.quantile(R, 0.95)), rmax=float(np.nanmax(R)),
                ess=float(np.mean(ess_list)), thS=gm(th, "S"), thW=gm(th, "W"), thN=gm(th, "N"),
                piS=gm(pih, "S"), piW=gm(pih, "W"), piN=gm(pih, "N"), low=low, minU=minU,
                sp=spearmanr(ts, th).statistic, kend=kendalltau(ts, th).statistic,
                topk=float(topk_overlap(ts, th, k)), ndcg=ndcg_at_k(ts, th, 50),
                mse=float(np.mean((th - ts) ** 2)), Tstop=T_conv)


def run_size(n, seed):
    global N, BATCH, T_MAX
    N = n; BATCH = SIZE_CFG[n]["BATCH"]; T_MAX = SIZE_CFG[n]["T_MAX"]
    graph, Y, B, u_0, ts, z, mu = load(n, seed)
    t0 = time.time()
    BtB = B.T @ B; ridge = 1e-6 * np.trace(BtB) / n
    P = np.linalg.solve(BtB + ridge * np.eye(n), np.eye(n)); P = 0.5 * (P + P.T)
    Lc = np.linalg.cholesky(P + 1e-10 * np.eye(n))
    a0 = E.alpha_find(u_0, Y, E.GRID)
    inits = [np.full(n, mu[2]), np.full(n, mu[1]), np.full(n, mu[0]),
             np.random.RandomState(7000 + seed).randn(n) * 1.5]              # μ_N/μ_W/μ_S/rand
    ctx = dict(Y=Y, B=B, u_0=u_0, BtB=BtB, P=P, Lc=Lc, a0=a0, inits=inits, seed_base=100 * seed)
    print(f"[n={n} seed={seed}] precond done ({int(time.time()-t0)}s), μ(S/W/N)={mu}", flush=True)

    # AWSGLD 먼저 → 조기종료 T_conv + cutoff, 계산 후 해제
    aw_states, T_conv = run_method("AWSGLD", ctx, target=None)
    statU = [float(np.median(E.energy_trace_common(aw_states[c]["ths"][BURN:aw_states[c]["t"]], Y, B, u_0, a0)))
             for c in range(len(aw_states))]
    cutoff = float(np.round(np.median(statU)))
    D = {"AWSGLD": metrics("AWSGLD", aw_states, ts, z, Y, B, u_0, a0, cutoff, T_conv)}
    del aw_states
    print(f"    AWSGLD T_conv={T_conv} cutoff={cutoff:.0f} R̂med={D['AWSGLD']['rmed']:.3f} ({int(time.time()-t0)}s)", flush=True)
    for m in METHODS:
        if m == "AWSGLD":
            continue
        sts, _ = run_method(m, ctx, target=T_conv)
        D[m] = metrics(m, sts, ts, z, Y, B, u_0, a0, cutoff, T_conv)
        del sts
        print(f"    {m:>8} R̂med={D[m]['rmed']:.3f} ({int(time.time()-t0)}s)", flush=True)

    rows = []
    for m in METHODS:
        d = D[m]
        rows.append([n, seed, m, round(d["rmed"], 4), round(d["rq95"], 4), round(d["rmax"], 4), round(d["ess"], 2),
                     round(d["thS"], 4), round(d["thW"], 4), round(d["thN"], 4),
                     round(d["piS"], 4), round(d["piW"], 4), round(d["piN"], 4), round(d["low"], 2),
                     round(d["sp"], 4), round(d["kend"], 4), round(d["topk"], 4), round(d["ndcg"], 4),
                     round(d["mse"], 4), d["Tstop"]])
    return rows


def main():
    n = int(sys.argv[1]); seed = int(sys.argv[2])
    tag = sys.argv[3] if len(sys.argv) > 3 else f"n{n}_s{seed}"
    rows = run_size(n, seed)
    out = os.path.join(_HERE, f"unified_scaling_{tag}.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["n", "seed", "method", "rhat_median", "rhat_q95", "rhat_max", "ess",
                    "th_S", "th_W", "th_N", "pi_S", "pi_W", "pi_N", "lowest_U",
                    "spearman", "kendall", "topk", "ndcg50", "mse_all", "T_stop"])
        w.writerows(rows)
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
