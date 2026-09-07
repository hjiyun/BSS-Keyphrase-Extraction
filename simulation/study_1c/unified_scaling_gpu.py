"""Study 1C n-scaling 통합표 — GPU(torch) 백엔드. 알고리즘은 CPU판(unified_scaling.py)과 동일,
연산만 GPU. 전체 크기(200/1500/10000)를 동일 백엔드·RNG로 통일 실행(일관성).

- 5 샘플러(SGLD/qSGLD/cycSGLD/SGHMC/AWSGLD), 4 연쇄(분산 init), 고정 T=T_MAX(1C 에선 조기종료가
  거의 안 걸려 고정 T 로 단순화). 지표 계산은 CPU(numpy, 기존 함수 재사용).
- 에너지/그래디언트는 kfa 를 torch 로 정확 이식(gU=−grad_ll+BtB(θ−u0)/σ²).
출력: unified_scaling_n{n}_s{seed}.csv (CPU판과 동일 포맷 → aggregate_scaling.py 그대로).
"""
import os, sys, time, csv
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_A0 = os.path.join(os.path.dirname(_HERE), "study_a0")
for p in (_A0, os.path.join(_A0, "_archive")):
    sys.path.insert(0, p)
import energy_diagnostics as E                 # energy_trace_common (LowU, CPU)  # noqa: E402
from extra_metrics import ess_per_node          # noqa: E402
sys.path.insert(0, _HERE)
import unified_scaling as US                    # split_rhat_coords/ndcg/topk 재사용(CPU)  # noqa: E402

DEV = torch.device("cuda")
DT = torch.float64
torch.set_default_dtype(DT)

SIZE_CFG = {200: dict(BATCH=50, T_MAX=8000), 1500: dict(BATCH=200, T_MAX=12000),
            10000: dict(BATCH=1000, T_MAX=12000)}
BURN = 1000; FLOOR = 1.0
M_REG = 1000; ZETA = 5.0; TAU = 1.0; DECAY = 100.0
SGLD_LR = 0.02; QSGLD_LR = 0.3; CYCSGLD_LR = 0.01; CYC_CYCLES = 10
SGHMC_LR = 0.01; SGHMC_FRICTION = 0.1
GRIDnp = (np.arange(10, 43) - 5) / np.arange(10, 43)


def load(n, seed):
    d = np.load(os.path.join(_HERE, f"data_n{n}_seed{seed}.npz"))
    z = np.array([str(x) for x in d["z"]])
    mu = (float(d["param_mu_S"]), float(d["param_mu_W"]), float(d["param_mu_N"]))
    return d["Y"].astype(float), d["B"].astype(float), d["u_0"].astype(float), d["theta_star"].astype(float), z, mu


def alpha_find_t(theta, Y, grid):
    pi = torch.clamp(torch.sigmoid(theta), 1e-10, 1 - 1e-10)
    temp = torch.clamp((1 - grid[:, None]) * pi[None, :], 1e-10, 1 - 1e-10)
    lk = (Y[None, :] * torch.log(temp) + (1 - Y)[None, :] * torch.log(1 - temp)).sum(1)
    return grid[torch.argmax(lk)]


def grad_t(theta, Y, u0, B, BtB, s2, alpha, batch_idx, n):
    pi = torch.clamp(torch.sigmoid(theta), 1e-10, 1 - 1e-10)
    dpi = pi * (1 - pi)
    temp = torch.clamp((1 - alpha) * pi, 1e-10, 1 - 1e-10)
    denom = torch.clamp(1 - temp, min=1e-10)
    full = torch.where(Y == 1, 1 - pi, -(1 - alpha) * dpi / denom)
    if batch_idx is None:
        gll = full
    else:
        gll = torch.zeros_like(theta); gll[batch_idx] = full[batch_idx] * (n / batch_idx.shape[0])
    gprior = -(BtB @ (theta - u0)) / s2
    return -(gll + gprior)


def energy_t(theta, Y, u0, B, s2, alpha):
    pi = torch.clamp(torch.sigmoid(theta), 1e-10, 1 - 1e-10)
    temp = torch.clamp((1 - alpha) * pi, 1e-10, 1 - 1e-10)
    loglik = (Y * torch.log(temp) + (1 - Y) * torch.log(1 - temp)).sum()
    Bd = B @ (theta - u0)
    return -loglik + (Bd @ Bd) / (2 * s2)


_GAMMA = {}


def sample_s2(theta, Y, u0, B, n):
    """s2 를 GPU 텐서로 반환(동기화 없음)."""
    Bd = B @ (theta - u0); C = Bd @ Bd
    if n not in _GAMMA:
        _GAMMA[n] = torch.distributions.Gamma(torch.tensor(n / 2 + 0.001, device=DEV), torch.tensor(1.0, device=DEV))
    g = _GAMMA[n].sample()
    return torch.clamp((C / 2 + 0.001) / g, min=FLOOR)


def run_chain(method, ini, Y, u0, B, BtB, P, Lc, grid, n, T, batch, seed):
    torch.manual_seed(seed)
    theta = ini.clone(); ths = torch.zeros((T, n), device=DEV); Js = np.full(T, -1)
    alpha = alpha_find_t(theta, Y, grid); v = torch.zeros(n, device=DEV)
    aw = (torch.arange(1, M_REG + 1, device=DEV) / M_REG) if method == "AWSGLD" else None
    warm = 300; es = []; emin = du = None; J = M_REG - 1
    for t in range(T):
        s2 = sample_s2(theta, Y, u0, B, n)
        bidx = torch.randint(0, n, (batch,), device=DEV) if batch < n else None
        gU = grad_t(theta, Y, u0, B, BtB, s2, alpha, bidx, n)
        noise = torch.randn(n, device=DEV)
        if method == "SGLD":
            eps = SGLD_LR / ((t + 1) ** 0.6 + 10); theta = theta - eps * gU + (2 * TAU * eps) ** 0.5 * noise
        elif method == "qSGLD":
            eps = QSGLD_LR / ((t + 1) ** 0.6 + 10); theta = theta - eps * (P @ gU) + (2 * TAU * eps) ** 0.5 * (Lc @ noise)
        elif method == "cycSGLD":
            cl = max(1, T // CYC_CYCLES); beta = (t % cl) / cl
            eps = CYCSGLD_LR / 2 * (np.cos(np.pi * min(beta, 0.8)) + 1); tk = TAU if beta >= 0.8 else TAU / 1e4
            theta = theta - eps * gU + (2 * tk * eps) ** 0.5 * noise
        elif method == "SGHMC":
            eta = SGHMC_LR / ((t + 1) ** 0.6 + 10); theta = theta + v
            v = (1 - SGHMC_FRICTION) * v - eta * gU + (2 * SGHMC_FRICTION * eta * TAU) ** 0.5 * noise
        elif method == "AWSGLD":
            U = float(energy_t(theta, Y, u0, B, s2, alpha)); gm = 1.0   # band J 위해 U 만 동기화(1회/스텝)
            if t < warm:
                es.append(U)
                if t == warm - 1:
                    lo, hi = min(es), max(es); rg = max(hi - lo, 1.0)
                    emin = lo - 0.5 * rg; du = max((hi + 0.5 * rg - emin) / M_REG, 1e-8); es = None
            else:
                J = int(np.clip((U - emin) / du + 1, 1, M_REG - 1))
                gm = torch.clamp(1 + (ZETA * TAU / du) * (torch.log(aw[J] + 1e-12) - torch.log(aw[J - 1] + 1e-12)), 0.1, 10.0)
            ek = 0.3 / ((t + 1) ** 0.6 + 10)
            theta = theta - ek * gm * (P @ gU) + (2 * TAU * ek) ** 0.5 * (Lc @ noise)
            if t >= warm:
                dec = min(1.0, DECAY / ((t + 1) ** 0.75 + 1000)); cw = aw[J].clone()
                ge = torch.arange(M_REG, device=DEV) >= J
                aw = torch.where(ge, aw + dec * cw * (1 - aw), aw - dec * cw * aw)
                aw = torch.clamp(aw, 1e-10, 1.0)
            Js[t] = J
        theta = torch.clamp(theta, -700, 700); ths[t] = theta
        alpha = alpha_find_t(theta, Y, grid)
    return ths.cpu().numpy(), Js, (aw.cpu().numpy() if aw is not None else None)


def run_size(n, seed):
    Yn, Bn, u0n, ts, z, mu = load(n, seed)
    cfg = SIZE_CFG[n]; T = cfg["T_MAX"]; batch = cfg["BATCH"]
    t0 = time.time()
    B = torch.tensor(Bn, device=DEV); Y = torch.tensor(Yn, device=DEV); u0 = torch.tensor(u0n, device=DEV)
    BtB = B.T @ B; ridge = 1e-6 * torch.trace(BtB) / n
    P = torch.linalg.solve(BtB + ridge * torch.eye(n, device=DEV), torch.eye(n, device=DEV)); P = 0.5 * (P + P.T)
    Lc = torch.linalg.cholesky(P + 1e-10 * torch.eye(n, device=DEV))
    grid = torch.tensor(GRIDnp, device=DEV)
    a0 = float(alpha_find_t(u0, Y, grid))
    inits = [torch.full((n,), mu[2], device=DEV), torch.full((n,), mu[1], device=DEV),
             torch.full((n,), mu[0], device=DEV),
             torch.tensor(np.random.RandomState(7000 + seed).randn(n) * 1.5, device=DEV)]
    print(f"[GPU n={n} seed={seed}] precond done ({int(time.time()-t0)}s) μ={mu} T={T}", flush=True)
    METHODS = ["SGLD", "qSGLD", "cycSGLD", "SGHMC", "AWSGLD"]
    D = {}; cutoff = None; statU_aw = None
    order = ["AWSGLD"] + [m for m in METHODS if m != "AWSGLD"]
    for m in order:
        posts = []; wts = []
        for ci in range(4):
            ths, Js, aw = run_chain(m, inits[ci], Y, u0, B, BtB, P, Lc, grid, n, T, batch, seed=100 * seed + ci)
            post = ths[BURN:]; posts.append(post)
            if m == "AWSGLD":
                Jp = Js[BURN:]; w = aw[Jp].copy(); w[Jp >= M_REG - 1] = 0.0; wts.append(w)
            else:
                wts.append(None)
        D[m] = _metrics(posts, wts, ts, z, Yn, Bn, u0n, a0, m)
        if m == "AWSGLD":
            cutoff = float(np.round(np.median(D[m]["statU"])))
        print(f"    {m:>8} R̂med={D[m]['rmed']:.3f} ESS={D[m]['ess']:.0f} ({int(time.time()-t0)}s)", flush=True)
    rows = []
    for m in METHODS:
        d = D[m]; low = float(np.mean([max(cutoff, v) for v in d["minU"]]))
        rows.append([n, seed, m, round(d["rmed"], 4), round(d["rq95"], 4), round(d["rmax"], 4), round(d["ess"], 2),
                     round(d["thS"], 4), round(d["thW"], 4), round(d["thN"], 4),
                     round(d["piS"], 4), round(d["piW"], 4), round(d["piN"], 4), round(low, 2),
                     round(d["sp"], 4), round(d["kend"], 4), round(d["topk"], 4), round(d["ndcg"], 4),
                     round(d["mse"], 4), T])
    return rows


def _metrics(posts, wts, ts, z, Yn, Bn, u0n, a0, method):
    from scipy.stats import spearmanr, kendalltau
    n = len(ts); pn = np.zeros(n); pd = 0.0; ess_list = []; minU = []; statU = []
    for ci in range(4):
        post = posts[ci]
        if wts[ci] is not None:
            sw = wts[ci].sum(); pn += (wts[ci][:, None] * post).sum(0); pd += sw
        else:
            pn += post.sum(0); pd += post.shape[0]
        ess_list.append(float(np.nanmedian(ess_per_node(post))))
        Utr = E.energy_trace_common(post, Yn, Bn, u0n, a0); minU.append(float(Utr.min())); statU.append(float(np.median(Utr)))
    th = pn / pd; R = US.split_rhat_coords(posts, wts)
    pih = 1.0 / (1.0 + np.exp(-np.clip(th, -700, 700)))
    k = max(1, int(np.sum(ts > 0))); gm = lambda a, g: float(np.mean(a[z == g]))
    return dict(rmed=float(np.median(R)), rq95=float(np.quantile(R, 0.95)), rmax=float(np.nanmax(R)),
                ess=float(np.mean(ess_list)), thS=gm(th, "S"), thW=gm(th, "W"), thN=gm(th, "N"),
                piS=gm(pih, "S"), piW=gm(pih, "W"), piN=gm(pih, "N"), minU=minU, statU=statU,
                sp=spearmanr(ts, th).statistic, kend=kendalltau(ts, th).statistic,
                topk=float(US.topk_overlap(ts, th, k)), ndcg=US.ndcg_at_k(ts, th, 50), mse=float(np.mean((th - ts) ** 2)))


def main():
    args = sys.argv[1:]
    sizes = [int(x) for x in args[0].split(",")] if args else [200, 1500, 10000]
    seeds = [int(x) for x in args[1].split(",")] if len(args) > 1 else [0, 1, 2, 3, 4]
    for n in sizes:
        for s in seeds:
            rows = run_size(n, s)
            out = os.path.join(_HERE, f"unified_scaling_n{n}_s{s}.csv")
            with open(out, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["n", "seed", "method", "rhat_median", "rhat_q95", "rhat_max", "ess",
                            "th_S", "th_W", "th_N", "pi_S", "pi_W", "pi_N", "lowest_U",
                            "spearman", "kendall", "topk", "ndcg50", "mse_all", "T_stop"])
                w.writerows(rows)
            print(f"  저장: {out}", flush=True)


if __name__ == "__main__":
    main()
