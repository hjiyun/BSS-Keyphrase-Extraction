"""A0(합성, n=100, 정답 θ* 존재) 기준 사후분포 vs AWSGLD 검증.
기준 사후분포를 세 독립 경로로 도출: A) Laplace N(θ*,H⁻¹)  B) equipartition  C) gold MCMC(MH보정 pMALA).
AWSGLD가 그 기준(평균·분산)에 맞는지 + 생성 정답 θ*_true 를 복원하는지.
σ²=1, α=a0 고정 → 볼록 사후분포 하나로 못박음.
"""
import os, sys
import numpy as np
from scipy.optimize import minimize

_HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(os.path.dirname(_HERE), "study_1a"), os.path.join(os.path.dirname(_HERE), "study_1b"),
          os.path.join(os.path.dirname(os.path.dirname(_HERE)), "code_JOC")):
    sys.path.insert(0, p)
import keyphrase_functions_awsgld as kfa       # noqa: E402
import unified_a0 as A0M                        # gen_a0 (A0 데이터 생성)  # noqa: E402

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
graph, Y, B, u_0, ts, scen, a0 = A0M.gen_a0(None, SEED)   # ts = 생성 정답 θ*_true
n = graph["n"]
BtB = B.T @ B; ridge = 1e-6 * np.trace(BtB) / n
P = np.linalg.solve(BtB + ridge * np.eye(n), np.eye(n)); P = 0.5 * (P + P.T)
Pinv = np.linalg.inv(P); Lc = np.linalg.cholesky(P + 1e-10 * np.eye(n))


def U(th):  return kfa.posterior_energy(Y, a0, th, u_0, B, 1.0)
def gU(th): return kfa.grad_posterior_energy(Y, a0, th, u_0, B, 1.0, BtB=BtB)


# 볼록성 + 유일 mode
modes = [minimize(U, ini, jac=gU, method="L-BFGS-B").x
         for ini in [np.zeros(n), np.full(n, 2.0), np.full(n, -2.0)] + [np.random.RandomState(s).randn(n) * 2 for s in range(3)]]
th_star = modes[0]; spread = max(np.linalg.norm(m - th_star) for m in modes)
H = np.zeros((n, n)); e = 1e-5
for i in range(n):
    d = np.zeros(n); d[i] = e; H[:, i] = (gU(th_star + d) - gU(th_star - d)) / (2 * e)
H = 0.5 * (H + H.T); mineig = np.linalg.eigvalsh(H).min()
Cov = np.linalg.inv(H); lap_sd = np.sqrt(np.diag(Cov)); Lcov = np.linalg.cholesky(Cov + 1e-10 * np.eye(n))

# 경로 A: Laplace
np.random.seed(0); lapU = np.mean([U(th_star + Lcov @ np.random.randn(n)) for _ in range(3000)])
# 경로 C: gold MCMC = MH 보정 pMALA (정규성 가정 없음)
def logq(x, m0, eps):
    mu = m0 - eps * (P @ gU(m0)); dd = x - mu; return -0.25 / eps * (dd @ (Pinv @ dd))
eps = 0.02; th = th_star.copy(); S = []; acc = 0; T = 30000
for t in range(T):
    prop = th - eps * (P @ gU(th)) + np.sqrt(2 * eps) * (Lc @ np.random.randn(n))
    if np.log(np.random.rand()) < (-U(prop) + U(th)) + (logq(th, prop, eps) - logq(prop, th, eps)):
        th = prop; acc += 1
    if t >= T // 3: S.append(th.copy())
S = np.array(S); gold_sd = S.std(0); goldU = np.mean([U(s) for s in S[::20]])

# AWSGLD (real, gm+precond), 4 분산연쇄, σ²=1·α=a0 고정
ZETA = kfa.ZETA; M_REG = kfa.M_REGIONS; DECAY = kfa.DECAY_LR; TAU = 1.0
inits = [np.full(n, -0.5), np.full(n, 0.5), np.full(n, 1.5), np.random.RandomState(7000).randn(n) * 1.5]
chains = []
for ci, ini in enumerate(inits):
    np.random.seed(ci); theta = ini.copy(); ths = []
    aw = np.arange(1, M_REG + 1, dtype=float) / M_REG; warm = 300; es = []; emin = du = None; J = M_REG - 1
    for t in range(8000):
        gg = gU(theta); Uv = U(theta); gm = 1.0
        if t < warm:
            es.append(Uv)
            if t == warm - 1:
                lo, hi = min(es), max(es); rg = max(hi - lo, 1.0); emin = lo - 0.5 * rg; du = max((hi + 0.5 * rg - emin) / M_REG, 1e-8)
        else:
            J = int(np.clip((Uv - emin) / du + 1, 1, M_REG - 1))
            gm = float(np.clip(1 + (ZETA * TAU / du) * (np.log(aw[J] + 1e-12) - np.log(aw[J - 1] + 1e-12)), 0.1, 10.0))
        ek = 0.3 / ((t + 1) ** 0.6 + 10)
        theta = np.clip(theta - ek * gm * (P @ gg) + np.sqrt(2 * TAU * ek) * (Lc @ np.random.randn(n)), -700, 700)
        if t >= warm:
            dec = min(1.0, DECAY / ((t + 1) ** 0.75 + 1000)); cw = aw[J]; aw[J:] += dec * cw * (1 - aw[J:]); aw[:J] -= dec * cw * aw[:J]; aw = np.clip(aw, 1e-10, 1)
        if t >= 8000 // 3: ths.append(theta.copy())
    chains.append(np.array(ths))
aw_sd = np.concatenate(chains, 0).std(0)
th_hat = np.mean([c.mean(0) for c in chains], 0)
m = np.array([c.mean(0) for c in chains]); v = np.array([c.var(0, ddof=1) for c in chains]); Ln = chains[0].shape[0]
R = np.median(np.sqrt(((Ln - 1) / Ln * v.mean(0) + m.var(0, ddof=1)) / np.maximum(v.mean(0), 1e-12)))
awU = np.mean([U(s) for c in chains for s in c[::20]])

print(f"\n===== A0 (합성, n={n}, seed={SEED}) 기준 사후분포 vs AWSGLD =====")
print(f"[정의] p(θ|Y) ∝ exp(-U),  U=-loglik+‖B(θ-u0)‖²/2σ²  (σ²=1,α={a0:.3f} 고정)")
print(f"[볼록성] mode 유일(시작점간 거리 {spread:.1e}),  Hessian 최소고유값 {mineig:+.3f} (>0 → 강볼록, 트랩 없음)")
print(f"[생성 정답] θ*_true(S/W/N)≈{scen['mu_S']}/{scen['mu_W']}/{scen['mu_N']}\n")
print(f"기준 사후분포 요약 (세 독립 경로):")
print(f"   A. Laplace N(θ*,H⁻¹) : ⟨U⟩={lapU:6.2f}   σ중앙={np.median(lap_sd):.3f}")
print(f"   B. equipartition     : ⟨U⟩={U(th_star)+n/2:6.2f}   (=U_min {U(th_star):.2f}+dof/2 {n/2:.0f})")
print(f"   C. gold MCMC(pMALA)  : ⟨U⟩={goldU:6.2f}   σ중앙={np.median(gold_sd):.3f}   (accept={acc/T:.2f})")
print(f"\nAWSGLD (4연쇄, 8k):")
print(f"   R̂med={R:.3f}   ⟨U⟩={awU:6.2f}   σ중앙={np.median(aw_sd):.3f} (기준대비 {np.median(aw_sd)/np.median(gold_sd)*100:.0f}%)")
print(f"   corr(θ̂, mode θ*)={np.corrcoef(th_hat, th_star)[0,1]:.4f}   corr(θ̂, 생성정답 θ*_true)={np.corrcoef(th_hat, ts)[0,1]:.4f}")
