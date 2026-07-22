"""Study 3 — 실데이터 trap 위 acMH vs AWSGLD 샘플러.

공정성 설계:
  - 두 샘플러 모두 같은 타깃 exp(-U_mix(θ)/TAU) 를 겨냥 (동일 온도)
  - 두 샘플러 모두 (BᵀB)^{-1} 기하 사용
      acMH : 제안 θ* ~ N(θ, s²·P)
      AWSGLD: θ ← θ - ε·gm·P∇U + sqrt(2·TAU·ε)·L·noise ,  L Lᵀ = P
  → 유일한 차이는 flat-histogram(적응 가중) 유무.
"""
import numpy as np
from numpy.linalg import solve, cholesky


def make_precond(BtB, n, ridge_scale=1e-6):
    ridge = ridge_scale * np.trace(BtB) / n
    P = solve(BtB + ridge * np.eye(n), np.eye(n))
    Psym = 0.5 * (P + P.T)
    L = cholesky(Psym + 1e-10 * np.eye(n))
    return P, L


def acmh(U, mode, ini, seed, step, T, P, L):
    """전처리된 랜덤워크 MH (타깃 exp(-U/TAU); U 는 이미 /TAU 된 함수)."""
    rng = np.random.RandomState(seed)
    n = len(ini); th = ini.copy(); Uc = U(th)
    ths = np.zeros((T, n)); mo = np.zeros(T, dtype=np.int8); acc = 0
    for t in range(T):
        star = th + step * (L @ rng.randn(n))
        Us = U(star)
        if np.log(rng.rand() + 1e-300) < (Uc - Us):
            th, Uc = star, Us; acc += 1
        ths[t] = th; mo[t] = mode(th)
    return dict(theta=ths, mode=mo, accept=acc / T)


def awsgld(U, gU, mode, ini, seed, T, P, L, TAU=1.0, ZETA=5.0, eps0=0.1,
           M=500, warmup=300, gm_clip=(0.05, 20.0)):
    """flat-histogram 전처리 Langevin (U, gU 는 이미 /TAU 된 함수)."""
    rng = np.random.RandomState(seed)
    n = len(ini); th = ini.copy()
    ths = np.zeros((T, n)); mo = np.zeros(T, dtype=np.int8)
    w = np.arange(1, M + 1, dtype=float) / M
    emin = None; du = None; J = M - 1; es = []
    nboundary = 0
    for t in range(T):
        eps = eps0 / ((t + 1) ** 0.5 + 10)
        Ut = U(th); g = gU(th)
        if t < warmup:
            es.append(Ut); gm = 1.0
            if t == warmup - 1:
                lo, hi = min(es), max(es); rg = max(hi - lo, 1.0)
                emin = lo - 0.5 * rg
                du = max((hi + 0.5 * rg - emin) / M, 1e-8)
                es = None
        else:
            J = int(np.clip((Ut - emin) / du + 1, 1, M - 1))
            if J <= 1 or J >= M - 1: nboundary += 1
            raw = 1.0 if J == 1 else 1 + (ZETA * TAU / du) * (
                np.log(w[J] + 1e-12) - np.log(w[J - 1] + 1e-12))
            gm = float(np.clip(raw, *gm_clip))
        th = th - eps * gm * (P @ g) + np.sqrt(2.0 * TAU * eps) * (L @ rng.randn(n))
        th = np.clip(th, -700, 700)
        if t >= warmup:
            dec = min(1.0, 100.0 / ((t + 1) ** 0.75 + 1000))
            cw = w[J]
            w[J:] = w[J:] + dec * cw * (1 - w[J:])
            w[:J] = w[:J] - dec * cw * w[:J]
            w = np.clip(w, 1e-10, 1.0)
        ths[t] = th; mo[t] = mode(th)
    return dict(theta=ths, mode=mo,
                boundary_rate=nboundary / max(T - warmup, 1))


def summarize(res, K, burn, ini_mode):
    """모드 방문/탈출 요약."""
    mo = res['mode'][burn:]
    frac = np.array([(mo == k).mean() for k in range(K)])
    visited = int((frac > 0.01).sum())
    full = res['mode']
    esc = np.where(full != ini_mode)[0]
    return dict(visited=visited, frac=frac,
                escape_iter=(int(esc[0]) if len(esc) else -1),
                switches=int((np.diff(mo.astype(int)) != 0).sum()))
