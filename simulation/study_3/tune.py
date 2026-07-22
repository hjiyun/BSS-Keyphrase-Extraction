"""Study 3 — 하이퍼파라미터 튜닝 스캔 (짧은 런, 1시드).
1) acMH 제안 스텝: 수용률 0.2~0.35 목표
2) 온도 TAU: acMH 가 갇히고 AWSGLD 는 탈출하는 구간 탐색
3) AWSGLD eps0/ZETA: 안정성 + 탈출
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trap_build as TB
from trap_samplers import make_precond, acmh, awsgld, summarize

TB.BASELINE = -2.0
SIG2 = 0.5; DOCS = ['1994', '212', '227']; T = 8000; BURN = 2000
mg = TB.build(DOCS, sigma2=SIG2)
Uraw, gUraw, mode = TB.energies(mg)
n = mg['n']; K = len(DOCS)
P, L = make_precond(mg['BtB'], n)
ini = mg['centers'][0].copy()          # doc 1994 basin 에서 출발 (가장 깊게 분리)
ini_mode = mode(ini)
print(f"Study3 튜닝: n={n}, 문서={DOCS}, 출발 basin=doc {DOCS[ini_mode]}, T={T}\n", flush=True)

def scaled(TAU):
    return (lambda th: Uraw(th) / TAU), (lambda th: gUraw(th) / TAU)

# ---------- 1) acMH 스텝 (TAU=1) ----------
print("[1] acMH 제안 스텝 (TAU=1) — 수용률 0.2~0.35 목표")
U1, _ = scaled(1.0)
best_step = None
for s in (0.02, 0.05, 0.1, 0.2, 0.4):
    r = acmh(U1, mode, ini, 0, s, 3000, P, L)
    sm = summarize(r, K, 500, ini_mode)
    print(f"   step={s:<5} 수용률={r['accept']:.3f}  방문={sm['visited']}  탈출={sm['escape_iter']}")
    if 0.18 <= r['accept'] <= 0.38 and best_step is None:
        best_step = s
if best_step is None: best_step = 0.05
print(f"   → 선택 step={best_step}\n", flush=True)

# ---------- 2) 온도 TAU ----------
print("[2] 온도 TAU — acMH 갇힘 & AWSGLD 탈출 구간")
print(f"   {'TAU':>5} | {'acMH수용':>8} {'ac방문':>6} {'ac탈출':>8} | {'aw방문':>6} {'aw탈출':>8} {'경계율':>7}")
cand = []
for TAU in (0.5, 1.0, 2.0, 4.0, 8.0):
    Us, gUs = scaled(TAU)
    ra = acmh(Us, mode, ini, 0, best_step, T, P, L); sa = summarize(ra, K, BURN, ini_mode)
    rw = awsgld(Us, gUs, mode, ini, 0, T, P, L, TAU=TAU, ZETA=5.0, eps0=0.1)
    sw = summarize(rw, K, BURN, ini_mode)
    print(f"   {TAU:>5.1f} | {ra['accept']:>8.3f} {sa['visited']:>6} {sa['escape_iter']:>8} | "
          f"{sw['visited']:>6} {sw['escape_iter']:>8} {rw['boundary_rate']:>7.3f}")
    cand.append((TAU, sa['visited'], sw['visited'], rw['boundary_rate']))
# acMH 방문 최소, AWSGLD 방문 최대인 TAU
cand.sort(key=lambda c: (c[1], -c[2]))
TAU_SEL = cand[0][0]
print(f"   → 선택 TAU={TAU_SEL} (acMH 방문 {cand[0][1]}, AWSGLD 방문 {cand[0][2]})\n", flush=True)

# ---------- 3) AWSGLD eps0 / ZETA ----------
print(f"[3] AWSGLD eps0 × ZETA (TAU={TAU_SEL})")
Us, gUs = scaled(TAU_SEL)
print(f"   {'eps0':>6} {'ZETA':>5} | {'방문':>5} {'탈출':>8} {'전환':>6} {'경계율':>7} {'발산':>5}")
bestaw = None
for eps0 in (0.03, 0.1, 0.3):
    for ZETA in (1.0, 5.0, 10.0):
        rw = awsgld(Us, gUs, mode, ini, 0, T, P, L, TAU=TAU_SEL, ZETA=ZETA, eps0=eps0)
        sw = summarize(rw, K, BURN, ini_mode)
        div = not np.isfinite(rw['theta'][-1]).all() or np.abs(rw['theta'][BURN:]).max() > 600
        print(f"   {eps0:>6.2f} {ZETA:>5.1f} | {sw['visited']:>5} {sw['escape_iter']:>8} "
              f"{sw['switches']:>6} {rw['boundary_rate']:>7.3f} {'예' if div else '아니오':>5}")
        if not div:
            score = (sw['visited'], sw['switches'])
            if bestaw is None or score > bestaw[0]: bestaw = (score, eps0, ZETA)
_, EPS_SEL, ZETA_SEL = bestaw if bestaw else ((0, 0), 0.1, 5.0)
print(f"\n→ 최종 선택: acMH step={best_step}, TAU={TAU_SEL}, AWSGLD eps0={EPS_SEL}, ZETA={ZETA_SEL}")
np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tuned.npz"),
         step=best_step, TAU=TAU_SEL, eps0=EPS_SEL, zeta=ZETA_SEL, docs=np.array(DOCS), sigma2=SIG2)
print("저장: tuned.npz")
