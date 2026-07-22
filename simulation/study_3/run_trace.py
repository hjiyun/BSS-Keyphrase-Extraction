"""Study 3 — 모드 방문 트레이스 시각화 (대표 시드).
왼쪽: acMH (갇힘) / 오른쪽: AWSGLD (탈출). 시간에 따른 basin index + 경로 좌표.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trap_build as TB
from trap_samplers import make_precond, acmh, awsgld

TB.BASELINE = -2.0
DOCS = ['1994', '212', '227']; SIG2 = 0.5
TAU = 1.0; STEP = 0.3; EPS0 = 12.0; ZETA = 5.0; T = 8000; BURN = 2000; SEED = 3
mg = TB.build(DOCS, sigma2=SIG2)
Uraw, gUraw, mode = TB.energies(mg)
n = mg['n']; K = 3; P, L = make_precond(mg['BtB'], n)
U = lambda th: Uraw(th) / TAU; gU = lambda th: gUraw(th) / TAU
ini = mg['centers'][1].copy()
ra = acmh(U, mode, ini, SEED, STEP, T, P, L)
rw = awsgld(U, gU, mode, ini, 1000 + SEED, T, P, L, TAU=TAU, ZETA=ZETA, eps0=EPS0)

CAT = ['#0072B2', '#E69F00', '#009E73']; INK, MUT, SURF = '#1a1a1a', '#6b6b6b', '#fcfcfb'
plt.rcParams.update({'font.size': 10, 'figure.facecolor': SURF, 'axes.facecolor': SURF,
                     'axes.edgecolor': '#c9c9c9', 'axes.linewidth': 0.9,
                     'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False})
fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.0), sharey=True)
for ax, (nm, r) in zip(axes, [('acMH', ra), ('AWSGLD', rw)]):
    mo = r['mode']
    ax.axhspan(-0.4, 2.4, color='none')
    # 방문 basin 밴드
    for k in range(K):
        ax.axhspan(k - 0.4, k + 0.4, color=CAT[k], alpha=0.08, lw=0)
    ax.scatter(np.arange(T), mo, s=4, c=[CAT[m] for m in mo], alpha=0.5, edgecolors='none')
    ax.axvline(BURN, color=MUT, ls='--', lw=1.0, alpha=0.7)
    ax.text(BURN, 2.62, 'burn-in', fontsize=8, color=MUT, ha='center')
    frac = np.array([(mo[BURN:] == k).mean() for k in range(K)])
    visited = int((frac > 0.01).sum())
    ax.set_title(f"{nm} — 방문 basin {visited}개\n체류: " +
                 "  ".join(f"doc{DOCS[k]} {frac[k]*100:.0f}%" for k in range(K)),
                 fontsize=11, color=INK)
    ax.set_xlabel('반복 (iteration)', color=MUT)
    ax.set_xlim(0, T); ax.set_ylim(-0.5, 2.75)
    ax.set_yticks(range(K)); ax.set_yticklabels([f"doc {d}" for d in DOCS])
    ax.tick_params(colors=MUT, labelsize=9)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
axes[0].set_ylabel('머무는 basin (가장 가까운 문서)', color=MUT)
fig.suptitle(f"Study 3 — 실데이터 trap 에서 모드 방문 (출발 doc 212, seed {SEED})\n"
             f"acMH 는 갇히고, AWSGLD 는 얕은 장벽(≈8)을 넘어 doc 227 탐색",
             fontsize=12.5, fontweight='bold', color=INK, y=0.99)
fig.subplots_adjust(top=0.80)
fig.tight_layout(rect=[0, 0, 1, 0.86])
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mode_visit.png")
fig.savefig(out, dpi=150, facecolor=SURF)
print("acMH 방문:", [round((ra['mode'][BURN:]==k).mean(),3) for k in range(K)])
print("AWSGLD 방문:", [round((rw['mode'][BURN:]==k).mean(),3) for k in range(K)])
print("저장:", out)
