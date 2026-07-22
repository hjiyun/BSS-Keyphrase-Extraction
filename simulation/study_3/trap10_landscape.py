"""Study 3 (10문서) — 실데이터 local trap 에너지 지형 (1B 스타일 1D 단면).
경로: 10개 문서 중심을 순서대로 잇는 꺾은선 (모든 중심을 정확히 통과).
출력: trap10_landscape.png / trap10_landscape.npz
"""
import os, sys, itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trap_build as TB

TB.BASELINE = -2.0
SIG2 = 0.5; K = 10
DOCS = TB.pick_low(K)
mg = TB.build(DOCS, sigma2=SIG2)
U, _, mode = TB.energies(mg)
C = np.array(mg['centers']); n = mg['n']
nb, bmax, bmin, cd, reps = TB.diagnose(mg)
print(f"K={K} docs={DOCS} n={n} basin={nb} 장벽 {bmin:.1f}~{bmax:.1f}", flush=True)

# ---- 꺾은선 경로 (중심 순서대로) ----
seg = [np.linalg.norm(C[i+1]-C[i]) for i in range(K-1)]
total = sum(seg); NPT = 1000
xs, THs, acc = [], [], 0.0
for i in range(K-1):
    m = max(int(round(NPT*seg[i]/total)), 12)
    lam = np.linspace(0, 1, m, endpoint=(i == K-2))
    for l in lam:
        THs.append((1-l)*C[i] + l*C[i+1]); xs.append(acc + l*seg[i])
    acc += seg[i]
xs = np.array(xs); THs = np.array(THs)
center_x = np.concatenate([[0.0], np.cumsum(seg)])
E = np.array([U(th) for th in THs])
WHICH = np.array([mode(th) for th in THs])

# 구간별 장벽
bars = []
for i in range(K-1):
    m = (xs >= center_x[i]) & (xs <= center_x[i+1])
    bars.append(E[m].max() - max(E[np.argmin(abs(xs-center_x[i]))], E[np.argmin(abs(xs-center_x[i+1]))]))

# ---- 그림 ----
import matplotlib.cm as cm
CAT = [cm.tab10(i) for i in range(10)]
LINE = '#2E9E75'; INK, MUT, SURF = '#1a1a1a', '#6b6b6b', '#fcfcfb'
plt.rcParams.update({'font.size': 10, 'figure.facecolor': SURF, 'axes.facecolor': SURF,
                     'axes.edgecolor': '#c9c9c9', 'axes.linewidth': 0.9,
                     'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False})
fig, ax = plt.subplots(figsize=(15.0, 6.4))
rng_E = np.ptp(E)
for k in range(K):
    m = WHICH == k
    if m.any():
        ax.fill_between(xs, E.min()-0.12*rng_E, E.max()+0.20*rng_E, where=m,
                        color=CAT[k], alpha=0.10, lw=0)
ax.plot(xs, E, lw=2.6, color=LINE, solid_capstyle='round', zorder=4)
for k in range(K):
    ax.axvline(center_x[k], color=CAT[k], ls='--', lw=1.3, alpha=0.9, zorder=3)
    ax.text(center_x[k], E.max()+0.075*rng_E, DOCS[k], ha='center', va='bottom',
            fontsize=9.5, fontweight='bold', color=CAT[k], rotation=0)
# 국소최소점 (근접 병합)
_raw = [i for i in range(1, len(E)-1) if E[i] < E[i-1] and E[i] <= E[i+1]]
loc = []
for i in sorted(_raw, key=lambda k: E[k]):
    if all(abs(xs[i]-xs[j]) > 0.025*np.ptp(xs) for j in loc): loc.append(i)
for i in loc:
    ax.scatter(xs[i], E[i], marker='v', s=95, color=LINE, edgecolors='white',
               linewidths=1.2, zorder=6)
# 장벽 라벨
for i in range(K-1):
    m = (xs >= center_x[i]) & (xs <= center_x[i+1])
    j = np.argmax(np.where(m, E, -np.inf))
    ax.annotate(f"{bars[i]:.1f}", (xs[j], E[j]), textcoords='offset points', xytext=(0, 9),
                ha='center', fontsize=8.8, fontweight='bold', color=INK,
                bbox=dict(fc='white', ec='#dcdcdc', boxstyle='round,pad=0.2', alpha=0.9))
ax.set_xlabel('문서 중심을 잇는 경로 좌표 (θ 공간 거리)', color=MUT)
ax.set_ylabel('Energy  $U_{mix}(\\theta) = -\\log\\sum_k e^{-U_k(\\theta)}$', color=MUT)
ax.set_title(f"Study 3 Target (10문서) — 실데이터 local trap 에너지 지형\n"
             f"Hulth {K}문서, 중심 = 문서별 TextRank(실산출), $\\sigma^2$={SIG2}, "
             f"baseline={TB.BASELINE}  —  basin {nb}개, 장벽 {bmin:.1f}~{bmax:.1f}  (숫자 = 구간 장벽)",
             fontsize=12.5, fontweight='bold', color=INK)
ax.set_xlim(xs.min(), xs.max())
ax.set_ylim(E.min()-0.12*rng_E, E.max()+0.20*rng_E)
ax.grid(alpha=0.16, lw=0.6); ax.set_axisbelow(True)
ax.tick_params(colors=MUT, labelsize=9)
for s in ('top', 'right'): ax.spines[s].set_visible(False)
fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trap10_landscape.png")
fig.savefig(out, dpi=150, facecolor=SURF)
np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), "trap10_landscape.npz"),
         xs=xs, E=E, which=WHICH, center_x=center_x, docs=np.array(DOCS),
         barriers=np.array(bars), n=n, basins=nb)
print("구간 장벽:", np.round(bars, 1))
print("저장:", out)
