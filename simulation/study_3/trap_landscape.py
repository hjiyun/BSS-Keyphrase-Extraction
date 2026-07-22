"""Study 3 — 실데이터 기반 local trap 에너지 지형 (Study 1B 스타일 1D 단면).

타깃 분포
--------
Hulth 문서 K편을 합집합 어휘로 통합하고, 각 문서 k 의 TextRank 해 u^(k) 를
그 문서 basin 의 중심으로 사용한다 (중심은 전부 실데이터 산출값).

  U_k(θ)   = -loglik(θ) + ||B(θ - u^(k))||^2 / (2σ²)
  U_mix(θ) = -log Σ_k exp(-U_k(θ))          ← log-sum-exp (CLAUDE.md 준수)

시각화
------
고차원 θ 를 문서 중심들을 잇는 **꺾은선 경로**로 1D 단면을 내어 1B 와 동일한
형태(에너지 곡선 + basin 음영 + 국소최소점 ▼)로 그린다. 경로는 모든 중심을
정확히 통과하므로 각 basin 과 그 사이 장벽이 그대로 드러난다.

출력: trap_landscape.npz / trap_landscape.png
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trap_build as TB

# ---- 구성 (고정) ----
TB.BASELINE = -2.0
SIG2 = 0.5
ORDER = ['1994', '212', '227']        # 경로 순서: 먼 문서 → 가까운 두 문서
NPT = 420                             # 경로 표본 수

mg = TB.build(ORDER, sigma2=SIG2)
U, _, mode = TB.energies(mg)
C = np.array(mg['centers'])
K, n = C.shape

# ---- 꺾은선 경로: c0 → c1 → c2 (모든 중심 정확 통과) ----
seg_len = [np.linalg.norm(C[i + 1] - C[i]) for i in range(K - 1)]
total = sum(seg_len)
xs, THs = [], []
acc = 0.0
for i in range(K - 1):
    m = int(round(NPT * seg_len[i] / total))
    lam = np.linspace(0, 1, m, endpoint=(i == K - 2))
    for l in lam:
        THs.append((1 - l) * C[i] + l * C[i + 1])
        xs.append(acc + l * seg_len[i])
    acc += seg_len[i]
xs = np.array(xs); THs = np.array(THs)
center_x = np.concatenate([[0.0], np.cumsum(seg_len)])

E = np.array([U(th) for th in THs])
WHICH = np.array([mode(th) for th in THs])       # 각 지점에서 가장 가까운(최소) basin

# ---- 국소최소점 (경로 위) ----
_raw = [i for i in range(1, len(E) - 1) if E[i] < E[i - 1] and E[i] <= E[i + 1]]
loc = []                                          # 근접 중복 최소점 병합
for i in sorted(_raw, key=lambda k: E[k]):
    if all(abs(xs[i] - xs[j]) > 0.09 * np.ptp(xs) for j in loc):
        loc.append(i)
# ---- 장벽 (인접 basin 경계의 최대) ----
bars = []
for i in range(K - 1):
    a, b = center_x[i], center_x[i + 1]
    m = (xs >= a) & (xs <= b)
    top = E[m].max()
    bars.append(top - max(E[np.argmin(np.abs(xs - a))], E[np.argmin(np.abs(xs - b))]))

# ================= 그림 (1B 스타일) =================
CAT = ['#0072B2', '#E69F00', '#009E73']          # 문서 정체성 (CVD 안전)
LINE = '#2E9E75'                                  # 1B 와 동일 계열 에너지 곡선
INK, MUT, SURF = '#1a1a1a', '#6b6b6b', '#fcfcfb'
plt.rcParams.update({'font.size': 10, 'figure.facecolor': SURF, 'axes.facecolor': SURF,
                     'axes.edgecolor': '#c9c9c9', 'axes.linewidth': 0.9,
                     'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False})
fig, ax = plt.subplots(figsize=(11.6, 6.2))

# basin 음영 (그 지점에서 어느 문서가 최소인지)
for k in range(K):
    m = WHICH == k
    if m.any():
        ax.fill_between(xs, E.min() - 0.06 * np.ptp(E), E.max() + 0.12 * np.ptp(E),
                        where=m, color=CAT[k], alpha=0.085, lw=0)

ax.plot(xs, E, lw=3.0, color=LINE, solid_capstyle='round', zorder=4)

# 문서 중심 표시
for k in range(K):
    ax.axvline(center_x[k], color=CAT[k], ls='--', lw=1.5, alpha=0.85, zorder=3)
    ax.text(center_x[k], E.max() + 0.075 * np.ptp(E), f"doc {ORDER[k]}",
            ha='center', va='bottom', fontsize=12, fontweight='bold', color=CAT[k])

# 국소최소점 ▼
for i in loc:
    ax.scatter(xs[i], E[i], marker='v', s=130, color=LINE, edgecolors='white',
               linewidths=1.4, zorder=6)
    ax.annotate(f"E={E[i]:.1f}", (xs[i], E[i]), textcoords='offset points',
                xytext=(0, -22), ha='center', fontsize=9, color=LINE, fontweight='bold')

# 장벽 표시
for i in range(K - 1):
    a, b = center_x[i], center_x[i + 1]
    m = (xs >= a) & (xs <= b)
    j = np.argmax(np.where(m, E, -np.inf))
    ax.annotate(f"장벽 {bars[i]:.1f}", (xs[j], E[j]), textcoords='offset points',
                xytext=(0, 14), ha='center', fontsize=10.5, fontweight='bold', color=INK,
                bbox=dict(fc='white', ec='#d8d8d8', boxstyle='round,pad=0.28', alpha=0.92))

ax.set_xlabel('문서 중심을 잇는 경로 좌표', color=MUT)
ax.set_ylabel('Energy  $U_{mix}(\\theta) = -\\log\\sum_k e^{-U_k(\\theta)}$', color=MUT)
ax.set_title("Study 3 Target — 실데이터 기반 local trap 에너지 지형\n"
             f"Hulth {K}문서, 중심 = 문서별 TextRank(실산출),  "
             f"$\\sigma^2$={SIG2},  baseline={TB.BASELINE}  —  ▼ local min",
             fontsize=12.5, fontweight='bold', color=INK)
ax.set_xlim(xs.min(), xs.max())
ax.set_ylim(E.min() - 0.13 * np.ptp(E), E.max() + 0.16 * np.ptp(E))
ax.grid(alpha=0.18, lw=0.6); ax.set_axisbelow(True)
ax.tick_params(colors=MUT, labelsize=9)
for s in ('top', 'right'): ax.spines[s].set_visible(False)

fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trap_landscape.png")
fig.savefig(out, dpi=155, facecolor=SURF)
np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), "trap_landscape.npz"),
         xs=xs, E=E, which=WHICH, center_x=center_x, docs=np.array(ORDER),
         barriers=np.array(bars), sigma2=SIG2, baseline=TB.BASELINE, n=n)
print(f"문서={ORDER}  n={n}  경로 국소최소점={len(loc)}개  장벽={[round(b,1) for b in bars]}")
print("저장:", out)
