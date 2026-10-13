"""AWSGLD vs acMH — 기본 seed0 trace+수렴, 두 그림 동일 y축(겹치지 않음)."""
import sys; sys.path.insert(0, "/home/jiyoon/BSS-Keyphrase-Extraction/simulation/study_fav")
_a = sys.argv; sys.argv = [sys.argv[0]]
import numpy as np, time
import opt_acmh_fair as of
import keyphrase_functions_awsgld as kfa
from numpy.linalg import inv, cholesky
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
_FP = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"; fm.fontManager.addfont(_FP)
plt.rcParams["font.family"] = fm.FontProperties(fname=_FP).get_name(); plt.rcParams["axes.unicode_minus"] = False
sys.argv = _a
SEED = 0; of.N = 1000; of.MU = 1.2; of.ALPHA = 0.35; of.LR_AWSGLD = 0.6
n, B, u0, Y, a0, ts, truth = of.gen(SEED)
U, gradU, BtB = of.make(n, B, u0, Y, a0)
P = inv(BtB + 1e-6 * np.trace(BtB) / n * np.eye(n)); P = 0.5 * (P + P.T); Lc = cholesky(P + 1e-10 * np.eye(n))
th = u0.copy()
for _ in range(8000): th = th - 0.05 * gradU(th)
Umin = U(th); thstar = th.copy()
order = np.argsort(thstar); hi = order[-1]; mid = order[n // 2]; lo = order[0]
sel = [(hi, 'High (keyphrase)', '#1f77b4'), (mid, 'Mid', '#ff7f0e'), (lo, 'Low', '#777777')]
MAXB = of.MAXB


def awsgld():
    M_REG = of.M_REG; DECAY = of.DECAY; ZETA = of.ZETA; rng = np.random.default_rng(0)
    theta = rng.standard_normal(n) * 1.5; aw = np.arange(1, M_REG + 1, dtype=float) / M_REG; warm = 300; es = []; emin = du = None; J = M_REG - 1
    tr = {i: [] for i, _, _ in sel}; bU = np.inf; bg = []
    for t in range(MAXB):
        Tt = max(1e-4, np.exp(-t / 400)); g = gradU(theta); Uv = U(theta); gm = 1.0
        if not np.isfinite(Uv): Uv = emin if emin is not None else 0.0
        if t < warm:
            es.append(Uv)
            if t == warm - 1: lo_, hi_ = min(es), max(es); rg = max(hi_ - lo_, 1.0); emin = lo_ - 0.5 * rg; du = max((hi_ + 0.5 * rg - emin) / M_REG, 1e-8)
        else:
            J = int(np.clip((Uv - emin) / du + 1, 1, M_REG - 1)); gm = float(np.clip(1 + (ZETA / du) * (np.log(aw[J] + 1e-12) - np.log(aw[J - 1] + 1e-12)), 0.1, 10.0))
        ek = of.LR_AWSGLD / ((t + 1) ** 0.6 + 10); theta = theta - ek * gm * (P @ g) + np.sqrt(2 * ek * gm * Tt) * (Lc @ rng.standard_normal(n))
        if t >= warm: dec = min(1.0, DECAY / ((t + 1) ** 0.75 + 1000)); cw = aw[J]; aw[J:] += dec * cw * (1 - aw[J:]); aw[:J] -= dec * cw * aw[:J]; aw = np.clip(aw, 1e-10, 1)
        theta = np.clip(theta, -700, 700); u = U(theta); bU = min(bU, u)
        for i, _, _ in sel: tr[i].append(theta[i])
        bg.append(bU - Umin)
    return tr, bg, warm


def acmh(scale=12.0):
    rng = np.random; rng.seed(0); theta = rng.standard_normal(n) * 1.5; hist = [theta.copy()]
    tr = {i: [] for i, _, _ in sel}; bU = np.inf; bg = []; lp = kfa.posterior_gibbstheta(Y, a0, theta, u0, B, 1.0)
    for t in range(MAXB):
        Tt = max(1e-4, np.exp(-t / 400))
        vars_i = np.ones(n) if t < 10 else np.sqrt(scale * (np.array(hist).var(0, ddof=1) + 0.01))
        for i in range(n):
            ov = theta[i]; theta[i] = ov + rng.normal(0, vars_i[i]); lp2 = kfa.posterior_gibbstheta(Y, a0, theta, u0, B, 1.0)
            if np.log(rng.uniform() + 1e-300) < (lp2 - lp) / Tt: lp = lp2
            else: theta[i] = ov
        hist.append(theta.copy()); u = U(theta); bU = min(bU, u)
        for i, _, _ in sel: tr[i].append(theta[i])
        bg.append(bU - Umin)
    return tr, bg


tA, gA, warm = awsgld(); t0 = time.time(); tM, gM = acmh(12.0); print(f'acMH {(time.time()-t0)/60:.1f}min')
allth = np.concatenate([np.array(tA[i]) for i, _, _ in sel] + [np.array(tM[i]) for i, _, _ in sel] + [thstar[[hi, mid, lo]]])
YL = (allth.min() - 0.3, allth.max() + 0.3)
GL = (min(min(gA), min(gM)) * 0.6, max(max(gA), max(gM)) * 1.5)


def draw(tr, bg, warm, method, fname, wlabel):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.5), gridspec_kw={'height_ratios': [2, 1]})
    for i, lab, c in sel:
        ax1.plot(tr[i], color=c, lw=0.7, alpha=0.85, label=f'{lab} (node {i})')
        ax1.axhline(thstar[i], color=c, ls='--', lw=1.1)
    ax1.axvline(warm, color='k', ls=':', lw=1, label=wlabel)
    ax1.set_ylim(*YL); ax1.set_ylabel(r'$\theta_j^{(t)}$'); ax1.legend(loc='upper right', fontsize=9)
    ax1.text(0.012, 0.96, f'{method}, seed {SEED}', transform=ax1.transAxes, va='top', fontsize=9, style='italic')
    ax2.plot(bg, color='#d62728', lw=1.5, label=r'$U_{\min}-U^*$ (opt.gap)')
    ax2.axhline(bg[-1], color='#d62728', ls='--', lw=0.9, alpha=0.7)
    ax2.axvline(warm, color='k', ls=':', lw=1)
    ax2.set_yscale('log'); ax2.set_ylim(*GL); ax2.set_xlabel('Iteration'); ax2.set_ylabel(r'$U-U^*$')
    ax2.legend(loc='upper right', fontsize=9)
    msg = f'final opt.gap = {bg[-1]:.3f}' if bg[-1] < 1 else f'final opt.gap = {bg[-1]:.1f}   (stalled, NOT at optimum)'
    ax2.text(MAXB * 0.42, bg[-1] * 1.8, msg, color='#d62728', fontsize=10, fontweight='bold')
    plt.tight_layout(); plt.savefig(fname, dpi=120); plt.close(); print("saved", fname, f"{bg[-1]:.3f}")


D = "/home/jiyoon/BSS-Keyphrase-Extraction/simulation/study_fav/"
draw(tA, gA, warm, 'AWSGLD', D + "trace_AWSGLD_basic.png", 'warm-up end (t=300)')
draw(tM, gM, 10, 'acMH', D + "trace_acMH_basic.png", 'proposal-adapt start (t=10)')
print(f'shared y: theta {YL}, gap {GL}')
