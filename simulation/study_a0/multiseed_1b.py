"""Study 1B 멀티시드 평균 — seed 0..4 를 돌려 표의 모든 열을 mean±sd 로 집계.

열: MSE / Spearman / R̂max / S 도달률 / S 도달시간(median) / NDCG@10 / NDCG@160 / 시간(초/chain)

방식
----
- seed 0: 기존 저장물(ava_/sgld_metric_summary.json, ndcg_summary.json) 재사용 (재계산 X).
- seed 1..4: acmh_vs_awsgld.py, sgld_only.py 를 subprocess 로 실행 → 요약 JSON 읽고 NDCG 계산.
- seed 0 산출물(2 npz + 3 json)은 시작 시 백업, 종료 시 복원 → 기존 그림 정합성 유지.

지표 출처는 원 러너와 100% 동일 (같은 코드가 만든 요약을 그대로 집계).
출력: multiseed_1b_summary.json, multiseed_1b.csv, 콘솔 표.
"""
import os, sys, json, shutil, subprocess, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
S1B = os.path.join(HERE, "..", "study_1b")   # 원본 샘플러(acmh_vs_awsgld/sgld_only)·체인·요약은 study_1b
SEEDS = [0, 1]
METHODS = ["acMH", "SGLD", "qSGLD", "cycSGLD", "SGHMC", "AWSGLD"]
NDCG_METHODS = ["acMH", "SGLD", "qSGLD", "cycSGLD", "SGHMC", "AWSGLD"]
K_NDCG = (10, 160)
BACKUP = os.path.join(S1B, "_seed0_backup")
FILES_S0 = ["ava_results.npz", "sgld_results.npz",
            "ava_metric_summary.json", "sgld_metric_summary.json", "ndcg_summary.json"]


def ndcg_at_k(theta_star, theta_hat, k):
    rel = np.argsort(np.argsort(theta_star)).astype(float) / max(len(theta_star) - 1, 1)
    pred = np.argsort(theta_hat)[::-1][:k]; ideal = np.argsort(theta_star)[::-1][:k]
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = np.sum((2 ** rel[pred] - 1) * disc); idcg = np.sum((2 ** rel[ideal] - 1) * disc)
    return float(dcg / idcg) if idcg > 0 else 0.0


def compute_ndcg(seed):
    """현재 폴더의 ava_/sgld_results.npz (해당 seed 실행 직후) 로 NDCG@k 계산."""
    data = np.load(os.path.join(S1B, f"data_seed{seed}.npz"))
    ava = np.load(os.path.join(S1B, "ava_results.npz"))
    sg = np.load(os.path.join(S1B, "sgld_results.npz"))
    ts = data["theta_star"]; out = {}
    src = {"acMH": ava, "AWSGLD": ava, "SGLD": sg, "qSGLD": sg, "cycSGLD": sg, "SGHMC": sg}
    for m in NDCG_METHODS:
        th = src[m][f"{m}_theta_hat"]
        out[m] = {k: ndcg_at_k(ts, th, k) for k in K_NDCG}
    return out


def read_summaries():
    ava = json.load(open(os.path.join(S1B, "ava_metric_summary.json")))
    sg = json.load(open(os.path.join(S1B, "sgld_metric_summary.json")))
    return ava, sg


def extract(ava, sg, ndcg):
    """method -> dict(mse, spearman, rhat, reach, reach_total, reach_step, ndcg10, ndcg160, wall)."""
    rec = {}
    for j in (ava, sg):
        for m, r in j["recovery"].items():
            rec.setdefault(m, {})["mse"] = r["mse_all"]; rec[m]["spearman"] = r["spearman"]
        for m, r in j["R_hat"].items():
            rec.setdefault(m, {})["rhat"] = r["R_hat_max"]
        for m, r in j["escape_summary"].items():
            s = r["S"]; rec.setdefault(m, {})["reach"] = s["n_escaped"]
            rec[m]["reach_total"] = s["n_total"]; rec[m]["reach_step"] = s["median_step"]
        for m, w in j["wall_time_sec"].items():
            rec.setdefault(m, {})["wall"] = float(np.mean(w))
    for m in NDCG_METHODS:
        rec[m]["ndcg10"] = ndcg[m][10]; rec[m]["ndcg160"] = ndcg[m][160]
    return rec


def run_seed(seed):
    print(f"  [seed {seed}] acmh_vs_awsgld.py ...", flush=True); t0 = time.time()
    subprocess.run([sys.executable, "acmh_vs_awsgld.py", str(seed)], cwd=S1B, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    print(f"  [seed {seed}] sgld_only.py ... ({int(time.time()-t0)}s)", flush=True)
    subprocess.run([sys.executable, "sgld_only.py", str(seed)], cwd=S1B, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    ndcg = compute_ndcg(seed); ava, sg = read_summaries()
    print(f"  [seed {seed}] done ({int(time.time()-t0)}s)", flush=True)
    return extract(ava, sg, ndcg)


def main():
    t_all = time.time()
    # seed 0 산출물 백업
    os.makedirs(BACKUP, exist_ok=True)
    for f in FILES_S0:
        p = os.path.join(S1B, f)
        if os.path.exists(p): shutil.copy2(p, os.path.join(BACKUP, f))
    print(f"[백업] seed0 산출물 {len(FILES_S0)}개 -> {BACKUP}", flush=True)

    per_seed = {}
    # seed 0: 기존 요약 JSON 재사용, NDCG 는 복원된 seed0 npz 에서 직접 계산(SGHMC 포함)
    ndcg0 = compute_ndcg(0)
    ava0, sg0 = read_summaries()
    per_seed[0] = extract(ava0, sg0, ndcg0)
    print("[seed 0] 기존 저장물 재사용 완료", flush=True)

    # seed 1..4 실행
    for s in SEEDS[1:]:
        per_seed[s] = run_seed(s)

    # seed 0 산출물 복원
    for f in FILES_S0:
        b = os.path.join(BACKUP, f)
        if os.path.exists(b): shutil.copy2(b, os.path.join(S1B, f))
    print("[복원] seed0 산출물 복원 완료", flush=True)

    # 집계
    keys = ["mse", "spearman", "rhat", "ndcg10", "ndcg160", "wall"]
    agg = {m: {} for m in METHODS}
    for m in METHODS:
        for k in keys:
            vals = [per_seed[s][m].get(k) for s in SEEDS if k in per_seed[s][m]]
            vals = [v for v in vals if v is not None]
            agg[m][k] = (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"), float("nan"))
        # S 도달률: 평균 n_escaped / n_total
        reach = [per_seed[s][m]["reach"] for s in SEEDS]
        total = per_seed[0][m]["reach_total"]
        agg[m]["reach_rate"] = (float(np.mean(reach)), total)
        # S 도달시간: 성공한 seed 의 median_step 평균 (실패=None 제외)
        steps = [per_seed[s][m]["reach_step"] for s in SEEDS if per_seed[s][m]["reach_step"] is not None]
        agg[m]["reach_step"] = float(np.mean(steps)) if steps else None

    # 출력
    print("\n" + "=" * 118)
    print(f"Study 1B — {len(SEEDS)} seed 평균 (mean±sd)")
    print("=" * 118)
    hdr = f"{'method':>8} | {'MSE':>12} {'Spearman':>13} {'Rhat_max':>13} | {'S도달률':>14} {'S도달time':>10} | {'NDCG@10':>13} {'NDCG@160':>13} | {'sec/chain':>11}"
    print(hdr); print("-" * len(hdr))
    for m in METHODS:
        a = agg[m]
        mse = f"{a['mse'][0]:.3f}±{a['mse'][1]:.2f}"
        sp = f"{a['spearman'][0]:.3f}±{a['spearman'][1]:.2f}"
        rh = f"{a['rhat'][0]:.2f}±{a['rhat'][1]:.2f}"
        rr = f"{a['reach_rate'][0]:.1f}/{a['reach_rate'][1]}"
        rs = "실패" if a["reach_step"] is None else f"{a['reach_step']:.0f}"
        n10 = f"{a['ndcg10'][0]:.3f}" if not np.isnan(a['ndcg10'][0]) else "—"
        n160 = f"{a['ndcg160'][0]:.3f}" if not np.isnan(a['ndcg160'][0]) else "—"
        wl = f"{a['wall'][0]:.2f}"
        print(f"{m:>8} | {mse:>12} {sp:>13} {rh:>13} | {rr:>14} {rs:>10} | {n10:>13} {n160:>13} | {wl:>11}")

    # 저장
    out = {"seeds": SEEDS, "per_seed": {str(s): per_seed[s] for s in SEEDS},
           "aggregate": {m: agg[m] for m in METHODS}}
    with open(os.path.join(HERE, "multiseed_1b_summary.json"), "w") as f:
        json.dump(out, f, indent=2, default=lambda o: o)
    import csv
    with open(os.path.join(HERE, "multiseed_1b.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "MSE_mean", "MSE_sd", "Spearman_mean", "Spearman_sd",
                    "Rhat_max_mean", "Rhat_max_sd", "S_reach_mean", "S_total",
                    "S_reach_step", "NDCG10_mean", "NDCG160_mean", "sec_per_chain"])
        for m in METHODS:
            a = agg[m]
            w.writerow([m, round(a['mse'][0], 4), round(a['mse'][1], 4),
                        round(a['spearman'][0], 4), round(a['spearman'][1], 4),
                        round(a['rhat'][0], 4), round(a['rhat'][1], 4),
                        round(a['reach_rate'][0], 2), a['reach_rate'][1],
                        ("" if a['reach_step'] is None else round(a['reach_step'], 1)),
                        ("" if np.isnan(a['ndcg10'][0]) else round(a['ndcg10'][0], 4)),
                        ("" if np.isnan(a['ndcg160'][0]) else round(a['ndcg160'][0], 4)),
                        round(a['wall'][0], 2)])
    print(f"\n저장: multiseed_1b_summary.json, multiseed_1b.csv  (총 {int(time.time()-t_all)}s)")


if __name__ == "__main__":
    main()
