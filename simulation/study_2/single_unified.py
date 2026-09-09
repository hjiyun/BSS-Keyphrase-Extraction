"""단일 문서(병합 없음) 6-sampler 통일 비교 — merge 와 동일 지표.
Hulth dense 문서를 1편씩 그래프로 만들어 6샘플러 실행. 20편 돌린 뒤 aggregate_single.py 에서
AWSGLD 가 가장 좋은 10편을 골라 mean±std.

사용: python3 single_unified.py pick        # 20편 doc id 출력
     python3 single_unified.py <doc_id>     # 한 편 실행 → single_unified_<doc>.csv
"""
import os, sys, csv
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import merge_unified as MU

MIN_TRUTH = 5; MAX_N = 260; N_DOC = 20


def pick_docs():
    docs = []
    for dc in MU.IDS:
        try:
            mg = MU.merge([dc], 0)
        except Exception:
            continue
        if len(mg["truth"]) >= MIN_TRUTH and mg["n"] <= MAX_N:
            docs.append(dc)
        if len(docs) >= N_DOC:
            break
    return docs


def run_doc(doc):
    mg = MU.merge([doc], 0)
    D, n, nt = MU.run_mg(mg, 0, label=f"doc={doc}")
    rows = []
    for m in ["acMH", "SGLD", "qSGLD", "cycSGLD", "SGHMC", "AWSGLD"]:
        d = D[m]
        rows.append([doc, n, nt, m, round(d["rmed"], 4), round(d["rq95"], 4), round(d["rmax"], 4),
                     round(d["ess"], 2), round(d["thK"], 4), round(d["thN"], 4), round(d["piK"], 4), round(d["piN"], 4),
                     round(d["low"], 2), round(d["P"], 4), round(d["R"], 4), round(d["ndcg"], 4),
                     round(d["auc"], 4), d["Tstop"]])
    return rows


def main():
    arg = sys.argv[1]
    if arg == "pick":
        print(" ".join(pick_docs()))
        return
    rows = run_doc(arg)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"single_unified_{arg}.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["doc", "n", "n_truth", "method", "rhat_median", "rhat_q95", "rhat_max", "ess",
                    "th_kw", "th_nonkw", "pi_kw", "pi_nonkw", "lowest_U", "P_at_k", "R_at_k", "ndcg_at_k", "auc", "T_stop"])
        w.writerows(rows)
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
