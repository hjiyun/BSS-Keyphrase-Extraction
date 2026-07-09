"""
기존 langevin_methods_comparison_summary.json (n=800, 5 methods:
SGLD/qSGLD/cycSGLD/AWSGLD/SGHMC) 에 acMH 결과만 추가한다.

acMH 는 다른 방법과 '동일한 시드'(SEED_BASE+r)로 생성된 graph/theta_star/Y
위에서 실행되므로 기존 결과와 그대로 비교 가능하다 (재현성 검증 완료).

원본은 건드리지 않고 새 파일로 저장:
  langevin_methods_comparison_with_acmh_summary.json
"""
import importlib.util
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "langevin_methods_comparison.py")
EXISTING = os.path.join(HERE, "langevin_methods_comparison_summary.json")
OUT = os.path.join(HERE, "langevin_methods_comparison_with_acmh_summary.json")


def load_module():
    spec = importlib.util.spec_from_file_location("lmc", SRC)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    m = load_module()
    with open(EXISTING, encoding="utf-8") as f:
        payload = json.load(f)

    for sc_cfg in m.SCENARIOS:
        m.BLOCK_PROBS = sc_cfg["block_probs"]
        m.DEFAULT_SCENARIO = sc_cfg["scenario"]
        scen_name = m.DEFAULT_SCENARIO["name"]
        print("=" * 72)
        print(f"acMH on {scen_name}  (R={m.R}, T={m.T}, Burn-in={m.BURN_IN}, n={m.DEFAULT_SCENARIO['n_total']})")
        print("=" * 72)

        acmh_results = []
        for r in range(m.R):
            rng = np.random.default_rng(m.SEED_BASE + r)
            np.random.seed(m.SEED_BASE + r)
            graph = m.build_block_graph(m.DEFAULT_SCENARIO, rng)
            theta_star = m.sample_theta_star(graph["group"], m.DEFAULT_SCENARIO, rng)
            Y, p_obs = m.generate_labels(theta_star, m.DEFAULT_SCENARIO["alpha_true"], rng)
            init_state = m.bss_initial_state(graph, Y)

            res = m.run_acmh_variant(graph, Y, theta_star, p_obs, init_state)
            acmh_results.append(res)
            print(f"[{scen_name} | Trial {r+1}/{m.R}] acMH | n_obs={res.n_obs} | "
                  f"MSE(theta)={res.mse_theta:.4f} | Spearman={res.spearman:.4f} | "
                  f"NDCG@k={res.ndcg_at_k:.4f} | time={res.wall_time_sec:.2f}s", flush=True)

        summary = m.summarize_trials(acmh_results)
        payload["scenarios"][scen_name]["methods"]["acMH"] = {
            "summary": summary,
            "trials": [m.trial_payload(r) for r in acmh_results],
        }
        print(f"[{scen_name}/acMH] MSE(theta)={summary['mse_theta']['mean']:.4f} | "
              f"Spear={summary['spearman']['mean']:.4f} | NDCG={summary['ndcg_at_k']['mean']:.4f} | "
              f"time={summary['wall_time_sec']['mean']:.2f}s\n", flush=True)

    # 기록용으로 acMH 가 동일 시드로 추가됐음을 명시
    payload.setdefault("settings_common", {})["acmh_added_seed_base"] = m.SEED_BASE

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved merged JSON -> {OUT}")


if __name__ == "__main__":
    main()
