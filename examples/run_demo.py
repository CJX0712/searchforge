"""SearchForge · 端到端演示（冒烟 + 多 seed 基准 + 确定性校验 + 消融 + 失败案例）。

星 (晨星) · 2026-10-07

输出：控制台表格 + benchmark.json（全部数字来自真实运行，无手填）。
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from searchforge.core.config import Config  # noqa: E402
from searchforge.core.errors import E400RerankError  # noqa: E402
from searchforge.core.seed import set_all  # noqa: E402
from searchforge.eval.metrics import ndcg_at_k  # noqa: E402
from searchforge.pipeline.pipeline import (  # noqa: E402
    FLAGSHIP,
    SearchPipeline,
    benchmark,
    summarize,
)
from searchforge.rerank.features import FEATURE_NAMES  # noqa: E402

try:
    import psutil  # noqa: F401

    _HAS_PSUTIL = True
except Exception:
    _HAS_PSUTIL = False


def _mem_mb() -> float:
    if not _HAS_PSUTIL:
        return 0.0
    import psutil

    return psutil.Process().memory_info().rss / (1024.0 * 1024.0)


def failure_cases(pipe: SearchPipeline, n: int = 3) -> list[dict]:
    """从真实结果派生典型误例：区分「召回天花板受限」与「排序能力受限」。"""
    rr = pipe._train_reranker()
    if rr is None:
        return []
    cases = []
    for qid in pipe.test_qids:
        grades = pipe.qrels.data.get(qid, {})
        if not any(g > 0 for g in grades.values()):
            continue
        bm25, dense, _lx, _dn, fused, rrf_pos = pipe._recall(qid)
        cand = fused[: pipe.cfg.rerank_candidates]
        qset = set(pipe.queries[qid].terms)
        fd = pipe.fe.feature_dict(qid, cand, bm25, dense, rrf_pos, qset)
        ordered = rr.rerank(qid, fd)
        seen = set(ordered)
        ranking = ordered + [d for d in fused if d not in seen]
        achieved = ndcg_at_k(ranking, grades, 10)

        # 天花板：在候选窗口内能做到的最好排序
        cand_sorted = sorted(cand, key=lambda d: -grades.get(d, 0))
        ceiling = ndcg_at_k(cand_sorted, grades, 10)
        # 初排（RRF 融合）在同一查询上的表现，用于区分「召回失败」与「重排失败」
        base_ndcg = ndcg_at_k(fused, grades, 10)
        n_rel_total = sum(1 for g in grades.values() if g > 0)
        n_rel_in_cand = sum(1 for d in cand if grades.get(d, 0) > 0)
        first_rel_rank = next((i for i, d in enumerate(cand) if grades.get(d, 0) > 0), -1)
        # 归因（严格按实测数字判定，不用预设结论）
        if n_rel_in_cand == 0:
            reason = "候选窗口零召回：相关文档全部落在重排窗口之外"
        elif achieved < base_ndcg - 1e-9:
            reason = "重排劣于初排：该查询上学习到的组合信号失效"
        elif first_rel_rank >= 10:
            reason = "召回失败：相关文档在窗口内但初排即被跨主题干扰词压到第11位之后"
        else:
            reason = "排序能力受限：候选已含高相关文档但重排未排到前列"
        cases.append(
            {
                "qid": int(qid),
                "ndcg@10_achieved": round(float(achieved), 4),
                "ndcg@10_ceiling": round(float(ceiling), 4),
                "ndcg@10_first_stage": round(float(base_ndcg), 4),
                "n_relevant_total": int(n_rel_total),
                "n_relevant_in_candidates": int(n_rel_in_cand),
                "first_relevant_rank_in_window": int(first_rel_rank),
                "attribution": reason,
            }
        )
    cases.sort(key=lambda c: c["ndcg@10_achieved"])
    return cases[:n]


def ablation(cfg, pipe: SearchPipeline) -> dict:
    """关键组件消融：去掉稠密特征 / 去掉词法特征 / 只用 RRF 不重排。"""
    groups = {
        "all": list(FEATURE_NAMES),
        "no_dense": [f for f in FEATURE_NAMES if f != "dense"],
        "no_bm25": [f for f in FEATURE_NAMES if f != "bm25"],
        "no_rrf": [f for f in FEATURE_NAMES if f != "rrf"],
    }
    out = {}
    for name, names in groups.items():
        res = pipe.run(feature_names=names)
        out[name] = {
            "ndcg@10": res["methods"][FLAGSHIP]["ndcg@10"],
            "map@10": res["methods"][FLAGSHIP]["map@10"],
        }
    # 不重排（纯 RRF）对照
    out["no_rerank(rrf)"] = {
        "ndcg@10": pipe.run()["methods"]["hybrid"]["ndcg@10"],
        "map@10": pipe.run()["methods"]["hybrid"]["map@10"],
    }
    return out


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    cfg = Config.from_env()
    print("=" * 78)
    print(" SearchForge · RankFuse 端到端演示   作者：晨星 (CJX0712)")
    print("=" * 78)

    # ---- 1. 多 seed 基准 ----
    bench = benchmark(cfg)
    print(summarize(bench))
    print()

    # ---- 2. 确定性校验（同 seed 两次运行核心指标一致）----
    scfg = replace(cfg, seed=cfg.seed)
    set_all(scfg.seed)
    p1 = SearchPipeline(scfg)
    r1 = p1.run()
    set_all(scfg.seed)
    p2 = SearchPipeline(scfg)
    r2 = p2.run()
    diffs = []
    for m in r1["methods"]:
        for k, v in r1["methods"][m].items():
            if k.startswith("_"):
                continue
            diffs.append(abs(v - r2["methods"][m][k]))
    max_diff = max(diffs) if diffs else 0.0
    print(
        f"[determinism] 同 seed 两次运行核心指标最大差 = {max_diff:.2e} "
        f"({'PASS' if max_diff <= 1e-9 else 'WARN'})"
    )
    print()

    # ---- 3. 消融 ----
    abl = ablation(cfg, p1)
    print("[ablation] 关键组件开关对照（单 seed，nDCG@10 / MAP@10）")
    for name, v in abl.items():
        print(f"  {name:<16} nDCG@10={v['ndcg@10']:.4f}  MAP@10={v['map@10']:.4f}")
    print()

    # ---- 4. 失败案例 ----
    try:
        fc = failure_cases(p1, n=3)
        print("[failure cases] 从结果派生的典型误例")
        for c in fc:
            print(
                f"  q={c['qid']:<5} achieved={c['ndcg@10_achieved']:.4f} "
                f"stage1={c['ndcg@10_first_stage']:.4f} "
                f"ceiling={c['ndcg@10_ceiling']:.4f} "
                f"rel_in_win={c['n_relevant_in_candidates']}/{c['n_relevant_total']} "
                f"first_rel@{c['first_relevant_rank_in_window']} "
                f"← {c['attribution']}"
            )
    except E400RerankError as e:
        fc = []
        print(f"[failure cases] 跳过：{e}")
    print()

    elapsed = time.time() - t0
    peak = _mem_mb()
    print(f"[budget] elapsed={elapsed:.1f}s (<=60s)  peak_rss={peak:.0f}MB (<=2048MB)")

    payload = {
        "benchmark": bench,
        "determinism": {"max_metric_diff": max_diff, "pass": bool(max_diff <= 1e-9)},
        "ablation": abl,
        "failure_cases": fc,
        "budget": {
            "elapsed_sec": round(elapsed, 2),
            "peak_rss_mb": round(peak, 1),
            "elapsed_ok": bool(elapsed <= 60.0),
            "memory_ok": bool(peak <= 2048.0),
        },
        "diagnosis": r1["diagnosis"],
        "backend": r1["backend"],
        "config": cfg.as_dict(),
    }
    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmark.json"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[output] benchmark.json -> {out_path}")
    return 0 if bench["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
