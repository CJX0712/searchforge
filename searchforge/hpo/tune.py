"""SearchForge · Optuna 超参优化（独立验证种子，防调参泄漏）。

星 (晨星) · 2026-10-07

HPO 使用独立的 ``hpo_seed`` 数据集（与 benchmark 种子不相交），
在验证查询上以 nDCG@10 为目标调 RRF k / LSI dim / LambdaMART 超参。
"""

from __future__ import annotations

from dataclasses import replace

from ..core.seed import set_all


def tune(cfg, n_trials: int = 8):
    """在独立验证集上调参，返回 (best_params, study)。"""
    set_all(cfg.hpo_seed)
    try:
        import optuna
        from optuna.samplers import TPESampler
    except Exception:
        return {}, None

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # 与最终配置同尺寸（缩小语料会改变相关文档密度，导致调参结果不迁移）
    vcfg = replace(cfg, seed=cfg.hpo_seed)

    from ..data.generator import SyntheticDataset
    from ..pipeline.pipeline import SearchPipeline

    ds = SyntheticDataset(vcfg)

    def objective(trial):
        rrf_k = trial.suggest_int("rrf_k", 10, 120)
        lsi_dim = trial.suggest_categorical("lsi_dim", [32, 64, 100, 128])
        num_leaves = trial.suggest_categorical("num_leaves", [15, 31, 63])
        lr = trial.suggest_categorical("learning_rate", [0.05, 0.1, 0.2])
        tcfg = replace(
            vcfg,
            rrf_k=rrf_k,
            lsi_dim=lsi_dim,
            ltr_epochs=min(cfg.ltr_epochs, 120),
        )
        try:
            pipe = SearchPipeline(tcfg, dataset=ds)
            res = pipe.run(rerank_params={"num_leaves": num_leaves, "learning_rate": lr})
            return res["methods"]["rankfuse"]["ndcg@10"]
        except Exception:
            return 0.0

    sampler = TPESampler(seed=cfg.hpo_seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study
