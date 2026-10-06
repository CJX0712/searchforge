# Changelog

作者：晨星 (CJX0712)

## v0.1.0 — 2026-10-07

首个发布版本。**SearchForge · 世界级神经/混合检索与学习重排系统**。

### 新增
- `core/`：types（dataclass）· errors（E100~E500）· config（`SEARCHFORGE_*` 环境变量覆盖 + schema 校验）· interfaces（Protocol）· seed（全局确定性）
- `data/`：确定性合成语料生成器 + **主题门控的分级相关性标注**（0~3 档）
- `retrieval/`：
  - `lexical`：BM25（`rank_bm25`），离线兜底 TF-IDF 余弦
  - `dense`：FAISS `IndexFlatIP` + LSI 嵌入，离线兜底 numpy 精确余弦
  - `hybrid`：Reciprocal Rank Fusion（Cormack et al., SIGIR 2009）
- `rerank/`：6 维 LTR 特征 + LightGBM LambdaMART，离线兜底 `sklearn.LogisticRegression`
- `hpo/`：Optuna 调参（独立验证种子 9042，防调参泄漏）
- `eval/`：nDCG@K / MAP@K / MRR@K / Recall@K
- `pipeline/`：端到端流水线 + 多 seed 基准 + 显著性门禁
- `cli.py`：`run` / `bench` / `tune`
- 文档：`docs/architecture.md` · `docs/model_card.md`
- 工程：CI（Python 3.12/3.13 矩阵）· Dockerfile · Makefile · 依赖锁定

### 实测结果（3 seeds：42/43/44）
- 旗舰 `rankfuse` nDCG@10 = **0.9086 ± 0.0175**
- 最强基线 BM25 = 0.8132 → **+11.73%**（门槛 +5%，统计显著）
- 确定性：同 seed 两次运行核心指标差 = 0.00e+00
- 预算：demo 37.4s，峰值内存 187MB

### 方法学修正（重要）
- 初版把相关性定义为"共享词数 ≥ 阈值"，导致 `overlap` 特征近乎等同于标签，
  产生 **+10~13% 伪增益**；改为**主题门控分级相关性**后增益回落为真实值。
- 初版 DGP 有 41% 文档相关（BM25 已 0.98、旗舰满分），经难度扫描重定甜点参数。
- 基线在独立验证种子上充分调参（LSI 维度 6 档 × RRF k 8 档），确认基线增益不可能来自调参不足。
