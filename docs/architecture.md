# SearchForge · 架构设计文档

作者：晨星 (CJX0712) · v0.1.0 · 2026-10-07

---

## 1. 系统定位

SearchForge 是一个**两段式神经检索与学习重排系统**（two-stage retrieval + learned reranking），
复刻现代搜索引擎与 RAG 检索栈的标准架构：

```
                  ┌─────────────── 离线建索引 ───────────────┐
语料 ──┬──→ 词法通路：BM25 (rank_bm25)                        │
       │                                                     │
       └──→ 稠密通路：TF-IDF → TruncatedSVD(LSI) → 归一化 ──→ FAISS IndexFlatIP
                                                     │
                  ┌───────────── 在线查询 ────────────┘
query ──→ 双路召回 top-K ──→ RRF 融合初排 ──→ top-N 候选 ──→ LambdaMART 重排 ──→ 最终排序
                                                      ▲
                                            6 维 LTR 特征（bm25/dense/rrf/overlap/doc_len/query_idf）
```

旗舰方法命名 **RankFuse**：双路召回 + RRF + 学习重排的融合重排器。

### 为什么这个架构是"世界顶级"的现实选择

| 环节 | 采用的顶级开源 | 权威依据 |
|------|---------------|---------|
| 词法召回 | `rank_bm25` (BM25Okapi) | Robertson & Zaragoza, *The Probabilistic Relevance Framework* (2009)；BM25 至今仍是词法检索不可绕过的强基线 |
| 稠密召回 | `faiss-cpu` (IndexFlatIP) | Johnson, Douze & Jégou, *Billion-scale similarity search with GPUs* (arXiv 1702.08734)；Meta 出品，业界 ANN 事实标准 |
| 稠密表示 | `TruncatedSVD` (LSI) | Deerwester et al., *Indexing by Latent Semantic Analysis* (1990) |
| 多路融合 | Reciprocal Rank Fusion | Cormack, Clarke & Büttcher, SIGIR 2009；免训练融合的 SOTA，被商用搜索引擎广泛采用 |
| 学习重排 | `lightgbm` LambdaMART | Burges, *From RankNet to LambdaRank to LambdaMART* (MSR-TR-2010-82)；Bing 等工业级 LTR 主力算法 |
| 超参优化 | `optuna` | Akiba et al., KDD 2019 |

**不自研 SOTA**：本系统不试图从头实现 BM25/ANN/LambdaMART，而是直接复用上述久经验证的实现，
把工程重心放在**把它们正确组合、可复现评测、可降级交付**上。

---

## 2. 模块结构（单向无环）

```
searchforge/
  core/        types(dataclass) · errors(E100~E500) · config(ENV 覆盖) · interfaces(Protocol) · seed
  data/        合成语料/查询生成 + 分级相关性标注（生成器固定 seed）
  retrieval/   lexical(BM25 / TF-IDF 兜底) · dense(FAISS-LSI / numpy 兜底) · hybrid(RRF)
  rerank/      features(LTR 特征) · lambdamart(LightGBM / sklearn 兜底)
  hpo/         Optuna 调参（独立验证种子）
  eval/        nDCG@K / MAP@K / MRR@K / Recall@K
  pipeline/    SearchPipeline.run() + benchmark() + summarize()
  cli.py       argparse 入口（run / bench / tune）
  examples/    run_demo.py（端到端：基准 + 确定性 + 消融 + 失败案例）
tests/         pytest（手算对照 / 不变量 / 降级路径 / 泄漏防护）
```

调用方向严格单向：

```
cli → pipeline → {data, retrieval, rerank, hpo, eval} → core
```

### 接口契约（Protocol）

- `Retriever`：`fit(corpus)` / `search(query, top_k)` / `available()`
- `Fusion`：`fuse({source: ranking}, top_k)` —— 输出对**召回路传入顺序置换不变**
- `Reranker`：`fit(corpus, qrels, train_qids, feat_by_qid)` / `rerank(qid, {doc_id: features})`

### 确定性

唯一入口 `core.seed.set_all(seed)` 一次设齐 `random` / `numpy` / `PYTHONHASHSEED`。
FAISS(IndexFlatIP) 与 TruncatedSVD 为确定性算法；LightGBM 以
`n_jobs=1` + `bagging_fraction=1.0` + `feature_fraction=1.0` + 固定 `random_state` 关闭随机性。
同 seed 两次运行核心指标差异实测 ≤ 1e-9（见 benchmark.json `determinism`）。

---

## 3. 数据生成（DGP）与泄漏控制

### 3.1 生成过程

1. 抽取跨主题**公共词池**（占每个主题支撑集的 `common_frac` 比例）；
2. 每个主题拥有大小 `topic_support` 的支撑集 = 公共词 + 私有词，支撑集上取 Dirichlet 分布；
3. 每篇文档随机分配主题，按主题分布抽 `doc_len` 个词；
4. 每条查询抽样自一篇源文档的同主题，抽 `query_len` 个词，并混入 `query_noise` 比例的背景均匀噪声。

### 3.2 相关性标注（关键设计）

```
grade = 0                              主题不匹配（即便词法高度重叠也只是诱导负例）
      = 1                              同主题
      = 2                              同主题 且 共享词数 >= overlap_threshold
      = 3                              同主题 且 共享词数 >= overlap_threshold + strong_offset
```

> **泄漏控制（这是本系统最重要的一条设计约束）**
>
> 早期版本把相关性直接定义为「共享词数 ≥ 阈值」，而 `overlap` 特征几乎就是标签本身，
> LambdaMART 因此拿到 +10~13% 的**伪增益**（实为"把标签当特征喂进去"）。
>
> 现版本把标签改为依赖**潜在主题**这一隐变量：任何单个特征（`bm25` / `dense` / `overlap`）
> 都只是它的噪声观测，无法单独决定标签。重排器的增益因此只能来自"学会组合两路信号"，
> 而非"记住标签"。增益从虚高的 +10~13% 回落到可核查的真实值。

### 3.3 训练/评测切分

查询按 `train_ratio=0.6` 划分，重排器**只**在 train 查询上拟合，在 held-out test 查询上评测；
单测 `test_reranker_trained_only_on_train_queries` 断言两者不相交。
语料（被检索的文档集合）是共享的——这符合真实检索设定（索引固定、查询不断到来），
且 TF-IDF/LSI 只在语料上拟合、查询只做 transform，不存在从查询侧泄漏。

---

## 4. 评估口径

| 指标 | 定义 |
|------|------|
| nDCG@K | gain = 2^grade − 1，discount = log2(rank+1)，按理想排序归一化（**分级**） |
| MAP@K | 二元相关性（grade>0），命中处 precision 的均值 / 相关总数 |
| MRR@K | 首个相关文档排名倒数 |
| Recall@K | 二元相关性，K 内命中占比 |

无相关文档的查询**从聚合中剔除**而非填 0（填 0 会把"作废"伪装成"最差"，比 nan 更危险）。

---

## 5. 难度甜点扫描（实测，非设计值）

所有数字均为真实运行输出。扫描语料 `n_docs=1200, n_queries=200, vocab=1500`，
指标 nDCG@10，单 seed。

### 5.1 难度旋钮（跨主题混淆 × 查询噪声）

| cmf | qnoi | sup | 相关数 | random | lexical | dense | hybrid | **rankfuse** | gain |
|-----|------|-----|--------|--------|---------|-------|--------|-------------|------|
| 0.50 | 0.15 | 200 | 60.5 | 0.021 | 0.882 | 0.772 | 0.872 | 0.938 | +6.38% |
| 0.65 | 0.15 | 200 | 61.4 | 0.019 | 0.830 | 0.775 | 0.859 | 0.909 | +5.81% |
| 0.80 | 0.15 | 200 | 61.1 | 0.027 | 0.770 | 0.730 | 0.796 | 0.820 | +3.03% |
| **0.50** | **0.30** | **200** | **60.5** | **0.021** | **0.790** | **0.705** | **0.815** | **0.879** | **+7.77%** |
| 0.65 | 0.30 | 200 | 61.4 | 0.016 | 0.743 | 0.666 | 0.773 | 0.833 | +7.77% |
| 0.80 | 0.30 | 200 | 61.1 | 0.029 | 0.613 | 0.625 | 0.665 | 0.703 | +5.59% |
| 0.65 | 0.45 | 200 | 61.4 | 0.015 | 0.617 | 0.579 | 0.668 | 0.696 | +4.08% |
| 0.80 | 0.30 | 300 | 61.2 | 0.021 | 0.580 | 0.546 | 0.625 | 0.656 | +5.11% |

**选定 `cmf=0.50, qnoi=0.30, sup=200`**：基线落在 0.70~0.82（明显低于天花板 1.0，
留出 headroom），随机对照 0.021（证明任务非平凡），重排器增益 +7.8%。

> 反面教训：首轮扫描曾得到「41% 文档都相关」的病态数据——BM25 已 0.98、
> 旗舰直接满分 1.0000，**全算法无区分度**。必须先做难度体检再定参数。

### 5.2 基线充分调参（防止"增益来自基线没调好"）

踩坑教训：基线超参网格过窄会让旗舰增益变成假象。故在**独立验证种子 9042** 上充分扫描：

| lsi_dim | dense | hybrid | lexical | rankfuse |
|---------|-------|--------|---------|----------|
| 32 | 0.5832 | 0.7864 | 0.7986 | **0.8811** |
| 64 | 0.6668 | 0.7929 | 0.7986 | 0.8731 |
| 100 | 0.6889 | **0.7933** | 0.7986 | 0.8661 |
| 128 | **0.6927** | 0.7901 | 0.7986 | 0.8633 |
| 160 | 0.6831 | 0.7837 | 0.7986 | 0.8558 |
| 200 | 0.6863 | 0.7810 | 0.7986 | 0.8548 |

| rrf_k | 1 | 5 | 10 | 30 | 60 | 100 | 200 | 500 |
|-------|---|---|----|----|----|-----|-----|-----|
| hybrid | 0.7743 | 0.7867 | 0.7943 | **0.7952** | 0.7933 | 0.7945 | 0.7919 | 0.7919 |

结论：**BM25 = 0.7986 是所有基线的实际上限**（高于 RRF 混合的最优 0.7952），
且 RRF 的 k 表现平坦（k∈[10,200] 波动 <0.003），与 Cormack 原文"RRF 对 k 不敏感"一致。
基线已充分调参，旗舰增益不可能来自基线调参不足。

### 5.3 HPO 与稳健性

Optuna（TPE，seed=9042，16 trials）在验证集上给出
`{lsi_dim: 32, rrf_k: 118, num_leaves: 31, learning_rate: 0.1}`。
其中 `rrf_k=118` 命中搜索空间上界（10~120），结合 §5.2 的平坦曲线判断为噪声，
故**改用 Cormack 原论文经典值 k=60**（实测不劣）。跨 seed 稳健性复核（seeds 42/43/44）：

| lsi | k | rankfuse(mean±std) | 最强基线 | gain |
|-----|---|--------------------|---------|------|
| **32** | **60** | **0.9068 ± 0.0186** | 0.8132 | **+11.51%** |
| 32 | 30 | 0.9062 ± 0.0152 | 0.8130 | +11.46% |
| 32 | 118 | 0.9046 ± 0.0194 | 0.8135 | +11.21% |
| 48 | 30 | 0.8957 ± 0.0185 | 0.8125 | +10.23% |
| 64 | 30 | 0.8953 ± 0.0176 | 0.8178 | +9.47% |

最终选定 **lsi_dim=32, rrf_k=60**。

### 5.4 门禁

- 指标：nDCG@10
- 门槛：旗舰相对**最强基线**提升 ≥ **+5%** 且统计显著
- 显著性判据：均值差 > ½(σ_旗舰 + σ_基线)
- 门槛在开工前按 pilot（+11.5%，std 0.018）定死，为实测值的 1/2.3，属结构性余量
- **定死后不回头迁就结果**

---

## 6. 离线降级矩阵

| 依赖缺失 | 降级路径 | 单测覆盖 |
|---------|---------|---------|
| `rank_bm25` | 词法改用 TF-IDF 余弦 (`TfidfCosineRetriever`) | `test_tfidf_fallback_works` |
| `faiss` | 稠密改用 numpy 精确余弦 `NumpyCosineRetriever`（**同一 LSI 嵌入，结果逐位一致**） | `test_faiss_and_numpy_fallback_are_equivalent` |
| `lightgbm` | 重排改用 `sklearn.LogisticRegression` 点对点打分 | `test_sklearn_fallback_rerank` |
| 全部缺失 | 仍可端到端跑通 | `test_offline_fallback_path_runs` |

Bench 中缺失的后端不伪造数字：降级路径照常运行并如实标注所用后端
（`res["backend"]` 记录 faiss/bm25 是否可用、`reranker` 名称）。

---

## 7. 性能预算

- demo 端到端（3 seeds 基准 + 确定性双跑 + 4 组消融 + 失败案例分析）实测见 benchmark.json `budget`
- 目标：CPU ≤ 60s，内存峰值 ≤ 2GB
- 语料规模经裁剪以同时满足统计意义与预算（相关文档均值 ≈60 ≥ 30）
