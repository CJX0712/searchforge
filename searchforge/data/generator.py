"""SearchForge · 合成语料与查询生成（确定性 DGP）。

星 (晨星) · 2026-10-07

数据生成过程（DGP）：
  1. 从 Dirichlet(topic_concentration) 抽 K 个主题-词分布。
  2. 每篇文档随机分配一个主题，按其主题分布独立抽 doc_len 个词。
  3. 每条查询抽样自一篇源文档的同一主题，按主题分布抽 query_len 个词，
     并混入 query_noise 比例的背景均匀噪声。
  4. 相关性 grade(q,d) = 查询与文档共享词数（带重数），上限 5；
     relevant ⟺ grade >= overlap_threshold。

该设定下：词法检索（BM25）直接捕获词重叠信号，稠密检索（LSI）捕获主题级
语义信号，二者互补；relevance 由数据本身决定，绝不泄漏任何模型信息。
"""

from __future__ import annotations

import numpy as np

from ..core.errors import E200DataError
from ..core.types import Corpus, Document, Qrels, Query


def generate_corpus(
    seed: int,
    n_docs: int,
    vocab_size: int,
    n_topics: int,
    doc_len: int,
    topic_concentration: float,
    topic_support: int = 120,
    common_frac: float = 0.25,
) -> Corpus:
    """生成带显式主题支撑集的语料。

    DGP 关键设计（难度甜点的可控旋钮）：
      * 每个主题有大小为 ``topic_support`` 的支撑集；``common_frac`` 比例的支撑词
        来自**跨主题共享的公共词池**，因此异主题文档会与查询产生词法重叠，构成
        「困难负例」——词法信号（BM25）因此成为必要但**不充分**的判据。
      * 同主题文档期望重叠 ≈ query_len × doc_len / topic_support，可直接调。
    """
    rng = np.random.default_rng(seed)
    if topic_concentration <= 0:
        raise E200DataError("topic_concentration 必须为正")
    if not (0.0 <= common_frac < 1.0):
        raise E200DataError("common_frac 必须属于 [0,1)")
    support_size = int(max(2, min(topic_support, vocab_size)))
    n_common = int(round(common_frac * support_size))
    common = (
        rng.choice(vocab_size, size=min(n_common, vocab_size), replace=False)
        if n_common > 0
        else np.array([], dtype=np.int64)
    )
    priv_n = max(1, support_size - len(common))
    topic_word = np.zeros((n_topics, vocab_size), dtype=np.float64)
    for z in range(n_topics):
        priv = rng.choice(vocab_size, size=priv_n, replace=False)
        sup = np.unique(np.concatenate([common, priv]))
        p = rng.dirichlet(np.full(len(sup), topic_concentration))
        topic_word[z, sup] = p
    doc_topic = rng.integers(0, n_topics, size=n_docs)
    docs: list[Document] = []
    for i in range(n_docs):
        z = int(doc_topic[i])
        terms = rng.choice(vocab_size, size=doc_len, p=topic_word[z]).tolist()
        docs.append(Document(id=i, terms=terms, topic=z))
    return Corpus(docs=docs, topic_word=topic_word, doc_topic=doc_topic)


def generate_queries(
    seed: int,
    corpus: Corpus,
    n_queries: int,
    vocab_size: int,
    query_len: int,
    query_noise: float,
) -> list[Query]:
    rng = np.random.default_rng(seed + 7919)  # 与语料种子错开，保证独立
    if n_queries > corpus.n_docs:
        src = rng.choice(corpus.n_docs, size=n_queries, replace=True)
    else:
        src = rng.choice(corpus.n_docs, size=n_queries, replace=False)
    uniform = np.full(vocab_size, 1.0 / vocab_size)
    queries: list[Query] = []
    for qi, s in enumerate(src):
        z = int(corpus.doc_topic[s])
        p = (1.0 - query_noise) * corpus.topic_word[z] + query_noise * uniform
        terms = rng.choice(vocab_size, size=query_len, p=p).tolist()
        queries.append(Query(id=qi, terms=terms, topic=z, source=int(s)))
    return queries


def _doc_term_counts(corpus: Corpus, vocab_size: int) -> np.ndarray:
    """构造 [n_docs, vocab_size] 词频矩阵（稠密，V 不大）。"""
    n = corpus.n_docs
    mat = np.zeros((n, vocab_size), dtype=np.int32)
    for d in corpus.docs:
        if not d.terms:
            continue
        t = np.asarray(d.terms, dtype=np.int64)
        uniq, cnt = np.unique(t, return_counts=True)
        mat[d.id, uniq] = cnt
    return mat


def build_qrels(
    corpus: Corpus,
    queries: list[Query],
    vocab_size: int,
    overlap_threshold: int,
    grade_cap: int = 5,
    require_topic: bool = True,
    strong_offset: int = 2,
) -> Qrels:
    """分级相关性（TREC 式 0..3 档）。

    相关性判据 = **潜在主题匹配**（语义）为门控，再按词法重叠分档：
        grade = 0                       主题不匹配（哪怕词法高度重叠也只是诱导负例）
              = 1                       同主题
              = 2                       同主题 且 共享词数 >= overlap_threshold
              = 3                       同主题 且 共享词数 >= overlap_threshold + strong_offset

    这样设计的关键（泄漏控制）：标签依赖**潜在主题**这一隐变量，而任何单个特征
    （bm25 / dense / overlap）都只是它的噪声观测，无法单独决定标签 ⟹ 避免
    "把标签当特征喂给模型"的伪增益，同时让词法与语义信号真正互补。
    """
    D = _doc_term_counts(corpus, vocab_size)
    doc_topic = np.asarray(corpus.doc_topic, dtype=np.int64)
    qrels = Qrels()
    thr1 = int(overlap_threshold)
    thr2 = int(overlap_threshold + strong_offset)
    for q in queries:
        qt = np.zeros(vocab_size, dtype=np.int32)
        t = np.asarray(q.terms, dtype=np.int64)
        uniq, cnt = np.unique(t, return_counts=True)
        qt[uniq] = cnt
        # overlap_t = min(D[:,t], q_t) 按词取最小，再求和
        # 只沿查询的非零词列计算，避免 [n_docs, V] 全量广播
        nz = np.flatnonzero(qt)
        shared = np.minimum(D[:, nz], qt[nz]).sum(axis=1)  # [n_docs]
        same_topic = doc_topic == int(q.topic)
        grades = np.where(
            same_topic,
            1 + (shared >= thr1).astype(np.int32) + (shared >= thr2).astype(np.int32),
            0,
        ).astype(np.int32)
        grades = np.minimum(grades, grade_cap)
        rel = {int(d): int(g) for d, g in enumerate(grades) if g > 0}
        qrels.data[q.id] = rel
    return qrels


class SyntheticDataset:
    """一次性生成可复现数据集。"""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.corpus = generate_corpus(
            seed=cfg.seed,
            n_docs=cfg.n_docs,
            vocab_size=cfg.vocab_size,
            n_topics=cfg.n_topics,
            doc_len=cfg.doc_len,
            topic_concentration=cfg.topic_concentration,
            topic_support=cfg.topic_support,
            common_frac=cfg.common_frac,
        )
        self.queries = generate_queries(
            seed=cfg.seed,
            corpus=self.corpus,
            n_queries=cfg.n_queries,
            vocab_size=cfg.vocab_size,
            query_len=cfg.query_len,
            query_noise=cfg.query_noise,
        )
        self.qrels = build_qrels(
            corpus=self.corpus,
            queries=self.queries,
            vocab_size=cfg.vocab_size,
            overlap_threshold=cfg.overlap_threshold,
            require_topic=getattr(cfg, "require_topic", True),
            strong_offset=getattr(cfg, "strong_offset", 2),
        )
        # 诊断：相关文档数分布
        self._diag = self._diagnose()

    def _diagnose(self) -> dict:
        rel_counts = [len(self.qrels.relevant(q.id)) for q in self.queries]
        arr = np.asarray(rel_counts, dtype=float)
        return {
            "queries": len(self.queries),
            "docs": self.corpus.n_docs,
            "mean_relevant": float(arr.mean()),
            "min_relevant": int(arr.min()),
            "max_relevant": int(arr.max()),
        }

    @property
    def diagnosis(self) -> dict:
        return self._diag
