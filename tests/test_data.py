"""SearchForge · 数据层单测（确定性 + 相关性门控 + 难度诊断）。

星 (晨星) · 2026-10-07
"""

import numpy as np
import pytest

from searchforge.core.config import Config
from searchforge.data.generator import SyntheticDataset, build_qrels, generate_corpus


def _small_cfg(**kw):
    base = dict(
        n_docs=120,
        n_queries=40,
        vocab_size=200,
        n_topics=6,
        doc_len=30,
        query_len=8,
        topic_support=60,
        common_frac=0.4,
        query_noise=0.2,
        overlap_threshold=2,
    )
    base.update(kw)
    return Config(**base)


def test_corpus_determinism():
    cfg = _small_cfg()
    a = generate_corpus(
        seed=cfg.seed,
        n_docs=cfg.n_docs,
        vocab_size=cfg.vocab_size,
        n_topics=cfg.n_topics,
        doc_len=cfg.doc_len,
        topic_concentration=cfg.topic_concentration,
        topic_support=cfg.topic_support,
        common_frac=cfg.common_frac,
    )
    b = generate_corpus(
        seed=cfg.seed,
        n_docs=cfg.n_docs,
        vocab_size=cfg.vocab_size,
        n_topics=cfg.n_topics,
        doc_len=cfg.doc_len,
        topic_concentration=cfg.topic_concentration,
        topic_support=cfg.topic_support,
        common_frac=cfg.common_frac,
    )
    assert [d.terms for d in a.docs] == [d.terms for d in b.docs]
    assert np.array_equal(a.doc_topic, b.doc_topic)


def test_dataset_determinism():
    cfg = _small_cfg()
    d1 = SyntheticDataset(cfg)
    d2 = SyntheticDataset(cfg)
    assert d1.qrels.data == d2.qrels.data
    assert [q.terms for q in d1.queries] == [q.terms for q in d2.queries]


def test_qrels_topic_gating():
    """主题门控：异主题文档即使词法重叠也必须为 grade 0（防"标签=特征"伪增益）。"""
    cfg = _small_cfg()
    ds = SyntheticDataset(cfg)
    q = ds.queries[0]
    doc_topic = np.asarray(ds.corpus.doc_topic)
    other = np.flatnonzero(doc_topic != q.topic)
    for d in other[:20]:
        assert ds.qrels.grade(q.id, int(d)) == 0


def test_qrels_same_topic_min_grade_one():
    cfg = _small_cfg(overlap_threshold=99)  # 阈值拉满，只保留 base 档
    ds = SyntheticDataset(cfg)
    q = ds.queries[0]
    doc_topic = np.asarray(ds.corpus.doc_topic)
    same = np.flatnonzero(doc_topic == q.topic)
    assert len(same) > 0
    for d in same[:20]:
        assert ds.qrels.grade(q.id, int(d)) >= 1


def test_qrels_grades_are_in_range():
    cfg = _small_cfg()
    ds = SyntheticDataset(cfg)
    for qid, rel in ds.qrels.data.items():
        for _d, g in rel.items():
            assert 1 <= g <= 3


def test_build_qrels_strong_offset():
    """strong_offset 增大应单调减少 grade=3 的文档数。"""
    cfg = _small_cfg()
    ds = SyntheticDataset(cfg)
    q = ds.queries[0]
    g2 = build_qrels(ds.corpus, ds.queries, cfg.vocab_size, cfg.overlap_threshold, strong_offset=2)
    g8 = build_qrels(ds.corpus, ds.queries, cfg.vocab_size, cfg.overlap_threshold, strong_offset=8)
    n3_a = sum(1 for v in g2.data[q.id].values() if v == 3)
    n3_b = sum(1 for v in g8.data[q.id].values() if v == 3)
    assert n3_a >= n3_b


def test_diagnosis_has_enough_relevant():
    """难度体检：相关文档绝对数量需够统计意义（>=30），否则指标退化为噪声。"""
    # 用足量语料（每主题 50 篇），小测试配置下相关数天然偏少
    ds = SyntheticDataset(_small_cfg(n_docs=300, n_topics=6))
    assert ds.diagnosis["mean_relevant"] >= 30
    assert ds.diagnosis["min_relevant"] >= 0


def test_production_config_relevance_is_statistically_meaningful():
    """生产默认配置的相关文档数必须达标（防难度旋钮被无意调坏）。"""
    ds = SyntheticDataset(Config())
    assert ds.diagnosis["mean_relevant"] >= 30


def test_topic_support_smaller_means_more_overlap():
    """支撑集越小 → 同主题重叠越多 → 相关文档越多（难度旋钮方向正确）。"""
    small = SyntheticDataset(_small_cfg(topic_support=40))
    large = SyntheticDataset(_small_cfg(topic_support=140))
    assert small.diagnosis["mean_relevant"] > large.diagnosis["mean_relevant"]
