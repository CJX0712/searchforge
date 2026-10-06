"""SearchForge · 核心层单测。

星 (晨星) · 2026-10-07
"""

import os

import pytest

from searchforge.core.config import Config
from searchforge.core.errors import E100ConfigError, E500EvalError
from searchforge.core.seed import get_seed, set_all


def test_set_all_sets_seed():
    assert set_all(7) == 7
    assert get_seed() == 7
    set_all(42)


def test_config_env_override(monkeypatch):
    monkeypatch.setenv("SEARCHFORGE_N_DOCS", "321")
    monkeypatch.setenv("SEARCHFORGE_TOPIC_SUPPORT", "150")
    monkeypatch.setenv("SEARCHFORGE_QUERY_NOISE", "0.42")
    monkeypatch.setenv("SEARCHFORGE_REQUIRE_TOPIC", "false")
    cfg = Config.from_env()
    assert cfg.n_docs == 321
    assert cfg.topic_support == 150
    assert abs(cfg.query_noise - 0.42) < 1e-12
    assert cfg.require_topic is False
    monkeypatch.delenv("SEARCHFORGE_N_DOCS")
    monkeypatch.delenv("SEARCHFORGE_TOPIC_SUPPORT")
    monkeypatch.delenv("SEARCHFORGE_QUERY_NOISE")
    monkeypatch.delenv("SEARCHFORGE_REQUIRE_TOPIC")


def test_config_env_eval_k(monkeypatch):
    monkeypatch.setenv("SEARCHFORGE_EVAL_K", "5,10,20")
    cfg = Config.from_env()
    assert cfg.eval_k == (5, 10, 20)
    monkeypatch.delenv("SEARCHFORGE_EVAL_K")


def test_config_guard_train_ratio():
    with pytest.raises(E100ConfigError):
        Config(train_ratio=0.0)


def test_config_guard_lsi_dim_too_large():
    with pytest.raises(E100ConfigError):
        Config(lsi_dim=10_000, vocab_size=100)


def test_eval_guard_k_nonpositive():
    from searchforge.eval.metrics import ndcg_at_k

    with pytest.raises(E500EvalError):
        ndcg_at_k([1, 2], {1: 1}, 0)
