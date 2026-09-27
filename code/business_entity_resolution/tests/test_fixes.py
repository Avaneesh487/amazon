"""Synthetic-data regression tests for the leakage / 1-to-1 / ordering fixes."""
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pipeline as PL  # noqa: E402


def _work_with_s1(tmp_path, n=20, seed=0):
    ids = np.random.default_rng(seed).choice(10**6, n, replace=False).astype(np.uint32)
    pl.DataFrame({"id": ids}).write_parquet(tmp_path / "train_s1.parquet")
    return str(tmp_path), pl.Series("id", ids)


def test_val_split_is_canonical_and_disjoint(tmp_path):
    work, ids = _work_with_s1(tmp_path)
    val = PL.val_s1_ids(work)
    expected = ids.filter(PL.is_val(ids))
    assert sorted(val.to_list()) == sorted(expected.to_list())
    train = ids.filter(~ids.is_in(val.implode()))
    assert set(train.to_list()).isdisjoint(val.to_list())
    assert len(train) + len(val) == len(ids)          # nobody dropped


def test_pruner_excludes_exactly_val_ids(tmp_path, monkeypatch):
    work, ids = _work_with_s1(tmp_path, n=40, seed=1)
    rows = []
    for k, i in enumerate(ids.to_list()):
        for j in range(3):
            rows.append(dict(id1=i, src=2, id2=k * 10 + j, nk=1, c_name=0.9 - 0.3 * j,
                             c_sq=0.5, c_addr=0.8 - 0.3 * j, c_num=float(j == 0), r1=k, r2=k * 3 + j))
    P = pl.DataFrame(rows).with_columns(pl.col("id1").cast(pl.UInt32), pl.col("src").cast(pl.UInt8),
                                        pl.col("id2").cast(pl.UInt32))
    P.write_parquet(tmp_path / "train_pre_X.parquet")
    gt = P.filter(pl.col("id2") % 10 == 0).select("id1", "src", "id2")
    seen = {}

    def fake_fit(Pf, y):
        seen["ids"] = set(Pf["id1"].to_list())
        return {"coef": [0.0] * 6, "intercept": 0.0, "features": []}

    monkeypatch.setattr(PL, "gt_pairs", lambda d: gt)
    monkeypatch.setattr(PL.blocking, "fit_pruner", fake_fit)
    PL.stage_pruner(type("A", (), {"work": work, "data": None})())
    val = set(PL.val_s1_ids(work).to_list())
    assert seen["ids"] == set(ids.to_list()) - val
    assert not (seen["ids"] & val)


def _F(rows):
    return pl.DataFrame(rows, schema={"id1": pl.UInt32, "id2": pl.UInt32, "src": pl.UInt8})


def test_decide_breaks_exact_ties_deterministically():
    F = _F([(5, 100, 2), (3, 100, 2), (7, 100, 3), (9, 200, 2)])
    prob = np.array([0.9, 0.9, 0.8, 0.95])
    outs = [PL.decide(F, prob, 0.5).sort("src", "id2") for _ in range(5)]
    for o in outs[1:]:
        assert o.equals(outs[0])
    got = outs[0].filter((pl.col("src") == 2) & (pl.col("id2") == 100))
    assert got.height == 1 and got["id1"][0] == 3      # lowest id1 wins the tie
    PL.assert_one_to_one(outs[0])


def test_guardrail_catches_double_assignment():
    D = _F([(1, 100, 2), (2, 100, 2)])
    try:
        PL.assert_one_to_one(D)
    except AssertionError:
        return
    raise AssertionError("guardrail did not fire")


def test_write_lists_is_byte_identical(tmp_path):
    s1 = pl.DataFrame({"id": pl.Series([3, 1, 2], dtype=pl.UInt32)})
    pairs = _F([(1, 50, 3), (1, 7, 2), (3, 9, 2), (1, 12, 2), (3, 1, 3)])
    a, b = tmp_path / "a.tsv", tmp_path / "b.tsv"
    PL.write_lists(str(a), ("source1_entity_id", "matched_entity_ids"), s1, pairs)
    PL.write_lists(str(b), ("source1_entity_id", "matched_entity_ids"), s1, pairs)
    assert a.read_bytes() == b.read_bytes()
    lines = a.read_text().splitlines()
    assert lines[0] == "source1_entity_id\tmatched_entity_ids"
    assert lines[1:] == ["S1-3\tS2-9,S3-1", "S1-1\tS2-12,S2-7,S3-50", "S1-2\t"]
