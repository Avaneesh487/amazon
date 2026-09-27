"""Synthetic-data tests for the selection/report split of the validation fold."""
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pipeline as PL  # noqa: E402


def _case(n_s1=200, seed=3):
    """each S1 has 3 candidates (2 true, 1 distractor); stage 1 and stage 2
    predictions differ only by small noise, so their scores are close."""
    rng = np.random.default_rng(seed)
    ids = pl.Series("id", rng.choice(10**6, n_s1, replace=False).astype(np.uint32))
    rows, y = [], []
    for k, i in enumerate(ids.to_list()):
        for j in range(3):
            rows.append((i, k * 3 + j, 2 + j % 2))
            y.append(j < 2)
    Fv = pl.DataFrame(rows, schema={"id1": pl.UInt32, "id2": pl.UInt32, "src": pl.UInt8},
                      orient="row")
    y = np.array(y)
    base = np.where(y, 0.75, 0.35) + rng.normal(0, 0.15, len(y))
    pv1 = np.clip(base, 0, 1)
    pv2 = np.clip(base + rng.normal(0, 0.02, len(y)), 0, 1)
    truth = Fv.filter(pl.Series(y)).select("id1", "src", "id2")
    return ids, Fv, pv1, pv2, truth


def test_subfolds_partition_validation_ids_deterministically():
    ids, *_ = _case()
    a1, b1 = PL.val_subfolds(ids)
    a2, b2 = PL.val_subfolds(ids)
    assert a1.equals(a2) and b1.equals(b2)
    assert set(a1.to_list()).isdisjoint(b1.to_list())
    assert sorted(a1.to_list() + b1.to_list()) == sorted(ids.to_list())
    assert len(a1) > 0 and len(b1) > 0


def test_selection_is_reproducible_and_report_uses_only_report_subfold():
    ids, Fv, pv1, pv2, truth = _case()
    r1 = PL.select_and_report(Fv, pv1, pv2, truth, ids)
    r2 = PL.select_and_report(Fv, pv1, pv2, truth, ids)
    assert r1 == r2                                          # deterministic
    assert abs(r1["val_f05_stage1_selection"] - r1["val_f05_stage2_selection"]) < 0.05  # close case
    # the reported score is exactly the chosen model scored on the report subfold only
    _, rep_ids = PL.val_subfolds(ids)
    chosen = pv2 if r1["use_stage2"] else pv1
    f_rep, _ = PL.macro_f05(PL.decide(Fv, chosen, r1["threshold"]), truth, rep_ids)
    assert r1["val_f05"] == f_rep
    assert r1["n_s1_selection"] + r1["n_s1_report"] == len(ids)
