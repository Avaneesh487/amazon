"""End-to-end driver: data -> blocking -> features -> model -> output files.

    python pipeline.py --data ../../../dataset --work <scratch dir> --out ../../../output all

Stages (each can be run on its own): prepare, cands, train, predict.
"""
import argparse
import gc
import json
import os
import time

import lightgbm as lgb
import numpy as np
import polars as pl

import blocking
import data
import features
import prepare

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


# --------------------------------------------------------------------------
def gt_pairs(data_dir):
    gt = pl.read_csv(os.path.join(data_dir, "train", "train_ground_truth.tsv"), separator="\t",
                     quote_char=None, schema_overrides={"matched_entity_ids": pl.Utf8})
    return (gt.with_columns(pl.col("matched_entity_ids").str.split(","))
              .explode("matched_entity_ids", empty_as_null=True).drop_nulls()
              .filter(pl.col("matched_entity_ids") != "")
              .select(pl.col("source1_entity_id").str.slice(3).cast(pl.UInt32).alias("id1"),
                      pl.col("matched_entity_ids").str.slice(1, 1).cast(pl.UInt8).alias("src"),
                      pl.col("matched_entity_ids").str.slice(3).cast(pl.UInt32).alias("id2")))


def stage_cands(args, split):
    """blocking + hand-score pre-pruning -> <split>_pre_<country>.parquet"""
    for ctry in data.countries(args.work, split):
        out = os.path.join(args.work, f"{split}_pre_{ctry}.parquet")
        if os.path.exists(out) and not args.force:
            log(f"{split}/{ctry}: {out} exists, skip")
            continue
        s1, o = data.load_split(args.work, split, cols=["id", "country", "core", "sq", "atok", "nums"],
                                country=ctry)
        log(f"{split}/{ctry}: {s1.height:,} S1, {o.height:,} S2/S3 - blocking")
        P = blocking.candidates(s1, o, log=log)
        P = P.with_columns(s1["id"].gather(P["r1"]).alias("id1"), o["id"].gather(P["r2"]).alias("id2"),
                           o["src"].gather(P["r2"]).alias("src"))
        P.write_parquet(out)
        del s1, o, P
        gc.collect()


def stage_pruner(args):
    """fit the 6-weight logistic blocking score on the training pre-pruned pairs."""
    gt = gt_pairs(args.data).with_columns(pl.lit(1, pl.UInt8).alias("y"))
    P = pl.concat([pl.read_parquet(os.path.join(args.work, f)) for f in os.listdir(args.work)
                   if f.startswith("train_pre_")], how="diagonal_relaxed")
    P = P.join(gt, on=["id1", "src", "id2"], how="left").with_columns(pl.col("y").fill_null(0))
    # never let the pruner see labels of validation S1 entities
    val = val_s1_ids(args.work)
    held = P["id1"].is_in(val.implode())
    log(f"pruner: excluding {len(val):,} validation S1 entities "
        f"({int(held.sum()):,} of {P.height:,} pre-pruned pairs)")
    P = P.filter(~held)
    w = blocking.fit_pruner(P, P["y"].to_numpy())
    json.dump(w, open(os.path.join(args.work, "pruner.json"), "w"), indent=1)
    log(f"pruner weights: {w}")


def stage_feats(args, split):
    """final pruning + pair features -> <split>_feat_<country>.parquet"""
    w = json.load(open(os.path.join(args.work, "pruner.json")))
    for ctry in data.countries(args.work, split):
        out = os.path.join(args.work, f"{split}_feat_{ctry}.parquet")
        if os.path.exists(out) and not args.force:
            log(f"{split}/{ctry}: {out} exists, skip")
            continue
        s1, o = data.load_split(args.work, split, country=ctry)
        P = blocking.prune(pl.read_parquet(os.path.join(args.work, f"{split}_pre_{ctry}.parquet")), w)
        log(f"{split}/{ctry}: final candidates {P.height:,} ({P.height / s1.height:.2f}/S1) - features")
        v1, v2 = features.record_view(s1), features.record_view(o)
        idf_n = features.idf_table(v1, v2, "core")
        idf_a = features.idf_table(v1, v2, "atok")
        F = features.pair_features(P.drop("src"), v1, v2, idf_n, idf_a)
        F = F.with_columns(pl.lit(ctry).alias("country"))
        F.write_parquet(out)
        log(f"{split}/{ctry}: wrote {out}")
        del s1, o, P, v1, v2, F
        gc.collect()


# --------------------------------------------------------------------------
NON_FEATURES = {"r1", "r2", "id1", "id2", "country", "y", "rank_s1"}


def load_feats(work, split):
    fs = [pl.read_parquet(os.path.join(work, f)) for f in sorted(os.listdir(work))
          if f.startswith(f"{split}_feat_") and f.endswith(".parquet")]
    return pl.concat(fs, how="diagonal_relaxed")


def feature_cols(F):
    return [c for c in F.columns if c not in NON_FEATURES]


def to_X(F, cols):
    return F.select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy()


def is_val(id_expr):
    return (id_expr.hash(seed=42) % 5) == 0


def val_s1_ids(work):
    """the single canonical validation split: ids of the held-out 20 % of train S1.
    Used by pruner fitting, LightGBM training, threshold tuning and reporting."""
    ids = pl.read_parquet(os.path.join(work, "train_s1.parquet"), columns=["id"])["id"]
    return ids.filter(is_val(ids))


def decide(F, prob, thr):
    """1-to-1: each S2/S3 record goes to its most probable S1; keep if >= thr."""
    D = F.select("id1", "id2", "src").with_columns(pl.Series("p", prob))
    # exactly one S1 per (src, id2): highest p, exact ties broken by lowest id1
    D = (D.sort(["src", "id2", "p", "id1"], descending=[False, False, True, False])
          .unique(subset=["src", "id2"], keep="first", maintain_order=True))
    return D.filter(pl.col("p") >= thr)


def assert_one_to_one(D):
    """guardrail: no S2/S3 record may be matched to more than one S1."""
    dup = D.group_by("src", "id2").agg(pl.col("id1").n_unique().alias("n")).filter(pl.col("n") > 1)
    if dup.height:
        raise AssertionError(f"{dup.height} S2/S3 records assigned to >1 S1, e.g. {dup.head(5).to_dicts()}")


def macro_f05(pred, truth, s1_ids):
    """pred/truth: (id1, id2, src) pairs; s1_ids: every S1 in the evaluation set."""
    key = pl.col("src").cast(pl.UInt64) * (1 << 32) + pl.col("id2").cast(pl.UInt64)
    pr = pred.select("id1", key.alias("m")).group_by("id1").agg(pl.col("m").alias("p"))
    tr = truth.select("id1", key.alias("m")).group_by("id1").agg(pl.col("m").alias("t"))
    E = (pl.DataFrame({"id1": s1_ids}).join(pr, on="id1", how="left").join(tr, on="id1", how="left")
           .with_columns(pl.col("p").fill_null([]), pl.col("t").fill_null([])))
    E = E.with_columns(pl.col("p").list.len().alias("np"), pl.col("t").list.len().alias("nt"),
                       pl.col("p").list.set_intersection("t").list.len().alias("tp"))
    prec = pl.col("tp") / pl.col("np")
    rec = pl.col("tp") / pl.col("nt")
    f = (1.25 * prec * rec / (0.25 * prec + rec)).fill_nan(0).fill_null(0)
    E = E.with_columns(pl.when((pl.col("nt") == 0) & (pl.col("np") == 0)).then(1.0)
                         .when((pl.col("nt") == 0) | (pl.col("np") == 0)).then(0.0)
                         .otherwise(f).alias("f"))
    return E["f"].mean(), E


LGB_PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  max_bin=255, num_threads=0, verbose=-1, seed=7)


def fit(X, y, Xv=None, yv=None, rounds=1500):
    dtr = lgb.Dataset(X, y, free_raw_data=True)
    kw = {}
    if Xv is not None:
        kw = dict(valid_sets=[lgb.Dataset(Xv, yv, reference=dtr)],
                  callbacks=[lgb.log_evaluation(100), lgb.early_stopping(50, verbose=False)])
    return lgb.train(LGB_PARAMS, dtr, num_boost_round=rounds, **kw)


def tune_threshold(Fv, pv, truth, s1_val, tag):
    best = (0.0, 0.5)
    for thr in np.arange(0.30, 0.951, 0.025):
        sc, _ = macro_f05(decide(Fv, pv, thr), truth, s1_val)
        best = max(best, (sc, float(round(thr, 3))))
    log(f"  [{tag}] best macro F0.5 = {best[0]:.5f} at thr={best[1]:.3f}")
    return best


def stage_train(args):
    """stage 1: LightGBM on pair features (+ 2-fold out-of-fold predictions);
    stage 2: LightGBM on pair features + cluster-consistency features.
    80 % of train S1 entities fit the models, 20 % (hash split) validate them."""
    import stage2
    F = load_feats(args.work, "train")
    gt = gt_pairs(args.data)
    F = F.join(gt.with_columns(pl.lit(1, pl.UInt8).alias("y")), on=["id1", "src", "id2"],
               how="left", maintain_order="left").with_columns(pl.col("y").fill_null(0))
    cols = feature_cols(F)
    s1_val = val_s1_ids(args.work)
    val = F["id1"].is_in(s1_val.implode())
    Ft, Fv = F.filter(~val), F.filter(val)
    del F
    gc.collect()
    truth = gt.filter(pl.col("id1").is_in(s1_val.implode()))
    log(f"train pairs {Ft.height:,} (pos {Ft['y'].mean():.3f}), val pairs {Fv.height:,}; "
        f"{len(cols)} stage-1 features")

    # ---- stage 1
    yt, yv = Ft["y"].to_numpy(), Fv["y"].to_numpy()
    Xt, Xv = to_X(Ft, cols), to_X(Fv, cols)
    m1 = fit(Xt, yt, Xv, yv, args.rounds)
    m1.save_model(os.path.join(args.work, "model_s1.txt"))
    pv1 = m1.predict(Xv)
    log(f"stage 1: {m1.best_iteration} rounds")
    best1 = tune_threshold(Fv, pv1, truth, s1_val, "stage 1")
    fold = (Ft["id1"].hash(seed=7) % 2 == 0).to_numpy()
    pt1 = np.zeros(Ft.height, dtype=np.float64)
    for k in (True, False):
        mk = fit(Xt[fold == k], yt[fold == k], rounds=max(m1.best_iteration, 50))
        pt1[fold != k] = mk.predict(Xt[fold != k])
        del mk
    del Xt, Xv
    gc.collect()

    # ---- stage 2
    Gt = stage2.group_features(Ft, pt1, args.work, "train")
    Gv = stage2.group_features(Fv, pv1, args.work, "train")
    cols2 = cols + [c for c in Gt.columns if c == "p1" or c.startswith("g_")]
    m2 = fit(to_X(Gt, cols2), yt, to_X(Gv, cols2), yv, args.rounds)
    m2.save_model(os.path.join(args.work, "model_s2.txt"))
    pv2 = m2.predict(to_X(Gv, cols2))
    log(f"stage 2: {m2.best_iteration} rounds")
    best2 = tune_threshold(Fv, pv2, truth, s1_val, "stage 2")

    use2 = best2[0] > best1[0]
    best = best2 if use2 else best1
    model = m2 if use2 else m1
    fcols = cols2 if use2 else cols
    ceil = int(yv.sum()) / truth.height
    imp = dict(zip(fcols, model.feature_importance("gain").round().astype(int).tolist()))
    rep = dict(val_f05=best[0], threshold=best[1], use_stage2=bool(use2),
               val_f05_stage1=best1[0], threshold_stage1=best1[1],
               val_f05_stage2=best2[0], threshold_stage2=best2[1],
               blocking_recall_val=ceil, candidates_per_s1_val=Fv.height / len(s1_val),
               n_train_pairs=Ft.height, features=cols, features2=cols2,
               importance=dict(sorted(imp.items(), key=lambda x: -x[1])))
    with open(os.path.join(args.work, "train_report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    log(f"VAL macro F0.5 = {best[0]:.5f} (stage {2 if use2 else 1}) at thr={best[1]:.3f}; "
        f"blocking recall={ceil:.4f}; candidates/S1={rep['candidates_per_s1_val']:.2f}")


# --------------------------------------------------------------------------
def write_lists(path, header, s1_ids, pairs):
    """one row per S1 id; pairs: (id1, src, id2)."""
    lists = (pairs.select("id1", pl.format("S{}-{}", pl.col("src"), pl.col("id2")).alias("e"))
                  .unique().sort("id1", "e").group_by("id1", maintain_order=True).agg(pl.col("e").str.join(",")))
    out = (s1_ids.join(lists, left_on="id", right_on="id1", how="left")
                 .select(pl.format("S1-{}", pl.col("id")).alias(header[0]),
                         pl.col("e").fill_null("").alias(header[1])))
    out.write_csv(path, separator="\t", quote_style="never")


def stage_predict(args):
    import stage2
    rep = json.load(open(os.path.join(args.work, "train_report.json")))
    F = load_feats(args.work, "test")
    m1 = lgb.Booster(model_file=os.path.join(args.work, "model_s1.txt"))
    prob = m1.predict(to_X(F, rep["features"]))
    if rep["use_stage2"]:
        m2 = lgb.Booster(model_file=os.path.join(args.work, "model_s2.txt"))
        F = stage2.group_features(F, prob, args.work, "test")
        prob = m2.predict(to_X(F, rep["features2"]))
    D = decide(F, prob, rep["threshold"])
    assert_one_to_one(D)
    # keep S1 ids in the original file order
    s1_ids = pl.read_parquet(os.path.join(args.work, "test_s1.parquet"), columns=["id"])
    os.makedirs(args.out, exist_ok=True)
    write_lists(os.path.join(args.out, "matching_results.tsv"),
                ("source1_entity_id", "matched_entity_ids"), s1_ids, D)
    write_lists(os.path.join(args.out, "candidate_pairs.tsv"),
                ("source1_entity_id", "candidate_entity_ids"), s1_ids, F.select("id1", "src", "id2"))
    log(f"test: {F.height:,} candidate pairs ({F.height / s1_ids.height:.2f}/S1), "
        f"{D.height:,} matches ({D.height / s1_ids.height:.2f}/S1), "
        f"S1 with no match: {1 - D['id1'].n_unique() / s1_ids.height:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "cands", "pruner", "feats", "train", "predict",
                                      "all"])
    ap.add_argument("--data", default="../../../dataset")
    ap.add_argument("--work", default=os.path.expanduser("~/er_work"))
    ap.add_argument("--out", default="../../../output")
    ap.add_argument("--rounds", type=int, default=1500)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--splits", default="train,test", help="splits for cands/feats stages")
    args = ap.parse_args()
    if args.stage in ("prepare", "all"):
        prepare.run(args.data, args.work)
    if args.stage in ("cands", "all"):
        for split in args.splits.split(","):
            stage_cands(args, split)
    if args.stage in ("pruner", "all"):
        stage_pruner(args)
    if args.stage in ("feats", "all"):
        for split in args.splits.split(","):
            stage_feats(args, split)
    if args.stage in ("train", "all"):
        stage_train(args)
    if args.stage in ("predict", "all"):
        stage_predict(args)


if __name__ == "__main__":
    main()
