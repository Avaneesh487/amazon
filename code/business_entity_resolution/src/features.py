"""Stage 3: pairwise features for (S1 record, S2/S3 record) candidate pairs.

All features are country-agnostic similarities; token weights are IDF computed
inside each country, so frequent region / city / generic-business words get low
weight everywhere (including the unseen France test data).
"""
import math

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

import normalize as N

CHUNK = 1_000_000


def record_view(d):
    """string / list columns used by the feature functions."""
    src = pl.col("src") if "src" in d.columns else pl.lit(1, pl.UInt8).alias("src")
    return d.select(
        "r", src,
        pl.col("core").list.join(" ").alias("cs"),
        "sq", "name_n", "core", "atok", "nums",
        pl.col("atok").list.join(" ").alias("as_"),
        pl.col("nums").list.first().fill_null("").alias("hn"),
        pl.col("core").list.eval(
            pl.element().map_elements(N.phonetic_key, return_dtype=pl.Utf8)).alias("ph"),
        pl.col("f_indic"), pl.col("f_domain"), pl.col("f_addr_empty"),
        pl.col("core").list.len().alias("ncore"),
        pl.col("atok").list.len().alias("natok"),
        pl.col("nums").list.len().alias("nnums"),
        pl.col("name_n").str.len_chars().alias("nlen"),
    )


def idf_table(v1, v2, col):
    n = v1.height + v2.height
    ex = pl.concat([v1.select(pl.col(col).list.unique().alias("t")),
                    v2.select(pl.col(col).list.unique().alias("t"))]).explode("t").drop_nulls()
    return (ex.group_by("t").agg(pl.len().alias("df"))
              .with_columns((pl.lit(math.log(n + 1)) - (pl.col("df") + 1).log()).cast(pl.Float32).alias("w"))
              .select("t", "w"))


def _idf_sum(d, lcol, idf, out):
    """sum of idf over list column lcol, one value per row index `i`."""
    s = (d.select("i", pl.col(lcol).alias("t")).explode("t").drop_nulls()
           .join(idf, on="t", how="left")
           .group_by("i").agg(pl.col("w").fill_null(0).sum().alias(out),
                              pl.col("w").fill_null(0).max().alias(out + "_max")))
    return s


def _cd(scorer, a, b, **kw):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32, **kw)


def pair_features(pairs, v1, v2, idf_n, idf_a):
    """pairs: (r1, r2, ...extra cols kept). Returns pairs + feature columns."""
    outs = []
    for s in range(0, pairs.height, CHUNK):
        p = pairs.slice(s, CHUNK).with_row_index("i")
        a = v1.join(p.select("i", pl.col("r1").alias("r")), on="r").sort("i")
        b = v2.join(p.select("i", pl.col("r2").alias("r")), on="r").sort("i")
        f = {}
        cs_a, cs_b = a["cs"].to_list(), b["cs"].to_list()
        f["nm_ratio"] = _cd(fuzz.ratio, cs_a, cs_b)
        f["nm_tsort"] = _cd(fuzz.token_sort_ratio, cs_a, cs_b)
        f["nm_tset"] = _cd(fuzz.token_set_ratio, cs_a, cs_b)
        f["nm_partial"] = _cd(fuzz.partial_ratio, cs_a, cs_b)
        sq_a, sq_b = a["sq"].to_list(), b["sq"].to_list()
        f["sq_ratio"] = _cd(fuzz.ratio, sq_a, sq_b)
        f["sq_partial"] = _cd(fuzz.partial_ratio, sq_a, sq_b)
        f["sq_jw"] = _cd(JaroWinkler.normalized_similarity, sq_a, sq_b)
        f["full_tset"] = _cd(fuzz.token_set_ratio, a["name_n"].to_list(), b["name_n"].to_list())
        ph_a = a["ph"].list.join(" ").to_list(); ph_b = b["ph"].list.join(" ").to_list()
        f["ph_tset"] = _cd(fuzz.token_set_ratio, ph_a, ph_b)
        as_a, as_b = a["as_"].to_list(), b["as_"].to_list()
        f["ad_tset"] = _cd(fuzz.token_set_ratio, as_a, as_b)
        f["ad_tsort"] = _cd(fuzz.token_sort_ratio, as_a, as_b)
        f["ad_partial"] = _cd(fuzz.partial_token_set_ratio, as_a, as_b)
        hn_a, hn_b = a["hn"].to_list(), b["hn"].to_list()
        f["hn_ratio"] = _cd(fuzz.ratio, hn_a, hn_b)
        F = pl.DataFrame(f)

        # token set algebra (vectorised in polars)
        T = pl.DataFrame({
            "i": p["i"],
            "c1": a["core"], "c2": b["core"], "t1": a["atok"], "t2": b["atok"],
            "m1": a["nums"], "m2": b["nums"],
        }).with_columns(
            pl.col("c1").list.set_intersection("c2").alias("c_and"),
            pl.col("c2").list.set_difference("c1").alias("c_only2"),
            pl.col("c1").list.set_difference("c2").alias("c_only1"),
            pl.col("t1").list.set_intersection("t2").alias("t_and"),
            pl.col("t2").list.set_difference("t1").alias("t_only2"),
            pl.col("m1").list.set_intersection("m2").list.len().alias("num_shared"),
            pl.col("m1").list.unique().list.len().alias("_nm1"),
            pl.col("m2").list.unique().list.len().alias("_nm2"),
        )
        W = T.select("i")
        for lcol, idf, out in (("c1", idf_n, "w_c1"), ("c2", idf_n, "w_c2"),
                               ("c_and", idf_n, "w_cand"), ("c_only2", idf_n, "w_conly2"),
                               ("c_only1", idf_n, "w_conly1"), ("t1", idf_a, "w_t1"),
                               ("t2", idf_a, "w_t2"), ("t_and", idf_a, "w_tand"),
                               ("t_only2", idf_a, "w_tonly2")):
            W = W.join(_idf_sum(T, lcol, idf, out), on="i", how="left")
        W = W.sort("i").fill_null(0)
        T = T.join(W, on="i").sort("i")
        eps = 1e-6
        G = T.select(
            (pl.col("w_cand") / (pl.col("w_c1") + eps)).alias("nm_w_rec1"),
            (pl.col("w_cand") / (pl.col("w_c2") + eps)).alias("nm_w_rec2"),
            (2 * pl.col("w_cand") / (pl.col("w_c1") + pl.col("w_c2") + eps)).alias("nm_w_dice"),
            pl.col("w_conly2").alias("nm_w_extra2"), pl.col("w_conly2_max").alias("nm_w_extra2_max"),
            pl.col("w_conly1").alias("nm_w_miss1"), pl.col("w_conly1_max").alias("nm_w_miss1_max"),
            pl.col("c_and").list.len().alias("nm_n_shared"),
            (pl.col("w_tand") / (pl.col("w_t1") + eps)).alias("ad_w_rec1"),
            (pl.col("w_tand") / (pl.col("w_t2") + eps)).alias("ad_w_rec2"),
            pl.col("w_tonly2").alias("ad_w_extra2"), pl.col("w_tonly2_max").alias("ad_w_extra2_max"),
            pl.col("w_tand").alias("ad_w_shared"),
            pl.col("num_shared"),
            (pl.col("num_shared") / (pl.col("_nm1") + pl.col("_nm2") - pl.col("num_shared") + eps)).alias("num_jacc"),
            ((pl.col("_nm1") > 0) & (pl.col("_nm2") > 0) & (pl.col("num_shared") == 0)).alias("num_conflict"),
            (pl.col("m1").list.first() == pl.col("m2").list.first()).fill_null(False).alias("hn_eq"),
            # b's first number equals any number of a (reordered addresses)
            pl.col("m1").list.contains(pl.col("m2").list.first()).fill_null(False).alias("hn2_in_1"),
            pl.col("m2").list.contains(pl.col("m1").list.first()).fill_null(False).alias("hn1_in_2"),
        )
        # digit-dropped / prefix numbers (424 vs 42)
        hn_pref = [bool(x and y and (x.startswith(y) or y.startswith(x))) for x, y in zip(hn_a, hn_b)]
        H = pl.DataFrame({"hn_prefix": hn_pref})
        meta = pl.DataFrame({
            "src": b["src"],
            "b_indic": b["f_indic"], "b_domain": b["f_domain"], "b_addr_empty": b["f_addr_empty"],
            "a_ncore": a["ncore"], "b_ncore": b["ncore"], "a_natok": a["natok"], "b_natok": b["natok"],
            "a_nnums": a["nnums"], "b_nnums": b["nnums"], "a_nlen": a["nlen"], "b_nlen": b["nlen"],
        })
        outs.append(pl.concat([p.drop("i"), F, G, H, meta], how="horizontal"))
    return pl.concat(outs)
