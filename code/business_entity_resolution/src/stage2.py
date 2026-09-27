"""Second-stage ("cluster consistency") features.

After a first LightGBM gives every candidate pair a probability p1, each pair
(a, b) is described by how the *other* candidates of the same S1 entity look:
S2/S3 variants of one business usually repeat each other's address / name form,
so a weak-looking b that closely resembles confidently matched siblings is
probably a true match, and a b that competes with a much stronger S1 is not.
"""
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

import data

HI = 0.5


def _cd(scorer, a, b):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)


def group_features(F, p1, work, split):
    """F: feature frame with r1, r2, country, id1, id2, src. p1: stage-1 prob."""
    F = F.with_columns(pl.Series("p1", p1.astype(np.float32)))
    F = F.with_columns(
        pl.col("p1").rank("ordinal", descending=True).over("country", "r1").cast(pl.UInt8).alias("g_rank"),
        ((pl.col("p1") > HI).sum().over("country", "r1") - (pl.col("p1") > HI).cast(pl.UInt32))
          .alias("g_n_hi_other"),
        (pl.col("p1").sum().over("country", "r1") - pl.col("p1")).alias("g_sum_other"),
        (pl.col("p1").max().over("country", "r2") ).alias("g_o_best"),
    )
    # best stage-1 prob of this S2/S3 record with a *different* S1
    F = F.with_columns(
        pl.when(pl.col("p1") >= pl.col("g_o_best"))
          .then(pl.col("p1").sort(descending=True).slice(1, 1).first().over("country", "r2"))
          .otherwise(pl.col("g_o_best")).fill_null(0).alias("g_o_comp"),
        (pl.col("p1") - pl.col("p1").max().over("country", "r1")).alias("g_d_best"),
    ).drop("g_o_best")

    # similarity of b to the confident siblings c (p1(c) > HI, c != b)
    outs = []
    for ctry in F["country"].unique().to_list():
        Fc = F.filter(pl.col("country") == ctry).select("r1", "r2", "p1")
        _, o = data.load_split(work, split, cols=["id", "country", "name_n", "addr_n"], country=ctry)
        sib = (Fc.join(Fc.filter(pl.col("p1") > HI).rename({"r2": "c", "p1": "pc"}), on="r1")
                 .filter(pl.col("c") != pl.col("r2")))
        nb = o["name_n"].gather(sib["r2"]).to_list(); nc = o["name_n"].gather(sib["c"]).to_list()
        ab = o["addr_n"].gather(sib["r2"]).to_list(); ac = o["addr_n"].gather(sib["c"]).to_list()
        sib = sib.with_columns(
            pl.Series("sn", _cd(fuzz.token_set_ratio, nb, nc)),
            pl.Series("sa", _cd(fuzz.token_set_ratio, ab, ac)),
            pl.Series("sa_eq", [x == y and x != "" for x, y in zip(ab, ac)]),
        )
        g = sib.group_by("r1", "r2").agg(
            pl.col("sn").max().alias("g_sib_name_max"),
            pl.col("sa").max().alias("g_sib_addr_max"),
            ((pl.col("sn") + pl.col("sa")) / 2).max().alias("g_sib_both_max"),
            pl.col("sa_eq").any().alias("g_sib_addr_eq"),
            ((pl.col("sn") * pl.col("pc")).sum() / pl.col("pc").sum()).alias("g_sib_name_w"),
            ((pl.col("sa") * pl.col("pc")).sum() / pl.col("pc").sum()).alias("g_sib_addr_w"),
        ).with_columns(pl.lit(ctry).alias("country"))
        outs.append(g)
        del o, sib, nb, nc, ab, ac
    G = pl.concat(outs)
    return F.join(G, on=["country", "r1", "r2"], how="left", maintain_order="left")
