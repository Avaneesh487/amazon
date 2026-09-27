"""Stage 2: candidate generation (blocking).

Records are only compared inside the same country. Every record gets a set of
hashed blocking keys; an (S1, S2/S3) pair is a *raw* candidate when it shares at
least one key whose block is small (big blocks are dropped, never enumerated).

Tokens are typed (n: name word, a: address word, #: number). "Rare" means lowest
document frequency inside the country; tokens seen only once (df == 1, i.e.
typos) are never used for blocking. Key families:

    U1  a single very rare token (df <= SINGLE_MAX) - usually only the entity's
        own cluster carries it (e.g. a distinctive surname)
    U2  any pair of the record's 6 rarest tokens
    A   house number x rare address word
    B   rare name word x house number
    C   pair of the 3 rarest name words
    D   squashed core name (meets domain-style names: jodiespub.com)
    E   rare name word x rarest address word

Pruning (two steps, both cheap):
  1. pre-pruning with a fixed hand-written similarity `cheap` (token-set name /
     squashed-name / address similarity + shared number): every S2/S3 record
     keeps its PRE_TOPK_O best S1 records above PRE_FLOOR.
  2. final pruning with `bscore`, a 6-weight logistic score over the same cheap
     similarities (+ number of shared blocking keys), fitted once on train.
     In the training ground truth an S2/S3 record never belongs to two S1
     entities, so each S2/S3 record keeps only its best S1 plus any runner-up
     within DELTA logits; a floor removes hopeless pairs and every S1 keeps at
     most TOPK_S1.
The output of step 2 is exactly what the matching model scores and what is
written to candidate_pairs.tsv.
"""
import gc

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

SINGLE_MAX = 12
NPARTS = 8
MAX_S1, MAX_O = 20, 80          # block-size caps
O_CHUNK = 600_000               # S2/S3 records per chunk
PRE_TOPK_O, PRE_FLOOR, PRE_TOPK_S1 = 5, 0.2, 60
DELTA, BFLOOR, TOPK_S1 = 1.0, -6.0, 12
PRUNE_FEATS = ["c_name", "c_sq", "c_addr", "c_num", "lnk", "na"]


# --------------------------------------------------------------------------
def _typed_hash(col, prefix):
    return pl.col(col).list.eval(pl.format(prefix + "{}", pl.element()).hash(seed=11))


def hashed_tokens(d):
    return d.select(
        "r",
        _typed_hash("core", "n:").alias("nh"),
        _typed_hash("atok", "a:").alias("ah"),
        _typed_hash("nums", "#:").alias("mh"),
        pl.col("nums").list.unique(maintain_order=True).list.head(2)
          .list.eval(pl.format("#:{}", pl.element()).hash(seed=11)).alias("hn"),
        pl.when(pl.col("sq").str.len_chars() >= 5).then(pl.col("sq").hash(seed=13))
          .alias("sqh"),
    )


def doc_freq(h1, h2):
    parts = []
    for h in (h1, h2):
        for c in ("nh", "ah", "mh"):
            parts.append(h.select(pl.col(c).list.unique().alias("u")).explode("u").drop_nulls())
    return pl.concat(parts).group_by("u").agg(pl.len().cast(pl.UInt32).alias("df"))


def rare_lists(h, df):
    ex = pl.concat([
        h.select("r", pl.col(c).list.unique().alias("u"), pl.lit(t, pl.UInt8).alias("typ"))
         .explode("u").drop_nulls()
        for t, c in enumerate(("nh", "ah", "mh"))])
    ex = ex.join(df, on="u").filter(pl.col("df") >= 2).sort(["r", "df", "u"])
    g = ex.group_by("r", maintain_order=True).agg(
        pl.col("u").filter(pl.col("typ") == 0).head(3).alias("rn"),
        pl.col("u").filter(pl.col("typ") == 1).head(2).alias("ra"),
        pl.col("u").head(6).alias("top"),
        pl.col("df").head(6).alias("topdf"),
    )
    out = h.select("r", "hn", "sqh").join(g, on="r", how="left")
    return out.with_columns(pl.col("rn", "ra", "top").fill_null(pl.lit([], pl.List(pl.UInt64))),
                            pl.col("topdf").fill_null(pl.lit([], pl.List(pl.UInt32))),
                            pl.col("top").list.sort().alias("tops"))


def _k(fam, a, b=None):
    fields = [pl.lit(fam, pl.UInt8).alias("f"), a.alias("a")]
    if b is not None:
        fields.append(b.alias("b"))
    return pl.struct(fields).hash(seed=5)


def gen_keys(R, part, nparts=NPARTS):
    """(r, k) rows of all key families, restricted to hash partition `part`."""
    frames = []

    def add(df):
        frames.append(df.filter((pl.col("k") % nparts) == part))

    def get(col, i):
        return pl.col(col).list.get(i, null_on_oob=True)

    # U1 single rare tokens
    add(R.select("r", "top", "topdf").explode("top", "topdf").drop_nulls()
          .filter(pl.col("topdf") <= SINGLE_MAX).select("r", _k(0, pl.col("top")).alias("k")))
    # U2 pairs of the 6 rarest tokens
    for i in range(6):
        for j in range(i + 1, 6):
            add(R.filter(pl.col("tops").list.len() > j)
                 .select("r", _k(1, get("tops", i), get("tops", j)).alias("k")))
    # A house number x rare address word, B rare name word x house number
    for i in range(2):
        for j in range(2):
            add(R.select("r", _k(2, get("hn", i), get("ra", j)).alias("k")).drop_nulls())
            add(R.select("r", _k(3, get("rn", i), get("hn", j)).alias("k")).drop_nulls())
    # C pairs among the 3 rarest name words (order-free)
    for i, j in ((0, 1), (0, 2), (1, 2)):
        a, b = get("rn", i), get("rn", j)
        add(R.select("r", _k(4, pl.min_horizontal(a, b), pl.max_horizontal(a, b)).alias("k"))
             .drop_nulls())
    # D squashed name
    add(R.select("r", _k(5, pl.col("sqh")).alias("k")).drop_nulls())
    # E rare name word x rarest address word
    for i in range(2):
        add(R.select("r", _k(6, get("rn", i), get("ra", 0)).alias("k")).drop_nulls())
    return pl.concat(frames).unique()


def s1_key_index(R1, R2):
    """S1 keys whose block is small on both sides: (r1, k)."""
    idx = []
    for part in range(NPARTS):
        k1 = gen_keys(R1, part)
        k2 = gen_keys(R2, part)
        c1 = k1.group_by("k").agg(pl.len().alias("n1")).filter(pl.col("n1") <= MAX_S1)
        c2 = k2.group_by("k").agg(pl.len().alias("n2")).filter(pl.col("n2") <= MAX_O)
        ok = c1.join(c2, on="k").select("k")
        idx.append(k1.join(ok, on="k").rename({"r": "r1"}))
        del k1, k2, c1, c2, ok
        gc.collect()
    return pl.concat(idx)


# --------------------------------------------------------------------------
# cheap similarity used for pruning (hand-written, no learning)
def _cd(scorer, a, b):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)


def cheap_view(d):
    return d.select(
        "r",
        pl.col("core").list.join(" ").alias("cs"),
        "sq",
        pl.concat_list([pl.col("atok"), pl.col("nums")]).list.join(" ").alias("aa"),
        "nums",
    )


def cheap_scores(p, v1, v2):
    """p: (r1, r2, nk). Adds c_name, c_sq, c_addr, c_num, cheap."""
    a = p.select(pl.col("r1").alias("r")).join(v1, on="r", how="left", maintain_order="left")
    b = p.select(pl.col("r2").alias("r")).join(v2, on="r", how="left", maintain_order="left")
    c_name = _cd(fuzz.token_set_ratio, a["cs"].to_list(), b["cs"].to_list()) / 100
    c_sq = _cd(fuzz.partial_ratio, a["sq"].to_list(), b["sq"].to_list()) / 100
    c_addr = _cd(fuzz.token_set_ratio, a["aa"].to_list(), b["aa"].to_list()) / 100
    c_num = (a["nums"].list.set_intersection(b["nums"]).list.len() > 0).cast(pl.Float32)
    name = np.maximum(c_name, 0.9 * c_sq)
    cheap = 0.55 * name + 0.30 * c_addr + 0.15 * c_num.to_numpy()
    return p.with_columns(pl.Series("c_name", c_name), pl.Series("c_sq", c_sq),
                          pl.Series("c_addr", c_addr), c_num.alias("c_num"),
                          pl.Series("cheap", cheap.astype(np.float32)))


def candidates(s1, o, topk_o=PRE_TOPK_O, topk_s1=PRE_TOPK_S1, floor=PRE_FLOOR, log=print):
    """s1/o: normalised frames of one country with dense row index r.
    Returns pre-pruned pairs (r1, r2, nk, c_*, cheap, n_o)."""
    h1, h2 = hashed_tokens(s1), hashed_tokens(o)
    df = doc_freq(h1, h2)
    R1, R2 = rare_lists(h1, df), rare_lists(h2, df)
    del h1, h2, df
    gc.collect()
    K1 = s1_key_index(R1, R2)
    log(f"    s1 key index: {K1.height:,} rows")
    v1, v2 = cheap_view(s1), cheap_view(o)
    out, n_raw = [], 0
    for lo in range(0, o.height, O_CHUNK):
        Rc = R2.filter((pl.col("r") >= lo) & (pl.col("r") < lo + O_CHUNK))
        kc = pl.concat([gen_keys(Rc, part) for part in range(NPARTS)])
        p = (kc.join(K1, on="k").group_by("r1", pl.col("r").alias("r2"))
               .agg(pl.len().cast(pl.UInt16).alias("nk")))
        n_raw += p.height
        p = cheap_scores(p, v1, v2)
        p = (p.filter(pl.col("cheap") >= floor)
              .sort(["r2", "cheap"], descending=[False, True])
              .with_columns(pl.int_range(pl.len()).over("r2").alias("_rk"),
                            pl.len().over("r2").cast(pl.UInt16).alias("n_o"))
              .filter(pl.col("_rk") < topk_o).drop("_rk"))
        out.append(p)
        del kc, p
        gc.collect()
    P = pl.concat(out)
    P = (P.sort(["r1", "cheap"], descending=[False, True])
          .filter(pl.int_range(pl.len()).over("r1") < topk_s1))
    log(f"    raw pairs {n_raw:,} ({n_raw / max(s1.height, 1):.1f}/S1) -> "
        f"pre-pruned {P.height:,} ({P.height / max(s1.height, 1):.2f}/S1)")
    return P


# --------------------------------------------------------------------------
# final pruning with a 6-weight logistic blocking score
def prune_matrix(P):
    return P.select(
        "c_name", "c_sq", "c_addr", "c_num",
        pl.col("nk").cast(pl.Float32).log1p().alias("lnk"),
        (pl.col("c_name") * pl.col("c_addr")).alias("na"),
    ).cast(pl.Float32).to_numpy()


def fit_pruner(P, y, n=3_000_000, seed=1):
    from sklearn.linear_model import LogisticRegression
    idx = np.random.default_rng(seed).permutation(P.height)[:n]
    lr = LogisticRegression(C=1.0, max_iter=1000).fit(prune_matrix(P[idx]), y[idx])
    return {"coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0]),
            "features": PRUNE_FEATS}


def prune(P, w, delta=DELTA, floor=BFLOOR, topk_s1=TOPK_S1):
    """final candidate set + blocking-context columns."""
    s = prune_matrix(P) @ np.asarray(w["coef"], dtype=np.float32) + w["intercept"]
    P = P.with_columns(pl.Series("bscore", s.astype(np.float32)))
    P = P.with_columns(pl.col("bscore").max().over("r2").alias("_ob"),
                       pl.col("bscore").sort(descending=True).slice(1, 1).first()
                         .over("r2").alias("_o2"))
    # margin over the best *competing* S1 for this S2/S3 record
    P = P.with_columns(
        pl.when(pl.col("bscore") >= pl.col("_ob"))
          .then(pl.col("bscore") - pl.col("_o2")).otherwise(pl.col("bscore") - pl.col("_ob"))
          .fill_null(10.0).alias("o_margin")).drop("_o2")
    P = P.filter((pl.col("_ob") - pl.col("bscore") <= delta) & (pl.col("bscore") >= floor))
    P = (P.sort(["r1", "bscore"], descending=[False, True])
          .with_columns(pl.int_range(pl.len()).over("r1").cast(pl.UInt16).alias("rank_s1"))
          .filter(pl.col("rank_s1") < topk_s1))
    # competition context (after pruning, i.e. what the model actually sees)
    return P.with_columns(
        (pl.col("bscore") - pl.col("_ob")).alias("o_d_best"),
        (pl.col("bscore") - pl.col("bscore").max().over("r1")).alias("s1_d_best"),
        pl.len().over("r1").cast(pl.UInt16).alias("n_s1"),
        pl.len().over("r2").cast(pl.UInt8).alias("n_o_kept"),
    ).drop("_ob")
