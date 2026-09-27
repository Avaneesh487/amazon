"""Stage 1: read the raw TSVs, normalise every record, write parquet.

Output per split (train/test) and source (1/2/3): <work>/<split>_s<k>.parquet with
    id        u32   numeric part of entity_id
    country   str
    name_n    str   folded name (lowercase, ascii, punctuation removed)
    ntok      list  canonical name tokens (Indic tokens transliterated)
    core      list  ntok without legal forms / stop-words
    sq        str   core tokens concatenated (meets domain-style names)
    addr_n    str   folded address
    atok      list  canonical alphabetic address tokens (no numbers / stop-words)
    nums      list  numeric address tokens without leading zeros
    f_indic, f_domain, f_addr_empty   bool flags

The Indic -> Latin token dictionary is learnt from the *training* ground truth
(word-aligned name pairs) and reused for the test split.
"""
import collections
import json
import os

import polars as pl

import normalize as N

TSV_OPTS = dict(separator="\t", quote_char=None,
                schema_overrides={"business_name": pl.Utf8, "business_address": pl.Utf8,
                                  "country": pl.Utf8})


def read_source(path):
    df = pl.read_csv(path, **TSV_OPTS)
    return df.with_columns(
        pl.col("entity_id").str.slice(3).cast(pl.UInt32).alias("id"),
        pl.col("country").fill_null("UNK"),
    )


def _indic_tokens(raw_name):
    """tokens of an Indic-script name, before transliteration."""
    s = N.re.sub(r"[^\wऀ-ൿ]+", " ", raw_name.lower())
    return [t for t in s.split() if t]


def learn_indic_dict(work, data_dir):
    """Align Indic S2/S3 names with their Latin S1 names in the train set."""
    gt = pl.read_csv(os.path.join(data_dir, "train", "train_ground_truth.tsv"),
                     separator="\t", quote_char=None,
                     schema_overrides={"matched_entity_ids": pl.Utf8})
    pairs = (gt.with_columns(pl.col("matched_entity_ids").str.split(","))
               .explode("matched_entity_ids").drop_nulls()
               .filter(pl.col("matched_entity_ids") != "")
               .select(pl.col("source1_entity_id").str.slice(3).cast(pl.UInt32).alias("id1"),
                       pl.col("matched_entity_ids").alias("mid")))
    s1 = pl.read_parquet(os.path.join(work, "train_s1.parquet"), columns=["id", "ntok"])
    counts = collections.defaultdict(collections.Counter)
    for k in (2, 3):
        raw = read_source(os.path.join(data_dir, "train", f"train_source{k}.tsv"))
        raw = raw.filter(pl.col("business_name").str.contains("[ऀ-ൿ]"))
        raw = raw.select(pl.col("entity_id").alias("mid"), "business_name")
        j = raw.join(pairs, on="mid").join(s1, left_on="id1", right_on="id")
        for name, lat in j.select("business_name", "ntok").iter_rows():
            itoks = _indic_tokens(name)
            if len(itoks) == len(lat):
                for a, b in zip(itoks, lat):
                    if N.INDIC_RE.search(a):
                        counts[a][b] += 1
    d = {}
    for a, c in counts.items():
        b, n = c.most_common(1)[0]
        if n >= 2 and n >= 0.5 * sum(c.values()):
            d[a] = b
    return d


def indic_to_latin(raw_name, dic):
    out = []
    for t in _indic_tokens(raw_name):
        if N.INDIC_RE.search(t):
            out.append(dic.get(t) or N.transliterate_indic(t))
        else:
            out.append(t)
    return " ".join(out)


def normalise(df, dic):
    has_indic = pl.col("business_name").fill_null("").str.contains("[ऀ-ൿ]")
    df = df.with_columns(
        has_indic.alias("f_indic"),
        pl.col("business_name").fill_null("")
          .str.contains(r"(?i)(\.(com|in|net|org|fr|co|biz|info|us|io)\b|www\.)").alias("f_domain"),
    )
    # transliterate the (few) Indic names in Python
    ind = df.filter(pl.col("f_indic")).select("id", "business_name")
    if ind.height:
        lat = [indic_to_latin(x, dic) for x in ind["business_name"].to_list()]
        ind = ind.with_columns(pl.Series("bn_lat", lat))
        df = df.join(ind.select("id", "bn_lat"), on="id", how="left")
        df = df.with_columns(pl.coalesce("bn_lat", "business_name").alias("business_name")).drop("bn_lat")

    df = df.with_columns(N.name_expr("business_name").alias("name_n"),
                         N.addr_expr("business_address").alias("addr_n"))
    df = df.with_columns(
        N.canon_tokens(pl.col("name_n"), N.LEGAL_CANON).alias("ntok"),
        pl.col("addr_n").str.replace_all(N.STATE_RE, "").alias("_a"),
        pl.col("addr_n").str.extract_all(N.STATE_RE).list.first().alias("_state"),
    )
    # multi-word canon (e.g. "pvtltd" -> "pvt ltd") -> re-split
    df = df.with_columns(pl.col("ntok").list.join(" ").str.split(" ").alias("ntok"))
    stop = list(N.NAME_STOP)
    df = df.with_columns(
        pl.col("ntok").list.eval(pl.element().filter(~pl.element().is_in(stop))).alias("core"),
    )
    df = df.with_columns(pl.col("core").list.join("").alias("sq"))

    atoks = pl.col("_a").str.split(" ")
    astop = list(N.ADDR_STOP)
    df = df.with_columns(
        atoks.list.eval(pl.element().filter(pl.element().str.contains(r"^\d+$"))
                        .str.replace(r"^0+(\d)", "$1")).alias("nums"),
        atoks.list.eval(pl.element().filter(pl.element().str.contains(r"^[a-z]{2,}$"))
                        .replace(N.ADDR_CANON))
             .list.eval(pl.element().filter(~pl.element().is_in(astop))).alias("atok"),
        pl.col("_state").replace(N.STATE_ALL).alias("state"),
    )
    df = df.with_columns((pl.col("addr_n").str.len_chars() == 0).alias("f_addr_empty"))
    return df.select("id", "country", "name_n", "ntok", "core", "sq", "addr_n", "atok",
                     "nums", "state", "f_indic", "f_domain", "f_addr_empty")


def run(data_dir, work, splits=("train", "test")):
    os.makedirs(work, exist_ok=True)
    dic_path = os.path.join(work, "indic_dict.json")
    for split in splits:
        # source 1 is always Latin; do it first (needed to learn the dictionary)
        for k in (1, 2, 3):
            if k == 2 and not os.path.exists(dic_path):
                if split != "train":
                    raise RuntimeError("run the train split first (learns the Indic dictionary)")
                dic = learn_indic_dict(work, data_dir)
                with open(dic_path, "w", encoding="utf-8") as f:
                    json.dump(dic, f, ensure_ascii=False)
                print(f"  indic dictionary: {len(dic)} tokens")
            dic = {}
            if os.path.exists(dic_path):
                with open(dic_path, encoding="utf-8") as f:
                    dic = json.load(f)
            src = os.path.join(data_dir, split, f"{split}_source{k}.tsv")
            out = os.path.join(work, f"{split}_s{k}.parquet")
            df = normalise(read_source(src), dic)
            df.write_parquet(out)
            print(f"  {split} s{k}: {df.height:,} rows -> {out}")
            del df
