"""Loading the normalised parquet files written by prepare.py."""
import os

import polars as pl

ALL_COLS = ["id", "country", "name_n", "core", "sq", "addr_n", "atok", "nums",
            "f_indic", "f_domain", "f_addr_empty"]


def countries(work, split):
    return (pl.scan_parquet(os.path.join(work, f"{split}_s1.parquet")).select("country")
              .unique().collect()["country"].sort().to_list())


def load_split(work, split, cols=ALL_COLS, country=None):
    """S1 frame and the concatenated S2+S3 frame (optionally one country only),
    each with a dense row index `r`; the S2/S3 frame has `src` in {2, 3}."""
    def scan(k):
        lf = pl.scan_parquet(os.path.join(work, f"{split}_s{k}.parquet")).select(cols)
        if country is not None:
            lf = lf.filter(pl.col("country") == country)
        return lf
    s1 = scan(1).collect().with_row_index("r")
    o = pl.concat([scan(2).with_columns(pl.lit(2, pl.UInt8).alias("src")),
                   scan(3).with_columns(pl.lit(3, pl.UInt8).alias("src"))]).collect()
    return s1, o.with_row_index("r")
