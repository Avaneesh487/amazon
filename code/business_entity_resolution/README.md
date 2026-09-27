# Business Entity Resolution: blocking + LightGBM matcher

This pipeline regenerates `output/matching_results.tsv` and `output/candidate_pairs.tsv`
from the raw challenge TSVs. It uses only the provided data. It makes no external
lookups, has no geocoding, and uses no pretrained models.

## Environment

* Python 3.11 (tested on Windows 11, 16 GB RAM, 12-core laptop CPU; no GPU needed)
* `pip install -r requirements.txt`

## Run

From `src/`:

```bash
python pipeline.py all --data <path>/dataset --work <scratch dir> --out <path>/output
```

`--work` holds intermediate parquet files (about 10 GB). Put it on a local disk
that isn't synced to the cloud. Each stage can also be run on its own, and a stage
skips any output that already exists unless you pass `--force`:

| stage     | what it does | output (in `--work`) |
|-----------|--------------|----------------------|
| `prepare` | normalises every record of train + test and learns the Indic→Latin token dictionary from train | `{split}_s{1,2,3}.parquet`, `indic_dict.json` |
| `cands`   | blocking: hashed multi-key blocking + hand-score pre-pruning, per country | `{split}_pre_{country}.parquet` |
| `pruner`  | fits the 6-weight logistic blocking score on train, **excluding the validation S1 entities** (see *Validation discipline*) | `pruner.json` |
| `feats`   | final candidate pruning + ~55 pairwise features | `{split}_feat_{country}.parquet` |
| `train`   | stage-1 LightGBM (+ 2-fold out-of-fold predictions) and stage-2 LightGBM with cluster-consistency features, fitted on the 80 % training S1 entities; threshold and stage choice made on half of the 20 % validation entities, score reported on the other half | `model_s1.txt`, `model_s2.txt`, `train_report.json` |
| `predict` | scores test candidates (stage 1 → stage 2), 1-to-1 assignment, threshold, writes both TSVs | `--out` |

`--splits train` / `--splits test` restricts `cands`/`feats` to one split.

Validate the outputs with the organisers' script (shipped unchanged in `utils/`).
From the folder that holds `output/` and `dataset/`:

```bash
python code/business_entity_resolution/utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids
```

`pipeline.py predict` also refuses to write outputs if any S2/S3 record would be
matched to more than one S1 entity (`assert_one_to_one`).

## Validation discipline

`val_s1_ids()` in `src/pipeline.py` defines the one validation split: a fixed hash
of the S1 id selects 20 % of the training S1 entities. The blocking pruner, both
LightGBM models and the thresholds never train on these entities. The 20 % is split
again by `val_subfolds()` into two halves:

* the **selection** half chooses the stage-1 vs stage-2 model and the threshold;
* the **report** half is used once, to compute the `val_f05` written to
  `train_report.json`.

The selection-half score is also recorded, as `val_f05_selection`, and is an upper
bound. Early stopping of the two LightGBM models still watches the whole 20 % fold.

## Tests

Unit tests use small synthetic data and don't need the dataset:

```bash
pytest tests/
```

Approximate runtimes on the machine above (end to end about 4.5 h, peak RAM about 8 GB):
prepare 4 min; cands 6–30 min per country and split (about 2 h total); feats 5–15 min per
country and split (about 1 h total); train 52 min; predict 16 min. Countries are
processed one at a time to bound memory, so don't run two stages concurrently on a 16 GB machine.

## Source layout

| file | role |
|------|------|
| `src/normalize.py` | string folding, legal-form / street-type canonicalisation, generic Indic-script transliteration, phonetic keys |
| `src/prepare.py`   | stage 1: TSV to normalised parquet, Indic dictionary learning |
| `src/data.py`      | loading helpers (per-country slices) |
| `src/blocking.py`  | stage 2: blocking keys, block-size caps, cheap scores, pruning |
| `src/features.py`  | stage 3: pairwise features (rapidfuzz + IDF-weighted token algebra) |
| `src/stage2.py`    | cluster-consistency features for the second LightGBM |
| `src/pipeline.py`  | driver: validation split, training, threshold search, 1-to-1 decoding, output writing |
| `tests/`           | pytest unit tests (synthetic data) |
| `utils/validate_submission.py` | organisers' submission validator, unmodified |

The method is described in `Documentation_template.md` at the root of the
submission zip.
