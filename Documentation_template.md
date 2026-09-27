# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** [Date]

---

## 1. Executive Summary

We use a classic blocking → pairwise-classifier pipeline, built so that 12 M
records fit on a 16 GB laptop and so that nothing depends on which countries are
in the data. Blocking uses **hashed multi-key blocking on rare typed tokens**
(name words, address words and house numbers; typos excluded via document
frequency). A cheap score then prunes candidates, using the structural fact that
**every S2/S3 record belongs to at most one S1 entity**. This yields
**~5 candidates per S1 entity at 96.7 % recall** (6.17 per S1 on test). A
two-stage LightGBM scores the candidates. The second stage adds
*cluster-consistency* features: how much a candidate resembles the other
confident matches of the same entity. 1-to-1 decoding and an F0.5-tuned threshold
give **macro F0.5 = 0.9809** on a held-out 20 % of the training entities.

---

## 2. Methodology

### 2.1 Problem Analysis

EDA on the 2.21 M training S1 entities (5.03 M S2, 5.29 M S3 records):

* **Cluster structure.** An S1 entity has 3.46 matches on average (1.67 from S2,
  1.79 from S3; at most 11). 5.6 % of S1 entities are singletons. **No S2/S3
  record is ever matched to two S1 entities** (7,638,365 matched pairs, 7,638,365
  distinct S2/S3 ids). **Matched records always share the country label.**
* **Distractors.** 26 % of S2/S3 records match nothing. Many of them are
  *deliberate near-duplicates* of a real entity, for example
  `jaramillo ready riverside inc, ##22207 EAST RD` next to S1
  `Jaramillo Ready Inc, 22206 East Road` (house number off by one, an extra
  name word). House numbers and "extra" words therefore matter a lot. Numbers
  are noisy in true matches too (`424`→`42`, `730`→`73`, `2031`→`2031C`,
  `703`→`00703`), so exact equality alone is not enough.
* **Name noise.** Token shuffles, typos (`Trerpies`, `J0die's Púb`), legal-form
  churn (`Pvt`/`Private`, `L.L.C.`, `(Ltd)`), added generic words (`Center`,
  `Services`, `Trading`), prefixes like `--`/`<<`, domain-style names
  (`jodiespub.com`, `kasseylaingdpm.com`), trailing phone numbers, and
  **Indic-script names**. 9 % of S2 names and 5 % of S3 names are in Devanagari,
  Bengali, Gujarati, Tamil, Kannada and others, while S1 is always Latin.
* **Address noise.** Component reordering (`Salt Lake City, 676 Hollywood Avenue, UT`),
  abbreviations (`Rd`/`Road`, `R.`/`Rue`), full vs. abbreviated states,
  state names in local script (`পশ্চিমবঙ্গ`), `null`/`<NULL>`, PO boxes added,
  missing street or number, and 3.3 % empty addresses. S1 India addresses are long,
  and S2/S3 often keep only a fragment (`#26, Howrah, Kolkata, WB`).
* **Test shift.** The test set is 47 % India, 38 % US and 15 % **France**, which is
  unseen in training. French records use region *or* department (`Nouvelle-Aquitaine`
  vs `Gironde`), `bis`/`ter`, `R.`/`Av`/`All.`, and very generic names
  (`Ecole`, `Amicale`, `Comite` + `SARL`/`SAS`/`EURL`).

### 2.2 Solution Strategy

**Approach Type:** Blocking + two-stage gradient-boosted classifier + constrained (1-to-1) decoding  
**Core Innovation:**
1. Memory-bounded **typed rare-token multi-key blocking**. Keys are u64 hashes and
   are joined in hash partitions and S2/S3 chunks, so large blocks are never
   enumerated.
2. **Assignment-aware pruning**. An S2/S3 record only stays a candidate of its
   best S1 (plus near-ties), which brings the candidate set down to about 5 per S1.
3. **Cluster-consistency second stage**, which scores a candidate in the context
   of its sibling candidates.
4. **Country-agnostic design**. Country is only used as a partition key, and all
   token weights are IDF computed within each country. Regional tokens therefore
   lose weight automatically in any country, including France.

Pipeline: `prepare` (normalisation) → `cands` (blocking + pre-pruning) →
`pruner` (fit blocking score) → `feats` (final pruning + features) → `train`
→ `predict`.

---

## 3. Candidate Generation (Blocking)

All comparisons happen **within a country** (never violated in the training ground truth).

**Normalisation (applied before blocking).** The steps are:

* NFKD accent folding and lower-casing.
* Punctuation removal. `l.l.c.` becomes `llc`, `jodie's` becomes `jodies`, and
  `jodiespub.com` becomes `jodiespub`.
* Legal-form canonicalisation (`corporation`→`corp`, `private`→`pvt`, …) and a
  "core name" with legal forms and stop-words removed.
* A squashed core name (`waidrestaurantterrell`) to meet domain-style names.
* Street-type canonicalisation (English, French and common Indian abbreviations)
  and US/Indian state names mapped to codes.
* Leading zeros stripped from numbers, and `2031C` split into `2031 c`.
* PO boxes and `null` removed.

**Indic names** go through a generic transliterator. It folds every ISCII-layout
script onto Devanagari by code-point offset, then transliterates with schwa
deletion. A token dictionary learned from word-aligned training pairs takes
precedence (1,312 tokens, e.g. `प्रोड्यूसर`→`producer`, `प्रा`→`pvt`).

**Blocking keys used.** Tokens are typed: `n:` name word, `a:` address word, `#:` number.
"Rare" means lowest document frequency in the country. Tokens with df = 1 (typos)
are never used. Each record emits:

| family | key | purpose |
|---|---|---|
| U1 | a single token with df ≤ 12 | distinctive words that essentially only the entity's own cluster carries |
| U2 | every pair of the record's 6 rarest tokens | general-purpose, robust to any single noisy field |
| A  | house number × rare address word | garbled or transliterated names |
| B  | rare name word × house number | sparse or partial addresses |
| C  | pair of the 3 rarest name words | missing or renumbered addresses |
| D  | squashed core name | domain-style names |
| E  | rare name word × rarest address word | no house number anywhere |

A key is used only if its block has ≤ 20 S1 and ≤ 80 S2/S3 records. Oversized
blocks are skipped rather than truncated. Implementation: keys are 64-bit hashes
of typed tokens, never strings. S1 keys are indexed per hash partition (8
partitions). S2/S3 records are processed in chunks of 600 k and joined against
that index, so peak memory stays around 8 GB.

**Pruning (the output of this step is `candidate_pairs.tsv`).**

1. *Pre-pruning.* A hand-written cheap score:
   `0.55·max(tokenset(core), 0.9·partial(squashed)) + 0.30·tokenset(address words + numbers) + 0.15·[shared number]`.
   Each S2/S3 record keeps its 5 best S1 records with score ≥ 0.2, and each S1 keeps at most 60.
2. *Final pruning.* A 6-weight logistic "blocking score" over the same cheap
   similarities, the log of the number of shared keys, and a name×address
   interaction, fitted once on train. Each S2/S3 record keeps its best S1 plus any
   S1 within 1 logit of it (near-ties), the score must be ≥ −6, and each S1 keeps
   at most 12. This relies on the 1-to-1 structure: in the ground truth an S2/S3
   record never has two S1 entities, so everything but the best-scoring S1 is
   almost always a distractor.

**Candidate pairs generated.**

| | raw key-sharing pairs / S1 | after pre-pruning / S1 | **final / S1** | recall of true pairs |
|---|---|---|---|---|
| train India | 134.8 | 22.4 | **5.03** | 95.9 % after pre-pruning |
| train US | 112.3 | 22.4 | **4.90** | 98.2 % after pre-pruning |
| train validation split (India + US) | | | **4.95** | **96.7 % final** |
| test France | 111.3 | 25.8 | **6.58** | n/a |
| test India | 147.7 | 27.0 | **6.23** | n/a |
| test US | 113.1 | 26.6 | **5.92** | n/a |
| **test total** | | | **6.17** (10,682,924 pairs) | n/a |

Against all within-country S1 × (S2 ∪ S3) comparisons (6.7·10¹² on test), the
reduction ratio is 1 − 1.6·10⁻⁶. Test has more S2/S3 records per S1 than train
(5.8 vs 4.7), hence slightly more candidates per S1.

**How we ensured true matches were not lost.**

* Seven complementary key families. Measured on train, each family alone recalls
  only 41–79 %, but the union recalls 96–98 %.
* Blocking ignores df = 1 tokens, so a typo cannot push the real tokens out of a
  record's rarest set.
* Pruning keeps near-tie runner-ups instead of a hard top-1.
* The pruning floor, δ and caps were chosen from recall-vs-size sweeps on train.
  For example, on India: top-1 only gives 94.9 % recall at 4.5/S1; δ = 1 with a
  cap of 12 gives 95.4 % at 5.0/S1; δ = 2 gives 95.5 % at 5.3/S1.
* The remaining losses are mostly unrecoverable: an S2/S3 record with an empty
  address and a very generic name (`heartland charities inc`, `pediatric clinic`)
  that exists hundreds of times in the country.

---

## 4. Matching Model

**Features used (58 in stage 1, 70 in stage 2).**

- **Name features:**
  - rapidfuzz ratio, token-sort, token-set and partial ratio on the core name;
    token-set on the full name.
  - ratio, partial ratio and Jaro–Winkler on the squashed name (domains, spacing).
  - token-set ratio on phonetic keys (for transliterated names).
  - IDF-weighted recall of S1 name tokens, recall of candidate name tokens, and Dice.
  - IDF mass and max-IDF of **extra** candidate words (`riverside` in a distractor)
    and of **missing** S1 words; number of shared words.
- **Address features:**
  - token-set, token-sort and partial token-set ratios on address words.
  - IDF-weighted overlap both ways, and IDF mass of extra address words.
  - Number features: shared-number count, number Jaccard, a *number conflict* flag
    (both sides have numbers, none shared), first-number equality, first number
    contained in the other side, a digit-drop prefix match (`424`/`42`), and a
    fuzzy ratio of the house numbers.
- **Other:**
  - Source (S2/S3), Indic-script / domain-name / empty-address flags, token and
    number counts, name lengths.
  - Blocking context: the cheap and logistic blocking scores, the number of
    shared blocking keys, **margin over the best competing S1 for this S2/S3
    record**, gap to the best candidate of this S1, and candidate counts.
  - **Stage-2 cluster-consistency features:**
    - the stage-1 probability p1, its rank within the S1, the count and sum of
      other confident siblings, and the gap to the best sibling;
    - the best p1 this S2/S3 record has with a different S1;
    - max and p1-weighted name and address similarity to confident siblings
      (p1 > 0.5), and whether the address is identical to a sibling's.

  The idea behind the stage-2 features: S2/S3 variants of one business usually
  repeat each other's address or name format, so a weak-looking record that
  mirrors confident siblings is likely a match.

**Model type:** LightGBM (MIT licence) binary classifiers. Settings: 127 leaves,
learning rate 0.08, bagging and feature fraction 0.8, early stopping on the
validation split. Stage 1 ran 1,500 rounds; stage 2 stopped at 862. Stage 2 is
trained on **out-of-fold** stage-1 probabilities (2-fold by S1 entity) to avoid
leakage. The whole model is a few MB of trees, well under the 8 B-parameter limit,
and no pretrained model is used.

**Decoding and threshold selection.** Each S2/S3 record is assigned to its single most
probable S1 (1-to-1 constraint). Exact probability ties go to the lowest S1 id, so
the assignment is deterministic. The pair is kept if p ≥ threshold. Before the
output files are written, a guardrail (`assert_one_to_one`) stops the run if any
S2/S3 record is matched to more than one S1. The threshold maximises **macro F0.5**,
counting singletons and entities without any candidate.

**Validation discipline.** `val_s1_ids()` is the single definition of the
validation fold: a deterministic hash selects 20 % of training S1 ids. None of the
following learns from these entities: the blocking pruner (excluded before its
logistic regression is fitted), the stage-1 model, the out-of-fold stage-1 models,
or the stage-2 model. Blocking still runs over *all* training records, so every
validation entity competes with the full distractor pool exactly as on test. The
fold is split once more by a second hash (`val_subfolds()`):

* the **selection** half tunes the thresholds and chooses between stage 1 and stage 2;
* the **report** half is used only to compute the reported score, so that number
  was not used for any decision.

Remaining known optimism: LightGBM early stopping still monitors the whole 20 % fold.

**Country handling (audit).** Records are compared only inside the same exact
country label; there is no canonicalisation step. On 2026-09-27 we checked this on
the released data:

* Train uses exactly `US` and `India` in all three sources.
* Test uses `US`, `India` and `France` in all three sources, with identical spelling
  (test S1: France 259,452, India 809,986, US 663,106 records).
* In an audited sample of 300 k training S1 entities, 0 ground-truth matches had a
  different country label from their S1 record.

So country canonicalisation was checked and found unnecessary for this dataset.
Test France has French street types and legal forms in the normalisation
vocabulary; regions and departments are not mapped and get little weight through
per-country IDF instead.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro), validation (441 k held-out S1 entities, India + US):**
  - stage 1: 0.97909
  - **stage 2: 0.98093**, at threshold 0.70 (stage 1 alone: 0.675)
  - blocking recall ceiling: 96.7 % of true pairs at 4.95 candidates per S1.
  - *These numbers come from the run that produced the submitted output files.
    That run predates the validation-discipline fixes: its pruner had also been fitted
    on the validation entities, and model choice, thresholds and the reported score
    all used the same 20 % fold. They are therefore slightly optimistic upper bounds.
    Re-running `pruner` → `feats` → `train` → `predict` with the current code
    produces a report-half score that no decision has used.*
- **Test output:** 3.27 matches per S1; 6.3 % of S1 entities predicted as
  singletons (5.8 % France, 6.8 % India, 5.9 % US; the train singleton rate is
  5.6 %). The per-country rates are very similar, which is a good sign that the
  country-agnostic features transfer to France.
- **Where the remaining F0.5 is lost.** Breakdown of the loss on the India
  validation entities with an early stage-1 model (loss = Σ(1 − F)/N):
  - partial recall (predicted a correct subset): 1.65 pts;
  - nothing predicted for an entity that has matches: 0.74 pts;
  - at least one false merge: 0.27 pts;
  - a match predicted for a true singleton: 0.08 pts.

  **Precision is already very high; missed matches dominate**, as intended by the
  F0.5 trade-off.
- **Common false positives (wrong merges):** planted near-duplicates whose only
  difference is a house number that could also be ordinary number noise
  (`22207` vs `22206`); sibling-like records of the same franchise name at the
  same street; generic French names differing only in a generic word
  (`FL Societe SAS` vs `FL Comite SAS` at the same address).
- **Common false negatives (missed matches):** candidates with an empty address
  and a generic name; heavily truncated Indian addresses where only the city
  survives; domain names that glue shuffled words (`aimeidirectmetro`); rule
  transliterations far from the English original that the dictionary does not
  cover; blocking misses (3.3 % of true pairs).

---

## 6. Conclusion

The candidate set is small (6.2 per S1 on test, about 1.8× the average true
cluster size) and keeps 96.7 % of true matches. Two things drive this: blocking
on *typed, rare, non-typo* tokens with many complementary key families, and
pruning that exploits the one-S1-per-record structure. A two-stage LightGBM that
looks at each candidate *in the context of its sibling candidates*, plus 1-to-1
decoding, reaches a validation macro F0.5 of 0.981. Lessons learned:

* Exploiting the assignment structure paid off more than any single similarity feature.
* Explicit memory engineering (hashed keys, partitioned joins, chunking) is what
  makes billion-comparison-scale ER feasible on commodity hardware.
* Keeping every feature country-agnostic is the only safe way to handle an unseen
  country.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`:

| file | role |
|---|---|
| `src/pipeline.py` | **entry point**: `python pipeline.py all --data <dataset> --work <scratch> --out <output>` runs every stage; stages: `prepare`, `cands`, `pruner`, `feats`, `train`, `predict` |
| `src/normalize.py` | folding, canonicalisation vocabularies, Indic transliteration, phonetic keys |
| `src/prepare.py` | TSV → normalised parquet; Indic dictionary learnt from train ground truth |
| `src/data.py` | per-country loaders |
| `src/blocking.py` | hashed multi-key blocking, block caps, cheap score, pre-pruning, logistic pruning |
| `src/features.py` | pairwise features |
| `src/stage2.py` | cluster-consistency features |
| `tests/` | pytest unit tests on synthetic data (validation split, pruner exclusion, 1-to-1 tie-break, output ordering, model-selection subfolds); run `pytest tests/` |
| `utils/validate_submission.py` | the organisers' validator, unmodified |
| `README.md`, `requirements.txt` | exact run instructions and pinned versions (Python 3.11, polars 1.44.2, rapidfuzz 3.14.6, LightGBM 4.7.0, scikit-learn 1.9.1, numpy 2.4.6, pyarrow 25.0.1) |

Only the provided files are used: no external data, APIs, geocoders or pretrained
models. Vocabularies such as street-type abbreviations and US/Indian state codes
are generic text-normalisation rules written in the code, not lookups.

### B. Additional Results

Fitted blocking-score weights (logistic, on `c_name, c_sq, c_addr, c_num, log1p(n_keys), c_name·c_addr`):
`[17.04, 6.73, 21.30, 1.74, 1.65, −18.74]`, intercept −30.19.

Top stage-2 features by gain: stage-1 probability, gap to the best sibling, sum and
count of other confident siblings, cheap blocking score, address similarity,
candidate count, max address similarity to a confident sibling, best competing-S1
probability, number Jaccard, and margin over the competing S1.

Top stage-1 features by gain (India-only run): number Jaccard, cheap address
similarity, squashed-name ratio, margin over the best competing S1, IDF mass of
extra candidate words, cheap score, core-name ratio, full-name token-set ratio.
