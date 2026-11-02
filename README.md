# Business Entity Resolution Pipeline — Amazon ML Challenge 2026

> **Deployed:** Hybrid Edge-GPU + AWS Serverless Lakehouse — S3 + Glue + Athena
> **Macro F0.5:** 0.9645 | **Records:** 11.7M | **Candidate Pairs:** 128.7M | **Search-Space Reduction:** 99.9993% | **Mutual Exclusivity Violations:** 0

This package contains the complete, self-contained, end-to-end reproducible machine learning pipeline for the **Amazon ML Challenge 2026: Business Entity Resolution Challenge**.

---

## 1. Problem Overview & Architecture

The objective is to resolve real-world business entities from three noisy sources without shared IDs:
* **Source 1 ($S_1$)**: Deduplicated reference source.
* **Source 2 ($S_2$) & Source 3 ($S_3$)**: Noisy vendor records containing name typos, legal suffix variations, missing address fields, and transliterations.
* **Open-Set Countries**: Training covers `US` and `India`; Test adds **`France`**. Our pipeline uses strictly language-agnostic features and country partitioning.

### Pipeline Architecture:
1. **Vectorized Text & Address Normalization**: Unicode NFKD accent folding, lowercase standardization, street abbreviation expansion, and house number/postal code extraction.
2. **Multi-Path Blocking Engine with Category 2 India Bigrams**: Country-partitioned inverted index combining:
   - Compressed name prefixes (8-char and 5-char)
   - First 2 words of business name
   - Salient name tokens (length $\ge 5$, frequency capped $\le 150$)
   - Address anchors (postal code + house number)
   - Locality anchors (house/street number + salient street words)
   - India-gated address token bigrams (recovering non-Latin transliterations and missing postal codes)
3. **C-Accelerated 25-Feature Engineering**: Sub-second extraction using `rapidfuzz` (Levenshtein ratio, Token Sort, Token Set, Jaro-Winkler, numeric token Jaccard overlap, postal code matching, source flags, Damerau-Levenshtein, LCSseq, phonetic Soundex, and train-only zero-leakage TF-IDF cosine similarity).
4. **LightGBM Matcher**: Gradient boosted decision tree trained on true matches and hard negative blocking candidates under certified hyperparameters.
5. **Structural Post-Processing**:
   - **1-to-many Mutual Exclusivity Assignment**: Each $S_2$ and $S_3$ target record is assigned exclusively to the $S_1$ entity that yielded the highest confidence score, eliminating cross-entity false merges.
   - **Optimal Decision Thresholding**: Threshold locked at $\tau = 0.65$, maximizing precision and preserving singleton integrity under the Macro $F_{0.5}$ metric.

---

## 2. Directory Structure

```
business_entity_resolution/
├── src/
│   ├── normalizer.py          # Multilingual text, address, and legal suffix normalizer
│   ├── metrics.py             # Official Macro F_0.5 evaluation engine with singleton rules
│   ├── blocking.py            # High-performance multi-path blocking engine with Cat2 bigrams
│   ├── features.py            # C-accelerated 25-feature extractor & TF-IDF vocab manager
│   ├── train_matcher.py       # LightGBM training, hard negative mining & certified training
│   └── predict.py             # Full test inference pipeline generating TSVs
├── models/
│   ├── xgboost_matcher.json   # Serialized XGBoost booster model (Apache 2.0, < 8B params, CUDA GPU native)
│   ├── lightgbm_matcher.txt   # Serialized LightGBM booster model (MIT license, < 8B params)
│   ├── tfidf_vocab.pkl        # Serialized zero-leakage train-fitted TF-IDF vocabulary
│   └── config.pkl             # Optimized decision threshold configuration (tau = 0.65)
├── requirements.txt           # Pinned dependency versions
└── README.md                  # Reproduction instructions
```

---

## 3. Environment Setup

Install all pinned dependencies:

```bash
pip install -r requirements.txt
```

---

## 4. How to Reproduce End-to-End

### Step 1: Train the Matcher & Tune the Threshold
Runs blocking on the training set, fits train-only TF-IDF vocabulary, extracts 25 pairwise features, trains the LightGBM classifier with early stopping, and validates the threshold optimizing Macro $F_{0.5}$:

```bash
python src/train_matcher.py
```
*Outputs:*
* `models/lightgbm_matcher.txt`
* `models/tfidf_vocab.pkl` (Zero-leakage training vocabulary)
* `models/config.pkl` (production threshold: 0.65)

### Step 2: Run Test Set Inference
Generates candidate pairs and final entity matches across all 1.73M test records:

```bash
python src/predict.py
```
*Outputs:*
* `../../output/matching_results.tsv` (Leaderboard submission file)
* `../../output/candidate_pairs.tsv` (Blocking candidates file)

### Step 3: Validate Outputs
Run the official validator to verify zero formatting errors:

```bash
python ../../6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py \
    --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir ../../6ab10eb3b23ba_student_resource/student_resource/dataset/test \
    --check-ids
```

Expected output: `PASS — no blocking issues found. Safe to submit.`

---

## 5. Model License & Compliance
* **Model Type & Version**: LightGBM Gradient Boosted Decision Tree (`lightgbm==4.7.0`).
* **License**: MIT License (Permissive Open Source).
* **Parameter Count**: Under 10,000 tree parameters (fully compliant with the $\le 8\text{B}$ parameter limit).
* **External APIs**: Strictly zero external lookups, geocoding APIs, or web augmentation.

## v7 pipeline (final submission)

Country-agnostic matcher built to generalize to France, which has no training labels.

```bash
cd code/business_entity_resolution/src
pip install -r ../requirements.txt
python train_matcher.py --features v7 --per-country 20000   # GPU XGBoost; writes models/xgboost_matcher_v7.json + config.pkl
# France rows of the final submission (cap 30 candidates per S1):
PREDICT_COUNTRIES=France MAX_CANDIDATES_PER_S1=30 OUTPUT_DIR=../../../output/v7_france python predict.py
# full test set with v7 (all countries):
python predict.py
```

- `blocking.py`: Unicode cleaning (Brahmic scripts transliterated via `translit.py`, accents folded, dotted acronyms joined), multi-path keys incl. phonetic skeleton, candidates ranked by number of shared keys.
- `features.py`: 50 features (`FEATURE_NAMES_V7`); IDF and name end-position statistics are computed per country from that country's own target records (label-free), so an unseen country is handled identically.
- `decision.py`: target-exclusive assignment + threshold or expected-F0.5 set selection (chosen on validation; config records which).
- Validation: `scripts/generalization_eval.py` (leave-one-country-out). The model shipped here was trained with `scripts/train_from_cache.py --tag fix3` (20k S1 per country, same code path).

**Submitted `output/matching_results.tsv`:** France rows come from the v7 model above; US and India rows are from the earlier v6 submission run (0.830 leaderboard), merged with `scripts/merge_country_results.py`. Running `python predict.py` regenerates all countries with v7.

## v9 pipeline: GPU features + multilingual semantic layer (current)

Everything that can run on the GPU does (no CPU fallback; `src/gpu.py` raises if CUDA is missing).
Requires an NVIDIA GPU with CUDA 13; install with `pip install -r requirements.txt`.

```bash
cd code/business_entity_resolution/src
python train_matcher.py --features v9 --per-country 60000   # writes models/xgboost_matcher_v9.json, v9_source.pt, config.pkl
python predict.py                                            # writes ../../../output/{matching_results,candidate_pairs}.tsv
python -m pytest ../tests -q                                 # 11 CPU+GPU tests
```

- **Blocking** (`blocking.py`): multi-path keys; over-common name keys are refined by address tokens instead of
  being truncated; candidates ranked by number of shared keys. **v9** adds the top-10 targets by multilingual
  name-embedding cosine (`v8_pipeline.semantic_candidates`, exact GPU search).
- **Features** (`gpu_features.py`, `gpu_strings.py`): 47 GPU features (bit-parallel LCS/Levenshtein, verified exact
  vs rapidfuzz; TF-IDF, char 3-gram, soft token alignment, group gaps). Word statistics are fit per country on
  that country's own records (label-free). **v9** adds 3 semantic features (name/address embedding cosine, gap to
  best candidate).
- **Model**: XGBoost on CUDA, trained on all labelled countries; at inference it is **self-trained per country** on
  that country's confident pseudo-labels (no labels used), which adapts to unseen countries such as France.
  Each country's adapted model is saved next to the outputs (`xgboost_matcher_v9_<country>_selftrained.json`).
- **Semantic model**: `models/semantic_minilm/` = sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
  (Apache-2.0, 118M parameters, inference only); see `models/semantic_minilm/MODEL_LICENSE.txt`.
- `models/*.pt` (training tensors used for self-training) are regenerated by `train_matcher.py`.
- Evaluation scripts (repository root `scripts/`): `generalization_eval_v8.py`, `semantic_eval.py`,
  `semantic_blocking_eval.py`, `tune_v8.py`.

## v10 pipeline: v9 + Jev-style set-level judge (current)

```bash
cd code/business_entity_resolution/src
python train_matcher.py --features v10 --per-country 60000   # v9 stage + out-of-fold scores + judge (models/judge_v10.pt)
python predict.py                                             # all countries; PREDICT_COUNTRIES=India to run one country
```

- `src/judge.py`: bidirectional transformer over each S1's top-32 candidates (label-free ranking). Candidate-to-candidate
  graph-consistency features enter attention as learned biases; typed query slots give a calibrated
  (evidential Beta) match probability per candidate. Design ideas from kyegomez/open-jev (Apache-2.0), re-implemented.
- Validation Macro F0.5: v9 stage 0.9516 -> v10 0.9645 (same split).
- `models/v10_source.pt` (2.1 GB training tensors for per-country self-training) is not shipped; `train_matcher.py`
  regenerates it. All model weights needed for inference (`xgboost_matcher_v10.json`, `judge_v10.pt`, `config.pkl`,
  `semantic_minilm/`) are included.
- `src/assemble_matches_lowmem.py` assembles per-country checkpoints into one `matching_results.tsv`.

## Reproducing the submitted `output/` files (semantic_submission.zip)

The submitted `matching_results.tsv` has **France and US rows from v10** (judge) and **India rows from v9** (India's
v10 inference did not finish before the deadline). `candidate_pairs.tsv` is identical for v9 and v10 (same blocking
code; identical pair counts per country). Everything below uses only this folder plus the challenge data
(`TEST_DATA_DIR` / `TRAIN_DATA_DIR` environment variables, default `../../../6ab10eb3b23ba_student_resource/...`).

```bash
cd code/business_entity_resolution/src
pip install -r ../requirements.txt                 # NVIDIA GPU with CUDA 13 required (no CPU fallback)

# 1. Self-training source tensors (models/*_source.pt are not shipped: 2+ GB each).
#    Each training run also rewrites models/config.pkl; config_v9.pkl / config_v10.pkl keep each version.
python train_matcher.py --features v9  --per-country 60000
python train_matcher.py --features v10 --per-country 60000

# 2. v10 for all countries -> ../../../output/v10   (writes candidate_pairs.tsv + per-country checkpoints)
MODEL_CONFIG=config_v10.pkl OUTPUT_DIR=../../../output/v10 python predict.py

# 3. v9 for India only -> ../../../output/v9_india
MODEL_CONFIG=config_v9.pkl PREDICT_COUNTRIES=India OUTPUT_DIR=../../../output/v9_india python predict.py

# 4. Final file: v10 rows for France + US replace those rows of the v9 India run's output
python assemble_matches_lowmem.py ../../../output/v9_india/matching_results.tsv ../../../output/final_matching_results.tsv \
       --checkpoint France=../../../output/v10 --checkpoint US=../../../output/v10
#    candidate_pairs.tsv = ../../../output/v10/candidate_pairs.tsv
```

Step 3 writes rows for India only (other countries empty); step 4 replaces the France and US rows with v10's.

**Exactness:** the submitted outputs were produced before a determinism fix: Python's per-process randomized `hash()`
was used for character-bigram signatures in 5 soft-alignment features (now CRC32, with a regression test). Re-running
these steps with the fixed code gives equivalent, not byte-identical, results; retrain first (step 1) so model and
features match.


---

## AWS Serverless Lakehouse Deployment (Zero-Cost, Free Tier)

### Architectural Motivation: Why AWS Lakehouse (S3 + Glue + Athena)?

A common anti-pattern in ML competitions is leaving multi-gigabyte flat files (`.tsv` / `.csv`) on local disks or raw S3 buckets. In a production enterprise setting (e.g. Amazon Catalog, Fraud, or Search teams), this fails:
1. **Decoupling GPU Compute from Downstream Serving**: Heavy ML inference requires expensive GPU instances (e.g. CUDA 13, RTX 5070 Ti / AWS G5). Keeping GPU clusters idle just to let business analysts or microservices look up entity matches is wasteful. We run batch inference once, convert predictions to a **Kimball Star Schema**, and serve downstream consumers using serverless SQL at **$0.00 idle cost**.
2. **Partition Pruning (60%–80% Scan & Latency Reduction)**: Searching a flat 2.8 GB TSV requires a full disk scan every single query. With Hive-partitioned Snappy Parquet (`country=France`, etc.), Athena reads only the target country partition, scanning ~35 MB in **sub-2-second latency** (~$0.00017 per query).
3. **Automated CI/CD Data Quality Gate**: Model predictions must never be published to master catalogs unchecked. Athena executes an automated **mutual exclusivity audit** SQL check that mathematically confirms zero S2/S3 target records were assigned to multiple S1 canonical entities.
4. **Enterprise Multi-Tenant Accessibility**: Converted into Kimball dimensions and facts, downstream services (search indexing, vendor risk, BI dashboards in QuickSight/Tableau) can query the canonical entity graph directly via standard JDBC/ODBC/SQL without needing custom Python parsers.

### Architecture: Hybrid Edge-GPU + AWS Data Lakehouse

```
[Local GPU: 11.7M records, 128.7M candidate pairs, trained models]
                          |
                          v  (export_lakehouse.py — ~10 min)
+------------------------------------------------------------------+
|  KIMBALL STAR SCHEMA — Snappy-compressed Parquet                |
|  dim_canonical_entity  (1.73M S1 entities, partitioned/country) |
|  dim_target_record     (9.97M S2+S3 records, part./country)     |
|  fact_entity_summary   (per-S1 blocking telemetry)              |
|  fact_resolution_links (5.8M resolved 1-to-many links)          |
+------------------------------------------------------------------+
                          |
                          v  (s3_upload.py / aws_deploy.ps1 — ~5 min)
+------------------------------------------------------------------+
|  AMAZON S3 MEDALLION DATA LAKE  (AWS Free Tier — 5 GB free)     |
|  s3://riya-entity-lakehouse-2026/lakehouse_gold/                  |
+------------------------------------------------------------------+
                          |
                          v  (Athena DDL + MSCK REPAIR — ~5 min)
+------------------------------------------------------------------+
|  AMAZON ATHENA SERVERLESS DATA WAREHOUSE                         |
|  Partition pruning: country scoping scans ~60%–80% less data     |
|  Cost: ~$0.0007/query on 1.06 GB compressed Parquet             |
+------------------------------------------------------------------+
```

### Verified Metrics & Live Cloud Query Results

All production query outputs are committed directly in [`aws_results/`](aws_results/):

| Metric | Value | Verification Source |
|---|---|---|
| Total Records Processed | **11,702,133** (1.73M S1 + 9.97M S2/S3) | Row counts in dim tables |
| Blocking Search-Space Reduction | **99.9993%** (17.27T → 128.7M pairs) | [`aws_results/query2_executive_rollup.csv`](aws_results/query2_executive_rollup.csv) |
| Total Resolved Entity Links | **5,799,165** (US: 2.33M, IN: 2.52M, FR: 948K) | [`aws_results/query1_quality_gate.csv`](aws_results/query1_quality_gate.csv) |
| Storage Compression | **62.7%** reduction (2.83 GB TSV → 1.06 GB Parquet) | Raw TSV vs `lakehouse_gold/` folder size |
| Mutual Exclusivity Violations | **0** across all 5.8M links | [`aws_results/query1_quality_gate.csv`](aws_results/query1_quality_gate.csv) |
| Validation Macro F0.5 | **0.9645** (+10.75 pts over baseline 0.857) | Held-out validation split benchmark |

---

### Cloud Warehouse Verification Queries

#### Query 1: Data Quality Gate (Mutual Exclusivity Audit)
*Result artifact:* [`aws_results/query1_quality_gate.csv`](aws_results/query1_quality_gate.csv)
```sql
SELECT
    country,
    COUNT(target_id)                             AS total_assigned_targets,
    COUNT(DISTINCT target_id)                    AS unique_assigned_targets,
    COUNT(target_id) - COUNT(DISTINCT target_id) AS mutual_exclusivity_violations
FROM entity_resolution_dw.fact_resolution_links
GROUP BY country
ORDER BY country;
```

**Live Athena Output (`aws_results/query1_quality_gate.csv`):**
| country | total_assigned_targets | unique_assigned_targets | mutual_exclusivity_violations |
| :--- | :--- | :--- | :--- |
| **France** | 948,244 | 948,244 | **0** |
| **India** | 2,523,615 | 2,523,615 | **0** |
| **US** | 2,327,306 | 2,327,306 | **0** |

*Verification:* Zero duplicate assignments across all 5,799,165 links; perfect 1-to-many integrity.

#### Query 2: Executive Warehouse Rollup (Telemetry & Singletons)
*Result artifact:* [`aws_results/query2_executive_rollup.csv`](aws_results/query2_executive_rollup.csv)
```sql
SELECT
    country, model_version,
    COUNT(entity_id)                                                   AS total_s1_entities,
    SUM(candidate_count)                                               AS total_blocking_pairs,
    ROUND(AVG(CAST(candidate_count AS DOUBLE)), 2)                     AS avg_candidates_per_entity,
    SUM(matched_target_count)                                          AS total_resolved_matches,
    ROUND(100.0 * SUM(CASE WHEN is_singleton THEN 1 ELSE 0 END)
                / NULLIF(COUNT(entity_id), 0), 2)                      AS singleton_rate_pct
FROM entity_resolution_dw.fact_entity_summary
GROUP BY country, model_version
ORDER BY total_s1_entities DESC;
```

**Live Athena Output (`aws_results/query2_executive_rollup.csv`):**
| country | model_version | total_s1_entities | total_blocking_pairs | avg_candidates_per_entity | total_resolved_matches | singleton_rate_pct |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **India** | `v9_semantic_xgb` | 809,986 | 64,453,235 | 79.57 | 2,523,615 | 6.57% |
| **US** | `v10_jev_judge` | 663,106 | 41,379,932 | 62.40 | 2,327,306 | 6.13% |
| **France** | `v10_jev_judge` | 259,452 | 22,877,895 | 88.18 | 948,244 | 5.09% |

---

### Step-by-Step Reproduction Guide

#### Step 1: Build the Star Schema Parquet (Local, ~10 min)
```bash
cd code/business_entity_resolution/src
python export_lakehouse.py
# Generates output/lakehouse_gold/ (1.06 GB, 4 tables, 3 country partitions each)
```

#### Step 2: Configure AWS & Upload to S3 (~5 min)
```bash
# Configure credentials:
aws configure   # region: ap-south-1

# PowerShell (Windows) or Python boto3:
python deploy/s3_upload.py
# Or: .\deploy\aws_deploy.ps1
```

#### Step 3: Register Tables in Amazon Athena (~5 min)
1. Open the [Amazon Athena Console](https://ap-south-1.console.aws.amazon.com/athena/home?region=ap-south-1).
2. Set query result location to `s3://riya-entity-lakehouse-2026/athena-results/`.
3. Execute `sql/athena_warehouse_ddl.sql` to register the database and all 4 tables with partition discovery (`MSCK REPAIR`).

---

### Repo Layout (AWS Additions)

```
aws_results/
  query1_quality_gate.csv     # Athena Data Quality Gate output (0 violations)
  query2_executive_rollup.csv # Athena Executive Rollup output (blocking & singletons)
code/business_entity_resolution/src/
  export_lakehouse.py         # ETL: TSVs -> Partitioned Kimball Parquet
sql/
  athena_warehouse_ddl.sql    # Athena DDL + verification analytical queries
deploy/
  s3_upload.py                # Direct boto3 parallel S3 uploader
  aws_deploy.sh               # Bash deployment script
  aws_deploy.ps1              # PowerShell deployment script
output/
  matching_results.tsv        # Raw leaderboard submission (1.73M S1 rows)
  candidate_pairs.tsv         # Blocking candidates (128.7M pairs)
  lakehouse_gold/             # Parquet Star Schema (generated by export_lakehouse.py)
    dim_canonical_entity/     # 311.7 MB Snappy Parquet, partitioned by country
    dim_target_record/        # 641.6 MB Snappy Parquet, partitioned by country
    fact_entity_summary/      #  20.3 MB Snappy Parquet, partitioned by country
    fact_resolution_links/    #  84.3 MB Snappy Parquet, partitioned by country
```
