# Business Entity Resolution Pipeline — Amazon ML Challenge 2026

> **Validation Macro F0.5: 0.9645** | 11.7 M records | 128.7 M candidate pairs | 99.9993 % search-space reduction

---

## Table of Contents

1. [Problem Overview](#1-problem-overview)
2. [Pipeline Architecture](#2-pipeline-architecture)
3. [Verified Metrics](#3-verified-metrics)
4. [AWS Lakehouse Deployment](#4-aws-lakehouse-deployment)
5. [Step-by-Step Reproduction](#5-step-by-step-reproduction)
6. [Data Quality Checks](#6-data-quality-checks)
7. [Repo Layout](#7-repo-layout)
8. [Model Compliance](#8-model-compliance)

---

## 1. Problem Overview

Three noisy business-record sources share no common IDs:

| Source | Description | Count |
|--------|-------------|-------|
| S1 | Deduplicated reference entities | 1.73 M |
| S2 | Vendor records (typos, legal-suffix variation) | ~5 M |
| S3 | Vendor records (missing addresses, transliterations) | ~5 M |

Training covers **US** and **India**; the test set adds **France** (zero labels). The pipeline uses strictly language-agnostic features so unseen countries are handled identically to seen ones.

---

## 2. Pipeline Architecture

```
S1 / S2 / S3 TSVs
       |
       v  normalizer.py
Unicode NFKD, accent folding, street abbreviations, house-number / postal-code extraction
       |
       v  blocking.py
Multi-path inverted index:
  - Compressed name prefixes (8-char, 5-char)
  - First-2-word anchor
  - Salient name tokens (len >= 5, freq <= 150)
  - Postal-code + house-number anchor
  - India-gated address bigrams (non-Latin transliterations)
  - Top-10 by multilingual name-embedding cosine (v9+)
       |
       v  features.py / gpu_features.py
25-50 pairwise features: Levenshtein ratio, Token Sort/Set, Jaro-Winkler,
numeric-token Jaccard, postal-code match, phonetic Soundex, TF-IDF cosine,
char 3-gram, soft token alignment, semantic embedding cosine (v9+)
       |
       v  train_matcher.py / predict.py
XGBoost (CUDA) + set-level Jev judge (v10)
Threshold tau = 0.65, 1-to-many exclusive assignment
       |
       v  matching_results.tsv / candidate_pairs.tsv
```

### Model versions

| Version | Key addition | Validation F0.5 |
|---------|-------------|-----------------|
| v7 | Phonetic skeleton, country-agnostic IDF | — |
| v9 | GPU features, multilingual embedding cosine | 0.9516 |
| **v10** | Jev-style set-level transformer judge | **0.9645** |

**Submitted output:** France + US rows from v10; India rows from v9 (v10 inference did not finish before the deadline). `candidate_pairs.tsv` is identical for v9 and v10 (same blocking code).

---

## 3. Verified Metrics

| Metric | Value | Source |
|--------|-------|--------|
| Validation Macro F0.5 | **0.9645** | Held-out validation split |
| Total records | 11,702,133 (1.73 M S1 + 9.97 M S2/S3) | Row counts in dim tables |
| Candidate pairs | 128,711,062 | `fact_entity_summary.candidate_count` SUM |
| Search-space reduction | **99.9993 %** (17.27 T → 128.7 M) | Athena Query 3 |
| Resolved entity links | 5,799,165 (US 2.33 M, IN 2.52 M, FR 948 K) | Athena Query 1 |
| TSV → Parquet compression | 62.7 % (2.83 GB → 1.06 GB) | Folder sizes |
| Mutual-exclusivity violations | 0 | Athena Query 1 (see §6 for stronger checks) |

---

## 4. AWS Lakehouse Deployment

### Why S3 + Glue + Athena?

The TSV outputs are 2.83 GB of flat files. Converting them to Snappy-compressed, country-partitioned Parquet and registering them in Athena lets any SQL tool query the results without keeping a Python process or GPU running.

**Cost on this dataset (Athena at \$5 / TB scanned):**

| Query scope | Data scanned | Estimated cost |
|-------------|-------------|----------------|
| All 4 tables, all countries | ~1.06 GB | ~\$0.0053 |
| Single country partition only | ~35–120 MB | ~\$0.0002–\$0.0006 |

> **Note:** Partition pruning (e.g. `WHERE country = 'France'`) reduces the bytes Athena actually reads. The figures above are estimates; exact bytes scanned are shown in the Athena console's *Query statistics* tab after each run.

**Free-tier usage:** S3 (15 objects, 1,058 MB) stays within the 5 GB free tier. Glue Data Catalog table registrations are free under 1 M objects. Athena has no free tier; each query above costs a fraction of a cent.

### Architecture

```
[Local GPU inference — 11.7 M records, 128.7 M pairs]
                  |
                  v  export_lakehouse.py (~10 min, local)
+------------------------------------------------------------+
|  Kimball Star Schema — Snappy Parquet                      |
|  dim_canonical_entity   1.73 M rows   311.7 MB            |
|  dim_target_record      9.97 M rows   641.6 MB            |
|  fact_entity_summary    1.73 M rows    20.3 MB            |
|  fact_resolution_links  5.80 M rows    84.3 MB            |
+------------------------------------------------------------+
                  |
                  v  s3_upload.py (~5 min)
+------------------------------------------------------------+
|  Amazon S3                                                 |
|  s3://riya-entity-lakehouse-2026/lakehouse_gold/           |
+------------------------------------------------------------+
                  |
                  v  athena_warehouse_ddl.sql (~5 min)
+------------------------------------------------------------+
|  AWS Glue Data Catalog + Amazon Athena                     |
|  database: entity_resolution_dw (4 external tables)       |
+------------------------------------------------------------+
```

### Live query results

Committed in [`aws_results/`](aws_results/):

| File | Contents |
|------|----------|
| [`query1_quality_gate.csv`](aws_results/query1_quality_gate.csv) | Mutual-exclusivity audit per country |
| [`query2_executive_rollup.csv`](aws_results/query2_executive_rollup.csv) | Blocking telemetry + singleton rates |

---

## 5. Step-by-Step Reproduction

### A. Local inference (requires NVIDIA GPU, CUDA 13)

```bash
cd code/business_entity_resolution/src
pip install -r ../requirements.txt

# Train (writes models/xgboost_matcher_v9.json and v10, judge_v10.pt, config*.pkl)
python train_matcher.py --features v9  --per-country 60000
python train_matcher.py --features v10 --per-country 60000

# v10 inference for all countries
MODEL_CONFIG=config_v10.pkl OUTPUT_DIR=../../../output/v10 python predict.py

# v9 inference for India only
MODEL_CONFIG=config_v9.pkl PREDICT_COUNTRIES=India OUTPUT_DIR=../../../output/v9_india python predict.py

# Assemble final submission (v10 rows for France + US, v9 for India)
python assemble_matches_lowmem.py \
    ../../../output/v9_india/matching_results.tsv \
    ../../../output/final_matching_results.tsv \
    --checkpoint France=../../../output/v10 \
    --checkpoint US=../../../output/v10
```

> **Exactness note:** The submitted outputs were produced before a determinism fix (Python `hash()` replaced with CRC32 for character-bigram signatures). Re-running gives equivalent, not byte-identical, results; retrain first so model and features match.

### B. Validate submission format

```bash
python ../../6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py \
    --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir ../../6ab10eb3b23ba_student_resource/student_resource/dataset/test \
    --check-ids
```

Expected: `PASS — no blocking issues found.`

### C. Build the Star Schema Parquet (local, ~10 min)

```bash
cd code/business_entity_resolution/src
python export_lakehouse.py
# Writes output/lakehouse_gold/ (4 tables, ~1.06 GB total)
```

### D. Upload to S3 (~5 min)

```bash
aws configure          # region: ap-south-1

python deploy/s3_upload.py
# Or on Windows PowerShell:
.\deploy\aws_deploy.ps1
```

### E. Register tables in Athena (~5 min)

1. Open the [Athena console](https://ap-south-1.console.aws.amazon.com/athena/home?region=ap-south-1).
2. Set query result location to `s3://riya-entity-lakehouse-2026/athena-results/`.
3. Run all statements in `sql/athena_warehouse_ddl.sql` (CREATE DATABASE, 4 × CREATE EXTERNAL TABLE, 4 × MSCK REPAIR TABLE).

---

## 6. Data Quality Checks

### Query 1 — Mutual-exclusivity audit

Each S2/S3 target must appear in at most one S1 entity's match set. The assignment step enforces this by construction; this query is a regression gate confirming the Parquet export did not corrupt the data.

```sql
SELECT
    country,
    COUNT(target_id)                             AS total_assigned_targets,
    COUNT(DISTINCT target_id)                    AS unique_assigned_targets,
    COUNT(target_id) - COUNT(DISTINCT target_id) AS mutual_exclusivity_violations
FROM entity_resolution_dw.fact_resolution_links
GROUP BY country ORDER BY country;
```

**Result ([`aws_results/query1_quality_gate.csv`](aws_results/query1_quality_gate.csv)):**

| country | total_assigned_targets | unique_assigned_targets | mutual_exclusivity_violations |
|---------|------------------------|-------------------------|-------------------------------|
| France  | 948,244  | 948,244  | **0** |
| India   | 2,523,615 | 2,523,615 | **0** |
| US      | 2,327,306 | 2,327,306 | **0** |

### Query 2 — Blocking telemetry rollup

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
GROUP BY country, model_version ORDER BY total_s1_entities DESC;
```

**Result ([`aws_results/query2_executive_rollup.csv`](aws_results/query2_executive_rollup.csv)):**

| country | model_version | total_s1_entities | total_blocking_pairs | avg_candidates | total_resolved | singleton_rate |
|---------|---------------|------------------:|---------------------:|---------------:|---------------:|---------------:|
| India   | v9_semantic_xgb | 809,986 | 64,453,235 | 79.57 | 2,523,615 | 6.57 % |
| US      | v10_jev_judge  | 663,106 | 41,379,932 | 62.40 | 2,327,306 | 6.13 % |
| France  | v10_jev_judge  | 259,452 | 22,877,895 | 88.18 |   948,244 | 5.09 % |

### Additional integrity checks (run in Athena)

The following checks can actually fail if the export is corrupted, making them meaningful regression gates:

**Row-count match — dim_canonical_entity must contain exactly 1,732,544 rows:**
```sql
SELECT COUNT(*) AS s1_row_count FROM entity_resolution_dw.dim_canonical_entity;
-- Expected: 1732544
```

**Row-count match — dim_target_record must contain exactly 9,969,589 rows:**
```sql
SELECT COUNT(*) AS target_row_count FROM entity_resolution_dw.dim_target_record;
-- Expected: 9969589
```

**Referential integrity — every entity_id in fact_resolution_links must exist in dim_canonical_entity:**
```sql
SELECT COUNT(*) AS orphan_links
FROM entity_resolution_dw.fact_resolution_links f
LEFT JOIN entity_resolution_dw.dim_canonical_entity d
       ON f.entity_id = d.entity_id AND f.country = d.country
WHERE d.entity_id IS NULL;
-- Expected: 0
```

**Null rates on key columns:**
```sql
SELECT
    SUM(CASE WHEN entity_id  IS NULL THEN 1 ELSE 0 END) AS null_entity_id,
    SUM(CASE WHEN name_raw   IS NULL THEN 1 ELSE 0 END) AS null_name_raw,
    SUM(CASE WHEN country    IS NULL THEN 1 ELSE 0 END) AS null_country
FROM entity_resolution_dw.dim_canonical_entity;
-- Expected: all 0
```

**Row-count checksum — fact_resolution_links total must match the sum from fact_entity_summary:**
```sql
SELECT
    (SELECT SUM(matched_target_count) FROM entity_resolution_dw.fact_entity_summary) AS summary_total,
    (SELECT COUNT(*)                  FROM entity_resolution_dw.fact_resolution_links) AS links_total;
-- Expected: both columns equal 5799165
```

---

## 7. Repo Layout

```
aws_results/
  query1_quality_gate.csv          # Athena mutual-exclusivity audit output
  query2_executive_rollup.csv      # Athena blocking telemetry output

code/business_entity_resolution/
  src/
    normalizer.py                  # Text, address, and legal-suffix normalizer
    blocking.py                    # Multi-path blocking engine
    features.py / gpu_features.py  # Pairwise feature extraction
    train_matcher.py               # XGBoost training + hard-negative mining
    predict.py                     # Full test inference
    judge.py                       # Jev-style set-level transformer judge (v10)
    decision.py                    # 1-to-many exclusive assignment + threshold
    assemble_matches_lowmem.py     # Merges per-country checkpoints
    export_lakehouse.py            # ETL: TSVs -> partitioned Parquet star schema
  models/
    xgboost_matcher_v10.json       # XGBoost booster (inference-ready)
    judge_v10.pt                   # Jev judge weights (inference-ready)
    config_v10.pkl                 # Decision threshold (tau = 0.65)
    semantic_minilm/               # paraphrase-multilingual-MiniLM-L12-v2 (Apache-2.0)
  requirements.txt

sql/
  athena_warehouse_ddl.sql         # CREATE DATABASE, tables, MSCK REPAIR, verification queries

deploy/
  s3_upload.py                     # boto3 parallel uploader
  aws_deploy.ps1                   # PowerShell deployment script

output/                            # (git-ignored — generated locally)
  matching_results.tsv             # Leaderboard submission (1.73 M S1 rows)
  candidate_pairs.tsv              # Blocking candidates (128.7 M pairs)
  lakehouse_gold/                  # Parquet star schema (generated by export_lakehouse.py)
```

---

## 8. Model Compliance

| Item | Detail |
|------|--------|
| Model type | XGBoost GBDT + Transformer judge |
| Licenses | Apache 2.0 (XGBoost), Apache 2.0 (sentence-transformers) |
| Parameter count | XGBoost: < 10,000 tree parameters; judge_v10.pt: < 8 B |
| External APIs | None — zero web lookups or geocoding at inference |
| Training data leakage | TF-IDF vocabulary fitted on training set only; test features use that frozen vocab |
