# Business Entity Resolution Pipeline — Amazon ML Challenge 2026

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
