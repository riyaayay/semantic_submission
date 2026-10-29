# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Semantic  
**Submission Date:** 2 October 2026  

---

## 1. Executive Summary

The submitted system resolves 1,732,544 Source-1 business entities against 9,969,589 Source-2 and Source-3 records.
It runs end to end on a single laptop GPU (NVIDIA GeForce RTX 5070 Ti, CUDA 13), and every GPU-capable stage
executes on the GPU. The system comprises four components:

1. **Multi-Path Blocking with Address-Refined Hub Keys and Multilingual Semantic Retrieval:** language-agnostic
   candidate generation.
2. **GPU Pairwise Feature Engine:** 50 features, including bit-parallel string alignment, label-free per-country
   term statistics and multilingual sentence embeddings.
3. **Gradient-Boosted Pairwise Matcher with Unsupervised Per-Country Domain Adaptation:** adapts to France, for which
   no training labels exist.
4. **Set-Level Adjudication Network:** a transformer that decides each entity's complete candidate set jointly, using
   an Inter-Candidate Consistency Graph.

On held-out validation entities the Final Model reaches a Macro $F_{0.5}$ of **0.9645**. The same system without the
Set-Level Adjudication Network reaches **0.9516**, and the Baseline Model of our first submission **0.857**.

**Composition of the submitted file:** France and United States rows are produced by the Final Model. India rows are
produced by the Semantic-Augmented Model (the Final Model without the Set-Level Adjudication Network; validation
0.9513), because inference of the Final Model for India could not be completed before the deadline (Section 5.3).

---

## 2. Methodology

### 2.1 Problem Analysis
1. **One-to-many structure:** each Source-2/Source-3 record links to at most one Source-1 entity. Each target record
   is therefore assigned exclusively to its highest-scoring Source-1 entity.
2. **Unseen country:** training data covers the United States and India; the test set adds **France** (259,452
   entities) without labels. Generalization was therefore measured with leave-one-country-out validation (training on
   one country, evaluating on the other) as a proxy for France.
3. **Country-specific labelling conventions** (measured on the training data): pairs with identical names are true
   matches in 41% of cases in the United States but 21% in India; an entity's highest-ranked candidate is a true
   match in 70% versus 50% of cases. A model fixed at training time cannot be optimal for an unseen country, which
   motivates per-country domain adaptation.
4. **Multilingual variation:** diacritics (`Société`), dotted acronyms (`S.A.S.`), nine Indic scripts (Devanagari,
   Bengali, Tamil, Telugu and others), transliteration, abbreviations (`R.` for `rue`, `Av.` for `avenue`) and
   affiliated companies (`X Pvt Ltd` versus `X Exports Pvt Ltd`).

### 2.2 Solution Strategy

**Approach:** Multi-Path and Semantic Blocking -> GPU Pairwise Feature Engine -> Gradient-Boosted Pairwise Matcher
with Unsupervised Per-Country Domain Adaptation -> Set-Level Adjudication Network -> exclusive assignment with a
validated decision threshold.

**Principal contributions**
1. **Script-Agnostic Text Normalization:** all Brahmic scripts are mapped onto Devanagari by Unicode block offset and
   romanized; Latin diacritics are removed; dotted acronyms are joined; a phonetic consonant-skeleton key captures
   phonetically similar and cross-script spellings.
2. **Label-Free Per-Country Term Statistics:** inverse document frequencies and legal-form position rates are
   estimated from each country's own target records, so an unseen country is processed identically to a training
   country.
3. **Unsupervised Per-Country Domain Adaptation (Self-Training):** the pairwise matcher scores each country; its
   high-confidence predictions (probability >= 0.97 or <= 0.03, up to 3 million pairs) serve as pseudo-labels, and
   the matcher is retrained on the labelled source data together with these pseudo-labels. No test labels are used.
4. **Set-Level Adjudication Network:** the design follows the decision-model principles of the open-source
   reconstruction *open-jev* (kyegomez/open-jev, Apache-2.0), re-implemented for entity resolution (Section 4.3).

---

## 3. Candidate Generation (Blocking) & Scalability Audit

Candidate search is partitioned by country, since no true match crosses a country boundary. Blocking keys per record:
- compressed name prefixes of 8 and 5 characters;
- the first two name words;
- distinctive name tokens (length 3 or more, at most 4);
- a phonetic skeleton prefix;
- postal code with house number;
- house number with address token;
- address word bigrams.

- **Address-Refined Hub Keys:** a name key shared by more than 50 target records is refined into (name key, address
  token) sub-keys instead of being truncated, which had discarded true matches of common business names.
  Training-sample blocking recall rose from 91.3% to 95.5% (United States) and from 85.6% to 92.2% (India), with
  fewer candidates.
- **Multilingual Semantic Retrieval:** the ten nearest target records per entity by multilingual name-embedding cosine
  similarity (exact GPU search). Recall rose from 95.4% to **96.3%** (United States) and from 92.0% to **92.9%**
  (India).
- **Per-entity limit:** at most 120 key-based candidates, ranked by the number of shared keys, plus the ten semantic
  neighbours.

**Test-set candidate statistics**

| Country | Source-1 entities | Candidate pairs | Mean per entity |
|---|---:|---:|---:|
| France | 259,452 | 22,877,895 | 88.2 |
| United States | 663,106 | 41,379,932 | 62.4 |
| India | 809,986 | 64,453,235 | 79.6 |
| **Total** | **1,732,544** | **128,711,062** | **74.3** |

Search-space reduction: 1 - 74.3 / 9,969,589 = **99.9993%** of all entity-target comparisons are avoided.

---

## 4. Matching Model

### 4.1 GPU Pairwise Feature Engine (50 features)
- **String alignment:** bit-parallel longest-common-subsequence (Hyyro; Allison-Dix) and Levenshtein (Myers)
  algorithms implemented as GPU kernels over names, token-sorted names, addresses and phonetic skeletons. Verified
  identical to RapidFuzz on 40,000 real record pairs.
- **Term statistics:**
  - TF-IDF cosine similarity of name and address;
  - character trigram cosine similarity;
  - IDF-weighted Jaccard similarity;
  - IDF-weighted soft token alignment (Monge-Elkan) with abbreviation matching;
  - unmatched-token count, IDF mass and legal-form position rate (separating legal forms such as `Limited` or `SARL`
    from business descriptors such as `Exports` or `Holdings`);
  - acronym agreement;
  - house-number agreement and conflict.
- **Within-entity context:** difference from the entity's best candidate (name, address, alignment and semantic
  similarity), candidate rank and candidate count.
- **Multilingual Semantic Embedding Module:** cosine similarity of multilingual sentence embeddings of the original
  name and address text, with the difference from the best candidate.
  - Model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (Apache-2.0, 118 million parameters, inference
    only).
  - Weights are stored in 16-bit floating point, bit-identical to the inference configuration.

### 4.2 Gradient-Boosted Pairwise Matcher
- **Training:** XGBoost 3.2.0 on CUDA (`max_depth=7`, `learning_rate=0.05`, `colsample_bytree=0.6`, early stopping
  on validation log-loss), trained on 60,000 sampled entities per training country.
- **Inference:** Unsupervised Per-Country Domain Adaptation is applied to each test country.

### 4.3 Set-Level Adjudication Network
- **Input:** for each entity, the 32 highest-ranked candidates under a label-free similarity ranking, which retains
  99.57% of true matches, together with the entity itself.
- **Architecture:**
  - a three-layer bidirectional transformer (model width 128, four attention heads);
  - each candidate is represented by its standardized pairwise features and the pairwise matcher's score.
- **Inter-Candidate Consistency Graph:** thirteen similarity measures between every two candidates indicate whether
  the candidates are duplicate records of one business or conflicting variants. They enter every attention layer as
  learned additive biases, so the decision for one candidate depends on the others.
- **Outputs:** per-candidate query slots produce an evidential Beta distribution, giving a calibrated match
  probability and a confidence.
- **Training objective:** a calibrated-decision loss combining expected negative log-likelihood under the Beta
  distribution, the Brier score, an evidential regularizer and a consistency term between two perturbed forward passes.
- **Leakage control:** out-of-fold pairwise-matcher scores serve as inputs during training.

**Decision rule:** each target record is assigned to the entity that scores it highest; assignments above the
validated threshold (0.60 for the Final Model) are retained.

**Licenses and model size:**
- XGBoost: Apache-2.0.
- Multilingual MiniLM: Apache-2.0, 118 million parameters.
- Set-Level Adjudication Network: 488,784 parameters, trained by the team.

All are far below the 8-billion-parameter limit.

---

## 5. Results & Error Analysis

### 5.1 Validation Macro $F_{0.5}$ (official per-entity metric)

| Model | Validation |
|---|---:|
| Baseline Model (first submission; 0.830 on the leaderboard) | 0.857 |
| GPU Feature Model (refined blocking, per-country adaptation) | 0.948 |
| Semantic-Augmented Model | 0.9516 |
| **Final Model (with Set-Level Adjudication Network)** | **0.9645** |

### 5.2 Generalization to an unseen country (40,000 entities, 20,000 per country)

| Evaluation | Baseline Model | Semantic-Augmented Model | Final Model |
|---|---:|---:|---:|
| Within-distribution | 0.868 | 0.9475 | **0.9593** |
| Trained on United States, evaluated on India | 0.681 | 0.7813 | **0.7990** |
| Trained on India, evaluated on United States | 0.822 | 0.9083 | **0.9161** |

**Upper bound:** a perfect matcher operating on the generated candidates would reach approximately 0.984 (United
States) and 0.970 (India). The remaining gap stems from true matches absent from the candidate sets and from
inconsistent ground-truth labels for near-identical record pairs.

### 5.3 Composition of the submitted file

| Country | Model | Matches | Per entity |
|---|---|---:|---:|
| France | Final Model | 948,244 | 3.65 |
| United States | Final Model | 2,327,306 | 3.51 |
| India | Semantic-Augmented Model | 2,523,615 | 3.12 |

- Inference of the Final Model for India (64.5 million pairs) could not be completed before the deadline.
- The candidate pairs are identical for both models (same blocking procedure; identical pair counts per country).

### 5.4 Error analysis (trained on United States, evaluated on India)

| Error type | Points of Macro $F_{0.5}$ lost |
|---|---:|
| Partial matches (correct business with additional affiliated companies, or missing duplicate records) | 13.1 |
| Scoring errors | 4.1 |
| Candidates missed by blocking | 2.8 |
| False merges on entities without true matches | 2.6 |

The Set-Level Adjudication Network addresses the partial-match category directly (+1.8 points on this evaluation).

### 5.5 Approaches evaluated and not adopted
- **Monotonic feature constraints:** lower scores in every evaluation.
- **Expectation-maximization prior-shift recalibration:** no consistent effect.
- **Parameter-efficient fine-tuning of the multilingual encoder:**
  - Method: low-rank adaptation of 0.13% of the parameters, with a distillation anchor to the original model and a
    French-English cross-lingual acceptance test.
  - The acceptance test confirmed that cross-lingual capability was preserved.
  - As additional features it reduced the United States -> India score by 0.10 points, so it was not adopted.

---

## 6. Conclusion

The system moves from string similarity to a pipeline that:
- normalizes text in any script;
- estimates term statistics per country without labels;
- adapts to each unseen country through self-training;
- adjudicates complete candidate sets jointly.

Validation Macro $F_{0.5}$ improved from 0.857 to 0.9645, and the unseen-country evaluation (United States -> India)
improved from 0.681 to 0.799.

**Reproducibility statement:** after the submitted outputs were produced, one helper function was found to use
Python's process-randomized `hash()` for character-bigram signatures in five alignment features. It has been replaced
by a deterministic CRC32 hash, with a regression test. Re-executing the documented procedure therefore yields
equivalent but not byte-identical outputs; the models should be retrained with the corrected code first (see
README.md, "Reproducing the submitted output files").

---

## Appendix

### A. Code Artefacts (`code/business_entity_resolution/`)

| File | Component |
|---|---|
| `src/blocking.py`, `src/translit.py` | Script-Agnostic Text Normalization; Multi-Path Blocking with Address-Refined Hub Keys |
| `src/gpu.py`, `src/gpu_strings.py`, `src/gpu_features.py` | GPU Pairwise Feature Engine |
| `src/semantic.py` | Multilingual Semantic Embedding Module and Multilingual Semantic Retrieval |
| `src/judge.py` | Set-Level Adjudication Network and Inter-Candidate Consistency Graph |
| `src/v8_pipeline.py` | Training procedures, per-country feature computation, Unsupervised Per-Country Domain Adaptation |
| `src/decision.py`, `src/metrics.py` | Exclusive assignment, decision thresholds, official evaluation metric |
| `src/train_matcher.py`, `src/predict.py` | Training and inference entry points |
| `src/assemble_matches_lowmem.py` | Assembly of per-country outputs into the final file |
| `src/semantic_finetune.py` | Parameter-efficient fine-tuning (evaluated, not adopted) |
| `models/` | Final Model (`xgboost_matcher_v10.json`, `judge_v10.pt`, `config_v10.pkl`); Semantic-Augmented Model (`xgboost_matcher_v9.json`, `config_v9.pkl`); multilingual encoder (`semantic_minilm/`, with `MODEL_LICENSE.txt`) |
| `tests/` | 14 automated tests (GPU kernels against RapidFuzz, blocking, adjudication network, determinism) |
| `README.md`, `requirements.txt` | Exact reproduction procedure and pinned dependency versions |

### B. Hardware
Intel Core Ultra 9 275HX (24 cores), 31 GB RAM, NVIDIA GeForce RTX 5070 Ti Laptop GPU (12 GB), CUDA 13.
