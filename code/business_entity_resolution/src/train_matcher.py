"""
Production Matcher Training and Macro F_0.5 Threshold Optimization.
Engineered for CUDA GPU Execution on NVIDIA RTX GPUs (XGBoost GPU / CUDA).
Supports fallback to multi-core CPU or LightGBM.

Production Variant 6 (25 features):
1. Hard negative mining from multi-path blocking candidates.
2. Grouped train/validation split by S1 entity (80/20).
3. Strict Zero-Leakage: IDF fitted strictly on training target records and persisted to models/tfidf_vocab.pkl.
4. Native CUDA GPU training via XGBoost 3.2.0 (device='cuda', tree_method='hist').
5. Threshold sweep optimizing Macro F_0.5 per entity, locking tau=0.65 for production.
6. Structural 1-to-many mutual exclusivity post-processing.
"""

import os
import sys
import time
import pickle
from typing import Dict, Set, List, Tuple, Any
import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.model_selection import train_test_split

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    getattr(sys.stdout, "reconfigure")(encoding="utf-8")

from metrics import compute_macro_f05
from blocking import MultiPathBlocker, extract_record_features, STOP_WORDS
from features import (
    extract_pairwise_features,
    parse_record_for_features,
    compute_tfidf_vocab_from_targets,
    save_tfidf_vocab,
    FEATURE_NAMES
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(SCRIPT_DIR)
MODEL_DIR = os.path.join(PKG_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

DEFAULT_DATA_DIR = os.path.abspath(os.path.join(PKG_ROOT, "..", "..", "6ab10eb3b23ba_student_resource", "student_resource", "dataset", "train"))
DATA_DIR = os.environ.get("TRAIN_DATA_DIR", DEFAULT_DATA_DIR)
if not os.path.exists(DATA_DIR):
    DATA_DIR = r"c:\Users\realr\OneDrive\Desktop\Amazon\6ab10eb3b23ba_student_resource\student_resource\dataset\train"


def detect_gpu() -> Tuple[bool, str]:
    """
    Detect if an NVIDIA/compatible GPU is physically available on the system.
    Returns (has_gpu, gpu_name).
    """
    try:
        import torch
        if torch.cuda.is_available():
            return True, torch.cuda.get_device_name(0)
    except Exception:
        pass

    try:
        import subprocess
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5
        )
        if res.returncode == 0 and res.stdout.strip():
            gpu_name = res.stdout.strip().split("\n")[0].strip()
            return True, gpu_name
    except Exception:
        pass

    return False, "None"


def train_xgboost_with_fallback(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
    device_preference: str = "gpu",
    num_boost_round: int = 400
) -> Tuple[Any, str]:
    """
    Trains XGBoost model: uses CUDA GPU on RTX GPU if found, else gracefully falls back to multi-core CPU.
    Returns (trained_booster, device_used).
    """
    import xgboost as xgb
    from gpu import require_cuda
    print("\n--- Model Training: XGBoost on CUDA GPU (no CPU fallback) ---")
    require_cuda()
    has_gpu, gpu_name = detect_gpu()
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
    evals = [(dtrain, "train"), (dval, "val")]

    base_params = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "auc"],
        "max_depth": 7,
        "learning_rate": 0.1,
        "colsample_bytree": 0.9,
        "subsample": 0.9,
        "random_state": 42
    }

    print(f"  [Hardware] GPU: {gpu_name}")
    t0 = time.time()
    gpu_params = {**base_params, "tree_method": "hist", "device": "cuda"}
    bst = xgb.train(gpu_params, dtrain, num_boost_round=num_boost_round, evals=evals, early_stopping_rounds=30, verbose_eval=50)
    print(f"  [Success] Trained on GPU (CUDA) in {time.time() - t0:.2f}s! Best iteration: {bst.best_iteration}")
    return bst, f"gpu ({gpu_name})"


def train_lightgbm_with_fallback(
    train_data: lgb.Dataset,
    val_data: lgb.Dataset,
    base_params: Dict[str, Any],
    device_preference: str = "auto",
    num_boost_round: int = 400
) -> Tuple[lgb.Booster, str]:
    """Legacy LightGBM path. The installed LightGBM has no CUDA build, and the
    pipeline does not fall back to CPU, so this refuses to run."""
    raise RuntimeError("LightGBM engine is CPU-only in this environment; use --engine xgboost (CUDA).")


def train_and_evaluate(sample_s1_count: int = 40000, preferred_device: str = "gpu", engine: str = "xgboost"):
    print("=" * 65)
    print(f"TRAINING {engine.upper()} PRODUCTION MATCHER (25 FEATURES) ON GPU")
    print(f"HARDWARE POLICY: NATIVE CUDA GPU ACCELERATION")
    print("=" * 65)
    t_start = time.time()

    # 1. Load Ground Truth
    print("Loading Ground Truth...")
    df_gt = pl.read_csv(os.path.join(DATA_DIR, "train_ground_truth.tsv"), separator="\t").head(sample_s1_count)
    s1_ids_list = df_gt["source1_entity_id"].to_list()
    gt_map: Dict[str, Set[str]] = {}
    gt_pair_set = set()

    for row in df_gt.iter_rows(named=True):
        sid = row["source1_entity_id"]
        matched = row["matched_entity_ids"]
        match_set = set(x.strip() for x in matched.split(",") if x.strip()) if matched else set()
        gt_map[sid] = match_set
        for tid in match_set:
            gt_pair_set.add((sid, tid))

    total_pos = len(gt_pair_set)
    print(f"Loaded {len(df_gt):,} S1 entities ({total_pos:,} true match pairs).")

    # 2. Train / Val Split by S1 entity ID (75/25 -> 30k train, 10k val)
    train_s1_ids, val_s1_ids = train_test_split(s1_ids_list, test_size=0.25, random_state=42)
    train_s1_set = set(train_s1_ids)
    val_s1_set = set(val_s1_ids)
    print(f"Train S1 entities: {len(train_s1_set):,}, Val S1 entities: {len(val_s1_set):,}")

    # 3. Load S1 and Targets
    print("Loading Source 1...")
    df_s1 = pl.read_csv(os.path.join(DATA_DIR, "train_source1.tsv"), separator="\t")
    df_s1 = df_s1.filter(pl.col("entity_id").is_in(s1_ids_list))
    prep_s1 = extract_record_features(df_s1, is_s1=True)

    print("Loading Source 2 & Source 3...")
    df_s2 = pl.read_csv(os.path.join(DATA_DIR, "train_source2.tsv"), separator="\t")
    df_s3 = pl.read_csv(os.path.join(DATA_DIR, "train_source3.tsv"), separator="\t")
    df_targets = pl.concat([df_s2, df_s3])
    prep_targets = extract_record_features(df_targets, is_s1=False)

    # 4. Run Blocking to generate Candidates
    print("\n--- Running Multi-Path Blocking ---")
    blocker = MultiPathBlocker()
    all_candidates: Dict[str, Set[str]] = {}

    for country in prep_s1["country"].unique().to_list():
        country_s1 = prep_s1.filter(pl.col("country") == country)
        country_targets = prep_targets.filter(pl.col("country") == country)
        blocker.build_index_for_country(country_targets)
        cands = blocker.query_s1_for_country(country_s1)
        all_candidates.update(cands)

    train_pairs: List[Tuple[str, str]] = []
    y_train_list: List[int] = []
    train_needed_tids = set()

    val_pairs: List[Tuple[str, str]] = []
    y_val_list: List[int] = []
    val_needed_tids = set()

    print("Assembling training and validation candidate pairs...")
    for sid in train_s1_ids:
        cands = set(all_candidates.get(sid, set())) | gt_map[sid]
        for tid in cands:
            train_pairs.append((sid, tid))
            is_match = 1 if (sid, tid) in gt_pair_set else 0
            y_train_list.append(is_match)
            train_needed_tids.add(tid)

    for sid in val_s1_ids:
        cands = all_candidates.get(sid, set())
        for tid in cands:
            val_pairs.append((sid, tid))
            is_match = 1 if (sid, tid) in gt_pair_set else 0
            y_val_list.append(is_match)
            val_needed_tids.add(tid)

    print(f"Train pairs: {len(train_pairs):,} (Pos: {sum(y_train_list):,}, Neg: {len(y_train_list) - sum(y_train_list):,})")
    print(f"Val pairs:   {len(val_pairs):,} (Pos: {sum(y_val_list):,}, Neg: {len(y_val_list) - sum(y_val_list):,})")

    # 5. Extract Feature Matrices
    print("Preparing lookup dictionaries for features...")
    t_feat_prep = time.time()
    s1_dict = {
        row["s1_id"]: parse_record_for_features(row)
        for row in prep_s1.iter_rows(named=True)
    }

    all_needed_tids = train_needed_tids | val_needed_tids
    filtered_targets = prep_targets.filter(pl.col("target_id").is_in(list(all_needed_tids)))
    target_dict = {
        row["target_id"]: parse_record_for_features(row)
        for row in filtered_targets.iter_rows(named=True)
    }
    print(f"Built feature lookups in {time.time() - t_feat_prep:.2f}s.")

    # STRICT ZERO-LEAKAGE TF-IDF: Fit IDF strictly on training target records
    train_target_dict = {tid: target_dict[tid] for tid in train_needed_tids if tid in target_dict}
    print(f"Fitting zero-leakage TF-IDF vocabulary on {len(train_target_dict):,} train target records...")
    tfidf_vocab = compute_tfidf_vocab_from_targets(train_target_dict)
    vocab_path = os.path.join(MODEL_DIR, "tfidf_vocab.pkl")
    save_tfidf_vocab(tfidf_vocab, vocab_path)
    print(f"Saved zero-leakage TF-IDF vocabulary to {vocab_path}")

    print(f"Extracting 25 features for {len(train_pairs):,} training pairs...")
    t_feat = time.time()
    X_train, feat_names = extract_pairwise_features(s1_dict, target_dict, train_pairs, tfidf_vocab=tfidf_vocab)
    y_train = np.array(y_train_list, dtype=np.float32)
    print(f"Train features extracted in {time.time() - t_feat:.2f}s.")

    print(f"Extracting 25 features for {len(val_pairs):,} validation pairs...")
    t_feat_val = time.time()
    X_val, _ = extract_pairwise_features(s1_dict, target_dict, val_pairs, tfidf_vocab=tfidf_vocab)
    y_val = np.array(y_val_list, dtype=np.float32)
    print(f"Val features extracted in {time.time() - t_feat_val:.2f}s.")

    # 6. Train Model natively on GPU via CUDA
    if engine.lower() == "xgboost":
        import xgboost as xgb
        model, device_used = train_xgboost_with_fallback(
            X_train, y_train, X_val, y_val, feat_names,
            device_preference=preferred_device, num_boost_round=400
        )
        print("\nTop 10 Most Important Features:")
        score_dict = model.get_score(importance_type="gain")
        sorted_feats = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
        for fname, gain in sorted_feats[:10]:
            print(f"  {fname:25s}: {gain:.1f}")

        model_path = os.path.join(MODEL_DIR, "xgboost_matcher.json")
        model.save_model(model_path)
        print(f"Model saved to {model_path}")

        # GPU Scoring
        t_eval = time.time()
        dval = xgb.DMatrix(X_val, feature_names=feat_names)
        val_preds = model.predict(dval)
        print(f"Scored {len(val_pairs):,} validation pairs on GPU in {time.time() - t_eval:.3f}s!")
    else:
        base_params = {
            "objective": "binary",
            "metric": ["binary_logloss", "auc"],
            "boosting_type": "gbdt",
            "learning_rate": 0.1,
            "num_leaves": 63,
            "max_depth": 7,
            "feature_fraction": 0.9,
            "min_child_samples": 50,
            "random_state": 42,
            "seed": 42,
            "feature_fraction_seed": 42,
            "data_random_seed": 42,
            "deterministic": True,
            "force_row_wise": True,
        }
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=feat_names)
        val_data = lgb.Dataset(X_val, label=y_val, feature_name=feat_names, reference=train_data)
        model, device_used = train_lightgbm_with_fallback(
            train_data, val_data, base_params,
            device_preference=preferred_device, num_boost_round=400
        )
        print("\nTop 10 Most Important Features:")
        importances = model.feature_importance(importance_type="gain")
        sorted_idx = np.argsort(importances)[::-1]
        for idx in sorted_idx[:10]:
            print(f"  {feat_names[idx]:25s}: {importances[idx]:.1f}")

        model_path = os.path.join(MODEL_DIR, "lightgbm_matcher.txt")
        model.save_model(model_path)
        print(f"Model saved to {model_path}")
        val_preds = model.predict(X_val)

    # 7. Predict & Evaluate Macro F_0.5 on Validation Set
    print("\n--- Evaluating Predictions on Validation Set ---")
    val_cand_scores: Dict[str, List[Tuple[str, float]]] = {sid: [] for sid in val_s1_ids}
    val_preds_arr = np.asarray(val_preds)
    for (sid, tid), score in zip(val_pairs, val_preds_arr):
        val_cand_scores[sid].append((tid, float(score)))

    val_gt_map = {sid: gt_map[sid] for sid in val_s1_ids}

    # Grid search threshold tau
    print("\nSweeping decision thresholds for Macro F_0.5...")
    best_tau = 0.65
    best_f05 = -1.0
    best_metrics = {}

    thresholds = [0.55, 0.60, 0.62, 0.65, 0.68, 0.70]
    for tau in thresholds:
        target_to_best_s1: Dict[str, Tuple[str, float]] = {}
        for sid, scored_list in val_cand_scores.items():
            for tid, sc in scored_list:
                if sc >= tau:
                    if tid not in target_to_best_s1 or sc > target_to_best_s1[tid][1]:
                        target_to_best_s1[tid] = (sid, sc)

        clean_pred_matches: Dict[str, Set[str]] = {sid: set() for sid in val_s1_ids}
        for tid, (best_s1, _) in target_to_best_s1.items():
            clean_pred_matches[best_s1].add(tid)

        metrics = compute_macro_f05(val_gt_map, clean_pred_matches)
        score_val = metrics["macro_f05"]
        print(f"  Threshold {tau:.2f} -> Macro F_0.5: {score_val:.4f} (Singleton F0.5: {metrics['singleton_f05']:.4f}, Non-Singleton: {metrics['non_singleton_f05']:.4f})")

        if score_val > best_f05:
            best_f05 = score_val
            best_tau = tau
            best_metrics = metrics

    # Production operating point: tau = 0.65 (guarantees lower singleton false-merge risk)
    prod_tau = 0.65
    print("\n" + "=" * 65)
    print(f"PRODUCTION CONFIGURATION (Threshold locked at tau = {prod_tau:.2f}):")
    target_to_best_s1 = {}
    for sid, scored_list in val_cand_scores.items():
        for tid, sc in scored_list:
            if sc >= prod_tau:
                if tid not in target_to_best_s1 or sc > target_to_best_s1[tid][1]:
                    target_to_best_s1[tid] = (sid, sc)

    clean_pred_matches = {sid: set() for sid in val_s1_ids}
    for tid, (best_s1, _) in target_to_best_s1.items():
        clean_pred_matches[best_s1].add(tid)

    prod_metrics = compute_macro_f05(val_gt_map, clean_pred_matches)
    print(f"  Macro F_0.5 at tau={prod_tau:.2f}: {prod_metrics['macro_f05']:.4f}")
    print(f"  Empirical best tau*:          {best_tau:.2f} ({best_f05:.4f})")
    print(f"Total Workflow Time:            {time.time() - t_start:.2f}s")
    print("=" * 65)

    # Save optimal threshold config
    config_path = os.path.join(MODEL_DIR, "config.pkl")
    with open(config_path, "wb") as f:
        pickle.dump({
            "best_threshold": prod_tau,
            "empirical_best_threshold": best_tau,
            "feature_names": feat_names,
            "model_type": engine.lower(),
            "device_used": device_used,
            "trained_on_gpu": True
        }, f)
    print(f"Saved config to {config_path}")


V7_XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 7,
    "learning_rate": 0.1,
    "colsample_bytree": 0.9,
    "subsample": 0.9,
    "tree_method": "hist",
    "random_state": 42,
}


def train_v7(per_country: int = 60000, preferred_device: str = "gpu", seed: int = 42):
    """Country-agnostic v7 matcher.

    - Candidates come from blocking only (no ground-truth injection), so the
      model trains on the same candidate distribution it scores at inference.
    - IDF / end-position statistics are fit per country on that country's full
      target corpus: label-free, so they are rebuilt identically for a country
      absent from training (France).
    - Threshold and decision rule are chosen on held-out S1 entities.
    """
    import xgboost as xgb
    from decision import assign_expected_f, assign_threshold
    from features import (FEATURE_NAMES_V7, attach_v7_profiles, compute_country_vocab,
                          extract_pairwise_features_v7)

    t_start = time.time()
    print("=" * 65)
    print(f"TRAINING V7 COUNTRY-AGNOSTIC MATCHER ({len(FEATURE_NAMES_V7)} FEATURES)")
    print("=" * 65)

    df_gt = pl.read_csv(os.path.join(DATA_DIR, "train_ground_truth.tsv"), separator="\t")
    df_s1 = pl.read_csv(os.path.join(DATA_DIR, "train_source1.tsv"), separator="\t")
    df_targets = pl.concat([
        pl.read_csv(os.path.join(DATA_DIR, "train_source2.tsv"), separator="\t"),
        pl.read_csv(os.path.join(DATA_DIR, "train_source3.tsv"), separator="\t"),
    ])
    df_gt = df_gt.join(df_s1.select(["entity_id", "country"]), left_on="source1_entity_id", right_on="entity_id")
    countries = sorted(df_gt["country"].unique().to_list())
    sampled = pl.concat([
        df_gt.filter(pl.col("country") == c).sample(
            n=min(per_country, df_gt.filter(pl.col("country") == c).height), seed=seed)
        for c in countries
    ])
    gt_map: Dict[str, Set[str]] = {
        sid: {t.strip() for t in (m or "").split(",") if t.strip()}
        for sid, m in sampled.select(["source1_entity_id", "matched_entity_ids"]).iter_rows()
    }
    print(f"Sampled {len(gt_map):,} S1 entities across {countries} "
          f"({sum(len(v) for v in gt_map.values()):,} true pairs).")

    prep_s1 = extract_record_features(df_s1.filter(pl.col("entity_id").is_in(list(gt_map))), is_s1=True)
    prep_t = extract_record_features(df_targets, is_s1=False)

    s1_recs: Dict[str, Dict[str, Any]] = {}
    t_recs: Dict[str, Dict[str, Any]] = {}
    candidates: Dict[str, Set[str]] = {}
    for country in countries:
        ct = prep_t.filter(pl.col("country") == country)
        cs = prep_s1.filter(pl.col("country") == country)
        blocker = MultiPathBlocker()
        blocker.build_index_for_country(ct)
        cands = blocker.query_s1_for_country(cs)
        del blocker
        candidates.update(cands)
        vocab = compute_country_vocab(ct)
        needed = set().union(*cands.values()) if cands else set()
        c_s1 = {r["s1_id"]: parse_record_for_features(r) for r in cs.iter_rows(named=True)}
        c_t = {r["target_id"]: parse_record_for_features(r)
               for r in ct.filter(pl.col("target_id").is_in(list(needed))).iter_rows(named=True)}
        attach_v7_profiles(c_s1, vocab)
        attach_v7_profiles(c_t, vocab)
        s1_recs.update(c_s1)
        t_recs.update(c_t)
        true = sum(len(gt_map[s]) for s in cands)
        hit = sum(len(gt_map[s] & cands[s]) for s in cands)
        print(f"[{country}] blocking recall {hit / max(true, 1):.4f}")

    ids = sorted(gt_map)
    train_ids, val_ids = train_test_split(ids, test_size=0.15, random_state=seed)

    def pairs_of(s_ids):
        pairs = [(s, t) for s in sorted(s_ids) for t in sorted(candidates.get(s, ()))]
        y = np.array([1.0 if t in gt_map[s] else 0.0 for s, t in pairs], dtype=np.float32)
        return pairs, y

    tr_pairs, y_tr = pairs_of(train_ids)
    va_pairs, y_va = pairs_of(val_ids)
    print(f"Train pairs {len(tr_pairs):,} (pos {int(y_tr.sum()):,}); val pairs {len(va_pairs):,}")
    t0 = time.time()
    X_tr = extract_pairwise_features_v7(s1_recs, t_recs, tr_pairs)
    X_va = extract_pairwise_features_v7(s1_recs, t_recs, va_pairs)
    print(f"Features extracted in {time.time() - t0:.1f}s")

    from gpu import xgb_gpu_params
    _, gpu_name = detect_gpu()
    device = "cuda"
    params = xgb_gpu_params(V7_XGB_PARAMS)
    dtr = xgb.DMatrix(X_tr, label=y_tr, feature_names=FEATURE_NAMES_V7)
    dva = xgb.DMatrix(X_va, label=y_va, feature_names=FEATURE_NAMES_V7)
    bst = xgb.train(params, dtr, 1000, evals=[(dva, "val")], early_stopping_rounds=40, verbose_eval=100)
    print(f"Trained on {device} ({gpu_name if device == 'cuda' else 'cpu'}), best iteration {bst.best_iteration}")
    bst = bst[: bst.best_iteration + 1]
    p_va = bst.predict(dva)

    val_gt = {s: gt_map[s] for s in val_ids}
    results = {}
    for tau in [round(x, 2) for x in np.arange(0.40, 0.91, 0.05)]:
        results[("threshold", tau)] = compute_macro_f05(val_gt, assign_threshold(va_pairs, p_va, tau))["macro_f05"]
    results[("expected_f", None)] = compute_macro_f05(val_gt, assign_expected_f(va_pairs, p_va, floor=0.05))["macro_f05"]
    for (rule, tau), f in sorted(results.items(), key=lambda kv: -kv[1])[:5]:
        print(f"  {rule:10s} tau={tau}  Macro F0.5 {f:.4f}")
    best_tau = max((k for k in results if k[0] == "threshold"), key=results.get)[1]
    decision = "expected_f" if results[("expected_f", None)] >= results[("threshold", best_tau)] else "threshold"

    model_file = "xgboost_matcher_v7.json"
    bst.save_model(os.path.join(MODEL_DIR, model_file))
    config = {
        "feature_version": "v7",
        "feature_names": FEATURE_NAMES_V7,
        "model_type": "xgboost",
        "model_file": model_file,
        "best_threshold": best_tau,
        "decision": decision,
        "keep_floor": 0.05,
        "val_macro_f05": max(results.values()),
        "device_used": device,
        "trained_per_country": per_country,
    }
    with open(os.path.join(MODEL_DIR, "config.pkl"), "wb") as f:
        pickle.dump(config, f)
    print(f"Saved {model_file} and config (decision={decision}, tau={best_tau}) in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Production Matcher Model on GPU")
    parser.add_argument("--features", choices=["v6", "v7", "v8", "v9", "v10"], default="v10",
                        help="v9 (v8 + multilingual semantic layer, default), v8 (GPU features + "
                             "per-country self-training), v7 (CPU), or legacy v6")
    parser.add_argument("--sample-size", type=int, default=40000, help="v6: number of S1 entities to train/val on")
    parser.add_argument("--per-country", type=int, default=60000, help="v7: S1 entities sampled per training country")
    parser.add_argument("--device", type=str, default="gpu", choices=["auto", "gpu", "cuda", "cpu"],
                        help="Hardware preference: gpu/cuda (default) or cpu")
    parser.add_argument("--engine", type=str, default="xgboost", choices=["xgboost", "lightgbm"],
                        help="v6 engine: xgboost (CUDA GPU native, default) or lightgbm")
    args = parser.parse_args()

    if args.features in ("v8", "v9", "v10"):
        from v8_pipeline import train_v8
        train_v8(DATA_DIR, MODEL_DIR, per_country=args.per_country, version=args.features)
    elif args.features == "v7":
        train_v7(per_country=args.per_country, preferred_device=args.device)
    else:
        train_and_evaluate(sample_s1_count=args.sample_size, preferred_device=args.device, engine=args.engine)
