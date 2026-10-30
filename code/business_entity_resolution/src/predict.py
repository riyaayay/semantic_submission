"""
Test Inference Pipeline for Amazon ML Challenge 2026.
Generates:
1. output/matching_results.tsv (scored on leaderboard)
2. output/candidate_pairs.tsv (blocking candidates)

Features:
- Natively GPU-accelerated via CUDA (XGBoost GPU tree_method='hist').
- High-speed precomputed TF-IDF profile dot products (10x faster feature extraction).
- Country-level checkpointing (instant resume if interrupted).
- Real-time batch progress logging.
"""

import os
import sys
import time
import pickle
import hashlib
import tempfile
from typing import Dict, Set, List, Tuple, Any, Optional
import numpy as np
import polars as pl
import lightgbm as lgb

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    getattr(sys.stdout, "reconfigure")(encoding="utf-8", line_buffering=True)

from blocking import MultiPathBlocker, extract_record_features
from decision import assign_expected_f, assign_threshold
from gpu import require_cuda, xgb_predict_gpu
from v8_pipeline import (features_into, pair_arrays, self_train, predict as v8_predict,
                         semantic_candidates, semantic_features_into, judge_scores)
from judge import load_judge, topm_groups, tt_edge_features
from gpu_features import build_country_vocab
import torch
from features import (
    extract_pairwise_features,
    extract_pairwise_features_v7,
    attach_tfidf_profiles,
    attach_v7_profiles,
    compute_country_vocab,
    parse_record_for_features,
    load_tfidf_vocab,
    FEATURE_NAMES,
    FEATURE_NAMES_V7,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(SCRIPT_DIR)
MODEL_DIR = os.path.join(PKG_ROOT, "models")

DEFAULT_DATA_DIR = os.path.abspath(os.path.join(PKG_ROOT, "..", "..", "6ab10eb3b23ba_student_resource", "student_resource", "dataset", "test"))
DATA_DIR = os.environ.get("TEST_DATA_DIR", DEFAULT_DATA_DIR)
if not os.path.exists(DATA_DIR):
    DATA_DIR = r"c:\Users\realr\OneDrive\Desktop\Amazon\6ab10eb3b23ba_student_resource\student_resource\dataset\test"

DEFAULT_OUTPUT_DIR = os.path.abspath(os.path.join(PKG_ROOT, "..", "..", "output"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", DEFAULT_OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def compute_run_fingerprint(paths: List[str], tau: float) -> str:
    """Hashes everything a country checkpoint depends on, so a retrained model,
    new threshold, or changed blocking/feature code invalidates old checkpoints."""
    h = hashlib.sha256(f"tau={tau}".encode())
    for p in paths:
        h.update(os.path.basename(p).encode())
        if os.path.exists(p):
            with open(p, "rb") as f:
                h.update(f.read())
    return h.hexdigest()


def load_checkpoint(path: str, fingerprint: str) -> Optional[Dict[str, List[str]]]:
    """Returns the checkpointed data, or None if missing or built by a different run."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        ckpt = pickle.load(f)
    if not isinstance(ckpt, dict) or ckpt.get("fingerprint") != fingerprint:
        return None
    return ckpt["data"]


def run_inference(batch_feature_size: int = 500000, data_dir: str = None, output_dir: str = None):
    t_start = time.time()
    data_dir = data_dir or DATA_DIR
    output_dir = output_dir or OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 65, flush=True)
    print("RUNNING HIGH-THROUGHPUT GPU INFERENCE ON TEST SET", flush=True)
    print(f"Data Directory:   {data_dir}", flush=True)
    print(f"Output Directory: {output_dir}", flush=True)
    print("=" * 65, flush=True)

    # 1. Load trained model, config, and zero-leakage TF-IDF vocabulary
    # MODEL_CONFIG selects a saved config (e.g. config_v9.pkl / config_v10.pkl); default = active config.pkl
    config_path = os.path.join(MODEL_DIR, os.environ.get("MODEL_CONFIG", "config.pkl"))
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found in {MODEL_DIR}. Please run train_matcher.py first.")

    with open(config_path, "rb") as f:
        config = pickle.load(f)
    tau = config.get("best_threshold", 0.65)
    model_type = config.get("model_type", "xgboost")
    feature_version = config.get("feature_version", "v6")
    feature_names = config.get("feature_names", FEATURE_NAMES_V7 if feature_version == "v7" else FEATURE_NAMES)
    v8_source = None
    is_gpu_pipeline = feature_version in ("v8", "v9", "v10")
    judge = judge_ck = None
    if feature_version == "v10":
        judge, judge_ck = load_judge(os.path.join(MODEL_DIR, config["judge_file"]))
        print(f"Loaded v10 set-level judge ({config['judge_file']}), threshold {tau}", flush=True)
    src_path = os.path.join(MODEL_DIR, f"{feature_version}_source.pt")
    if is_gpu_pipeline and config.get("self_train", False):
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"{src_path} missing: run train_matcher.py --features {feature_version} to regenerate it.")
        v8_source = torch.load(src_path, weights_only=False)
        print(f"Self-training enabled: {len(v8_source['y']):,} labelled source pairs loaded.", flush=True)
    decision = config.get("decision", "threshold")
    # expected-F needs low-probability candidates too; threshold only needs >= tau
    keep_floor = config.get("keep_floor", 0.05) if decision == "expected_f" else tau
    print(f"Feature set {feature_version} ({len(feature_names)} features), decision rule: {decision}", flush=True)

    vocab_path = os.path.join(MODEL_DIR, "tfidf_vocab.pkl")
    tfidf_vocab = load_tfidf_vocab(vocab_path) if feature_version == "v6" else None
    if feature_version == "v7":
        print("Using per-country label-free vocabularies built from each country's targets.", flush=True)
    elif tfidf_vocab:
        n_name = len(tfidf_vocab.get("name_idf", {}))
        n_addr = len(tfidf_vocab.get("addr_idf", {}))
        print(f"Loaded train-fitted zero-leakage TF-IDF vocabulary ({n_name:,} name terms, {n_addr:,} address terms).", flush=True)
    else:
        print("Warning: tfidf_vocab.pkl not found in models directory; TF-IDF cosine will default to 0.0.", flush=True)

    if model_type == "xgboost":
        import xgboost as xgb
        model_path = os.path.join(MODEL_DIR, config.get("model_file", "xgboost_matcher.json"))
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Please run train_matcher.py first.")
        model = xgb.Booster()
        model.load_model(model_path)
        require_cuda()
        model.set_param({"device": "cuda"})
        print(f"Loaded XGBoost model on GPU (CUDA). Decision threshold tau = {tau:.2f}", flush=True)
    else:
        raise RuntimeError("LightGBM models are CPU-only in this environment; retrain with XGBoost (CUDA).")

    run_fingerprint = compute_run_fingerprint(
        [model_path, config_path, vocab_path] +
        [os.path.join(SCRIPT_DIR, f) for f in ("blocking.py", "features.py", "translit.py", "decision.py",
                                               "gpu_features.py", "gpu_strings.py", "v8_pipeline.py", "semantic.py", "judge.py")] +
        [src_path] + ([os.path.join(MODEL_DIR, config["judge_file"])] if config.get("judge_file") else []),
        tau,
    )

    # 2. Read Test Source 1 to know required entities and order
    s1_path = os.path.join(data_dir, "test_source1.tsv")
    print(f"Reading test reference: {s1_path}", flush=True)
    df_s1_full = pl.read_csv(s1_path, separator="\t")
    s1_all_ids = df_s1_full["entity_id"].to_list()
    print(f"Total test S1 entities: {len(s1_all_ids):,}", flush=True)
    print(f"Country breakdown: {df_s1_full['country'].value_counts().to_dicts()}", flush=True)

    # 3. Read Test Source 2 and Source 3
    print("Reading test Source 2 and Source 3...", flush=True)
    t_load = time.time()
    df_s2 = pl.read_csv(os.path.join(data_dir, "test_source2.tsv"), separator="\t")
    df_s3 = pl.read_csv(os.path.join(data_dir, "test_source3.tsv"), separator="\t")
    df_targets_full = pl.concat([df_s2, df_s3])
    print(f"Loaded {len(df_targets_full):,} target records in {time.time() - t_load:.2f}s.", flush=True)

    final_matches: Dict[str, List[str]] = {sid: [] for sid in s1_all_ids}
    final_candidates: Dict[str, List[str]] = {sid: [] for sid in s1_all_ids}

    # 4. Process Country by Country (France, US, India)
    countries = df_s1_full["country"].unique().to_list()
    countries.sort(key=lambda c: len(df_s1_full.filter(pl.col("country") == c)))
    only = [c.strip() for c in os.environ.get("PREDICT_COUNTRIES", "").split(",") if c.strip()]
    if only:
        countries = [c for c in countries if c in only]
        print(f"Restricting to countries: {countries}", flush=True)
    max_per_s1 = int(os.environ.get("MAX_CANDIDATES_PER_S1", "0")) or None

    for country in countries:
        print("\n" + "-" * 55, flush=True)
        print(f"PROCESSING COUNTRY: {country.upper()}", flush=True)
        print("-" * 55, flush=True)
        t_country = time.time()

        # Check for existing checkpoint
        country_ckpt_match = os.path.join(output_dir, f"matches_{country}.pkl")
        country_ckpt_cand = os.path.join(output_dir, f"cands_{country}.pkl")
        c_matches = load_checkpoint(country_ckpt_match, run_fingerprint)
        c_cands = load_checkpoint(country_ckpt_cand, run_fingerprint)
        if c_matches is not None and c_cands is not None:
            print(f"  [Resume] Found checkpoint for {country} from this model/config. Loading...", flush=True)
            final_matches.update(c_matches)
            final_candidates.update(c_cands)
            match_cnt = sum(len(v) for v in c_matches.values())
            print(f"  [Resume] Loaded {len(c_matches):,} entities ({match_cnt:,} matches) in {time.time() - t_country:.2f}s.", flush=True)
            continue
        if os.path.exists(country_ckpt_match) or os.path.exists(country_ckpt_cand):
            print(f"  [Resume] Ignoring stale checkpoint for {country} (model, config or code changed).", flush=True)

        df_s1_country = df_s1_full.filter(pl.col("country") == country)
        df_targets_country = df_targets_full.filter(pl.col("country") == country)
        print(f"  {country}: {len(df_s1_country):,} S1 entities vs {len(df_targets_country):,} target records", flush=True)

        # Feature preparation
        prep_s1 = extract_record_features(df_s1_country, is_s1=True)
        prep_targets = extract_record_features(df_targets_country, is_s1=False)

        # Multi-Path Blocking
        blocker = MultiPathBlocker(max_per_s1=max_per_s1) if max_per_s1 else MultiPathBlocker()
        blocker.build_index_for_country(prep_targets)
        cands_map = blocker.query_s1_for_country(prep_s1)
        del blocker   # the inverted index is several GB for large countries

        if is_gpu_pipeline:
            sem = None
            if feature_version in ("v9", "v10"):
                # Semantic layer: add top-k multilingual name neighbours to the candidates
                sem_cands, Es, Et = semantic_candidates(prep_s1, prep_targets)
                for sid, ts in sem_cands.items():
                    cands_map.setdefault(sid, set()).update(ts)
                del sem_cands
                sem = (Es, Et)
            # Integer pair arrays + disk-backed feature matrix keep host RAM bounded
            # (a country can have 40M+ candidate pairs).
            s_ids, t_ids, ia, ib = pair_arrays(cands_map)
            for sid in s_ids:
                final_candidates[sid] = sorted(cands_map[sid])
            del cands_map
            n_pairs = len(ia)
            print(f"  Total candidate pairs for {country}: {n_pairs:,}", flush=True)
            if n_pairs == 0:
                print(f"  No candidate pairs generated for {country}. All singletons.", flush=True)
                continue
            # Local temp dir, not output/: output/ may live in a synced folder (OneDrive),
            # which makes multi-GB writes crawl.
            feat_dir = os.environ.get("V8_FEATURE_DIR", tempfile.gettempdir())
            mm_path = os.path.join(feat_dir, f"{feature_version}_features_{country}.f32")
            X_country = np.memmap(mm_path, dtype=np.float32, mode="w+", shape=(n_pairs, len(feature_names)))
            features_into(prep_s1, prep_targets, s_ids, t_ids, ia, ib, X_country)
            if sem is not None:
                semantic_features_into(prep_s1, prep_targets, s_ids, t_ids, ia, ib, X_country, *sem)
            scores_all = v8_predict(model, X_country)
            if v8_source is not None:
                # Label-free adaptation to this country's own labelling conventions
                country_model = self_train(model, v8_source, X_country, scores_all)
                scores_all = v8_predict(country_model, X_country)
                # Keep the exact model behind this country's predictions
                country_model.save_model(os.path.join(output_dir, f"xgboost_matcher_{feature_version}_{country}_selftrained.json"))
            if judge is not None:
                # v10: Jev-style set judge over each S1's top-m candidates + graph-consistency edges
                t_judge = time.time()
                g = topm_groups(ia.astype(np.int64), X_country, feature_names)
                pt_list = prep_targets["target_id"].to_list()
                pt_pos = {tid: i for i, tid in enumerate(pt_list)}
                tpos = np.fromiter((pt_pos[x] for x in t_ids), np.int64, len(t_ids))
                gt_idx = np.where(g >= 0, tpos[ib[np.where(g >= 0, g, 0)]], -1)
                voc_j = build_country_vocab(prep_targets)
                edges_fn = lambda lo, hi: tt_edge_features(prep_targets, voc_j, gt_idx[lo:hi], pt_list, sem[1])  # noqa: E731
                scores_all = judge_scores(judge, judge_ck, X_country, scores_all, g, edges_fn)
                print(f"  Judge applied to {len(g):,} S1 sets in {time.time() - t_judge:.0f}s", flush=True)
                del g, gt_idx, pt_pos
            del sem
            keep = np.nonzero(scores_all >= keep_floor)[0]
            kept_pairs = [(s_ids[ia[i]], t_ids[ib[i]]) for i in keep]
            kept_scores = scores_all[keep].tolist()
            print(f"  Scored {n_pairs:,} pairs ({len(kept_pairs):,} kept).", flush=True)
            del X_country, scores_all, ia, ib
            os.remove(mm_path)
        else:
            country_candidate_pairs: List[Tuple[str, str]] = []
            for sid, c_set in cands_map.items():
                c_list = sorted(list(c_set))
                final_candidates[sid] = c_list
                for tid in c_list:
                    country_candidate_pairs.append((sid, tid))

            print(f"  Total candidate pairs for {country}: {len(country_candidate_pairs):,}", flush=True)

            if not country_candidate_pairs:
                print(f"  No candidate pairs generated for {country}. All singletons.", flush=True)
                continue

            # Build feature lookup dictionaries with precomputed TF-IDF profiles
            print("  Building feature lookup dictionaries...", flush=True)
            t_dict_prep = time.time()
            needed_tids = {tid for _, tid in country_candidate_pairs}
            s1_dict = {
                row["s1_id"]: parse_record_for_features(row)
                for row in prep_s1.iter_rows(named=True)
            }
            target_dict = {
                row["target_id"]: parse_record_for_features(row)
                for row in prep_targets.filter(pl.col("target_id").is_in(list(needed_tids))).iter_rows(named=True)
            }
            if feature_version == "v7":
                # Label-free statistics from this country's own targets: works the
                # same for a country the model never saw in training.
                country_vocab = compute_country_vocab(prep_targets)
                attach_v7_profiles(s1_dict, country_vocab)
                attach_v7_profiles(target_dict, country_vocab)
            else:
                attach_tfidf_profiles(s1_dict, tfidf_vocab)
                attach_tfidf_profiles(target_dict, tfidf_vocab)
            print(f"  Profiles built in {time.time() - t_dict_prep:.2f}s.", flush=True)

            # Batches end on S1 boundaries so v7 group features see every candidate
            # of an entity (pairs are grouped by S1 above).
            batches: List[List[Tuple[str, str]]] = [[]]
            for i, pair in enumerate(country_candidate_pairs):
                if len(batches[-1]) >= batch_feature_size and pair[0] != country_candidate_pairs[i - 1][0]:
                    batches.append([])
                batches[-1].append(pair)
            total_pairs = len(country_candidate_pairs)
            print(f"  Extracting features and scoring {total_pairs:,} pairs in {len(batches)} batches...", flush=True)
            t_score = time.time()
            kept_pairs: List[Tuple[str, str]] = []
            kept_scores: List[float] = []
            done = 0

            for b_idx, batch_pairs in enumerate(batches):
                t_b0 = time.time()
                if feature_version == "v7":
                    X_batch = extract_pairwise_features_v7(s1_dict, target_dict, batch_pairs)
                else:
                    X_batch, _ = extract_pairwise_features(s1_dict, target_dict, batch_pairs, tfidf_vocab=tfidf_vocab)
                preds_batch = xgb_predict_gpu(model, X_batch).cpu().numpy()

                for pair, score in zip(batch_pairs, preds_batch):
                    if score >= keep_floor:
                        kept_pairs.append(pair)
                        kept_scores.append(float(score))

                done += len(batch_pairs)
                print(f"    [{country}] Batch {b_idx+1}/{len(batches)} ({done / total_pairs * 100:.1f}%): {len(batch_pairs):,} pairs scored in {time.time() - t_b0:.2f}s", flush=True)

            print(f"  Scoring completed in {time.time() - t_score:.2f}s ({len(kept_pairs):,} pairs kept).", flush=True)

        # Each target joins at most its best-scoring S1; then per-S1 selection
        scores_arr = np.asarray(kept_scores, dtype=np.float64)
        if decision == "expected_f":
            chosen = assign_expected_f(kept_pairs, scores_arr, floor=keep_floor)
        else:
            chosen = assign_threshold(kept_pairs, scores_arr, tau)

        match_count = 0
        country_s1_ids = df_s1_country["entity_id"].to_list()
        for sid, tids in chosen.items():
            final_matches[sid].extend(sorted(tids))
            match_count += len(tids)

        print(f"  Final matches for {country}: {match_count:,} in {time.time() - t_country:.2f}s.", flush=True)

        # Checkpoint country results to disk
        with open(country_ckpt_match, "wb") as f:
            pickle.dump({"fingerprint": run_fingerprint,
                         "data": {sid: final_matches[sid] for sid in country_s1_ids}}, f)
        with open(country_ckpt_cand, "wb") as f:
            pickle.dump({"fingerprint": run_fingerprint,
                         "data": {sid: final_candidates[sid] for sid in country_s1_ids}}, f)
        print(f"  Checkpointed {country} to {country_ckpt_match}", flush=True)

    # 5. Write Final Output Deliverables
    print("\n--- Writing Final Deliverable TSV Files ---", flush=True)
    match_file = os.path.join(output_dir, "matching_results.tsv")
    cand_file = os.path.join(output_dir, "candidate_pairs.tsv")

    print(f"Writing {match_file}...", flush=True)
    with open(match_file, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in s1_all_ids:
            m_list = final_matches.get(sid, [])
            f.write(f"{sid}\t{','.join(m_list)}\n")

    print(f"Writing {cand_file}...", flush=True)
    with open(cand_file, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in s1_all_ids:
            c_list = final_candidates.get(sid, [])
            f.write(f"{sid}\t{','.join(c_list)}\n")

    total_matches = sum(len(v) for v in final_matches.values())
    total_singletons = sum(1 for v in final_matches.values() if not v)
    total_candidates = sum(len(v) for v in final_candidates.values())

    print("\n" + "=" * 65, flush=True)
    print("GPU INFERENCE SUMMARY:", flush=True)
    print(f"  Total S1 Entities Processed: {len(s1_all_ids):,}", flush=True)
    print(f"  Total Candidate Pairs:      {total_candidates:,}", flush=True)
    print(f"  Total Final Matches:        {total_matches:,}", flush=True)
    print(f"  Predicted Singletons:       {total_singletons:,} ({total_singletons / len(s1_all_ids) * 100:.2f}%)", flush=True)
    print(f"  Total Pipeline Time:        {time.time() - t_start:.2f}s", flush=True)
    print("=" * 65, flush=True)


if __name__ == "__main__":
    run_inference()
