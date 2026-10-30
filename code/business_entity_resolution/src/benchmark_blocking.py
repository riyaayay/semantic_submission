"""
Blocking Benchmark on Training Data.
Measures Recall Ceiling and Candidate Set Size across different blocking keys.
"""

import os
import sys
import time
from collections import defaultdict
from typing import Dict, Set, List
import pandas as pd
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from normalizer import normalize_business_name, normalize_address

DATA_DIR = r"c:\Users\realr\OneDrive\Desktop\Amazon\6ab10eb3b23ba_student_resource\student_resource\dataset\train"


def run_blocking_benchmark(sample_size: int = 50000):
    print(f"Loading training data (evaluating on {sample_size:,} S1 entities)...")
    t0 = time.time()
    
    # Read ground truth
    df_gt = pd.read_csv(os.path.join(DATA_DIR, "train_ground_truth.tsv"), sep="\t").fillna("")
    sample_gt = df_gt.head(sample_size)
    
    gt_map: Dict[str, Set[str]] = {}
    target_s2_s3 = set()
    for _, row in sample_gt.iterrows():
        s1_id = row["source1_entity_id"]
        m = row["matched_entity_ids"].strip()
        matches = set(x.strip() for x in m.split(",") if x.strip()) if m else set()
        gt_map[s1_id] = matches
        target_s2_s3.update(matches)

    sample_s1_ids = set(sample_gt["source1_entity_id"])
    total_true_matches = sum(len(v) for v in gt_map.values())
    print(f"Sample contains {len(sample_s1_ids):,} S1 entities and {total_true_matches:,} true match links.")
    
    # Read S1 sample
    df_s1 = pd.read_csv(os.path.join(DATA_DIR, "train_source1.tsv"), sep="\t")
    df_s1 = df_s1[df_s1["entity_id"].isin(sample_s1_ids)].copy()
    
    # Read S2 and S3 (only for the countries present in sample_s1, or full)
    countries_in_sample = set(df_s1["country"].unique())
    print(f"Countries in sample: {countries_in_sample}")
    
    # Read S2 and S3
    print("Reading Source 2 and Source 3...")
    df_s2 = pd.read_csv(os.path.join(DATA_DIR, "train_source2.tsv"), sep="\t")
    df_s2 = df_s2[df_s2["country"].isin(countries_in_sample)].copy()
    
    df_s3 = pd.read_csv(os.path.join(DATA_DIR, "train_source3.tsv"), sep="\t")
    df_s3 = df_s3[df_s3["country"].isin(countries_in_sample)].copy()
    
    print(f"Loaded: S1={len(df_s1):,}, S2={len(df_s2):,}, S3={len(df_s3):,} in {time.time() - t0:.2f}s")
    
    # Normalize S1
    print("Normalizing S1...")
    s1_norm = {}
    for _, r in df_s1.iterrows():
        eid = r["entity_id"]
        cn, cns, _ = normalize_business_name(r["business_name"])
        ca, pc, num = normalize_address(r["business_address"], r["country"])
        s1_norm[eid] = {
            "name": cn, "name_ns": cns, "addr": ca, "pc": pc, "num": num,
            "country": r["country"]
        }

    # Normalize S2/S3 and build index
    print("Normalizing S2/S3 and building blocking indexes...")
    # Index keys:
    # 1. (country, name_prefix_12): first 12 chars of name without suffix
    # 2. (country, postal_code, first_num): if pc and num exist
    # 3. (country, first_2_name_words): if >= 2 words
    
    idx_prefix = defaultdict(list)
    idx_pc_num = defaultdict(list)
    idx_words = defaultdict(list)
    
    for df, prefix in [(df_s2, "S2"), (df_s3, "S3")]:
        for _, r in df.iterrows():
            eid = r["entity_id"]
            country = r["country"]
            cn, cns, _ = normalize_business_name(r["business_name"])
            ca, pc, num = normalize_address(r["business_address"], country)
            
            # Key 1: name prefix
            if len(cns) >= 4:
                k1 = (country, cns[:12])
                idx_prefix[k1].append(eid)
                
            # Key 2: postal code + house number
            if pc and num:
                k2 = (country, pc, num[0])
                idx_pc_num[k2].append(eid)
                
            # Key 3: first 2 words
            words = cns.split()
            if len(words) >= 2:
                k3 = (country, words[0], words[1])
                idx_words[k3].append(eid)

    print("Querying index for S1 candidates...")
    candidates: Dict[str, Set[str]] = defaultdict(set)
    for s1_id, d in s1_norm.items():
        country = d["country"]
        cns = d["name_ns"]
        pc = d["pc"]
        num = d["num"]
        
        cands = set()
        # Key 1: name prefix
        if len(cns) >= 4:
            k1 = (country, cns[:12])
            if k1 in idx_prefix:
                cands.update(idx_prefix[k1][:100])
                
        # Key 2: postal code + number
        if pc and num:
            k2 = (country, pc, num[0])
            if k2 in idx_pc_num:
                cands.update(idx_pc_num[k2][:100])
                
        # Key 3: first 2 words
        words = cns.split()
        if len(words) >= 2:
            k3 = (country, words[0], words[1])
            if k3 in idx_words:
                cands.update(idx_words[k3][:100])
                
        candidates[s1_id] = cands

    # Evaluate blocking recall
    total_candidates = sum(len(c) for c in candidates.values())
    captured_matches = 0
    for s1_id, true_set in gt_map.items():
        cands = candidates.get(s1_id, set())
        captured_matches += len(true_set.intersection(cands))

    recall_ceiling = captured_matches / total_true_matches if total_true_matches > 0 else 0.0
    avg_cands = total_candidates / len(sample_s1_ids)

    print("\n" + "=" * 50)
    print("BLOCKING RESULTS (INITIAL KEYS):")
    print(f"Total S1 entities: {len(sample_s1_ids):,}")
    print(f"Average candidates per S1 entity: {avg_cands:.1f}")
    print(f"Total candidates: {total_candidates:,}")
    print(f"True matches in GT: {total_true_matches:,}")
    print(f"Captured matches: {captured_matches:,}")
    print(f"Blocking Recall Ceiling: {recall_ceiling * 100:.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    run_blocking_benchmark(30000)
