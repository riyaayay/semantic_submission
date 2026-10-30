"""
Enhanced Multi-Path Blocking Strategy.
Combines Name, Address, Acronym, and Compressed Domain Blocking.
"""

import os
import sys
import re
import time
from collections import defaultdict
from typing import Dict, Set, List, Tuple
import pandas as pd
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "code", "business_entity_resolution", "src"))
from normalizer import normalize_business_name, normalize_address

DATA_DIR = r"c:\Users\realr\OneDrive\Desktop\Amazon\6ab10eb3b23ba_student_resource\student_resource\dataset\train"

# Common generic stop words to ignore in single-token blocking
STOP_WORDS = {
    "the", "and", "inc", "incorporated", "corp", "corporation", "ltd", "limited",
    "co", "company", "llc", "llp", "pvt", "private", "services", "enterprises",
    "solutions", "group", "holdings", "management", "international", "associates",
    "road", "rd", "street", "st", "avenue", "ave", "lane", "ln", "near", "opp",
    "floor", "fl", "unit", "apt", "plot", "sector", "building", "bldg", "shop", "no"
}

def get_acronym(words: List[str]) -> str:
    """Generates acronym for words, e.g. ['zander', 'blue'] -> 'zb'."""
    if len(words) >= 2:
        return "".join(w[0] for w in words if w and w[0].isalnum())[:5]
    return ""

def get_salient_tokens(text: str, min_len: int = 4) -> List[str]:
    """Returns tokens of length >= min_len excluding generic stop words."""
    return [w for w in text.split() if len(w) >= min_len and w not in STOP_WORDS]


def test_enhanced_blocking(sample_size: int = 30000):
    print(f"--- Running Enhanced Blocking Benchmark ({sample_size:,} S1 entities) ---")
    t0 = time.time()
    
    # Read ground truth
    df_gt = pd.read_csv(os.path.join(DATA_DIR, "train_ground_truth.tsv"), sep="\t").fillna("")
    sample_gt = df_gt.head(sample_size)
    
    gt_map: Dict[str, Set[str]] = {}
    for _, row in sample_gt.iterrows():
        s1_id = row["source1_entity_id"]
        m = row["matched_entity_ids"].strip()
        matches = set(x.strip() for x in m.split(",") if x.strip()) if m else set()
        gt_map[s1_id] = matches

    sample_s1_ids = set(sample_gt["source1_entity_id"])
    total_true_matches = sum(len(v) for v in gt_map.values())
    
    # Read S1, S2, S3
    df_s1 = pd.read_csv(os.path.join(DATA_DIR, "train_source1.tsv"), sep="\t")
    df_s1 = df_s1[df_s1["entity_id"].isin(sample_s1_ids)].copy()
    countries = set(df_s1["country"].unique())
    
    df_s2 = pd.read_csv(os.path.join(DATA_DIR, "train_source2.tsv"), sep="\t")
    df_s2 = df_s2[df_s2["country"].isin(countries)].copy()
    
    df_s3 = pd.read_csv(os.path.join(DATA_DIR, "train_source3.tsv"), sep="\t")
    df_s3 = df_s3[df_s3["country"].isin(countries)].copy()
    
    print(f"Data loaded in {time.time() - t0:.2f}s. Normalizing and indexing...")
    
    # Pre-extract record representations
    def extract_record_keys(name: str, address: str, country: str):
        cn, cns, _ = normalize_business_name(name)
        ca, pc, nums = normalize_address(address, country)
        
        name_words = cns.split()
        compressed_name = re.sub(r"[^a-z0-9]", "", cns)
        acronym = get_acronym(name_words)
        
        salient_name_tokens = get_salient_tokens(cns)
        salient_addr_tokens = get_salient_tokens(ca)
        
        keys = []
        # Key 1: Exact first 2 words of name
        if len(name_words) >= 2:
            keys.append(("w2", country, name_words[0], name_words[1]))
            
        # Key 2: Compressed name prefix (8 chars)
        if len(compressed_name) >= 6:
            keys.append(("cpref", country, compressed_name[:8]))
            
        # Key 3: First salient name token (if rare/long >= 5)
        if salient_name_tokens and len(salient_name_tokens[0]) >= 5:
            keys.append(("fw", country, salient_name_tokens[0]))
            
        # Key 4: Address Number + Salient Address Token
        if nums and salient_addr_tokens:
            for st in salient_addr_tokens[:2]:
                keys.append(("num_tok", country, nums[0], st))
                
        # Key 5: Postal code + Address Number
        if pc and nums:
            keys.append(("pc_num", country, pc, nums[0]))
            
        # Key 6: Acronym + Salient Address Token
        if acronym and len(acronym) >= 2 and salient_addr_tokens:
            keys.append(("acr_tok", country, acronym, salient_addr_tokens[0]))
            
        return keys, compressed_name

    # Build index for S2 and S3
    t_idx = time.time()
    index = defaultdict(list)
    
    for df in [df_s2, df_s3]:
        for _, r in df.iterrows():
            eid = r["entity_id"]
            country = r["country"]
            keys, _ = extract_record_keys(r["business_name"], r["business_address"], country)
            for k in keys:
                index[k].append(eid)
                
    print(f"Index built with {len(index):,} distinct keys in {time.time() - t_idx:.2f}s.")
    
    # Query index for S1 candidates
    print("Generating candidates for S1...")
    t_query = time.time()
    candidates: Dict[str, Set[str]] = defaultdict(set)
    MAX_PER_KEY = 60 # cap to avoid explosion on overly common keys
    
    for _, r in df_s1.iterrows():
        s1_id = r["entity_id"]
        country = r["country"]
        keys, _ = extract_record_keys(r["business_name"], r["business_address"], country)
        
        cands = set()
        for k in keys:
            if k in index:
                bucket = index[k]
                if len(bucket) <= MAX_PER_KEY:
                    cands.update(bucket)
                else:
                    cands.update(bucket[:MAX_PER_KEY])
        candidates[s1_id] = cands
        
    print(f"Candidate generation completed in {time.time() - t_query:.2f}s.")
    
    # Evaluate
    total_candidates = sum(len(c) for c in candidates.values())
    captured_matches = 0
    for s1_id, true_set in gt_map.items():
        cands = candidates.get(s1_id, set())
        captured_matches += len(true_set.intersection(cands))
        
    recall_ceiling = captured_matches / total_true_matches if total_true_matches > 0 else 0.0
    avg_cands = total_candidates / len(sample_s1_ids)

    print("\n" + "=" * 55)
    print("ENHANCED MULTI-PATH BLOCKING RESULTS:")
    print(f"Total S1 entities: {len(sample_s1_ids):,}")
    print(f"Average candidates per S1 entity: {avg_cands:.1f}")
    print(f"Total candidates generated: {total_candidates:,}")
    print(f"True matches in GT: {total_true_matches:,}")
    print(f"Captured matches: {captured_matches:,}")
    print(f"Recall Ceiling: {recall_ceiling * 100:.2f}% (Previously 57.66%)")
    print("=" * 55)


if __name__ == "__main__":
    test_enhanced_blocking(30000)
