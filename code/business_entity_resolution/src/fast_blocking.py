"""
Fast Vectorized Multi-Path Blocking using Polars.
Evaluates recall and runtime on training split.
"""

import os
import sys
import time
import polars as pl

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = r"c:\Users\realr\OneDrive\Desktop\Amazon\6ab10eb3b23ba_student_resource\student_resource\dataset\train"


def prepare_features(df: pl.DataFrame, is_s1: bool = False) -> pl.DataFrame:
    """Vectorized feature and key extraction using Polars."""
    id_col = "source1_entity_id" if "source1_entity_id" in df.columns else "entity_id"
    out_id = "s1_id" if is_s1 else "target_id"

    # Stop words regex for first words
    stop_pattern = r"^(the|and|inc|llc|ltd|corp|co|pvt|hotel|restaurant|store|shop|dr|mr|ms)\b"

    df_clean = df.select([
        pl.col(id_col).alias(out_id),
        pl.col("country"),
        pl.col("business_name").fill_null("").alias("name_raw"),
        pl.col("business_address").fill_null("").alias("addr_raw"),
    ]).with_columns([
        pl.col("name_raw").str.to_lowercase().str.replace_all(r"[^a-z0-9\s]", " ").str.strip_chars().alias("name_clean"),
        pl.col("addr_raw").str.to_lowercase().str.replace_all(r"[^a-z0-9\s]", " ").str.strip_chars().alias("addr_clean"),
    ]).with_columns([
        # Key 1: 8-char compressed name prefix
        pl.col("name_clean").str.replace_all(r"\s+", "").str.slice(0, 8).alias("comp_name_8"),
        # Key 2: 12-char compressed name prefix
        pl.col("name_clean").str.replace_all(r"\s+", "").str.slice(0, 12).alias("comp_name_12"),
        # Key 3: First word (if not stop word and length >= 4)
        pl.col("name_clean").str.split(" ").list.first().alias("raw_first_word"),
        # Key 4: First 2 words of name
        pl.col("name_clean").str.split(" ").list.slice(0, 2).list.join("_").alias("first_2_words"),
        # Key 5: Postal code
        pl.col("addr_raw").str.extract(r"\b(\d{5,6})\b", 1).alias("postal_code"),
        # Key 6: First house / plot / street number
        pl.col("addr_clean").str.extract(r"\b(\d+[a-z]?)\b", 1).alias("house_num"),
    ]).with_columns([
        pl.when(
            pl.col("raw_first_word").str.len_chars() >= 4,
            ~pl.col("raw_first_word").str.contains(stop_pattern)
        ).then(pl.col("raw_first_word")).otherwise(pl.lit(None)).alias("valid_first_word"),
        
        pl.when(
            pl.col("comp_name_8").str.len_chars() >= 5
        ).then(pl.col("comp_name_8")).otherwise(pl.lit(None)).alias("valid_comp_8"),
    ])

    return df_clean


def run_fast_blocking(sample_s1_count: int = 50000):
    print("=" * 60)
    print(f"FAST POLARS BLOCKING BENCHMARK ({sample_s1_count:,} S1 entities)")
    print("=" * 60)
    
    t0 = time.time()
    # 1. Load ground truth
    print("Loading Ground Truth...")
    df_gt = pl.read_csv(os.path.join(DATA_DIR, "train_ground_truth.tsv"), separator="\t")
    if sample_s1_count:
        df_gt = df_gt.head(sample_s1_count)
    
    sample_s1_ids = set(df_gt["source1_entity_id"].to_list())
    
    # Parse true matches into a set of (s1_id, target_id) pairs
    gt_pairs = set()
    for row in df_gt.iter_rows(named=True):
        s1 = row["source1_entity_id"]
        matched = row["matched_entity_ids"]
        if matched:
            for mid in matched.split(","):
                mid = mid.strip()
                if mid:
                    gt_pairs.add((s1, mid))
                    
    total_true_matches = len(gt_pairs)
    print(f"Sample contains {len(df_gt):,} S1 entities and {total_true_matches:,} true match pairs.")

    # 2. Load and prep S1
    print("Loading and preparing Source 1...")
    t_load = time.time()
    df_s1 = pl.read_csv(os.path.join(DATA_DIR, "train_source1.tsv"), separator="\t")
    df_s1 = df_s1.filter(pl.col("entity_id").is_in(list(sample_s1_ids)))
    prep_s1 = prepare_features(df_s1, is_s1=True)
    print(f"Prepared S1 in {time.time() - t_load:.2f}s.")

    # 3. Load and prep S2 & S3
    print("Loading and preparing Source 2 & Source 3 (10+ million records)...")
    t_targets = time.time()
    df_s2 = pl.read_csv(os.path.join(DATA_DIR, "train_source2.tsv"), separator="\t")
    df_s3 = pl.read_csv(os.path.join(DATA_DIR, "train_source3.tsv"), separator="\t")
    df_targets = pl.concat([df_s2, df_s3])
    print(f"Read {len(df_targets):,} target records in {time.time() - t_targets:.2f}s.")

    t_prep = time.time()
    prep_targets = prepare_features(df_targets, is_s1=False)
    print(f"Prepared targets in {time.time() - t_prep:.2f}s.")

    # 4. Perform Multi-Path Hash Joins in Polars
    print("\n--- Executing Multi-Path Hash Joins in Polars ---")
    join_results = []

    # Path 1: (country, valid_comp_8)
    t_j1 = time.time()
    j1 = prep_s1.filter(pl.col("valid_comp_8").is_not_null()).join(
        prep_targets.filter(pl.col("valid_comp_8").is_not_null()),
        on=["country", "valid_comp_8"],
        how="inner"
    ).select(["s1_id", "target_id"])
    print(f"Path 1 (Compressed Name Prefix 8) generated {len(j1):,} pairs in {time.time() - t_j1:.2f}s.")
    join_results.append(j1)

    # Path 2: (country, first_2_words)
    t_j2 = time.time()
    j2 = prep_s1.filter(pl.col("first_2_words").is_not_null()).join(
        prep_targets.filter(pl.col("first_2_words").is_not_null()),
        on=["country", "first_2_words"],
        how="inner"
    ).select(["s1_id", "target_id"])
    print(f"Path 2 (First 2 Words) generated {len(j2):,} pairs in {time.time() - t_j2:.2f}s.")
    join_results.append(j2)

    # Path 3: (country, valid_first_word)
    t_j3 = time.time()
    # Filter first words to only those appearing <= 100 times in targets to avoid hub explosion
    fw_counts = prep_targets.group_by(["country", "valid_first_word"]).len()
    rare_fw_targets = prep_targets.join(
        fw_counts.filter(pl.col("len") <= 80),
        on=["country", "valid_first_word"],
        how="inner"
    )
    j3 = prep_s1.filter(pl.col("valid_first_word").is_not_null()).join(
        rare_fw_targets,
        on=["country", "valid_first_word"],
        how="inner"
    ).select(["s1_id", "target_id"])
    print(f"Path 3 (Rare First Word) generated {len(j3):,} pairs in {time.time() - t_j3:.2f}s.")
    join_results.append(j3)

    # Path 4: (country, postal_code, house_num) -> address anchor
    t_j4 = time.time()
    j4 = prep_s1.filter(pl.col("postal_code").is_not_null() & pl.col("house_num").is_not_null()).join(
        prep_targets.filter(pl.col("postal_code").is_not_null() & pl.col("house_num").is_not_null()),
        on=["country", "postal_code", "house_num"],
        how="inner"
    ).select(["s1_id", "target_id"])
    print(f"Path 4 (Postal Code + House Number) generated {len(j4):,} pairs in {time.time() - t_j4:.2f}s.")
    join_results.append(j4)

    # 5. Union & Deduplicate Candidates
    t_union = time.time()
    candidates_df = pl.concat(join_results).unique(subset=["s1_id", "target_id"])
    print(f"Union & Deduplication completed in {time.time() - t_union:.2f}s.")
    print(f"Total Unique Candidate Pairs: {len(candidates_df):,}")
    print(f"Average candidates per S1: {len(candidates_df) / len(df_gt):.1f}")

    # 6. Evaluate Recall Ceiling
    t_eval = time.time()
    cand_pairs_set = set(zip(candidates_df["s1_id"].to_list(), candidates_df["target_id"].to_list()))
    captured = len(gt_pairs.intersection(cand_pairs_set))
    recall = captured / total_true_matches if total_true_matches > 0 else 0.0

    print("\n" + "=" * 60)
    print("FAST POLARS BLOCKING RESULTS:")
    print(f"Total True Matches: {total_true_matches:,}")
    print(f"Captured True Matches: {captured:,}")
    print(f"Blocking Recall Ceiling: {recall * 100:.2f}%")
    print(f"Total Runtime: {time.time() - t0:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    run_fast_blocking(50000)
