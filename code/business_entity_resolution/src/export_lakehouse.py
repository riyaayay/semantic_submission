"""
src/export_lakehouse.py
=======================
Transforms existing entity resolution outputs into a partitioned Parquet
Kimball Star Schema ready for Amazon S3 + AWS Glue + Amazon Athena.

Zero model re-training required -- reads only:
  - dataset/test/test_source{1,2,3}.tsv  (raw test records)
  - output/matching_results.tsv          (1.73M S1 rows -> matched target IDs)
  - output/candidate_pairs.tsv           (1.73M S1 rows -> candidate target IDs)

Writes Snappy-compressed, country-partitioned Parquet under:
  output/lakehouse_gold/
    dim_canonical_entity/   (S1 reference dimension)
    dim_target_record/      (S2+S3 vendor dimension)
    fact_entity_summary/    (per-S1 blocking & match telemetry)
    fact_resolution_links/  (granular 1-to-many bridge table)

Usage:
    python src/export_lakehouse.py

Environment overrides (optional):
    TEST_DATA_DIR  -- path to folder containing test_source{1,2,3}.tsv
    OUTPUT_DIR     -- path to folder containing matching_results.tsv & candidate_pairs.tsv
                     (lakehouse_gold/ will be written here too)
"""

import os
import sys
import time
import polars as pl

# Force UTF-8 output on Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# -- Path resolution ---------------------------------------------------------
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SRC_DIR, "..", "..", ".."))

TEST_DATA_DIR = os.environ.get(
    "TEST_DATA_DIR",
    os.path.join(_REPO_ROOT, "dataset", "test"),
)
OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR",
    os.path.join(_REPO_ROOT, "output"),
)
LAKE_DIR = os.path.join(OUTPUT_DIR, "lakehouse_gold")


# -- Feature extraction (mirrors blocking.py but avoids the CUDA import) ----

def _clean_name(s: pl.Expr) -> pl.Expr:
    """Lowercase, strip punctuation, collapse whitespace."""
    return (
        s.str.to_lowercase()
        .str.replace_all(r"[^\w\s]", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )


_SKEL_RULES = [
    (r"chh", "ch"), (r"ph", "f"), (r"bh", "b"), (r"kh", "k"),
    (r"gh", "g"), (r"jh", "j"), (r"th", "t"), (r"dh", "d"),
    (r"sh", "s"), (r"ck", "k"), (r"q", "k"), (r"x", "ks"),
    (r"w", "v"), (r"z", "s"),
]


def _phonetic_skeleton(expr: pl.Expr) -> pl.Expr:
    """Consonant skeleton key (language-agnostic sound-alike key)."""
    for pat, rep in _SKEL_RULES:
        expr = expr.str.replace_all(pat, rep)
    # Strip interior vowels after a consonant (no lookbehind: use capture group)
    expr = expr.str.replace_all(r"([bcdfghjklmnprstv])[aeiouy]+", "${1}")
    return expr


def _extract_s1_features(df: pl.DataFrame) -> pl.DataFrame:
    """Extract blocking keys + metadata for S1 entities."""
    return df.select([
        pl.col("entity_id"),
        pl.col("country"),
        pl.col("business_name").fill_null("").alias("name_raw"),
        pl.col("business_address").fill_null("").alias("addr_raw"),
    ]).with_columns([
        _clean_name(pl.col("name_raw")).alias("name_clean"),
    ]).with_columns([
        _phonetic_skeleton(pl.col("name_clean")).alias("name_skel"),
        pl.col("addr_raw")
          .str.extract(r"\b(\d{5,6})\b", 1)
          .alias("postal_code"),
        _clean_name(pl.col("addr_raw"))
          .str.extract(r"\b0*(\d+[a-z]?)\b", 1)
          .alias("house_num"),
    ])


def _extract_target_features(df: pl.DataFrame, source_system: str) -> pl.DataFrame:
    """Extract metadata for S2/S3 target records."""
    return df.select([
        pl.col("entity_id").alias("target_id"),
        pl.lit(source_system).alias("source_system"),
        pl.col("country"),
        pl.col("business_name").fill_null("").alias("name_raw"),
        pl.col("business_address").fill_null("").alias("addr_raw"),
    ]).with_columns([
        pl.col("addr_raw")
          .str.extract(r"\b(\d{5,6})\b", 1)
          .alias("postal_code"),
    ])


# -- Parquet writer helper ---------------------------------------------------

def _write_partitioned(df: pl.DataFrame, table_name: str, partition_col: str = "country") -> None:
    """
    Write a country-partitioned Parquet dataset using PyArrow's
    write_to_dataset (produces Hive-style partition paths that Athena needs).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_dir = os.path.join(LAKE_DIR, table_name)
    os.makedirs(out_dir, exist_ok=True)

    arrow_table = df.to_arrow()
    pq.write_to_dataset(
        arrow_table,
        root_path=out_dir,
        partition_cols=[partition_col],
        compression="snappy",
        existing_data_behavior="overwrite_or_ignore",
    )
    print("   -> {}: {:,} rows -> {}".format(table_name, len(df), out_dir))


# -- Star-schema build -------------------------------------------------------

def build_star_schema() -> None:
    t_total = time.time()
    os.makedirs(LAKE_DIR, exist_ok=True)

    # 1. dim_canonical_entity (S1 reference) ---------------------------------
    print("\n[1/4] Building dim_canonical_entity ...")
    t0 = time.time()
    df_s1_raw = pl.read_csv(
        os.path.join(TEST_DATA_DIR, "test_source1.tsv"),
        separator="\t",
        encoding="utf8-lossy",
    )
    dim_entity = _extract_s1_features(df_s1_raw)
    _write_partitioned(dim_entity, "dim_canonical_entity")
    print("   Done in {:.1f}s -- {:,} S1 entities".format(time.time() - t0, len(dim_entity)))

    # 2. dim_target_record (S2 + S3 vendors) ---------------------------------
    print("\n[2/4] Building dim_target_record ...")
    t0 = time.time()
    df_s2_raw = pl.read_csv(
        os.path.join(TEST_DATA_DIR, "test_source2.tsv"),
        separator="\t",
        encoding="utf8-lossy",
    )
    df_s3_raw = pl.read_csv(
        os.path.join(TEST_DATA_DIR, "test_source3.tsv"),
        separator="\t",
        encoding="utf8-lossy",
    )
    dim_target = pl.concat([
        _extract_target_features(df_s2_raw, "S2"),
        _extract_target_features(df_s3_raw, "S3"),
    ])
    _write_partitioned(dim_target, "dim_target_record")
    print("   Done in {:.1f}s -- {:,} target records".format(time.time() - t0, len(dim_target)))

    # 3. fact_entity_summary (per-S1 blocking telemetry & match counts) ------
    print("\n[3/4] Building fact_entity_summary ...")
    t0 = time.time()

    # Country lookup: entity_id -> country (from S1)
    country_map = dim_entity.select(["entity_id", "country"])

    # Load matching results: source1_entity_id, matched_entity_ids (comma-sep, nullable)
    df_matches = pl.read_csv(
        os.path.join(OUTPUT_DIR, "matching_results.tsv"),
        separator="\t",
        encoding="utf8-lossy",
        null_values=[""],
    )

    # Load candidate pairs: source1_entity_id, candidate_entity_ids
    df_cands = pl.read_csv(
        os.path.join(OUTPUT_DIR, "candidate_pairs.tsv"),
        separator="\t",
        encoding="utf8-lossy",
        null_values=[""],
    )

    # Compute candidate_count per S1 entity
    df_cand_counts = df_cands.with_columns(
        pl.when(pl.col("candidate_entity_ids").is_not_null())
        .then(pl.col("candidate_entity_ids").str.count_matches(",") + 1)
        .otherwise(pl.lit(0))
        .alias("candidate_count")
    ).select(["source1_entity_id", "candidate_count"])

    # Compute matched_target_count
    df_match_counts = df_matches.with_columns(
        pl.when(pl.col("matched_entity_ids").is_not_null())
        .then(pl.col("matched_entity_ids").str.count_matches(",") + 1)
        .otherwise(pl.lit(0))
        .alias("matched_target_count")
    ).select(["source1_entity_id", "matched_target_count"])

    # Assemble fact_entity_summary
    fact_summary = (
        df_match_counts
        .join(df_cand_counts, on="source1_entity_id", how="left")
        .join(country_map, left_on="source1_entity_id", right_on="entity_id", how="left")
        .with_columns([
            (pl.col("matched_target_count") == 0).alias("is_singleton"),
            pl.when(pl.col("country") == "India")
              .then(pl.lit("v9_semantic_xgb"))
              .otherwise(pl.lit("v10_jev_judge"))
              .alias("model_version"),
            pl.col("source1_entity_id").alias("entity_id"),
        ])
        .select([
            "entity_id",
            "country",
            "matched_target_count",
            "is_singleton",
            "model_version",
            "candidate_count",
        ])
        .filter(pl.col("country").is_not_null())
    )
    _write_partitioned(fact_summary, "fact_entity_summary")
    n_singletons = fact_summary.filter(pl.col("is_singleton"))["entity_id"].len()
    print("   Done in {:.1f}s -- {:,} S1 summaries ({:,} singletons)".format(
        time.time() - t0, len(fact_summary), n_singletons))

    # 4. fact_resolution_links (granular 1-to-many bridge table) -------------
    print("\n[4/4] Building fact_resolution_links ...")
    t0 = time.time()

    fact_links = (
        df_matches
        .filter(pl.col("matched_entity_ids").is_not_null())
        .with_columns(
            pl.col("matched_entity_ids").str.split(",").alias("target_list")
        )
        .explode("target_list")
        .rename({"target_list": "target_id"})
        .with_columns(
            pl.col("target_id").str.slice(0, 2).alias("source_system")
        )
        .join(country_map, left_on="source1_entity_id", right_on="entity_id", how="left")
        .with_columns(
            pl.when(pl.col("country") == "India")
              .then(pl.lit("v9_semantic_xgb"))
              .otherwise(pl.lit("v10_jev_judge"))
              .alias("model_version"),
            pl.col("source1_entity_id").alias("entity_id"),
        )
        .select([
            "entity_id",
            "country",
            "target_id",
            "source_system",
            "model_version",
        ])
        .filter(pl.col("country").is_not_null())
    )
    _write_partitioned(fact_links, "fact_resolution_links")
    print("   Done in {:.1f}s -- {:,} resolved match links".format(
        time.time() - t0, len(fact_links)))

    # -- Summary -------------------------------------------------------------
    total_elapsed = time.time() - t_total
    print("\n" + "=" * 60)
    print("DONE: Lakehouse build complete in {:.0f}s".format(total_elapsed))
    print("  Output: {}".format(LAKE_DIR))
    print("  Tables: dim_canonical_entity, dim_target_record,")
    print("          fact_entity_summary, fact_resolution_links")

    total_bytes = 0
    for root, dirs, files in os.walk(LAKE_DIR):
        for f in files:
            total_bytes += os.path.getsize(os.path.join(root, f))
    print("  Total compressed size: {:.1f} MB".format(total_bytes / 1024 / 1024))
    print("=" * 60)

    print("\nPer-country breakdown (fact_resolution_links):")
    breakdown = fact_links.group_by("country").agg(
        pl.len().alias("resolved_links"),
        pl.col("entity_id").n_unique().alias("s1_entities"),
    ).sort("country")
    for row in breakdown.rows(named=True):
        print("  {:8s}: {:>8,} S1 entities, {:>9,} match links".format(
            row["country"], row["s1_entities"], row["resolved_links"]))

    print("\nPer-country breakdown (fact_entity_summary):")
    summ_breakdown = fact_summary.group_by("country").agg(
        pl.len().alias("total_s1"),
        pl.col("candidate_count").sum().alias("total_candidates"),
        pl.col("matched_target_count").sum().alias("total_matches"),
        pl.col("is_singleton").sum().alias("singletons"),
    ).sort("country")
    for row in summ_breakdown.rows(named=True):
        singleton_pct = 100.0 * row["singletons"] / row["total_s1"] if row["total_s1"] else 0
        print("  {:8s}: {:>8,} S1 | {:>12,} candidates | {:>9,} matches | {:.1f}% singletons".format(
            row["country"], row["total_s1"],
            row["total_candidates"], row["total_matches"], singleton_pct))


if __name__ == "__main__":
    build_star_schema()
