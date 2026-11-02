"""
deploy/glue_etl_job.py
======================
AWS Glue PySpark ETL job — ports the heavy step of export_lakehouse.py.

What it does
------------
Reads matching_results.tsv, candidate_pairs.tsv, and test_source{1,2,3}.tsv
from S3, applies the same Kimball star-schema transforms as export_lakehouse.py,
and writes Snappy-compressed, country-partitioned Parquet back to S3.

How to deploy
-------------
1. Upload this file to S3 (the bootstrap script does this automatically):
       s3://<BUCKET>/glue-scripts/glue_etl_job.py

2. Create the Glue job (bootstrap_aws_pipeline.py does this):
       Name:           entity-resolution-etl
       Role:           arn:aws:iam::<ACCOUNT>:role/GlueS3Role
       Script:         s3://<BUCKET>/glue-scripts/glue_etl_job.py
       Worker type:    G.1X  (1 DPU)
       Number workers: 2
       Glue version:   4.0  (Spark 3.3)

   Cost: Glue bills $0.44/DPU-hour.  2 DPUs x ~12 min = ~$0.18 total.
   This is not free-tier; set a billing alarm before running.

3. Or run manually in the Glue console:
       AWS Glue -> Jobs -> entity-resolution-etl -> Run

Environment / job parameters (set in Glue job definition or at run time)
-------------------------------------------------------------------------
--BUCKET          S3 bucket name         (default: riya-entity-lakehouse-2026)
--TSV_PREFIX      S3 prefix for raw TSVs (default: raw-input)
--GOLD_PREFIX     S3 prefix for output   (default: lakehouse_gold)
"""

import sys
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import functions as F
from pyspark.sql.types import LongType

# --------------------------------------------------------------------------- #
# Glue boilerplate
# --------------------------------------------------------------------------- #
args = getResolvedOptions(
    sys.argv,
    ["JOB_NAME", "BUCKET", "TSV_PREFIX", "GOLD_PREFIX"],
)
sc         = SparkContext()
glueCtx    = GlueContext(sc)
spark      = glueCtx.spark_session
job        = Job(glueCtx)
job.init(args["JOB_NAME"], args)

BUCKET      = args["BUCKET"]
TSV_PREFIX  = args.get("TSV_PREFIX",  "raw-input")
GOLD_PREFIX = args.get("GOLD_PREFIX", "lakehouse_gold")

S3_TSV  = f"s3://{BUCKET}/{TSV_PREFIX}"
S3_GOLD = f"s3://{BUCKET}/{GOLD_PREFIX}"

print(f"[glue_etl_job] BUCKET={BUCKET}  TSV_PREFIX={TSV_PREFIX}  GOLD_PREFIX={GOLD_PREFIX}")


# --------------------------------------------------------------------------- #
# Helper: write a Spark DF as Snappy Parquet partitioned by country
# --------------------------------------------------------------------------- #
def write_gold(df, table_name: str) -> None:
    path = f"{S3_GOLD}/{table_name}"
    (
        df.repartition("country")               # one file per country partition
          .write
          .mode("overwrite")
          .partitionBy("country")
          .option("compression", "snappy")
          .parquet(path)
    )
    count = df.count()
    print(f"  -> {table_name}: {count:,} rows -> {path}")


# --------------------------------------------------------------------------- #
# Text normalisation helpers (mirrors export_lakehouse.py)
# --------------------------------------------------------------------------- #
_SKEL_RULES = [
    ("chh", "ch"), ("ph", "f"), ("bh", "b"), ("kh", "k"),
    ("gh",  "g"),  ("jh", "j"), ("th", "t"), ("dh", "d"),
    ("sh",  "s"),  ("ck", "k"), ("q",  "k"), ("x",  "ks"),
    ("w",   "v"),  ("z",  "s"),
]


def clean_name(col):
    """Lowercase, strip punctuation, collapse whitespace."""
    return (
        F.trim(F.regexp_replace(
            F.regexp_replace(F.lower(col), r"[^\w\s]", " "),
            r"\s+", " "
        ))
    )


def phonetic_skeleton(col):
    """Consonant skeleton key."""
    expr = col
    for pat, rep in _SKEL_RULES:
        expr = F.regexp_replace(expr, pat, rep)
    # Strip vowels after a consonant (capture-group approach, no look-behind)
    expr = F.regexp_replace(expr, r"([bcdfghjklmnprstv])[aeiouy]+", r"$1")
    return expr


# --------------------------------------------------------------------------- #
# 1. dim_canonical_entity
# --------------------------------------------------------------------------- #
print("\n[1/4] Building dim_canonical_entity ...")
df_s1 = (
    spark.read
    .option("header", "true")
    .option("sep", "\t")
    .csv(f"{S3_TSV}/test_source1.tsv")
    .fillna({"business_name": "", "business_address": ""})
)

dim_entity = (
    df_s1
    .withColumn("name_raw",    F.col("business_name"))
    .withColumn("addr_raw",    F.col("business_address"))
    .withColumn("name_clean",  clean_name(F.col("name_raw")))
    .withColumn("name_skel",   phonetic_skeleton(F.col("name_clean")))
    .withColumn("postal_code", F.regexp_extract(F.col("addr_raw"), r"\b(\d{5,6})\b", 1))
    .withColumn("house_num",   F.regexp_extract(clean_name(F.col("addr_raw")), r"\b0*(\d+[a-z]?)\b", 1))
    .select("entity_id", "country", "name_raw", "addr_raw",
            "name_clean", "name_skel", "postal_code", "house_num")
)
write_gold(dim_entity, "dim_canonical_entity")

# Keep a lightweight country map for joins below
country_map = dim_entity.select("entity_id", "country").cache()


# --------------------------------------------------------------------------- #
# 2. dim_target_record
# --------------------------------------------------------------------------- #
print("\n[2/4] Building dim_target_record ...")

def load_target(path: str, source_system: str):
    return (
        spark.read
        .option("header", "true")
        .option("sep", "\t")
        .csv(path)
        .fillna({"business_name": "", "business_address": ""})
        .withColumn("source_system", F.lit(source_system))
        .withColumn("target_id",     F.col("entity_id"))
        .withColumn("name_raw",      F.col("business_name"))
        .withColumn("addr_raw",      F.col("business_address"))
        .withColumn("postal_code",   F.regexp_extract(F.col("business_address"), r"\b(\d{5,6})\b", 1))
        .select("target_id", "source_system", "country", "name_raw", "addr_raw", "postal_code")
    )

dim_target = load_target(f"{S3_TSV}/test_source2.tsv", "S2").union(
             load_target(f"{S3_TSV}/test_source3.tsv", "S3"))
write_gold(dim_target, "dim_target_record")


# --------------------------------------------------------------------------- #
# 3. fact_entity_summary
# --------------------------------------------------------------------------- #
print("\n[3/4] Building fact_entity_summary ...")

df_matches = (
    spark.read
    .option("header", "true")
    .option("sep", "\t")
    .csv(f"{S3_TSV}/matching_results.tsv")
)
df_cands = (
    spark.read
    .option("header", "true")
    .option("sep", "\t")
    .csv(f"{S3_TSV}/candidate_pairs.tsv")
)

# Count matches per S1 entity
match_counts = (
    df_matches
    .withColumn(
        "matched_target_count",
        F.when(
            F.col("matched_entity_ids").isNotNull() & (F.col("matched_entity_ids") != ""),
            F.size(F.split(F.col("matched_entity_ids"), ","))
        ).otherwise(F.lit(0)).cast(LongType())
    )
    .select("source1_entity_id", "matched_target_count")
)

# Count candidates per S1 entity
cand_counts = (
    df_cands
    .withColumn(
        "candidate_count",
        F.when(
            F.col("candidate_entity_ids").isNotNull() & (F.col("candidate_entity_ids") != ""),
            F.size(F.split(F.col("candidate_entity_ids"), ","))
        ).otherwise(F.lit(0)).cast(LongType())
    )
    .select("source1_entity_id", "candidate_count")
)

fact_summary = (
    match_counts
    .join(cand_counts, on="source1_entity_id", how="left")
    .join(country_map, match_counts["source1_entity_id"] == country_map["entity_id"], how="left")
    .withColumn("is_singleton", F.col("matched_target_count") == 0)
    .withColumn(
        "model_version",
        F.when(F.col("country") == "India", F.lit("v9_semantic_xgb"))
         .otherwise(F.lit("v10_jev_judge"))
    )
    .withColumnRenamed("source1_entity_id", "entity_id_alias")
    .select(
        country_map["entity_id"],
        "country",
        "matched_target_count",
        "is_singleton",
        "model_version",
        "candidate_count",
    )
    .filter(F.col("country").isNotNull())
)
write_gold(fact_summary, "fact_entity_summary")


# --------------------------------------------------------------------------- #
# 4. fact_resolution_links
# --------------------------------------------------------------------------- #
print("\n[4/4] Building fact_resolution_links ...")

fact_links = (
    df_matches
    .filter(F.col("matched_entity_ids").isNotNull() & (F.col("matched_entity_ids") != ""))
    .withColumn("target_id", F.explode(F.split(F.col("matched_entity_ids"), ",")))
    .withColumn("source_system", F.substring(F.col("target_id"), 1, 2))
    .join(country_map, df_matches["source1_entity_id"] == country_map["entity_id"], how="left")
    .withColumn(
        "model_version",
        F.when(F.col("country") == "India", F.lit("v9_semantic_xgb"))
         .otherwise(F.lit("v10_jev_judge"))
    )
    .select(
        country_map["entity_id"],
        "country",
        "target_id",
        "source_system",
        "model_version",
    )
    .filter(F.col("country").isNotNull())
)
write_gold(fact_links, "fact_resolution_links")


# --------------------------------------------------------------------------- #
# Done
# --------------------------------------------------------------------------- #
print("\nGlue ETL job complete.")
job.commit()

