"""
deploy/redshift_verify.py
=========================
Loads the gold Parquet tables into Amazon Redshift Serverless and runs
the same two verification queries as Athena so you can compare runtimes.

Prerequisites
-------------
1. Create a Redshift Serverless workgroup (free 90-day trial available):
     https://console.aws.amazon.com/redshift/home#/serverless-dashboard
   - Workgroup name:  entity-resolution-wg
   - Namespace name:  entity-resolution-ns
   - Base RPU:        8  (minimum; scale to 0 if idle billing is a concern)
   - VPC / subnet:    default VPC is fine for a trial
   - Admin username:  admin
   - Set REDSHIFT_PASSWORD env var to the admin password you choose

2. Set a billing alarm BEFORE starting Redshift Serverless.
   Redshift Serverless bills by RPU-second when queries run.
   8 RPU x ~10 min = ~$0.13 for this script.  There is a 90-day free trial
   for new accounts; check https://aws.amazon.com/redshift/pricing/ first.

3. Install the Redshift Data API client (already in requirements.txt):
       pip install boto3>=1.34

Usage
-----
    # Set credentials either via aws configure or env vars, then:
    python deploy/redshift_verify.py

Environment variables
---------------------
BUCKET              S3 bucket (default: riya-entity-lakehouse-2026)
GOLD_PREFIX         S3 prefix for Parquet (default: lakehouse_gold)
REDSHIFT_WORKGROUP  Redshift Serverless workgroup name (default: entity-resolution-wg)
REDSHIFT_DB         Database name (default: dev)
REDSHIFT_ROLE_ARN   IAM role ARN that Redshift uses to read S3
                    (must have s3:GetObject on the bucket)

What the script does
--------------------
1. Creates an external schema pointing at the Glue Data Catalog
   (so the same Parquet files are visible without re-COPY).
2. Runs Query 1 (mutual exclusivity) and Query 2 (blocking rollup)
   via the Redshift Data API (no JDBC driver needed).
3. Prints the results and the execution time reported by Redshift.
4. Writes results to aws_results/redshift_query{1,2}.csv.
5. Prints a teardown reminder.
"""

import csv
import json
import os
import sys
import time
import boto3
from botocore.exceptions import ClientError

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
BUCKET     = os.environ.get("BUCKET",             "riya-entity-lakehouse-2026")
GOLD_PREFIX= os.environ.get("GOLD_PREFIX",        "lakehouse_gold")
WORKGROUP  = os.environ.get("REDSHIFT_WORKGROUP",  "entity-resolution-wg")
DB         = os.environ.get("REDSHIFT_DB",         "dev")
ROLE_ARN   = os.environ.get("REDSHIFT_ROLE_ARN",   "")   # required

S3_GOLD    = f"s3://{BUCKET}/{GOLD_PREFIX}"

# Output directory (repo root aws_results/)
_HERE       = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.dirname(_HERE)
RESULTS_DIR = os.path.join(_REPO_ROOT, "aws_results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# --------------------------------------------------------------------------- #
# Redshift Data API helpers
# --------------------------------------------------------------------------- #
def _client():
    return boto3.client("redshift-data", region_name=os.environ.get("AWS_DEFAULT_REGION", "ap-south-1"))


def run_sql(client, sql: str, label: str = "") -> dict:
    """Execute SQL via the Redshift Data API; return result dict with timing."""
    resp = client.execute_statement(
        WorkgroupName=WORKGROUP,
        Database=DB,
        Sql=sql,
    )
    sid = resp["Id"]
    print(f"  [{label}] submitted (id={sid[:8]}...)", flush=True)

    # Poll until done
    t0 = time.time()
    while True:
        desc  = client.describe_statement(Id=sid)
        state = desc["Status"]
        if state in ("FINISHED", "FAILED", "ABORTED"):
            break
        time.sleep(2)
    elapsed = time.time() - t0

    if state != "FINISHED":
        err = desc.get("Error", "no error detail")
        raise RuntimeError(f"  [{label}] FAILED: {err}")

    # Fetch results
    result = client.get_statement_result(Id=sid)
    columns = [c["label"] for c in result["ColumnMetadata"]]
    rows    = []
    for row in result["Records"]:
        rows.append({columns[i]: list(field.values())[0] for i, field in enumerate(row)})

    rs_ms = desc.get("Duration", 0) / 1_000_000   # nanoseconds -> ms
    print(f"  [{label}] done in {elapsed:.1f}s wall / {rs_ms:.0f} ms Redshift execution")
    return {"columns": columns, "rows": rows, "wall_s": elapsed, "rs_ms": rs_ms}


def save_csv(result: dict, path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=result["columns"])
        w.writeheader()
        w.writerows(result["rows"])
    print(f"  Saved -> {path}")


# --------------------------------------------------------------------------- #
# SQL statements
# --------------------------------------------------------------------------- #
CREATE_EXTERNAL_SCHEMA = f"""
CREATE EXTERNAL SCHEMA IF NOT EXISTS entity_gold
FROM DATA CATALOG
DATABASE 'entity_resolution_dw'
IAM_ROLE '{ROLE_ARN}'
CREATE EXTERNAL DATABASE IF NOT EXISTS;
"""

QUERY1_MUTUAL_EXCLUSIVITY = """
SELECT
    country,
    COUNT(target_id)                             AS total_assigned_targets,
    COUNT(DISTINCT target_id)                    AS unique_assigned_targets,
    COUNT(target_id) - COUNT(DISTINCT target_id) AS mutual_exclusivity_violations
FROM entity_gold.fact_resolution_links
GROUP BY country
ORDER BY country;
"""

QUERY2_BLOCKING_ROLLUP = """
SELECT
    country,
    model_version,
    COUNT(entity_id)                                                   AS total_s1_entities,
    SUM(candidate_count)                                               AS total_blocking_pairs,
    ROUND(AVG(CAST(candidate_count AS FLOAT)), 2)                      AS avg_candidates_per_entity,
    SUM(matched_target_count)                                          AS total_resolved_matches,
    ROUND(100.0 * SUM(CASE WHEN is_singleton THEN 1 ELSE 0 END)
                / NULLIF(COUNT(entity_id), 0), 2)                      AS singleton_rate_pct
FROM entity_gold.fact_entity_summary
GROUP BY country, model_version
ORDER BY total_s1_entities DESC;
"""


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    if not ROLE_ARN:
        print("ERROR: Set REDSHIFT_ROLE_ARN env var to the IAM role Redshift uses to read S3.")
        print("  Example: arn:aws:iam::123456789012:role/RedshiftS3ReadRole")
        sys.exit(1)

    client = _client()

    print("=" * 64)
    print("Entity Resolution — Redshift Serverless Verification")
    print(f"  Workgroup : {WORKGROUP}")
    print(f"  Database  : {DB}")
    print(f"  S3 gold   : {S3_GOLD}")
    print("=" * 64)

    # 1. Create external schema (points at Glue Catalog)
    print("\n[1/3] Creating external schema entity_gold ...")
    try:
        run_sql(client, CREATE_EXTERNAL_SCHEMA, "create-schema")
    except RuntimeError as e:
        # Schema may already exist — safe to continue
        print(f"  Note: {e} (may already exist, continuing)")

    # 2. Query 1 — mutual exclusivity
    print("\n[2/3] Running Query 1: mutual-exclusivity audit ...")
    r1 = run_sql(client, QUERY1_MUTUAL_EXCLUSIVITY, "q1-exclusivity")
    save_csv(r1, os.path.join(RESULTS_DIR, "redshift_query1_quality_gate.csv"))

    violations = sum(int(row.get("mutual_exclusivity_violations", 0)) for row in r1["rows"])
    print(f"  Total violations: {violations} ({'PASS' if violations == 0 else 'FAIL'})")

    # 3. Query 2 — blocking rollup
    print("\n[3/3] Running Query 2: blocking telemetry rollup ...")
    r2 = run_sql(client, QUERY2_BLOCKING_ROLLUP, "q2-rollup")
    save_csv(r2, os.path.join(RESULTS_DIR, "redshift_query2_executive_rollup.csv"))

    # Summary
    print("\n" + "=" * 64)
    print("RESULTS")
    print(f"  Q1 (exclusivity) — Redshift execution: {r1['rs_ms']:.0f} ms")
    print(f"  Q2 (rollup)      — Redshift execution: {r2['rs_ms']:.0f} ms")
    print(f"  Violations: {violations}")
    print("  CSVs written to aws_results/")
    print()
    print("TEARDOWN REMINDER")
    print("  Redshift Serverless bills by RPU-second while queries run.")
    print("  When done, delete the workgroup in the Redshift console or run:")
    print(f"    aws redshift-serverless delete-workgroup --workgroup-name {WORKGROUP}")
    print(f"    aws redshift-serverless delete-namespace --namespace-name entity-resolution-ns")
    print("=" * 64)

    return 0 if violations == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

