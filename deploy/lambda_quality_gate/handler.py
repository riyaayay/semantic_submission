"""
deploy/lambda_quality_gate/handler.py
======================================
AWS Lambda function — triggered by an S3 ObjectCreated event or an
EventBridge scheduled rule.

What it does
------------
1. Starts an Athena query (the mutual-exclusivity + row-count integrity check).
2. Waits up to 55 s for the query to finish (Lambda default timeout: 60 s).
3. Writes the result CSV to s3://<BUCKET>/athena-results/quality-gate-latest.csv
4. Returns the pass/fail status as the Lambda response.

If any check fails (violations > 0 or row counts wrong) the function returns
HTTP 500 so EventBridge / SNS can alert.

Deployment
----------
See bootstrap_aws_pipeline.py for automated deployment.
Manual steps:
  1. Zip this file:    Compress-Archive handler.py lambda_quality_gate.zip
  2. Create function:
       aws lambda create-function \
         --function-name entity-resolution-quality-gate \
         --runtime python3.12 \
         --handler handler.lambda_handler \
         --role arn:aws:iam::<ACCOUNT>:role/LambdaAthenaRole \
         --zip-file fileb://lambda_quality_gate.zip \
         --timeout 60 \
         --environment "Variables={BUCKET=riya-entity-lakehouse-2026,
                                   ATHENA_DB=entity_resolution_dw,
                                   ATHENA_RESULTS=s3://riya-entity-lakehouse-2026/athena-results/}"
  3. Add S3 trigger (ObjectCreated on prefix lakehouse_gold/) or
     EventBridge rule (schedule: rate(1 day)).

IAM permissions required on the Lambda execution role
------------------------------------------------------
  athena:StartQueryExecution, athena:GetQueryExecution, athena:GetQueryResults
  s3:GetObject, s3:PutObject, s3:GetBucketLocation  on the bucket
  glue:GetDatabase, glue:GetTable  (Athena reads Glue Catalog)

Cost
----
Lambda: first 1 M invocations/month free; this function runs for < 60 s.
Athena: ~$0.0002 per quality-gate run (scans a single country partition).
Total per day: effectively $0.00.
"""

import json
import os
import time
import boto3

BUCKET          = os.environ.get("BUCKET",          "riya-entity-lakehouse-2026")
ATHENA_DB       = os.environ.get("ATHENA_DB",       "entity_resolution_dw")
ATHENA_RESULTS  = os.environ.get("ATHENA_RESULTS",  f"s3://{BUCKET}/athena-results/")

athena = boto3.client("athena")
s3     = boto3.client("s3")

# --------------------------------------------------------------------------- #
# Queries
# --------------------------------------------------------------------------- #
QUALITY_GATE_SQL = """
SELECT
    country,
    COUNT(target_id)                             AS total_assigned_targets,
    COUNT(DISTINCT target_id)                    AS unique_assigned_targets,
    COUNT(target_id) - COUNT(DISTINCT target_id) AS mutual_exclusivity_violations
FROM {db}.fact_resolution_links
GROUP BY country
ORDER BY country
""".format(db=ATHENA_DB)

ROW_COUNT_S1_SQL = """
SELECT COUNT(*) AS s1_row_count FROM {db}.dim_canonical_entity
""".format(db=ATHENA_DB)

ROW_COUNT_TARGET_SQL = """
SELECT COUNT(*) AS target_row_count FROM {db}.dim_target_record
""".format(db=ATHENA_DB)

REFERENTIAL_INTEGRITY_SQL = """
SELECT COUNT(*) AS orphan_links
FROM {db}.fact_resolution_links f
LEFT JOIN {db}.dim_canonical_entity d
       ON f.entity_id = d.entity_id AND f.country = d.country
WHERE d.entity_id IS NULL
""".format(db=ATHENA_DB)

CHECKSUM_SQL = """
SELECT
    (SELECT CAST(SUM(matched_target_count) AS BIGINT)
       FROM {db}.fact_entity_summary)   AS summary_total,
    (SELECT COUNT(*)
       FROM {db}.fact_resolution_links) AS links_total
""".format(db=ATHENA_DB)

# Expected counts (from the original export)
EXPECTED_S1_ROWS     = 1_732_544
EXPECTED_TARGET_ROWS = 9_969_589


# --------------------------------------------------------------------------- #
# Athena helpers
# --------------------------------------------------------------------------- #
def run_query(sql: str) -> str:
    """Start an Athena query and return the QueryExecutionId."""
    resp = athena.start_query_execution(
        QueryString=sql,
        QueryExecutionContext={"Database": ATHENA_DB},
        ResultConfiguration={"OutputLocation": ATHENA_RESULTS},
    )
    return resp["QueryExecutionId"]


def wait_for_query(qid: str, timeout_s: int = 50) -> str:
    """Poll until the query succeeds or fails. Returns final state."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        resp  = athena.get_query_execution(QueryExecutionId=qid)
        state = resp["QueryExecution"]["Status"]["State"]
        if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
            return state
        time.sleep(2)
    return "TIMEOUT"


def fetch_rows(qid: str) -> list[dict]:
    """Return query result rows as a list of dicts."""
    resp    = athena.get_query_results(QueryExecutionId=qid)
    columns = [c["Label"] for c in resp["ResultSet"]["ResultSetMetadata"]["ColumnInfo"]]
    rows    = []
    for row in resp["ResultSet"]["Rows"][1:]:   # skip header
        rows.append({columns[i]: d.get("VarCharValue", "") for i, d in enumerate(row["Data"])})
    return rows


# --------------------------------------------------------------------------- #
# Individual checks
# --------------------------------------------------------------------------- #
def check_mutual_exclusivity() -> tuple[bool, list[dict]]:
    qid   = run_query(QUALITY_GATE_SQL)
    state = wait_for_query(qid)
    if state != "SUCCEEDED":
        return False, [{"error": f"query {state}"}]
    rows   = fetch_rows(qid)
    passed = all(int(r["mutual_exclusivity_violations"]) == 0 for r in rows)
    return passed, rows


def check_row_count(sql: str, col: str, expected: int) -> tuple[bool, int]:
    qid   = run_query(sql)
    state = wait_for_query(qid)
    if state != "SUCCEEDED":
        return False, -1
    rows  = fetch_rows(qid)
    actual = int(rows[0][col])
    return actual == expected, actual


def check_referential_integrity() -> tuple[bool, int]:
    qid   = run_query(REFERENTIAL_INTEGRITY_SQL)
    state = wait_for_query(qid)
    if state != "SUCCEEDED":
        return False, -1
    rows   = fetch_rows(qid)
    orphans = int(rows[0]["orphan_links"])
    return orphans == 0, orphans


def check_cross_table_checksum() -> tuple[bool, dict]:
    qid   = run_query(CHECKSUM_SQL)
    state = wait_for_query(qid)
    if state != "SUCCEEDED":
        return False, {}
    rows = fetch_rows(qid)
    r    = rows[0]
    passed = r["summary_total"] == r["links_total"]
    return passed, r


# --------------------------------------------------------------------------- #
# Lambda handler
# --------------------------------------------------------------------------- #
def lambda_handler(event, context):
    results = {}
    all_passed = True

    # 1. Mutual-exclusivity
    ok, mx_rows = check_mutual_exclusivity()
    results["mutual_exclusivity"] = {"passed": ok, "rows": mx_rows}
    all_passed = all_passed and ok

    # 2. Row count — dim_canonical_entity
    ok, actual = check_row_count(ROW_COUNT_S1_SQL, "s1_row_count", EXPECTED_S1_ROWS)
    results["s1_row_count"] = {"passed": ok, "expected": EXPECTED_S1_ROWS, "actual": actual}
    all_passed = all_passed and ok

    # 3. Row count — dim_target_record
    ok, actual = check_row_count(ROW_COUNT_TARGET_SQL, "target_row_count", EXPECTED_TARGET_ROWS)
    results["target_row_count"] = {"passed": ok, "expected": EXPECTED_TARGET_ROWS, "actual": actual}
    all_passed = all_passed and ok

    # 4. Referential integrity
    ok, orphans = check_referential_integrity()
    results["referential_integrity"] = {"passed": ok, "orphan_links": orphans}
    all_passed = all_passed and ok

    # 5. Cross-table checksum
    ok, counts = check_cross_table_checksum()
    results["cross_table_checksum"] = {"passed": ok, "counts": counts}
    all_passed = all_passed and ok

    payload = {
        "status":     "PASS" if all_passed else "FAIL",
        "all_passed": all_passed,
        "checks":     results,
    }

    # Write result to S3 so it's queryable / auditable
    s3.put_object(
        Bucket=BUCKET,
        Key="athena-results/quality-gate-latest.json",
        Body=json.dumps(payload, indent=2).encode(),
        ContentType="application/json",
    )

    print(json.dumps(payload, indent=2))

    status_code = 200 if all_passed else 500
    return {"statusCode": status_code, "body": json.dumps(payload)}

