"""
deploy/bootstrap_aws_pipeline.py
=================================
One-shot script that provisions the AWS pipeline components in order:

  Step 1 — Upload the Glue script to S3
  Step 2 — Create the Glue ETL job (entity-resolution-etl)
  Step 3 — Create a Glue crawler (entity-resolution-crawler)
  Step 4 — Package and deploy the Lambda quality-gate function
  Step 5 — Create an EventBridge scheduled rule (daily at 02:00 UTC)
            that invokes the Lambda

Prerequisites
-------------
1. AWS credentials configured (aws configure or env vars).
2. IAM roles already created:
     GLUE_ROLE_ARN   — attached policies: AWSGlueServiceRole, S3 read/write on bucket
     LAMBDA_ROLE_ARN — attached policies: AWSLambdaBasicExecutionRole,
                        AmazonAthenaFullAccess, S3 read/write on bucket
3. Run export_lakehouse.py + s3_upload.py first so the Parquet data is in S3.

Usage
-----
    # Set the two role ARNs as env vars, then:
    python deploy/bootstrap_aws_pipeline.py

    # Or pass them on the command line:
    GLUE_ROLE_ARN=arn:aws:iam::123:role/GlueRole \
    LAMBDA_ROLE_ARN=arn:aws:iam::123:role/LambdaRole \
    python deploy/bootstrap_aws_pipeline.py

Cost notes (ap-south-1 pricing at time of writing)
---------------------------------------------------
Service              Free tier?   Estimated cost for this project
-------------------  ----------   -------------------------------------------
S3 storage           Yes (5 GB)   $0.00 (1.06 GB within free tier)
Glue Data Catalog    Yes          $0.00 (< 1 M objects)
Glue ETL job         No           ~$0.18 per run (2 DPU x 12 min x $0.44/DPU-hr)
Lambda               Yes          $0.00 (< 1 M invocations/month free)
EventBridge          Yes          $0.00 (< 14 M events/month free)
Athena (via Lambda)  No free tier ~$0.0002 per quality-gate run
Redshift Serverless  90-day trial ~$0.13 per verification run (8 RPU x 10 min)

Set a billing alarm before running Glue or Redshift.
"""

import io
import json
import os
import sys
import time
import zipfile
import boto3
from botocore.exceptions import ClientError

# --------------------------------------------------------------------------- #
# Config — override via environment variables
# --------------------------------------------------------------------------- #
BUCKET           = os.environ.get("BUCKET",            "riya-entity-lakehouse-2026")
REGION           = os.environ.get("AWS_DEFAULT_REGION","ap-south-1")
GLUE_ROLE_ARN    = os.environ.get("GLUE_ROLE_ARN",    "")   # required
LAMBDA_ROLE_ARN  = os.environ.get("LAMBDA_ROLE_ARN",  "")   # required

GLUE_JOB_NAME     = "entity-resolution-etl"
GLUE_CRAWLER_NAME = "entity-resolution-crawler"
LAMBDA_FUNC_NAME  = "entity-resolution-quality-gate"
EVENTBRIDGE_RULE  = "entity-resolution-daily-gate"

GOLD_PREFIX  = "lakehouse_gold"
TSV_PREFIX   = "raw-input"
GLUE_SCRIPT  = "glue-scripts/glue_etl_job.py"

_HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _clients():
    session = boto3.Session(region_name=REGION)
    return {
        "s3":      session.client("s3"),
        "glue":    session.client("glue"),
        "lambda":  session.client("lambda"),
        "events":  session.client("events"),
        "sts":     session.client("sts"),
    }


def banner(text: str) -> None:
    print(f"\n{'=' * 64}\n  {text}\n{'=' * 64}")


def _wait(label: str, check_fn, timeout_s: int = 120) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        ok, msg = check_fn()
        print(f"  {label}: {msg}")
        if ok:
            return True
        time.sleep(5)
    print(f"  TIMEOUT waiting for {label}")
    return False


# --------------------------------------------------------------------------- #
# Step 1 — Upload Glue script to S3
# --------------------------------------------------------------------------- #
def upload_glue_script(s3) -> None:
    banner("Step 1: Upload Glue script to S3")
    script_path = os.path.join(_HERE, "glue_etl_job.py")
    if not os.path.exists(script_path):
        print(f"ERROR: {script_path} not found")
        sys.exit(1)
    s3.upload_file(script_path, BUCKET, GLUE_SCRIPT)
    print(f"  Uploaded -> s3://{BUCKET}/{GLUE_SCRIPT}")


# --------------------------------------------------------------------------- #
# Step 2 — Create Glue ETL job
# --------------------------------------------------------------------------- #
def create_glue_job(glue) -> None:
    banner("Step 2: Create Glue ETL job")
    job_args = {
        "--BUCKET":       BUCKET,
        "--TSV_PREFIX":   TSV_PREFIX,
        "--GOLD_PREFIX":  GOLD_PREFIX,
        "--job-bookmark-option": "job-bookmark-enable",
    }
    try:
        glue.create_job(
            Name=GLUE_JOB_NAME,
            Role=GLUE_ROLE_ARN,
            Command={
                "Name":           "glueetl",
                "ScriptLocation": f"s3://{BUCKET}/{GLUE_SCRIPT}",
                "PythonVersion":  "3",
            },
            DefaultArguments=job_args,
            GlueVersion="4.0",
            WorkerType="G.1X",
            NumberOfWorkers=2,
            Timeout=30,   # minutes; kills the job if it runs too long
            Description="Entity resolution TSV -> partitioned Parquet ETL",
        )
        print(f"  Created Glue job: {GLUE_JOB_NAME}")
        print(f"  Worker: G.1X x 2 DPUs  (~$0.18 per run at current pricing)")
    except ClientError as e:
        if e.response["Error"]["Code"] == "AlreadyExistsException":
            print(f"  Job '{GLUE_JOB_NAME}' already exists — skipping create")
        else:
            raise


# --------------------------------------------------------------------------- #
# Step 3 — Create Glue crawler (schema discovery)
# --------------------------------------------------------------------------- #
def create_glue_crawler(glue) -> None:
    banner("Step 3: Create Glue crawler (schema discovery)")
    try:
        glue.create_crawler(
            Name=GLUE_CRAWLER_NAME,
            Role=GLUE_ROLE_ARN,
            DatabaseName="entity_resolution_dw",
            Targets={
                "S3Targets": [
                    {"Path": f"s3://{BUCKET}/{GOLD_PREFIX}/dim_canonical_entity/"},
                    {"Path": f"s3://{BUCKET}/{GOLD_PREFIX}/dim_target_record/"},
                    {"Path": f"s3://{BUCKET}/{GOLD_PREFIX}/fact_entity_summary/"},
                    {"Path": f"s3://{BUCKET}/{GOLD_PREFIX}/fact_resolution_links/"},
                ]
            },
            SchemaChangePolicy={
                "UpdateBehavior": "UPDATE_IN_DATABASE",
                "DeleteBehavior": "LOG",
            },
            Description="Discovers Parquet partitions and updates Glue Data Catalog",
        )
        print(f"  Created crawler: {GLUE_CRAWLER_NAME}")
        print("  Run it manually from the Glue console, or it runs with the ETL job.")
    except ClientError as e:
        if e.response["Error"]["Code"] == "AlreadyExistsException":
            print(f"  Crawler '{GLUE_CRAWLER_NAME}' already exists — skipping")
        else:
            raise


# --------------------------------------------------------------------------- #
# Step 4 — Package and deploy Lambda
# --------------------------------------------------------------------------- #
def deploy_lambda(lmb) -> str:
    banner("Step 4: Deploy Lambda quality-gate function")
    handler_path = os.path.join(_HERE, "lambda_quality_gate", "handler.py")
    if not os.path.exists(handler_path):
        print(f"ERROR: {handler_path} not found")
        sys.exit(1)

    # Zip the handler in memory
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(handler_path, "handler.py")
    zip_bytes = buf.getvalue()

    env_vars = {
        "BUCKET":         BUCKET,
        "ATHENA_DB":      "entity_resolution_dw",
        "ATHENA_RESULTS": f"s3://{BUCKET}/athena-results/",
    }

    try:
        resp = lmb.create_function(
            FunctionName=LAMBDA_FUNC_NAME,
            Runtime="python3.12",
            Role=LAMBDA_ROLE_ARN,
            Handler="handler.lambda_handler",
            Code={"ZipFile": zip_bytes},
            Timeout=60,
            MemorySize=256,
            Environment={"Variables": env_vars},
            Description="Runs Athena integrity checks after each Parquet upload",
        )
        func_arn = resp["FunctionArn"]
        print(f"  Created Lambda: {LAMBDA_FUNC_NAME}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceConflictException":
            # Update existing function
            lmb.update_function_code(
                FunctionName=LAMBDA_FUNC_NAME,
                ZipFile=zip_bytes,
            )
            lmb.update_function_configuration(
                FunctionName=LAMBDA_FUNC_NAME,
                Environment={"Variables": env_vars},
                Timeout=60,
            )
            resp    = lmb.get_function(FunctionName=LAMBDA_FUNC_NAME)
            func_arn = resp["Configuration"]["FunctionArn"]
            print(f"  Updated existing Lambda: {LAMBDA_FUNC_NAME}")
        else:
            raise

    print(f"  ARN: {func_arn}")
    return func_arn


# --------------------------------------------------------------------------- #
# Step 5 — EventBridge scheduled rule (daily at 02:00 UTC)
# --------------------------------------------------------------------------- #
def create_eventbridge_rule(events, lmb, func_arn: str) -> None:
    banner("Step 5: Create EventBridge daily schedule -> Lambda")

    # Create / update the rule
    try:
        rule_resp = events.put_rule(
            Name=EVENTBRIDGE_RULE,
            ScheduleExpression="cron(0 2 * * ? *)",   # 02:00 UTC every day
            State="ENABLED",
            Description="Daily trigger for entity-resolution quality gate",
        )
        rule_arn = rule_resp["RuleArn"]
        print(f"  Rule ARN: {rule_arn}")
    except ClientError as e:
        print(f"  put_rule failed: {e}")
        raise

    # Add Lambda as the target
    events.put_targets(
        Rule=EVENTBRIDGE_RULE,
        Targets=[{"Id": "quality-gate-target", "Arn": func_arn}],
    )
    print(f"  Target set: {func_arn}")

    # Grant EventBridge permission to invoke the Lambda
    try:
        lmb.add_permission(
            FunctionName=LAMBDA_FUNC_NAME,
            StatementId="EventBridgeDailyInvoke",
            Action="lambda:InvokeFunction",
            Principal="events.amazonaws.com",
            SourceArn=rule_arn,
        )
        print("  Lambda invoke permission granted to EventBridge")
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceConflictException":
            print("  Permission already exists — OK")
        else:
            raise


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    if not GLUE_ROLE_ARN or not LAMBDA_ROLE_ARN:
        print("ERROR: Set GLUE_ROLE_ARN and LAMBDA_ROLE_ARN environment variables.")
        print()
        print("  Quick IAM setup (one-time):")
        print("  1. Go to IAM -> Roles -> Create role -> AWS service -> Glue")
        print("     Attach: AWSGlueServiceRole + S3 full access on your bucket")
        print("     Name it: GlueS3Role   -> copy ARN to GLUE_ROLE_ARN")
        print()
        print("  2. Go to IAM -> Roles -> Create role -> AWS service -> Lambda")
        print("     Attach: AWSLambdaBasicExecutionRole + AmazonAthenaFullAccess")
        print("             + S3 GetObject/PutObject on your bucket")
        print("     Name it: LambdaAthenaRole -> copy ARN to LAMBDA_ROLE_ARN")
        sys.exit(1)

    c = _clients()

    # Verify credentials
    try:
        identity = c["sts"].get_caller_identity()
        print(f"AWS credentials OK — account {identity['Account']}, region {REGION}")
    except Exception as e:
        print(f"ERROR: AWS credentials check failed: {e}")
        sys.exit(1)

    upload_glue_script(c["s3"])
    create_glue_job(c["glue"])
    create_glue_crawler(c["glue"])
    func_arn = deploy_lambda(c["lambda"])
    create_eventbridge_rule(c["events"], c["lambda"], func_arn)

    print("\n" + "=" * 64)
    print("Bootstrap complete. Resources created:")
    print(f"  Glue job     : {GLUE_JOB_NAME}")
    print(f"  Glue crawler : {GLUE_CRAWLER_NAME}")
    print(f"  Lambda       : {LAMBDA_FUNC_NAME}")
    print(f"  EventBridge  : {EVENTBRIDGE_RULE}  (daily 02:00 UTC)")
    print()
    print("Next steps:")
    print("  1. Run the Glue job once to (re)generate Parquet from raw TSVs:")
    print(f"     aws glue start-job-run --job-name {GLUE_JOB_NAME}")
    print()
    print("  2. Run the crawler to refresh the Glue Data Catalog:")
    print(f"     aws glue start-crawler --name {GLUE_CRAWLER_NAME}")
    print()
    print("  3. Test the Lambda manually:")
    print(f"     aws lambda invoke --function-name {LAMBDA_FUNC_NAME} out.json")
    print("     cat out.json")
    print()
    print("  4. (Optional) Run Redshift verification:")
    print("     REDSHIFT_ROLE_ARN=<arn> python deploy/redshift_verify.py")
    print("=" * 64)


if __name__ == "__main__":
    main()

