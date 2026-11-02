"""
deploy/s3_upload.py
===================
Uploads output/lakehouse_gold/ Parquet Star Schema to Amazon S3.
Uses boto3 (already in requirements) -- no AWS CLI needed.

Prerequisites:
    aws configure   (sets ~/.aws/credentials and ~/.aws/config)
    OR set environment variables:
        AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION

Usage (from repo root):
    python deploy/s3_upload.py

Optional env overrides:
    BUCKET   -- S3 bucket name (default: riya-entity-lakehouse-2026)
    REGION   -- AWS region    (default: ap-southeast-2)
    LAKE_DIR -- local path to lakehouse_gold/
"""

import os
import sys
import time
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

# -- Config ------------------------------------------------------------------
BUCKET   = os.environ.get("BUCKET",   "riya-entity-lakehouse-2026")
REGION   = os.environ.get("REGION",   "ap-southeast-2")

_SCRIPT  = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.dirname(_SCRIPT)
LAKE_DIR = os.environ.get("LAKE_DIR", os.path.join(_ROOT, "output", "lakehouse_gold"))


def sizeof_fmt(num_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024:
            return "{:.1f} {}".format(num_bytes, unit)
        num_bytes /= 1024
    return "{:.1f} TB".format(num_bytes)


def create_bucket_if_not_exists(s3, bucket: str, region: str) -> None:
    try:
        if region == "us-east-1":
            s3.create_bucket(Bucket=bucket)
        else:
            s3.create_bucket(
                Bucket=bucket,
                CreateBucketConfiguration={"LocationConstraint": region},
            )
        print("  Created bucket s3://{}".format(bucket))
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
            print("  Bucket s3://{} already exists -- OK".format(bucket))
        else:
            raise


def collect_files(lake_dir: str):
    """Walk lakehouse_gold/ and collect (local_path, s3_key) pairs."""
    files = []
    for root, dirs, filenames in os.walk(lake_dir):
        for fname in filenames:
            if fname.startswith(".") or fname == "__pycache__":
                continue
            local = os.path.join(root, fname)
            # s3 key: gold/<table>/country=XX/<file>.parquet
            rel   = os.path.relpath(local, os.path.dirname(lake_dir))
            s3key = rel.replace("\\", "/")   # Windows path fix
            files.append((local, s3key))
    return sorted(files)


def upload_all(lake_dir: str, bucket: str, region: str) -> None:
    print("=" * 62)
    print(" Entity Resolution Lakehouse -- S3 Upload")
    print(" Source : {}".format(lake_dir))
    print(" Dest   : s3://{}/gold/".format(bucket))
    print(" Region : {}".format(region))
    print("=" * 62)

    if not os.path.isdir(lake_dir):
        print("ERROR: lakehouse_gold not found at {}.".format(lake_dir))
        print("  Run: python code/business_entity_resolution/src/export_lakehouse.py")
        sys.exit(1)

    # Init boto3
    session = boto3.Session(region_name=region)
    s3 = session.client("s3")

    # Verify credentials early
    try:
        session.client("sts").get_caller_identity()
        print("  AWS credentials OK")
    except NoCredentialsError:
        print("\nERROR: No AWS credentials found.")
        print("  Run:  aws configure")
        print("  Or set: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION")
        sys.exit(1)

    # Create bucket
    print("\n[1/3] Creating bucket (if needed)...")
    create_bucket_if_not_exists(s3, bucket, region)

    # Collect files
    files = collect_files(lake_dir)
    total_bytes = sum(os.path.getsize(p) for p, _ in files)
    print("\n[2/3] Uploading {:,} Parquet files ({})...".format(len(files), sizeof_fmt(total_bytes)))

    uploaded = 0
    uploaded_bytes = 0
    skipped = 0
    t0 = time.time()

    for local, s3key in files:
        fsize = os.path.getsize(local)
        # Check if already uploaded (skip if same size)
        try:
            head = s3.head_object(Bucket=bucket, Key=s3key)
            if head["ContentLength"] == fsize:
                skipped += 1
                continue
        except ClientError as e:
            if e.response["Error"]["Code"] != "404":
                raise

        s3.upload_file(
            local, bucket, s3key,
            ExtraArgs={"ContentType": "application/octet-stream"},
        )
        uploaded += 1
        uploaded_bytes += fsize
        elapsed = time.time() - t0
        speed = uploaded_bytes / elapsed / 1024 / 1024 if elapsed > 0 else 0
        print("  [{:>3}/{:>3}] {} ({}) -- {:.1f} MB/s".format(
            uploaded + skipped, len(files),
            os.path.basename(local), sizeof_fmt(fsize), speed))

    elapsed = time.time() - t0
    print("\n[3/3] Verifying upload (listing gold/ prefix)...")
    resp = s3.list_objects_v2(Bucket=bucket, Prefix="gold/")
    n_objects = resp.get("KeyCount", 0)
    remote_bytes = sum(o["Size"] for o in resp.get("Contents", []))
    print("  Objects in s3://{}/gold/ : {:,}".format(bucket, n_objects))
    print("  Total remote size        : {}".format(sizeof_fmt(remote_bytes)))

    print("\n" + "=" * 62)
    print("UPLOAD COMPLETE")
    print("  Uploaded : {:,} files  ({})".format(uploaded, sizeof_fmt(uploaded_bytes)))
    print("  Skipped  : {:,} files (already current)".format(skipped))
    print("  Time     : {:.0f}s".format(elapsed))
    print()
    print("Next steps:")
    print("  1. Open Athena console:")
    print("     https://ap-southeast-2.console.aws.amazon.com/athena/home?region=ap-southeast-2")
    print("  2. Set results: s3://{}/athena-results/".format(bucket))
    print("  3. Run: sql/athena_warehouse_ddl.sql")
    print("=" * 62)


if __name__ == "__main__":
    upload_all(LAKE_DIR, BUCKET, REGION)

