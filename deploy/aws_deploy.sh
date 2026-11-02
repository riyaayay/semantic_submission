#!/usr/bin/env bash
# =============================================================================
# deploy/aws_deploy.sh
# One-shot AWS Serverless Lakehouse Deployment
# Amazon S3 + AWS Glue Data Catalog + Amazon Athena (Free Tier)
#
# Prerequisites:
#   1. AWS CLI installed: https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html
#   2. Configured: aws configure  (region: ap-southeast-2, Mumbai)
#   3. ETL run: python src/export_lakehouse.py
#
# Usage (from repo root):
#   bash deploy/aws_deploy.sh
# =============================================================================

set -euo pipefail

BUCKET="riya-entity-lakehouse-2026"
REGION="ap-southeast-2"
LAKE_DIR="../../output/lakehouse_gold"
ATHENA_RESULTS="s3://${BUCKET}/athena-results/"

echo "============================================================"
echo " Entity Resolution Lakehouse -- AWS Deployment"
echo " Bucket : s3://${BUCKET}"
echo " Region : ${REGION}"
echo "============================================================"

# Step 1: Create S3 bucket (idempotent)
echo ""
echo "[1/3] Creating S3 bucket (if not exists)..."
aws s3 mb "s3://${BUCKET}" --region "${REGION}" 2>/dev/null || \
  echo "  Bucket already exists, continuing."

# Enable versioning for data integrity
aws s3api put-bucket-versioning \
  --bucket "${BUCKET}" \
  --versioning-configuration Status=Enabled 2>/dev/null || true

# Step 2: Sync Parquet Star Schema to Gold zone
echo ""
echo "[2/3] Syncing lakehouse_gold/ to S3 Gold zone..."
echo "  Source : ${LAKE_DIR}"
echo "  Dest   : s3://${BUCKET}/gold/"

aws s3 sync "${LAKE_DIR}" "s3://${BUCKET}/gold/" \
  --exclude "*.DS_Store" \
  --exclude "__pycache__/*" \
  --no-progress

echo "  Sync complete."

# Step 3: Verify upload
echo ""
echo "[3/3] Verifying upload..."
aws s3 ls "s3://${BUCKET}/gold/" --recursive --human-readable --summarize | tail -5

echo ""
echo "============================================================"
echo "DEPLOYMENT COMPLETE"
echo ""
echo "Next steps:"
echo "  1. Open AWS Athena console:"
echo "     https://ap-southeast-2.console.aws.amazon.com/athena/home"
echo "  2. Set query results location to: ${ATHENA_RESULTS}"
echo "  3. Run: sql/athena_warehouse_ddl.sql (create tables + MSCK REPAIR)"
echo "  4. Run verification queries (save screenshots for README)"
echo "============================================================"
