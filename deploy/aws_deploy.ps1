# deploy/aws_deploy.ps1
# =============================================================================
# One-shot AWS Serverless Lakehouse Deployment (PowerShell for Windows)
# Amazon S3 + AWS Glue Data Catalog + Amazon Athena (Free Tier)
#
# Prerequisites:
#   1. AWS CLI v2: https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html
#   2. Run: aws configure  (use ap-southeast-2 region)
#   3. ETL: python src/export_lakehouse.py  (generates output/lakehouse_gold/)
#
# Usage (from repo root, in PowerShell):
#   .\deploy\aws_deploy.ps1
# =============================================================================

$ErrorActionPreference = "Stop"

$BUCKET   = "riya-entity-lakehouse-2026"
$REGION   = "ap-southeast-2"

# Resolve lakehouse_gold relative to this script's repo root
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$REPO_ROOT  = Split-Path -Parent $SCRIPT_DIR
$LAKE_DIR   = Join-Path $REPO_ROOT "output\lakehouse_gold"

Write-Host "============================================================"
Write-Host " Entity Resolution Lakehouse -- AWS Deployment (PowerShell)"
Write-Host " Bucket : s3://$BUCKET"
Write-Host " Region : $REGION"
Write-Host " Source : $LAKE_DIR"
Write-Host "============================================================"

# Validate lakehouse_gold exists
if (-not (Test-Path $LAKE_DIR)) {
    Write-Error "lakehouse_gold not found at $LAKE_DIR. Run 'python src/export_lakehouse.py' first."
    exit 1
}

# Step 1: Create S3 bucket
Write-Host ""
Write-Host "[1/3] Creating S3 bucket (if not exists)..."
try {
    aws s3 mb "s3://$BUCKET" --region $REGION 2>$null
} catch {
    Write-Host "  Bucket already exists, continuing."
}

# Step 2: Sync Parquet to S3
Write-Host ""
Write-Host "[2/3] Syncing Parquet Star Schema to S3 Gold zone..."
aws s3 sync $LAKE_DIR "s3://$BUCKET/gold/" --exclude "*.DS_Store"
Write-Host "  Sync complete."

# Step 3: Verify
Write-Host ""
Write-Host "[3/3] Verifying upload..."
aws s3 ls "s3://$BUCKET/gold/" --recursive --human-readable --summarize | Select-Object -Last 5

Write-Host ""
Write-Host "============================================================"
Write-Host "DEPLOYMENT COMPLETE"
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Open Athena: https://ap-southeast-2.console.aws.amazon.com/athena/home"
Write-Host "  2. Set results bucket: s3://$BUCKET/athena-results/"
Write-Host "  3. Run: sql/athena_warehouse_ddl.sql"
Write-Host "  4. Run verification queries + screenshot results"
Write-Host "============================================================"
