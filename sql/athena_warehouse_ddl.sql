-- ============================================================
-- athena_warehouse_ddl.sql
-- Amazon Athena DDL for the Business Entity Resolution
-- Serverless Data Lakehouse (S3 + Glue + Athena).
--
-- Before running: set Athena query result location to
--   s3://riya-entity-lakehouse-2026/athena-results/
--
-- Run each statement sequentially in the Athena console.
-- ============================================================

-- 0. Database
CREATE DATABASE IF NOT EXISTS entity_resolution_dw;

-- ============================================================
-- DIMENSION TABLES
-- ============================================================

-- 1. dim_canonical_entity — 1.73M S1 reference entities
CREATE EXTERNAL TABLE IF NOT EXISTS entity_resolution_dw.dim_canonical_entity (
    entity_id   STRING      COMMENT 'Source-1 entity ID (S1-XXXXXXXXX)',
    name_raw    STRING      COMMENT 'Original business name (UTF-8)',
    addr_raw    STRING      COMMENT 'Original business address',
    name_clean  STRING      COMMENT 'Lowercased, punctuation-stripped name',
    name_skel   STRING      COMMENT 'Phonetic consonant skeleton key',
    postal_code STRING      COMMENT 'Extracted 5-6 digit postal code',
    house_num   STRING      COMMENT 'Extracted house / plot number'
)
PARTITIONED BY (country STRING)
STORED AS PARQUET
LOCATION 's3://riya-entity-lakehouse-2026/lakehouse_gold/dim_canonical_entity/'
TBLPROPERTIES (
    'parquet.compression' = 'SNAPPY',
    'classification'      = 'parquet'
);

MSCK REPAIR TABLE entity_resolution_dw.dim_canonical_entity;

-- 2. dim_target_record — 9.97M S2 + S3 vendor records
CREATE EXTERNAL TABLE IF NOT EXISTS entity_resolution_dw.dim_target_record (
    target_id     STRING  COMMENT 'Target entity ID (S2-XXXXXXXXX or S3-XXXXXXXXX)',
    source_system STRING  COMMENT 'S2 or S3',
    name_raw      STRING  COMMENT 'Original vendor business name',
    addr_raw      STRING  COMMENT 'Original vendor address',
    postal_code   STRING  COMMENT 'Extracted 5-6 digit postal code'
)
PARTITIONED BY (country STRING)
STORED AS PARQUET
LOCATION 's3://riya-entity-lakehouse-2026/lakehouse_gold/dim_target_record/'
TBLPROPERTIES (
    'parquet.compression' = 'SNAPPY',
    'classification'      = 'parquet'
);

MSCK REPAIR TABLE entity_resolution_dw.dim_target_record;

-- ============================================================
-- FACT TABLES
-- ============================================================

-- 3. fact_entity_summary — per-S1 blocking telemetry & match counts
CREATE EXTERNAL TABLE IF NOT EXISTS entity_resolution_dw.fact_entity_summary (
    entity_id            STRING   COMMENT 'Source-1 entity ID',
    matched_target_count BIGINT   COMMENT 'Number of S2/S3 records resolved to this entity',
    is_singleton         BOOLEAN  COMMENT 'TRUE if entity matched zero target records',
    model_version        STRING   COMMENT 'Model that produced this prediction',
    candidate_count      BIGINT   COMMENT 'Blocking candidate pairs generated for this entity'
)
PARTITIONED BY (country STRING)
STORED AS PARQUET
LOCATION 's3://riya-entity-lakehouse-2026/lakehouse_gold/fact_entity_summary/'
TBLPROPERTIES (
    'parquet.compression' = 'SNAPPY',
    'classification'      = 'parquet'
);

MSCK REPAIR TABLE entity_resolution_dw.fact_entity_summary;

-- 4. fact_resolution_links — granular 1-to-many bridge table
CREATE EXTERNAL TABLE IF NOT EXISTS entity_resolution_dw.fact_resolution_links (
    entity_id     STRING  COMMENT 'Source-1 entity ID',
    target_id     STRING  COMMENT 'Resolved S2 or S3 target ID',
    source_system STRING  COMMENT 'S2 or S3',
    model_version STRING  COMMENT 'Model that produced this link'
)
PARTITIONED BY (country STRING)
STORED AS PARQUET
LOCATION 's3://riya-entity-lakehouse-2026/lakehouse_gold/fact_resolution_links/'
TBLPROPERTIES (
    'parquet.compression' = 'SNAPPY',
    'classification'      = 'parquet'
);

MSCK REPAIR TABLE entity_resolution_dw.fact_resolution_links;

-- ============================================================
-- VERIFICATION QUERIES (save results / screenshots)
-- ============================================================

-- Query 1: Data Quality Gate
-- Verifies ZERO mutual-exclusivity violations across all 5.8M match links.
-- Each S2/S3 target must appear in at most one S1 entity's match set.
-- Expected: mutual_exclusivity_violations = 0 for every country.
SELECT
    country,
    COUNT(target_id)                              AS total_assigned_targets,
    COUNT(DISTINCT target_id)                     AS unique_assigned_targets,
    COUNT(target_id) - COUNT(DISTINCT target_id)  AS mutual_exclusivity_violations
FROM entity_resolution_dw.fact_resolution_links
GROUP BY country
ORDER BY country;

-- Query 2: Executive Warehouse Rollup
-- Blocking reduction + singleton distribution per country & model.
SELECT
    country,
    model_version,
    COUNT(entity_id)                                                      AS total_s1_entities,
    SUM(candidate_count)                                                  AS total_blocking_pairs,
    ROUND(AVG(CAST(candidate_count AS DOUBLE)), 2)                        AS avg_candidates_per_entity,
    SUM(matched_target_count)                                             AS total_resolved_matches,
    ROUND(
        100.0 * SUM(CASE WHEN is_singleton THEN 1 ELSE 0 END)
              / NULLIF(COUNT(entity_id), 0),
        2
    )                                                                     AS singleton_rate_pct
FROM entity_resolution_dw.fact_entity_summary
GROUP BY country, model_version
ORDER BY total_s1_entities DESC;

-- Query 3: Search-Space Reduction Audit
-- Validates the 99.9993% blocking reduction claim.
-- Total target records in dim_target_record = 9,969,589
-- Cartesian space = 1,732,544 x 9,969,589 = ~17.27 Trillion pairs
SELECT
    'Cartesian Space'                                           AS metric,
    CAST(1732544 AS BIGINT) * CAST(9969589 AS BIGINT)           AS value
UNION ALL
SELECT
    'Blocking Candidates (SUM)',
    SUM(candidate_count)
FROM entity_resolution_dw.fact_entity_summary
UNION ALL
SELECT
    'Reduction %',
    CAST(
        ROUND(
            100.0 * (1.0 - SUM(candidate_count)
                      / (CAST(1732544 AS DOUBLE) * CAST(9969589 AS DOUBLE))),
            6
        ) AS BIGINT
    )
FROM entity_resolution_dw.fact_entity_summary;

-- Query 4: Match Distribution by Source System
SELECT
    country,
    source_system,
    COUNT(*) AS resolved_links
FROM entity_resolution_dw.fact_resolution_links
GROUP BY country, source_system
ORDER BY country, source_system;
