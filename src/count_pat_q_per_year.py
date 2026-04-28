"""Country × year table for OECD Patent Quality Index.

Reads family-level quality data produced by patent_quality.py and aggregates
to a complete country-year panel, zero-filling missing entries.
Two quality index variants are included:
  - Hall et al. (2001) generality
  - OECD modified-HHI generality
"""

import polars as pl
from patent_count_utils import (
    build_complete_grid,
    add_country_3digit,
    print_summary,
)

YEAR_RANGE = (1985, 2025)

# ------------------------------------------------------------------
# 1. Load family-level quality data
# ------------------------------------------------------------------
print("Loading family-level quality data ...")
df_family = pl.read_parquet("PATSTAT2025FALL/output/patent_quality_family.parquet")
print(f"  Families loaded: {df_family.height}")

# ------------------------------------------------------------------
# 2. Explode countries and aggregate at country-year level
# ------------------------------------------------------------------
df_country_patents = (
    df_family.explode("countries")
    .filter(pl.col("countries").is_not_null() & (pl.col("countries") != ""))
)

VALUE_COLS = [
    "num_patents",
    "avg_forward_cites_5yr", "avg_family_size", "avg_claims",
    "avg_generality", "avg_generality_oecd",
    "avg_norm_cites", "avg_norm_family", "avg_norm_claims",
    "avg_norm_generality", "avg_norm_generality_oecd",
    "country_pqi_hall", "country_pqi_hall_median",
    "country_pqi_oecd", "country_pqi_oecd_median",
    "total_citations_5yr",
    "quality_std_hall", "quality_std_oecd",
]

df_country_year = (
    df_country_patents
    .group_by(["countries", "family_year"])
    .agg([
        pl.len().alias("num_patents"),

        # Raw indicators
        pl.col("fwd_citations_5yr").mean().alias("avg_forward_cites_5yr"),
        pl.col("family_size").mean().alias("avg_family_size"),
        pl.col("claims").mean().alias("avg_claims"),
        pl.col("generality").mean().alias("avg_generality"),
        pl.col("generality_oecd").mean().alias("avg_generality_oecd"),

        # Normalised indicators
        pl.col("norm_forward_cites").mean().alias("avg_norm_cites"),
        pl.col("norm_family_size").mean().alias("avg_norm_family"),
        pl.col("norm_claims").mean().alias("avg_norm_claims"),
        pl.col("norm_generality").mean().alias("avg_norm_generality"),
        pl.col("norm_generality_oecd").mean().alias("avg_norm_generality_oecd"),

        # Quality indices (Hall generality)
        pl.col("patent_quality_index_4").mean().alias("country_pqi_hall"),
        pl.col("patent_quality_index_4").median().alias("country_pqi_hall_median"),

        # Quality indices (OECD generality)
        pl.col("patent_quality_index_4_oecd").mean().alias("country_pqi_oecd"),
        pl.col("patent_quality_index_4_oecd").median().alias("country_pqi_oecd_median"),

        pl.col("fwd_citations_5yr").sum().alias("total_citations_5yr"),
        pl.col("patent_quality_index_4").std().alias("quality_std_hall"),
        pl.col("patent_quality_index_4_oecd").std().alias("quality_std_oecd"),
    ])
)

# ------------------------------------------------------------------
# 3. Build complete grid, add 3-digit codes, rank
# ------------------------------------------------------------------
df_complete = build_complete_grid(
    df_country_year, "countries", "family_year", VALUE_COLS, YEAR_RANGE,
    zero_fill_cols=["num_patents", "total_citations_5yr"],
)
df_complete = add_country_3digit(df_complete, "countries")

df_complete = df_complete.with_columns([
    pl.when(pl.col("country_pqi_hall") > 0)
    .then(
        pl.col("country_pqi_hall")
        .rank(method="ordinal", descending=True)
        .over("family_year")
    )
    .otherwise(None)
    .alias("quality_rank_hall"),

    pl.when(pl.col("country_pqi_oecd") > 0)
    .then(
        pl.col("country_pqi_oecd")
        .rank(method="ordinal", descending=True)
        .over("family_year")
    )
    .otherwise(None)
    .alias("quality_rank_oecd"),
])

# ------------------------------------------------------------------
# 4. Summary and export
# ------------------------------------------------------------------
print_summary(df_complete, "num_patents", "family_year",
              "OECD Patent Quality Index — Country-Year Table")

OUT = "PATSTAT2025FALL/output/oecd_patent_quality_country_year_complete.csv"
df_complete.write_csv(OUT)
print(f"\nExported to: {OUT}")
