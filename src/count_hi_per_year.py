"""Country × year table for high-influential green patent family counts.

Prioritises person_ctry_code (inventor country); falls back to appln_auth
(filing office) for applications without person data.  Regional offices
(EP, WO, …) are excluded from the fallback.
"""

import polars as pl
from patent_count_utils import (
    REGIONAL_OFFICES,
    INVALID_COUNTRY_CODES,
    build_complete_grid,
    add_country_3digit,
    inventor_fractional_country_year,
    print_summary,
)

YEAR_RANGE = (1985, 2025)

# ------------------------------------------------------------------
# 1. Load high-influential green patents (application-level rows)
# ------------------------------------------------------------------
print("Loading high-influential green patents ...")
hi = pl.read_parquet("PATSTAT2025FALL/output/high_influence_green_patents.parquet")
print(f"  Rows loaded: {hi.height}  "
      f"({hi['docdb_family_id'].n_unique()} unique families)")

# ------------------------------------------------------------------
# 2. Extract country per application: person_ctry_code → appln_auth
# ------------------------------------------------------------------
base = hi.select(["docdb_family_id", "person_ctry_code", "appln_auth",
                   "earliest_filing_date"]).with_columns(
    pl.col("earliest_filing_date").str.slice(0, 4).cast(pl.Int16).alias("family_year"),
)

with_person = (
    base
    .filter(pl.col("person_ctry_code").is_not_null() & (pl.col("person_ctry_code") != ""))
    .with_columns(pl.col("person_ctry_code").str.split(","))
    .explode("person_ctry_code")
    .with_columns(pl.col("person_ctry_code").str.strip_chars().alias("countries"))
    .filter(~pl.col("countries").is_in(list(INVALID_COUNTRY_CODES)))
    .select(["docdb_family_id", "countries", "family_year"])
)

without_person = (
    base
    .filter(pl.col("person_ctry_code").is_null() | (pl.col("person_ctry_code") == ""))
    .join(pl.read_parquet("PATSTAT2025FALL/output/docdb_family_country.parquet"), on="docdb_family_id", how="inner")
    .filter(
        pl.col("country").is_not_null()
        & (~pl.col("country").is_in(list(INVALID_COUNTRY_CODES)))
        & (~pl.col("country").is_in(list(REGIONAL_OFFICES)))
    )
    .select(["docdb_family_id", pl.col("country").alias("countries"), "family_year"])
)

n_with = base.filter(
    pl.col("person_ctry_code").is_not_null() & (pl.col("person_ctry_code") != "")
).height
print(f"  Country source: person_ctry_code {n_with}/{base.height} apps "
      f"({n_with/base.height*100:.1f}%), appln_auth fallback for the rest")

# ------------------------------------------------------------------
# 3. Deduplicate to (family, country) with earliest year, then count
# ------------------------------------------------------------------
df_country_year = (
    pl.concat([with_person, without_person])
    .group_by(["docdb_family_id", "countries"])
    .agg(pl.col("family_year").min())
    .group_by(["countries", "family_year"])
    .agg(pl.col("docdb_family_id").n_unique().alias("num_hi_patents"))
)

hi_family_year = (
    base
    .group_by("docdb_family_id")
    .agg(pl.col("family_year").min())
)
df_inventor_frac = inventor_fractional_country_year(
    hi_family_year,
    "family_year",
    "inventor_frac_hi_patents",
)

country_year_keys = pl.concat([
    df_country_year.select(["countries", "family_year"]),
    df_inventor_frac.select(["countries", "family_year"]),
]).unique()

df_country_year = (
    country_year_keys
    .join(df_country_year, on=["countries", "family_year"], how="left")
    .join(df_inventor_frac, on=["countries", "family_year"], how="left")
)

# ------------------------------------------------------------------
# 4. Build complete grid and add 3-digit codes
# ------------------------------------------------------------------
df_complete = build_complete_grid(
    df_country_year,
    "countries",
    "family_year",
    ["num_hi_patents", "inventor_frac_hi_patents"],
    YEAR_RANGE,
)
df_complete = add_country_3digit(df_complete, "countries")
df_complete = df_complete.with_columns([
    pl.col("num_hi_patents").alias("whole_count_hi_patents"),
    pl.col("inventor_frac_hi_patents").alias("inventor_fractional_count_hi_patents"),
])

# ------------------------------------------------------------------
# 5. Summary and export
# ------------------------------------------------------------------
print_summary(df_complete, "num_hi_patents", "family_year",
              "High-Influential Green Patents — Country-Year Table")

OUT = "PATSTAT2025FALL/output/hi_patent_country_year_complete.csv"
df_complete.write_csv(OUT)
print(f"\nExported to: {OUT}")
