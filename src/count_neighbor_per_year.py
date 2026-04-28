"""Country × year table for non-green neighbor patent family counts.

Prioritises person_ctry_code (inventor country) via the join chain
  neighbor_families → neighbor_index → neighbor_persons_agg
Falls back to appln_auth (filing office) from neighbor_families for
families without person data.  Regional offices (EP, WO, …) excluded.
"""

import polars as pl
from patent_count_utils import (
    REGIONAL_OFFICES,
    INVALID_COUNTRY_CODES,
    build_complete_grid,
    add_country_3digit,
    print_summary,
)

YEAR_RANGE = (1985, 2025)

# ------------------------------------------------------------------
# 1. Load data sources
# ------------------------------------------------------------------
print("Loading neighbor patent families ...")
neighbor = pl.read_parquet("PATSTAT2025FALL/output/neighbor_families.parquet").select(
    ["docdb_family_id", "year", "appln_auth"]
)
print(f"  Families loaded: {neighbor.height}")

print("Loading neighbor_index (docdb_family_id → appln_id) ...")
neighbor_index = pl.read_parquet("PATSTAT2025FALL/output/neighbor_index.parquet")

print("Loading neighbor_persons_agg (appln_id → person_ctry_code) ...")
persons = pl.read_parquet("PATSTAT2025FALL/output/neighbor_persons_agg.parquet")

# ------------------------------------------------------------------
# 2. person_ctry_code per family (primary source)
# ------------------------------------------------------------------
family_person_ctry = (
    neighbor.select("docdb_family_id").unique()
    .join(neighbor_index, on="docdb_family_id", how="left")
    .join(persons, on="appln_id", how="left")
    .filter(pl.col("person_ctry_code").is_not_null() & (pl.col("person_ctry_code") != ""))
    .with_columns(pl.col("person_ctry_code").str.split(","))
    .explode("person_ctry_code")
    .with_columns(pl.col("person_ctry_code").str.strip_chars().alias("country"))
    .filter(~pl.col("country").is_in(list(INVALID_COUNTRY_CODES)))
    .select(["docdb_family_id", "country"])
    .unique()
)

families_with_person = family_person_ctry.select("docdb_family_id").unique()
n_person = families_with_person.height

# ------------------------------------------------------------------
# 3. appln_auth fallback for families WITHOUT person data
# ------------------------------------------------------------------
family_auth = (
    neighbor
    .join(families_with_person, on="docdb_family_id", how="anti")
    .select(["docdb_family_id", "appln_auth"])
    .with_columns(pl.col("appln_auth").str.split(","))
    .explode("appln_auth")
    .with_columns(pl.col("appln_auth").str.strip_chars().alias("country"))
    .filter(
        pl.col("country").is_not_null()
        & (~pl.col("country").is_in(list(INVALID_COUNTRY_CODES)))
        & (~pl.col("country").is_in(list(REGIONAL_OFFICES)))
    )
    .select(["docdb_family_id", "country"])
    .unique()
)

n_fallback = family_auth.select("docdb_family_id").unique().height
print(f"  Country source: person_ctry_code {n_person}/{neighbor.height} families "
      f"({n_person/neighbor.height*100:.1f}%), "
      f"appln_auth fallback {n_fallback}")

# ------------------------------------------------------------------
# 4. Combine and count per (country, year)
# ------------------------------------------------------------------
family_country = pl.concat([family_person_ctry, family_auth]).unique()

df_country_year = (
    neighbor.select(["docdb_family_id", "year"])
    .join(family_country, on="docdb_family_id", how="inner")
    .rename({"country": "countries"})
    .group_by(["countries", "year"])
    .agg(pl.col("docdb_family_id").n_unique().alias("num_neighbor_patents"))
)

# ------------------------------------------------------------------
# 5. Build complete grid and add 3-digit codes
# ------------------------------------------------------------------
df_complete = build_complete_grid(
    df_country_year, "countries", "year", ["num_neighbor_patents"], YEAR_RANGE
)
df_complete = add_country_3digit(df_complete, "countries")

# ------------------------------------------------------------------
# 6. Summary and export
# ------------------------------------------------------------------
print_summary(df_complete, "num_neighbor_patents", "year",
              "Neighbor Patent Families — Country-Year Table")

OUT = "PATSTAT2025FALL/output/neighbor_patent_country_year_complete.csv"
df_complete.write_csv(OUT)
print(f"\nExported to: {OUT}")
