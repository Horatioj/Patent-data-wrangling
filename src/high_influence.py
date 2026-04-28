import polars as pl
import gc

# ============================================================================
# High-influential green patents (DOCDB family level)
#   Criterion 1: forward citations (5-year window) >= 5
#   Criterion 2: TOP 10 most-cited in each (sector, year) group
#   Criterion 3: triadic patent families (filed in JP + EP + US)
# ============================================================================
# Note: because patents in 2022 and 2021 are small, we may not be able to find enough high-influence patents. So in 2021 and 2022, we don't consider CRITERION 2.
# --- Step 1: Build a docdb_family_id → earliest_year lookup from FULL tls201 ---
# We need this for ALL families (not just green) because citing families can be any patent.
tls201_files = [
    "Z:/PATSTAT Global 2025 Autumn/tls201_appln_part01.csv",
    "Z:/PATSTAT Global 2025 Autumn/tls201_appln_part02.csv",
    "Z:/PATSTAT Global 2025 Autumn/tls201_appln_part03.csv",
]

(
    pl.concat(
        [pl.scan_csv(f, schema_overrides={"docdb_family_id": pl.Int32})
         .select(["docdb_family_id", "earliest_filing_date"])
         for f in tls201_files],
        rechunk=False,
    )
    .with_columns(
        pl.col("earliest_filing_date").str.slice(0, 4).cast(pl.Int16).alias("family_year")
    )
    .group_by("docdb_family_id")
    .agg(pl.col("family_year").min())
    .sink_parquet("PATSTAT2025FALL/output/docdb_family_year.parquet", compression="zstd", engine="streaming")
)
gc.collect()

# --- Step 2: Load green patent families with year and sector ---
green_patents = pl.read_parquet("PATSTAT2025FALL/output/green_patent8526.parquet", columns=[
    "docdb_family_id", "earliest_filing_date", "sector",
])

green_families = (
    green_patents
    .with_columns(
        pl.col("earliest_filing_date").str.slice(0, 4).cast(pl.Int16).alias("cited_year")
    )
    .group_by("docdb_family_id")
    .agg([
        pl.col("cited_year").min(),
        pl.col("sector").first(),
    ])
)
del green_patents
gc.collect()

# --- Step 3: Filter tls228 to citations of green families, compute 5-year window ---
citn_fam = pl.scan_csv(
    "Z:/PATSTAT Global 2025 Autumn/tls228_docdb_fam_citn_part01.csv",
    schema_overrides={"docdb_family_id": pl.Int32, "cited_docdb_family_id": pl.Int32},
)

green_fam_ids = green_families.select("docdb_family_id").rename(
    {"docdb_family_id": "cited_docdb_family_id"}
)

family_year_lookup = pl.scan_parquet("PATSTAT2025FALL/output/docdb_family_year.parquet")

# Join: citations → cited family year (from green) + citing family year (from lookup)
citations_windowed = (
    citn_fam
    .join(
        green_fam_ids.lazy(),
        on="cited_docdb_family_id",
        how="semi",
    )
    .join(
        green_families
        .select(["docdb_family_id", "cited_year", "sector"])
        .rename({"docdb_family_id": "cited_docdb_family_id"})
        .lazy(),
        on="cited_docdb_family_id",
        how="inner",
    )
    .join(
        family_year_lookup.rename({"family_year": "citing_year"}),
        on="docdb_family_id",
        how="inner",
    )
    .filter(
        (pl.col("citing_year") >= pl.col("cited_year"))
        & (pl.col("citing_year") <= pl.col("cited_year") + 5)
    )
)

fwd_citations_5yr = (
    citations_windowed
    .group_by("cited_docdb_family_id")
    .agg([
        pl.len().alias("fwd_citations_5yr"),
        pl.col("cited_year").first(),
        pl.col("sector").first(),
    ])
    .collect(engine="streaming")
)
gc.collect()

print(f"Families with any 5-year forward citation: {fwd_citations_5yr.height}")

# --- Step 4: Criterion 1 — forward citations (5-year window) >= 5 ---
crit1 = fwd_citations_5yr.filter(pl.col("fwd_citations_5yr") >= 5)
print(f"Criterion 1 (fwd_citations_5yr >= 5): {crit1.height} families")

# --- Step 5: Criterion 2 — TOP 10 most-cited per (sector, year) ---
# Exclude 2021-2022: too few patents with enough citation accumulation time
fwd_exploded = (
    fwd_citations_5yr
    .filter(pl.col("cited_year") <= 2023)
    .with_columns(pl.col("sector").str.split(","))
    .explode("sector")
    .filter(pl.col("sector").is_not_null() & (pl.col("sector") != ""))
)

crit2 = (
    fwd_exploded
    .sort("fwd_citations_5yr", descending=True)
    .group_by(["sector", "cited_year"])
    .head(10)
)
print(f"Criterion 2 (top-10 per sector-year, excl. 2024-2025): {crit2.height} entries "
      f"({crit2['cited_docdb_family_id'].n_unique()} unique families)")

# --- Step 6: Criterion 3 — triadic patent families (filed in JP + EP + US) ---
green_full = pl.read_parquet("PATSTAT2025FALL/output/green_patent8526.parquet")

triadic_families = (
    green_full
    .filter(pl.col("appln_auth").is_in(["JP", "EP", "US"]))
    .group_by("docdb_family_id")
    .agg(pl.col("appln_auth").unique().alias("auth_set"))
    .filter(pl.col("auth_set").list.len() >= 3)
    .select("docdb_family_id")
)
print(f"Criterion 3 (triadic families JP+EP+US): {triadic_families.height} families")

# --- Step 7: Union all three criteria ---
high_influence_ids = pl.concat([
    crit1.select("cited_docdb_family_id").rename({"cited_docdb_family_id": "docdb_family_id"}),
    crit2.select("cited_docdb_family_id").rename({"cited_docdb_family_id": "docdb_family_id"}),
    triadic_families.select("docdb_family_id"),
]).unique()

print(f"Total high-influential families (union of 3 criteria): {high_influence_ids.height}")

# --- Step 8: Join back to green patent data to get full records ---
high_influence_patents = green_full.join(
    high_influence_ids, on="docdb_family_id", how="semi"
)

# Attach 5-year citation count
high_influence_patents = high_influence_patents.join(
    fwd_citations_5yr.select(["cited_docdb_family_id", "fwd_citations_5yr"])
    .rename({"cited_docdb_family_id": "docdb_family_id"}),
    on="docdb_family_id",
    how="left",
)

# Tag which criteria each family meets
high_influence_patents = high_influence_patents.with_columns([
    pl.col("docdb_family_id").is_in(
        crit1["cited_docdb_family_id"].rename("docdb_family_id")
    ).alias("meets_crit1_citations"),
    pl.col("docdb_family_id").is_in(
        crit2["cited_docdb_family_id"].unique().rename("docdb_family_id")
    ).alias("meets_crit2_top10"),
    pl.col("docdb_family_id").is_in(
        triadic_families["docdb_family_id"]
    ).alias("meets_crit3_triadic"),
])

print(f"High-influential green patents: {high_influence_patents.height} rows "
      f"({high_influence_patents['docdb_family_id'].n_unique()} families)")

high_influence_patents.write_parquet("PATSTAT2025FALL/output/high_influence_green_patents.parquet", compression="zstd")

# --- Step 9: Save 4 index files (appln_id, docdb_family_id) ---
index_cols = ["appln_id", "appln_auth", "receiving_office", "docdb_family_id", "appln_nr_original", "appln_title", "earliest_filing_date"]

# Index for each criterion
crit1_ids = set(crit1["cited_docdb_family_id"].to_list())
crit2_ids = set(crit2["cited_docdb_family_id"].unique().to_list())
crit3_ids = set(triadic_families["docdb_family_id"].to_list())

idx_crit1 = green_full.filter(
    pl.col("docdb_family_id").is_in(crit1_ids)
).select(index_cols).unique()
idx_crit1.write_parquet("PATSTAT2025FALL/output/hi_index_crit1_citations.parquet", compression="zstd")
print(f"Saved hi_index_crit1_citations.parquet: {idx_crit1.height} rows "
      f"({idx_crit1['docdb_family_id'].n_unique()} families)")

idx_crit2 = green_full.filter(
    pl.col("docdb_family_id").is_in(crit2_ids)
).select(index_cols).unique()
idx_crit2.write_parquet("PATSTAT2025FALL/output/hi_index_crit2_top10.parquet", compression="zstd")
print(f"Saved hi_index_crit2_top10.parquet: {idx_crit2.height} rows "
      f"({idx_crit2['docdb_family_id'].n_unique()} families)")

idx_crit3 = green_full.filter(
    pl.col("docdb_family_id").is_in(crit3_ids)
).select(index_cols).unique()
idx_crit3.write_parquet("PATSTAT2025FALL/output/hi_index_crit3_triadic.parquet", compression="zstd")
print(f"Saved hi_index_crit3_triadic.parquet: {idx_crit3.height} rows "
      f"({idx_crit3['docdb_family_id'].n_unique()} families)")

# Families satisfying ALL three criteria
all_three_ids = crit1_ids & crit2_ids & crit3_ids
idx_all_three = green_full.filter(
    pl.col("docdb_family_id").is_in(all_three_ids)
).select(index_cols).unique()
idx_all_three.write_parquet("PATSTAT2025FALL/output/hi_index_all_three_criteria.parquet", compression="zstd")
print(f"Saved hi_index_all_three_criteria.parquet: {idx_all_three.height} rows "
      f"({idx_all_three['docdb_family_id'].n_unique()} families)")
print("satify all three criteria: ", idx_all_three.head(5))
# ============================================================================
# Descriptive statistics → Excel
# ============================================================================

# --- Total counts at DOCDB family level ---
all_families = pl.read_parquet("PATSTAT2025FALL/output/tls201.parquet", columns=["docdb_family_id"])
n_total_families = all_families["docdb_family_id"].n_unique()
del all_families

n_green_families = green_full["docdb_family_id"].n_unique()
n_high_influence_families = high_influence_patents["docdb_family_id"].n_unique()

n_crit1 = crit1.height
n_crit2_unique = crit2["cited_docdb_family_id"].n_unique()
n_crit3 = triadic_families.height

# Overlap between criteria
crit1_set = set(crit1["cited_docdb_family_id"].to_list())
crit2_set = set(crit2["cited_docdb_family_id"].unique().to_list())
crit3_set = set(triadic_families["docdb_family_id"].to_list())
n_crit1_and_2 = len(crit1_set & crit2_set)
n_crit1_and_3 = len(crit1_set & crit3_set)
n_crit2_and_3 = len(crit2_set & crit3_set)
n_all_three = len(crit1_set & crit2_set & crit3_set)

summary_rows = [
    {"Metric": "Total patent families (granted, family_size≥2)", "Value": n_total_families},
    {"Metric": "Green patent families", "Value": n_green_families},
    {"Metric": "High-influential green patent families (union)", "Value": n_high_influence_families},
    {"Metric": "  Crit 1: fwd citations (5yr) ≥ 5", "Value": n_crit1},
    {"Metric": "  Crit 2: top-10 per sector-year", "Value": n_crit2_unique},
    {"Metric": "  Crit 3: triadic (JP+EP+US)", "Value": n_crit3},
    {"Metric": "  Overlap: Crit 1 ∩ Crit 2", "Value": n_crit1_and_2},
    {"Metric": "  Overlap: Crit 1 ∩ Crit 3", "Value": n_crit1_and_3},
    {"Metric": "  Overlap: Crit 2 ∩ Crit 3", "Value": n_crit2_and_3},
    {"Metric": "  Overlap: Crit 1 ∩ Crit 2 ∩ Crit 3", "Value": n_all_three},
    {"Metric": "Green share of total families (%)", "Value": round(n_green_families / n_total_families * 100, 2)},
    {"Metric": "High-influence share of green (%)", "Value": round(n_high_influence_families / n_green_families * 100, 2)},
]
df_summary = pl.DataFrame(summary_rows)

# --- By sector breakdown ---
hi_exploded = (
    high_influence_patents
    .select(["docdb_family_id", "sector"])
    .unique()
    .with_columns(pl.col("sector").str.split(","))
    .explode("sector")
    .filter(pl.col("sector").is_not_null() & (pl.col("sector") != ""))
)
df_by_sector = (
    hi_exploded
    .group_by("sector")
    .agg(pl.col("docdb_family_id").n_unique().alias("n_families"))
    .sort("n_families", descending=True)
)

# --- By year breakdown ---
hi_year = (
    high_influence_patents
    .with_columns(
        pl.col("earliest_filing_date").str.slice(0, 4).cast(pl.Int16).alias("year")
    )
    .group_by("year")
    .agg(pl.col("docdb_family_id").n_unique().alias("n_families"))
    .sort("year")
)

# --- Top countries by high-influential families ---
hi_countries = (
    high_influence_patents
    .select(["docdb_family_id", "person_ctry_code"])
    .unique()
    .with_columns(pl.col("person_ctry_code").str.split(","))
    .explode("person_ctry_code")
    .filter(pl.col("person_ctry_code").is_not_null() & (pl.col("person_ctry_code") != ""))
    .group_by("person_ctry_code")
    .agg(pl.col("docdb_family_id").n_unique().alias("n_families"))
    .sort("n_families", descending=True)
    .head(30)
)

# --- Citation distribution of high-influential families ---
hi_fam_stats = (
    high_influence_patents
    .group_by("docdb_family_id")
    .agg([
        pl.col("fwd_citations_5yr").first(),
        pl.col("docdb_family_size").first(),
        pl.col("nb_inventors").first(),
        pl.col("nb_applicants").first(),
    ])
)
citation_stats = hi_fam_stats.select([
    pl.col("fwd_citations_5yr").fill_null(0).mean().alias("mean"),
    pl.col("fwd_citations_5yr").fill_null(0).median().alias("median"),
    pl.col("fwd_citations_5yr").fill_null(0).std().alias("std"),
    pl.col("fwd_citations_5yr").fill_null(0).min().alias("min"),
    pl.col("fwd_citations_5yr").fill_null(0).max().alias("max"),
    pl.col("fwd_citations_5yr").fill_null(0).quantile(0.25).alias("p25"),
    pl.col("fwd_citations_5yr").fill_null(0).quantile(0.75).alias("p75"),
    pl.col("fwd_citations_5yr").fill_null(0).quantile(0.90).alias("p90"),
    pl.col("fwd_citations_5yr").fill_null(0).quantile(0.99).alias("p99"),
]).unpivot(variable_name="statistic", value_name="fwd_citations_5yr")

family_size_stats = hi_fam_stats.select([
    pl.col("docdb_family_size").cast(pl.Float64).mean().alias("mean"),
    pl.col("docdb_family_size").cast(pl.Float64).median().alias("median"),
    pl.col("docdb_family_size").cast(pl.Float64).max().alias("max"),
]).unpivot(variable_name="statistic", value_name="docdb_family_size")

# --- Mitigation vs adaptation breakdown ---
df_mit_adapt = (
    high_influence_patents
    .select(["docdb_family_id", "mitigation_adaptation"])
    .unique()
    .group_by("mitigation_adaptation")
    .agg(pl.col("docdb_family_id").n_unique().alias("n_families"))
    .sort("n_families", descending=True)
)

# --- Write all sheets to Excel ---
import pandas as pd

with pd.ExcelWriter("PATSTAT2025FALL/output/high_influence_statistics.xlsx", engine="openpyxl") as writer:
    df_summary.to_pandas().to_excel(writer, sheet_name="Summary", index=False)
    df_by_sector.to_pandas().to_excel(writer, sheet_name="By Sector", index=False)
    hi_year.to_pandas().to_excel(writer, sheet_name="By Year", index=False)
    hi_countries.to_pandas().to_excel(writer, sheet_name="Top Countries", index=False)
    citation_stats.to_pandas().to_excel(writer, sheet_name="Citation Distribution", index=False)
    family_size_stats.to_pandas().to_excel(writer, sheet_name="Family Size", index=False)
    df_mit_adapt.to_pandas().to_excel(writer, sheet_name="Mitigation vs Adaptation", index=False)

print(f"\nStatistics saved to high_influence_statistics.xlsx")
del green_full, high_influence_patents
gc.collect()
