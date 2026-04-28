import polars as pl
import pandas as pd
import gc

# ============================================================================
# Citation network: categorise patent families into 4 groups
#   H = High-influential green patents
#   G = Green patents (not high-influential)
#   N = Non-green neighboring patents (cite or are cited by green families)
#   O = Other patents (no citation link to green families)
# All analysis at DOCDB family level using tls228.
# ============================================================================

# --- Step 1: Load green and high-influential family IDs ---
green_full = pl.read_parquet("PATSTAT2025FALL/output/green_patent8526.parquet")
green_fam_set = set(
    green_full.select("docdb_family_id").unique()["docdb_family_id"].to_list()
)

hi_fam_set = set(
    pl.read_parquet("PATSTAT2025FALL/output/high_influence_green_patents.parquet", columns=["docdb_family_id"])
    ["docdb_family_id"].unique().to_list()
)

print(f"Green families: {len(green_fam_set)}")
print(f"High-influential green families: {len(hi_fam_set)}")

# --- Step 2: Load tls228 (family-level citations) ---
citn_fam = pl.scan_csv(
    "Z:/PATSTAT Global 2025 Autumn/tls228_docdb_fam_citn_part01.csv",
    schema_overrides={"docdb_family_id": pl.Int32, "cited_docdb_family_id": pl.Int32},
)

# --- Step 3: Find all citation edges involving green families ---
# Case A: green family IS CITED by another family (forward citations)
#   docdb_family_id → cited_docdb_family_id, where cited_docdb_family_id ∈ green
fwd_edges = (
    citn_fam
    .filter(pl.col("cited_docdb_family_id").is_in(list(green_fam_set)))
    .select(["docdb_family_id", "cited_docdb_family_id"])
    .collect(engine="streaming")
)
print(f"Forward citation edges (citing → green): {fwd_edges.height}")

# Case B: green family CITES another family (backward citations)
#   docdb_family_id → cited_docdb_family_id, where docdb_family_id ∈ green
bwd_edges = (
    citn_fam
    .filter(pl.col("docdb_family_id").is_in(list(green_fam_set)))
    .select(["docdb_family_id", "cited_docdb_family_id"])
    .collect(engine="streaming")
)
print(f"Backward citation edges (green → cited): {bwd_edges.height}")
gc.collect()

# --- Step 4: Identify non-green neighboring families (N) ---
# Neighbors = families that cite or are cited by green families, but are NOT green themselves
citing_families = set(fwd_edges["docdb_family_id"].to_list())
cited_by_green = set(bwd_edges["cited_docdb_family_id"].to_list())

all_neighbor_candidates = (citing_families | cited_by_green) - green_fam_set
neighbor_fam_set = all_neighbor_candidates

print(f"Non-green neighboring families (N): {len(neighbor_fam_set)}")

# --- Step 5: Build the full edge list with category tags ---
all_edges = pl.concat([
    fwd_edges.with_columns(pl.lit("forward").alias("direction")),
    bwd_edges.with_columns(pl.lit("backward").alias("direction")),
]).unique(subset=["docdb_family_id", "cited_docdb_family_id", "direction"])

del fwd_edges, bwd_edges
gc.collect()

def classify_family(fam_id: int) -> str:
    if fam_id in hi_fam_set:
        return "H"
    elif fam_id in green_fam_set:
        return "G"
    elif fam_id in neighbor_fam_set:
        return "N"
    else:
        return "O"

# Tag both sides of each edge
all_edges = all_edges.with_columns([
    pl.col("docdb_family_id").map_elements(classify_family, return_dtype=pl.Utf8)
      .alias("citing_category"),
    pl.col("cited_docdb_family_id").map_elements(classify_family, return_dtype=pl.Utf8)
      .alias("cited_category"),
])

all_edges.write_parquet("PATSTAT2025FALL/output/citation_edges_categorized.parquet", compression="zstd")
print(f"Total categorized edges: {all_edges.height}")

# --- Step 6: Get metadata for neighbor families from tls201 ---
family_year_lookup = pl.read_parquet("PATSTAT2025FALL/output/docdb_family_year.parquet")

tls201_files = [
    "Z:/PATSTAT Global 2025 Autumn/tls201_appln_part01.csv",
    "Z:/PATSTAT Global 2025 Autumn/tls201_appln_part02.csv",
    "Z:/PATSTAT Global 2025 Autumn/tls201_appln_part03.csv",
]
patstat_null_values = ["", "null"]

# Build a neighbor metadata table from tls201: one row per neighbor family
neighbor_fam_list = list(neighbor_fam_set)

neighbor_meta = (
    pl.concat(
        [pl.scan_csv(f,
                     schema_overrides={
                         "docdb_family_id": pl.Int32,
                         "appln_id": pl.Int32,
                         "docdb_family_size": pl.Int16,
                         "nb_applicants": pl.Int16,
                         "nb_inventors": pl.Int16,
                         "nb_citing_docdb_fam": pl.Int32,
                     },
                     null_values=patstat_null_values)
         .select([
             "appln_id", "docdb_family_id", "appln_auth",
             "earliest_filing_date", "ipr_type", "granted",
             "docdb_family_size", "nb_applicants", "nb_inventors",
             "nb_citing_docdb_fam",
         ])
         for f in tls201_files],
        rechunk=False,
    )
    .filter(pl.col("docdb_family_id").is_in(neighbor_fam_list))
    .collect(engine="streaming")
)
gc.collect()

neighbor_families_agg = (
    neighbor_meta
    .with_columns(
        pl.col("earliest_filing_date").str.slice(0, 4).cast(pl.Int16).alias("year")
    )
    .group_by("docdb_family_id")
    .agg([
        pl.col("year").min(),
        pl.col("appln_auth").unique().sort().cast(pl.Utf8).str.join(","),
        pl.col("docdb_family_size").first(),
        pl.col("nb_applicants").max(),
        pl.col("nb_inventors").max(),
        pl.col("nb_citing_docdb_fam").max(),
        pl.col("granted").first(),
        pl.col("ipr_type").first(),
    ])
)

# Save neighbor index
neighbor_index = neighbor_meta.select(["appln_id", "docdb_family_id"]).unique()
neighbor_index.write_parquet("PATSTAT2025FALL/output/neighbor_index.parquet", compression="zstd")
print(f"Saved neighbor_index.parquet: {neighbor_index.height} rows "
      f"({neighbor_index['docdb_family_id'].n_unique()} families)")

# --- Step 6b: Build person data for neighbor applications ---
# persons_agg.parquet only covers applications from the filtered tls201
# (granted, PI/UM, family_size>=2), so most neighbor appln_ids are missing.
# Build a dedicated lookup from the raw tls207 + tls206 tables.
print("\nBuilding person data for neighbor applications ...")
TLS207_FILE = (
    "Z:/PATSTAT Global 2025 Autumn/tls207_pers_appln_part01.csv"
)
neighbor_appln_ids = neighbor_index.select("appln_id").unique()

neighbor_persons_joined = (
    pl.scan_csv(
        TLS207_FILE,
        schema_overrides={"appln_id": pl.Int32, "person_id": pl.Int32},
    )
    .join(neighbor_appln_ids.lazy(), on="appln_id", how="semi")
    .join(pl.scan_parquet("PATSTAT2025FALL/output/tls206.parquet"), on="person_id", how="inner")
    .collect(engine="streaming")
)
gc.collect()

neighbor_persons_agg = (
    neighbor_persons_joined
    .with_columns(
        pl.when(
            pl.col("person_ctry_code").is_null()
            | pl.col("person_ctry_code").cast(pl.Utf8).str.strip_chars().is_in(["", "null"])
        )
        .then(None)
        .otherwise(pl.col("person_ctry_code").cast(pl.Utf8).str.strip_chars())
        .alias("person_ctry_code")
    )
    .group_by("appln_id")
    .agg(
        pl.col("person_ctry_code")
        .drop_nulls()
        .unique()
        .sort()
        .str.join(",")
    )
)

neighbor_persons_agg.write_parquet(
    "PATSTAT2025FALL/output/neighbor_persons_agg.parquet", compression="zstd"
)
print(f"Saved neighbor_persons_agg.parquet: {neighbor_persons_agg.height} rows")

del neighbor_meta, neighbor_index, neighbor_persons_joined, neighbor_persons_agg
gc.collect()

# --- Step 7: Citation flow summary (category × category) ---
flow_matrix = (
    all_edges
    .group_by(["citing_category", "cited_category"])
    .agg(pl.len().alias("n_edges"))
    .sort(["citing_category", "cited_category"])
)
print("\nCitation flow matrix (citing → cited):")
print(flow_matrix)

# --- Step 8: Neighbor descriptive statistics ---

# 8a. How many neighbors cite green vs are cited by green
neighbors_citing_green = len(citing_families - green_fam_set)
neighbors_cited_by_green = len(cited_by_green - green_fam_set)
neighbors_both_directions = len((citing_families & cited_by_green) - green_fam_set)

# 8b. Degree distribution: how many green families does each neighbor connect to
neighbor_degree_fwd = (
    all_edges
    .filter(
        (pl.col("citing_category") == "N") &
        (pl.col("cited_category").is_in(["G", "H"]))
    )
    .group_by("docdb_family_id")
    .agg(pl.col("cited_docdb_family_id").n_unique().alias("n_green_cited"))
)

neighbor_degree_bwd = (
    all_edges
    .filter(
        (pl.col("cited_category") == "N") &
        (pl.col("citing_category").is_in(["G", "H"]))
    )
    .group_by("cited_docdb_family_id")
    .agg(pl.col("docdb_family_id").n_unique().alias("n_green_citing"))
    .rename({"cited_docdb_family_id": "docdb_family_id"})
)

# 8c. Year distribution of neighbors
neighbor_by_year = (
    neighbor_families_agg
    .group_by("year")
    .agg(pl.len().alias("n_families"))
    .sort("year")
)

# 8d. Country distribution of neighbors
neighbor_by_country = (
    neighbor_families_agg
    .with_columns(pl.col("appln_auth").str.split(","))
    .explode("appln_auth")
    .filter(pl.col("appln_auth").is_not_null() & (pl.col("appln_auth") != ""))
    .group_by("appln_auth")
    .agg(pl.col("docdb_family_id").n_unique().alias("n_families"))
    .sort("n_families", descending=True)
    .head(30)
)

# 8e. Family size and citation stats for neighbors
neighbor_stats = neighbor_families_agg.select([
    pl.lit("docdb_family_size").alias("metric"),
    pl.col("docdb_family_size").cast(pl.Float64).mean().alias("mean"),
    pl.col("docdb_family_size").cast(pl.Float64).median().alias("median"),
    pl.col("docdb_family_size").cast(pl.Float64).std().alias("std"),
    pl.col("docdb_family_size").cast(pl.Float64).min().alias("min"),
    pl.col("docdb_family_size").cast(pl.Float64).max().alias("max"),
])

neighbor_citation_stats = neighbor_families_agg.select([
    pl.lit("nb_citing_docdb_fam").alias("metric"),
    pl.col("nb_citing_docdb_fam").cast(pl.Float64).mean().alias("mean"),
    pl.col("nb_citing_docdb_fam").cast(pl.Float64).median().alias("median"),
    pl.col("nb_citing_docdb_fam").cast(pl.Float64).std().alias("std"),
    pl.col("nb_citing_docdb_fam").cast(pl.Float64).min().alias("min"),
    pl.col("nb_citing_docdb_fam").cast(pl.Float64).max().alias("max"),
])

neighbor_descriptive = pl.concat([neighbor_stats, neighbor_citation_stats])

# 8f. Degree stats: how connected are neighbors to the green network
degree_fwd_stats = neighbor_degree_fwd.select([
    pl.lit("green families cited by neighbor").alias("metric"),
    pl.col("n_green_cited").cast(pl.Float64).mean().alias("mean"),
    pl.col("n_green_cited").cast(pl.Float64).median().alias("median"),
    pl.col("n_green_cited").cast(pl.Float64).max().alias("max"),
])

degree_bwd_stats = neighbor_degree_bwd.select([
    pl.lit("green families citing neighbor").alias("metric"),
    pl.col("n_green_citing").cast(pl.Float64).mean().alias("mean"),
    pl.col("n_green_citing").cast(pl.Float64).median().alias("median"),
    pl.col("n_green_citing").cast(pl.Float64).max().alias("max"),
])

# --- Step 9: Summary table ---
summary_rows = [
    {"Metric": "Green families (G+H)", "Value": len(green_fam_set)},
    {"Metric": "  High-influential (H)", "Value": len(hi_fam_set)},
    {"Metric": "  Green non-high-influential (G)", "Value": len(green_fam_set - hi_fam_set)},
    {"Metric": "Non-green neighbor families (N)", "Value": len(neighbor_fam_set)},
    {"Metric": "  Neighbors citing green", "Value": neighbors_citing_green},
    {"Metric": "  Neighbors cited by green", "Value": neighbors_cited_by_green},
    {"Metric": "  Neighbors in both directions", "Value": neighbors_both_directions},
    {"Metric": "Total citation edges (green-related)", "Value": all_edges.height},
]
df_summary = pl.DataFrame(summary_rows)

# --- Step 10: Save all to Excel ---
with pd.ExcelWriter("PATSTAT2025FALL/output/neighbor_statistics.xlsx", engine="openpyxl") as writer:
    df_summary.to_pandas().to_excel(writer, sheet_name="Summary", index=False)
    flow_matrix.to_pandas().to_excel(writer, sheet_name="Citation Flow Matrix", index=False)
    neighbor_by_year.to_pandas().to_excel(writer, sheet_name="Neighbors By Year", index=False)
    neighbor_by_country.to_pandas().to_excel(writer, sheet_name="Neighbors Top Countries", index=False)
    neighbor_descriptive.to_pandas().to_excel(writer, sheet_name="Neighbor Descriptive Stats", index=False)
    degree_fwd_stats.to_pandas().to_excel(writer, sheet_name="Neighbor Degree (fwd)", index=False)
    degree_bwd_stats.to_pandas().to_excel(writer, sheet_name="Neighbor Degree (bwd)", index=False)

print(f"\nStatistics saved to neighbor_statistics.xlsx")

# Save categorized edges and neighbor family metadata
neighbor_families_agg.write_parquet("PATSTAT2025FALL/output/neighbor_families.parquet", compression="zstd")
print(f"Saved neighbor_families.parquet: {neighbor_families_agg.height} families")

del all_edges, green_full, neighbor_families_agg
gc.collect()
