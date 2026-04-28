import polars as pl
import gc
from pathlib import Path

# ============================================================================
# OECD Patent Quality Composite Index (4 components) — DOCDB family level
#
#   1. Forward citations (5-year window)
#   2. Patent family size
#   3. Number of claims 
#   4. Generality index (1 − HHI of technology-class distribution among citers)
#
# Data:
#   green_patent9223.parquet       — green patents (application level)
#   TLS201 (appln)                 — earliest filing year per family
#   TLS225 (docdb_fam_cpc)         — CPC codes at family level (≈ IPC subclass)
#   TLS228 (docdb_fam_citn)        — family-level forward citations
# ============================================================================

PATSTAT_DIR = "Z:/PATSTAT Global 2025 Autumn"

TLS201_FILES = [
    f"{PATSTAT_DIR}/tls201_appln_part0{i}.csv"
    for i in range(1, 4)
]
TLS225_FILES = [
    f"{PATSTAT_DIR}/tls225_docdb_fam_cpc_part0{i}.csv"
    for i in range(1, 3)
]
TLS228_FILE = (
    f"{PATSTAT_DIR}/tls228_docdb_fam_citn_part01.csv"
)

# ============================================================================
# Phase 1 — Build intermediate lookup tables (skip if files already exist)
# ============================================================================

# --- 1a. docdb_family_id → earliest filing year ---
if not Path("PATSTAT2025FALL/output/docdb_family_year.parquet").exists():
    print("Building docdb_family_year.parquet from TLS201 …")
    (
        pl.concat(
            [
                pl.scan_csv(f, schema_overrides={"docdb_family_id": pl.Int32})
                .select(["docdb_family_id", "earliest_filing_date"])
                for f in TLS201_FILES
            ],
            rechunk=False,
        )
        .with_columns(
            pl.col("earliest_filing_date")
            .str.slice(0, 4)
            .cast(pl.Int16)
            .alias("family_year")
        )
        .group_by("docdb_family_id")
        .agg(pl.col("family_year").min())
        .sink_parquet(
            "PATSTAT2025FALL/output/docdb_family_year.parquet", compression="zstd", engine="streaming"
        )
    )
    gc.collect()
    print("  Done.")
else:
    print("Using existing docdb_family_year.parquet")

# --- 1b. docdb_family_id → primary CPC subclass (4-char) ---
# CPC subclass ≈ IPC subclass at this level; used to classify citing families.
if not Path("PATSTAT2025FALL/output/family_cpc4.parquet").exists():
    print("Building family_cpc4.parquet from TLS225 …")
    (
        pl.concat(
            [
                pl.scan_csv(f, schema_overrides={"docdb_family_id": pl.Int32})
                .select(["docdb_family_id", "cpc_class_symbol"])
                for f in TLS225_FILES
            ],
            rechunk=False,
        )
        .with_columns(
            pl.col("cpc_class_symbol")
            .str.replace_all(" ", "")
            .str.to_uppercase()
            .str.slice(0, 4)
            .alias("cpc_4digit")
        )
        .group_by("docdb_family_id")
        .agg(pl.col("cpc_4digit").first().alias("primary_cpc4"))
        .sink_parquet(
            "PATSTAT2025FALL/output/family_cpc4.parquet", compression="zstd", engine="streaming"
        )
    )
    gc.collect()
    print("  Done.")
else:
    print("Using existing family_cpc4.parquet")


# ============================================================================
# Phase 2 — Load green patent families and aggregate to DOCDB-family level
# ============================================================================

print("\nLoading green patent data ...")
green = pl.read_parquet("PATSTAT2025FALL/output/green_patent8526.parquet")

green_families = (
    green
    .with_columns(
        pl.col("earliest_filing_date")
        .str.slice(0, 4)
        .cast(pl.Int16)
        .alias("family_year"),
    )
    .group_by("docdb_family_id")
    .agg([
        pl.col("family_year").min(),
        pl.col("docdb_family_size").max().alias("family_size"),
        pl.col("publn_claims").max().alias("claims"),
    ])
)

# Countries per family — prioritise person_ctry_code (inventor country);
# fall back to appln_auth (filing office) for applications without person data.
REGIONAL_OFFICES = {"EP", "WO", "EA", "OA", "AP", "GC", "BX"}

# Applications WITH person_ctry_code
with_person = (
    green
    .select(["docdb_family_id", "person_ctry_code"])
    .filter(pl.col("person_ctry_code").is_not_null() & (pl.col("person_ctry_code") != ""))
    .with_columns(pl.col("person_ctry_code").str.split(","))
    .explode("person_ctry_code")
    .with_columns(pl.col("person_ctry_code").str.strip_chars().alias("country"))
    .filter(pl.col("country") != "")
    .select(["docdb_family_id", "country"])
)

# Applications WITHOUT person_ctry_code
without_person = (
    green
    .select(["docdb_family_id", "appln_auth", "person_ctry_code"])
    .filter(pl.col("person_ctry_code").is_null() | (pl.col("person_ctry_code") == ""))
    .filter(
        pl.col("appln_auth").is_not_null()
        & (pl.col("appln_auth") != "")
        & (~pl.col("appln_auth").is_in(list(REGIONAL_OFFICES)))
    )
    .select(["docdb_family_id", pl.col("appln_auth").alias("country")])
)

n_total_apps = green.height
n_with = green.filter(
    pl.col("person_ctry_code").is_not_null() & (pl.col("person_ctry_code") != "")
).height
print(f"  Country source: person_ctry_code {n_with}/{n_total_apps} apps "
      f"({n_with/n_total_apps*100:.1f}%), appln_auth fallback for the rest")

country_by_family = (
    pl.concat([with_person, without_person])
    .group_by("docdb_family_id")
    .agg(pl.col("country").unique().alias("countries"))
)

# Technology field per family (first character of first IPC code → A–H section)
# ipc is List[String] in the parquet
tech_field_by_family = (
    green
    .select(["docdb_family_id", "ipc"])
    .explode("ipc")
    .with_columns(
        pl.col("ipc").str.strip_chars().str.slice(0, 1).alias("tech_field")
    )
    .filter(
        pl.col("tech_field").is_not_null() & (pl.col("tech_field") != "")
    )
    .group_by("docdb_family_id")
    .agg(pl.col("tech_field").first().alias("main_tech_field"))
)

del green
gc.collect()
print(f"  Green patent families: {green_families.height}")


# ============================================================================
# Phase 3 — Compute forward citations with 5-year window (from TLS228)
# ============================================================================

print("\nComputing 5-year forward citations …")

citn_fam = pl.scan_csv(
    TLS228_FILE,
    schema_overrides={
        "docdb_family_id": pl.Int32,
        "cited_docdb_family_id": pl.Int32,
    },
)

green_fam_ids = green_families.select("docdb_family_id").rename(
    {"docdb_family_id": "cited_docdb_family_id"}
)

family_year_lookup = pl.scan_parquet("PATSTAT2025FALL/output/docdb_family_year.parquet")

citations_windowed = (
    citn_fam
    # keep only citations whose cited family is a green patent
    .join(green_fam_ids.lazy(), on="cited_docdb_family_id", how="semi")
    # attach cited-family year
    .join(
        green_families
        .select(["docdb_family_id", "family_year"])
        .rename({
            "docdb_family_id": "cited_docdb_family_id",
            "family_year": "cited_year",
        })
        .lazy(),
        on="cited_docdb_family_id",
        how="inner",
    )
    # attach citing-family year
    .join(
        family_year_lookup.rename({"family_year": "citing_year"}),
        on="docdb_family_id",
        how="inner",
    )
    # 5-year window
    .filter(
        (pl.col("citing_year") >= pl.col("cited_year"))
        & (pl.col("citing_year") <= pl.col("cited_year") + 5)
    )
    .collect(engine="streaming")
)
gc.collect()

fwd_cit_5yr = (
    citations_windowed
    .group_by("cited_docdb_family_id")
    .agg(pl.len().alias("fwd_citations_5yr"))
    .rename({"cited_docdb_family_id": "docdb_family_id"})
)

print(f"  Families with ≥1 five-year citation: {fwd_cit_5yr.height}")


# ============================================================================
# Phase 4 — Compute generality index
#   Generality_i = 1 − Σ_j (n_ij / N_i)²
#   where n_ij = citations of family i from CPC subclass j,
#         N_i  = total 5-year forward citations of family i.
# ============================================================================

print("\nComputing generality index ...")

cpc_lookup = pl.read_parquet("PATSTAT2025FALL/output/family_cpc4.parquet")

# Attach primary CPC subclass of each *citing* family
citations_with_cpc = citations_windowed.join(
    cpc_lookup,
    on="docdb_family_id",   # citing family
    how="left",
)

n_total_hall = citations_with_cpc.height
n_has_cpc_hall = citations_with_cpc.filter(pl.col("primary_cpc4").is_not_null()).height
print(f"  Hall: {n_has_cpc_hall} / {n_total_hall} citations have CPC tags "
      f"({n_has_cpc_hall / n_total_hall * 100:.1f}%), "
      f"{n_total_hall - n_has_cpc_hall} dropped")

# Citation counts by (cited family, CPC subclass of citing family)
class_counts = (
    citations_with_cpc
    .filter(pl.col("primary_cpc4").is_not_null())
    .group_by(["cited_docdb_family_id", "primary_cpc4"])
    .agg(pl.len().alias("n_ij"))
)

del citations_with_cpc
gc.collect()

total_counts = (
    class_counts
    .group_by("cited_docdb_family_id")
    .agg(pl.col("n_ij").sum().alias("N_i"))
)

generality = (
    class_counts
    .join(total_counts, on="cited_docdb_family_id", how="left")
    .with_columns(
        (pl.col("n_ij") / pl.col("N_i")).pow(2).alias("share_sq")
    )
    .group_by("cited_docdb_family_id")
    .agg([
        (1.0 - pl.col("share_sq").sum()).alias("generality"),
        pl.col("N_i").first(),
    ])
    .rename({"cited_docdb_family_id": "docdb_family_id"})
)

del class_counts, total_counts
gc.collect()

print(f"  Families with valid generality (Hall): {generality.height}")


# ============================================================================
# Phase 4b — OECD generality index (modified HHI)
#
#   Unlike Hall et al. where each citing patent counts as 1 in a single class,
#   the OECD method weights each citing patent i's contribution to 4-digit
#   class j by  β_ji = T_ji^n / T_i^n  (share of its full CPC codes in that
#   class).  The generality of focal patent X with N citers is:
#
#       G_X = 1 − Σ_j [ (1/N) Σ_{i=1}^{N} β_ji ]²
#
#   T_i^n  = total CPC codes (finest level) in citing patent i
#   T_ji^n = CPC codes in citing patent i that belong to 4-digit class j
#   M      = set of all 4-digit classes across all citers of X
# ============================================================================

print("\nComputing OECD generality index …")

# Get the full CPC distribution for each citing family from TLS225
citing_ids = citations_windowed.select("docdb_family_id").unique()

citing_cpc_dist = (
    pl.concat(
        [
            pl.scan_csv(f, schema_overrides={"docdb_family_id": pl.Int32})
            .select(["docdb_family_id", "cpc_class_symbol"])
            for f in TLS225_FILES
        ],
        rechunk=False,
    )
    .with_columns(
        pl.col("cpc_class_symbol")
        .str.replace_all(" ", "")
        .str.to_uppercase()
        .alias("cpc_clean")
    )
    .with_columns(
        pl.col("cpc_clean").str.slice(0, 4).alias("cpc_4digit")
    )
    .join(citing_ids.lazy(), on="docdb_family_id", how="semi")
    .group_by(["docdb_family_id", "cpc_4digit"])
    .agg(pl.col("cpc_clean").n_unique().alias("T_ji"))
    .collect(engine="streaming")
)
gc.collect()

# N counts only citing families that have CPC data (consistent with β sum)
citing_with_cpc_set = citing_cpc_dist.select("docdb_family_id").unique()
n_total_oecd = citations_windowed.height
n_has_cpc_oecd = citations_windowed.join(
    citing_with_cpc_set, on="docdb_family_id", how="semi"
).height
print(f"  OECD: {n_has_cpc_oecd} / {n_total_oecd} citations have CPC tags "
      f"({n_has_cpc_oecd / n_total_oecd * 100:.1f}%), "
      f"{n_total_oecd - n_has_cpc_oecd} dropped")

N_per_cited = (
    citations_windowed
    .join(citing_with_cpc_set, on="docdb_family_id", how="semi")
    .group_by("cited_docdb_family_id")
    .agg(pl.len().alias("N"))
)

# T_i^n per citing family (total CPC codes at the finest level)
citing_totals = (
    citing_cpc_dist
    .group_by("docdb_family_id")
    .agg(pl.col("T_ji").sum().alias("T_i"))
)

# β_ji = T_ji / T_i for each (citing family, 4-digit class)
citing_betas = (
    citing_cpc_dist
    .join(citing_totals, on="docdb_family_id", how="left")
    .with_columns(
        (pl.col("T_ji") / pl.col("T_i")).alias("beta_ji")
    )
    .select(["docdb_family_id", "cpc_4digit", "beta_ji"])
)

del citing_cpc_dist, citing_totals
gc.collect()

# Expand: one row per (cited_family, citing_family, cpc_4digit) with β_ji
citations_with_betas = (
    citations_windowed
    .select(["docdb_family_id", "cited_docdb_family_id"])
    .join(citing_betas, on="docdb_family_id", how="left")
)

del citations_windowed
gc.collect()

# For each (cited_family, cpc_4digit): sum β_ji across all citing families
# then divide by N and square  →  [ (1/N) Σ β_ji ]²
avg_beta_sq = (
    citations_with_betas
    .filter(pl.col("beta_ji").is_not_null())
    .group_by(["cited_docdb_family_id", "cpc_4digit"])
    .agg(pl.col("beta_ji").sum().alias("sum_beta"))
    .join(N_per_cited, on="cited_docdb_family_id", how="left")
    .with_columns(
        (pl.col("sum_beta") / pl.col("N")).pow(2).alias("avg_beta_sq")
    )
)

# G_X = 1 − Σ_j [ avg_beta_j² ]
generality_oecd = (
    avg_beta_sq
    .group_by("cited_docdb_family_id")
    .agg((1.0 - pl.col("avg_beta_sq").sum()).alias("generality_oecd"))
    .rename({"cited_docdb_family_id": "docdb_family_id"})
)

del citations_with_betas, citing_betas, avg_beta_sq, N_per_cited
gc.collect()

print(f"  Families with valid generality (OECD): {generality_oecd.height}")


# ============================================================================
# Phase 5 — Assemble quality index at family level
# ============================================================================

print("\nAssembling patent quality index …")

df_family = (
    green_families
    .join(fwd_cit_5yr, on="docdb_family_id", how="left")
    .join(
        generality.select(["docdb_family_id", "generality"]),
        on="docdb_family_id",
        how="left",
    )
    .join(
        generality_oecd.select(["docdb_family_id", "generality_oecd"]),
        on="docdb_family_id",
        how="left",
    )
    .join(tech_field_by_family, on="docdb_family_id", how="left")
    .join(country_by_family, on="docdb_family_id", how="left")
    .with_columns([
        pl.col("fwd_citations_5yr").fill_null(0),
        pl.col("generality").fill_null(0.0),
        pl.col("generality_oecd").fill_null(0.0),
    ])
)

# --- Cohort normalisation (filing year × IPC section) ---

cohort_stats = (
    df_family
    .group_by(["family_year", "main_tech_field"])
    .agg([
        pl.col("fwd_citations_5yr").quantile(0.99).alias("p99_cites"),
        pl.col("family_size").quantile(0.99).alias("p99_family"),
        pl.col("claims").quantile(0.99).alias("p99_claims"),
        pl.col("generality").quantile(0.99).alias("p99_generality"),
        pl.col("generality_oecd").quantile(0.99).alias("p99_generality_oecd"),
        pl.len().alias("cohort_size"),
    ])
)

df_with_cohort = df_family.join(
    cohort_stats, on=["family_year", "main_tech_field"], how="left"
)

# Winsorise at 99th percentile and scale to [0, 1] (OECD methodology)
df_with_cohort = df_with_cohort.with_columns([
    pl.when(pl.col("p99_cites") > 0)
    .then(
        pl.min_horizontal(
            pl.col("fwd_citations_5yr").cast(pl.Float64),
            pl.col("p99_cites"),
        )
        / pl.col("p99_cites")
    )
    .otherwise(0.0)
    .alias("norm_forward_cites"),

    pl.when(pl.col("p99_family") > 0)
    .then(
        pl.min_horizontal(
            pl.col("family_size").cast(pl.Float64),
            pl.col("p99_family"),
        )
        / pl.col("p99_family")
    )
    .otherwise(0.0)
    .alias("norm_family_size"),

    pl.when(pl.col("p99_claims") > 0)
    .then(
        pl.min_horizontal(
            pl.col("claims").cast(pl.Float64),
            pl.col("p99_claims"),
        )
        / pl.col("p99_claims")
    )
    .otherwise(0.0)
    .alias("norm_claims"),

    pl.when(pl.col("p99_generality") > 0)
    .then(
        pl.min_horizontal(
            pl.col("generality"),
            pl.col("p99_generality"),
        )
        / pl.col("p99_generality")
    )
    .otherwise(0.0)
    .alias("norm_generality"),

    pl.when(pl.col("p99_generality_oecd") > 0)
    .then(
        pl.min_horizontal(
            pl.col("generality_oecd"),
            pl.col("p99_generality_oecd"),
        )
        / pl.col("p99_generality_oecd")
    )
    .otherwise(0.0)
    .alias("norm_generality_oecd"),
])

# Quality index: equal-weighted average of the four normalised components
# Two variants: one using Hall et al. generality, one using OECD generality
df_with_cohort = df_with_cohort.with_columns([
    (
        (
            pl.col("norm_forward_cites")
            + pl.col("norm_family_size")
            + pl.col("norm_claims")
            + pl.col("norm_generality")
        )
        / 4
    ).alias("patent_quality_index_4"),
    (
        (
            pl.col("norm_forward_cites")
            + pl.col("norm_family_size")
            + pl.col("norm_claims")
            + pl.col("norm_generality_oecd")
        )
        / 4
    ).alias("patent_quality_index_4_oecd"),
])


# Save family-level quality data for downstream per-year analysis
df_with_cohort.write_parquet("PATSTAT2025FALL/output/patent_quality_family.parquet", compression="zstd")
print(f"  Family-level quality data saved to patent_quality_family.parquet")

# ============================================================================
# Phase 6 — Country-level aggregation and output
# ============================================================================

print("\nAggregating to country level …")

df_country_patents = (
    df_with_cohort.explode("countries")
    .filter(pl.col("countries").is_not_null() & (pl.col("countries") != ""))
)

df_country_quality = (
    df_country_patents
    .group_by("countries")
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

        # Quality index (Hall generality)
        pl.col("patent_quality_index_4").mean().alias("country_patent_quality_index"),
        pl.col("patent_quality_index_4").median().alias("country_patent_quality_median"),

        # Quality index (OECD generality)
        pl.col("patent_quality_index_4_oecd").mean().alias("country_pqi_oecd"),
        pl.col("patent_quality_index_4_oecd").median().alias("country_pqi_oecd_median"),

        # Supplementary
        pl.col("fwd_citations_5yr").sum().alias("total_citations_5yr"),
        pl.col("patent_quality_index_4").std().alias("quality_std"),
        pl.col("patent_quality_index_4_oecd").std().alias("quality_std_oecd"),
        pl.col("family_year").min().alias("earliest_year"),
        pl.col("family_year").max().alias("latest_year"),
    ])
    .sort("country_pqi_oecd", descending=True)
)

df_country_quality = df_country_quality.with_columns([
    pl.col("country_patent_quality_index")
    .rank(method="ordinal", descending=True)
    .alias("quality_rank")
])

print("\nPatent Quality Index (4) — Country Rankings:")
print("=" * 80)
print(
    df_country_quality.select([
        "quality_rank", "countries", "country_patent_quality_index",
        "country_pqi_oecd",
        "num_patents", "avg_forward_cites_5yr", "avg_family_size",
        "avg_claims", "avg_generality", "avg_generality_oecd",
    ]).head(20)
)

print(f"\nTotal countries analysed: {len(df_country_quality)}")
print(f"Total patent families: {df_country_quality['num_patents'].sum()}")

print("\nMethodology:")
print("  Forward citations : 5-year window from earliest filing date (TLS228)")
print("  Generality (Hall) : 1 − HHI; each citing family → single primary CPC class")
print("  Generality (OECD) : 1 − Σ[(1/N) Σ β_ji]²; β_ji = CPC-share weight per citer")
print("  Normalisation     : Winsorised at p99 within (year × IPC section) cohorts")
print("  Quality index     : Equal-weighted average of 4 normalised components")
print("  Country aggregation: Mean of family-level quality indices")

df_country_quality.write_csv("PATSTAT2025FALL/output/oecd_patent_quality_country_rankings.csv")
print("\nResults exported to: oecd_patent_quality_country_rankings.csv")
