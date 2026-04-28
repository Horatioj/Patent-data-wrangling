"""Citation spillover analysis — decompose forward citations by origin country.

For green patents of a target country (e.g. US), this script:
  1. Identifies all citation edges where the cited patent belongs to the target
  2. Assigns a country to each citing family (via appln_auth from TLS201)
  3. Aggregates citations by (year × citing_country)
  4. Computes counterfactual curves: "what if no citations from DE / JP / CN?"
  5. Produces two visualisations:
     a) Counterfactual line plot (solid total + dashed without-country lines)
     b) Stacked area chart of citation composition over time

Data requirements (produced by prior pipeline steps):
  green_patent8526.parquet    — green patents with inventor country
  docdb_family_year.parquet   — docdb_family_id → earliest filing year
  TLS201 (PATSTAT)            — application metadata (appln_auth)
  TLS228 (PATSTAT)            — DOCDB family-level citation edges

Methodology note:
  Citing-family country is determined by appln_auth (filing authority) of
  the first non-regional application in each DOCDB family.  This is the
  standard proxy when inventor-level person data is unavailable for all
  families.  Each citation edge is counted exactly once to avoid
  double-counting multi-country families in the counterfactual analysis.
"""

import polars as pl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import gc
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

PATSTAT_DIR = "Z:/PATSTAT Global 2025 Autumn"
TLS228_FILE = f"{PATSTAT_DIR}/tls228_docdb_fam_citn_part01.csv"
TLS201_FILES = [
    f"{PATSTAT_DIR}/tls201_appln_part0{i}.csv"
    for i in range(1, 4)
]

REGIONAL_OFFICES = {"EP", "WO", "EA", "OA", "AP", "GC", "BX"}

COUNTRY_NAMES = {
    "US": "United States", "JP": "Japan", "DE": "Germany",
    "CN": "China", "KR": "South Korea", "GB": "United Kingdom",
    "FR": "France", "CA": "Canada", "AU": "Australia",
    "IN": "India", "TW": "Taiwan", "CH": "Switzerland",
    "NL": "Netherlands", "SE": "Sweden", "IT": "Italy",
    "FI": "Finland", "DK": "Denmark", "AT": "Austria",
    "BE": "Belgium", "IL": "Israel", "ES": "Spain",
    "BR": "Brazil", "RU": "Russia", "SG": "Singapore",
}

plt.rcParams.update({
    "figure.dpi": 200,
    "font.size": 10,
    "axes.titlesize": 11,
})


# ============================================================================
# Data loading
# ============================================================================

def load_green_family_countries(
    green_parquet: str = "PATSTAT2025FALL/output/green_patent8526.parquet",
) -> pl.DataFrame:
    """Load green patents and build family → list[country] mapping.

    Uses person_ctry_code (inventor country) with appln_auth fallback,
    consistent with patent_quality.py methodology.

    Returns
    -------
    DataFrame with columns [docdb_family_id, countries]
    """
    green = pl.read_parquet(green_parquet)

    with_person = (
        green
        .select(["docdb_family_id", "person_ctry_code"])
        .filter(
            pl.col("person_ctry_code").is_not_null()
            & (pl.col("person_ctry_code") != "")
        )
        .with_columns(pl.col("person_ctry_code").str.split(","))
        .explode("person_ctry_code")
        .with_columns(
            pl.col("person_ctry_code").str.strip_chars().alias("country")
        )
        .filter(pl.col("country") != "")
        .select(["docdb_family_id", "country"])
    )

    without_person = (
        green
        .select(["docdb_family_id", "appln_auth", "person_ctry_code"])
        .filter(
            pl.col("person_ctry_code").is_null()
            | (pl.col("person_ctry_code") == "")
        )
        .filter(
            pl.col("appln_auth").is_not_null()
            & (pl.col("appln_auth") != "")
            & (~pl.col("appln_auth").is_in(list(REGIONAL_OFFICES)))
        )
        .select(["docdb_family_id", pl.col("appln_auth").alias("country")])
    )

    country_by_family = (
        pl.concat([with_person, without_person])
        .group_by("docdb_family_id")
        .agg(pl.col("country").unique().alias("countries"))
    )

    del green
    gc.collect()
    return country_by_family


def identify_target_families(
    country_by_family: pl.DataFrame,
    target_country: str,
) -> pl.DataFrame:
    """Return unique docdb_family_id for families with ≥1 inventor in *target_country*."""
    return (
        country_by_family
        .explode("countries")
        .filter(pl.col("countries") == target_country)
        .select("docdb_family_id")
        .unique()
    )


def build_family_country_lookup(
    cache_path: str = "PATSTAT2025FALL/output/docdb_family_country.parquet",
) -> pl.DataFrame:
    """Map every DOCDB family to a single primary country (appln_auth).

    Uses the first non-regional-office appln_auth encountered per family.
    Result is cached to *cache_path* after initial build.

    Returns
    -------
    DataFrame with columns [docdb_family_id, country]
    """
    if Path(cache_path).exists():
        print(f"  Using cached {cache_path}")
        return pl.read_parquet(cache_path)

    print(f"  Building {cache_path} from TLS201 (one-time cost) ...")
    df = (
        pl.concat(
            [
                pl.scan_csv(f, schema_overrides={"docdb_family_id": pl.Int32})
                .select(["docdb_family_id", "appln_auth"])
                for f in TLS201_FILES
            ],
            rechunk=False,
        )
        .filter(
            pl.col("appln_auth").is_not_null()
            & (~pl.col("appln_auth").is_in(list(REGIONAL_OFFICES)))
        )
        .group_by("docdb_family_id")
        .agg(pl.col("appln_auth").first().alias("country"))
        .collect(engine="streaming")
    )

    df.write_parquet(cache_path, compression="zstd")
    print(f"  Saved {cache_path}: {df.height:,} families")
    gc.collect()
    return df


def load_citations_to_target(
    target_family_ids: pl.DataFrame,
    citation_window: int | None = None,
) -> pl.DataFrame:
    """Load all citation edges where the cited family is in *target_family_ids*.

    Parameters
    ----------
    target_family_ids : DataFrame with column ``docdb_family_id``
    citation_window   : If set, keep only citations where
                        citing_year − cited_year ≤ window (e.g. 5).

    Returns
    -------
    DataFrame [docdb_family_id (citing), cited_docdb_family_id,
               citing_year, cited_year]
    """
    citn = pl.scan_csv(
        TLS228_FILE,
        schema_overrides={
            "docdb_family_id": pl.Int32,
            "cited_docdb_family_id": pl.Int32,
        },
    )

    target_renamed = target_family_ids.rename(
        {"docdb_family_id": "cited_docdb_family_id"}
    )
    family_year = pl.scan_parquet("PATSTAT2025FALL/output/docdb_family_year.parquet")

    query = (
        citn
        .join(target_renamed.lazy(), on="cited_docdb_family_id", how="semi")
        .join(
            family_year.rename({"family_year": "citing_year"}),
            on="docdb_family_id",
            how="inner",
        )
        .join(
            family_year.rename({
                "docdb_family_id": "cited_docdb_family_id",
                "family_year": "cited_year",
            }),
            on="cited_docdb_family_id",
            how="inner",
        )
    )

    if citation_window is not None:
        query = query.filter(
            (pl.col("citing_year") >= pl.col("cited_year"))
            & (pl.col("citing_year") <= pl.col("cited_year") + citation_window)
        )

    edges = query.collect(engine="streaming")
    gc.collect()
    return edges


# ============================================================================
# Analysis
# ============================================================================

def build_spillover_table(
    citation_edges: pl.DataFrame,
    family_country: pl.DataFrame,
    year_col: str = "citing_year",
) -> pl.DataFrame:
    """Aggregate citation edges by (*year_col*, citing_country).

    Each edge is counted exactly once using the citing family's primary country.

    Returns
    -------
    DataFrame [year_col, citing_country, n_citations]
    """
    merged = (
        citation_edges
        .join(family_country, on="docdb_family_id", how="left")
        .filter(pl.col("country").is_not_null())
    )

    return (
        merged
        .group_by([year_col, "country"])
        .agg(pl.len().alias("n_citations"))
        .sort([year_col, "country"])
        .rename({"country": "citing_country"})
    )


def compute_counterfactual_series(
    spillover: pl.DataFrame,
    year_col: str,
    exclude_countries: list[str],
    year_range: tuple[int, int] = (1985, 2027),
) -> dict[str, pl.DataFrame]:
    """Total citation series + counterfactual series excluding each country.

    Returns
    -------
    dict with keys:
      ``"total"``       → DataFrame[year_col, n_citations]
      ``"excl_{CC}"``   → same, after dropping all CC citations
    """
    grid = pl.DataFrame(
        {year_col: list(range(year_range[0], year_range[1] + 1))}
    )

    def _fill(df: pl.DataFrame) -> pl.DataFrame:
        agg = (
            df.group_by(year_col)
            .agg(pl.col("n_citations").sum())
            .sort(year_col)
        )
        return grid.join(agg, on=year_col, how="left").with_columns(
            pl.col("n_citations").fill_null(0)
        )

    result: dict[str, pl.DataFrame] = {"total": _fill(spillover)}
    for cc in exclude_countries:
        excl = spillover.filter(pl.col("citing_country") != cc)
        result[f"excl_{cc}"] = _fill(excl)

    return result


# ============================================================================
# Plotting
# ============================================================================

_COUNTERFACTUAL_COLORS = {
    "DE": "#D4A017", "JP": "#E74C3C", "CN": "#2ECC71",
    "KR": "#9B59B6", "GB": "#3498DB", "FR": "#E67E22",
}
_COUNTERFACTUAL_STYLES: dict = {
    "DE": "--", "JP": "-.", "CN": ":",
    "KR": (0, (3, 1, 1, 1)), "GB": (0, (5, 2)), "FR": (0, (1, 1)),
}
_STACKED_PALETTE = [
    "#2C3E50", "#E74C3C", "#2ECC71", "#3498DB",
    "#9B59B6", "#E67E22", "#D4A017", "#1ABC9C", "#95A5A6",
]


def plot_counterfactual(
    series: dict[str, pl.DataFrame],
    year_col: str,
    target_country: str = "US",
    exclude_countries: list[str] | None = None,
    year_range: tuple[int, int] = (1985, 2027),
    output_path: str | None = None,
    title_suffix: str = "",
) -> str:
    """Line plot: solid total curve + dashed counterfactual lines.

    The shaded area between each counterfactual and the total indicates
    the contribution of the excluded country.
    """
    if exclude_countries is None:
        exclude_countries = ["DE", "JP", "CN"]

    target_name = COUNTRY_NAMES.get(target_country, target_country)

    fig, ax = plt.subplots(figsize=(10, 6))

    total = series["total"]
    years = total[year_col].to_list()
    total_vals = total["n_citations"].to_list()
    ax.plot(
        years, total_vals,
        color="#2C3E50", linewidth=2.2,
        label=f"All citations to {target_name}",
        zorder=10,
    )

    for cc in exclude_countries:
        key = f"excl_{cc}"
        if key not in series:
            continue
        excl_df = series[key]
        cc_name = COUNTRY_NAMES.get(cc, cc)
        excl_vals = excl_df["n_citations"].to_list()
        color = _COUNTERFACTUAL_COLORS.get(cc, "#95A5A6")

        ax.plot(
            excl_df[year_col].to_list(), excl_vals,
            color=color,
            linestyle=_COUNTERFACTUAL_STYLES.get(cc, "--"),
            linewidth=1.6, alpha=0.85,
            label=f"Without {cc_name}",
        )
        ax.fill_between(years, total_vals, excl_vals, color=color, alpha=0.07)

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Number of citations", fontsize=12)
    ax.set_title(
        f"Citation spillovers to {target_name} green patents{title_suffix}",
        fontsize=13, fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(year_range)
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )

    plt.tight_layout()
    if output_path is None:
        output_path = (
            f"PATSTAT2025FALL/output/citation_spillover_{target_country.lower()}_counterfactual.png"
        )
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")
    return output_path


def plot_composition_stacked(
    spillover: pl.DataFrame,
    year_col: str,
    target_country: str = "US",
    top_n: int = 8,
    year_range: tuple[int, int] = (1985, 2027),
    output_path: str | None = None,
) -> str:
    """Stacked area chart showing citation origin composition over time.

    Domestic (target-country) citations are placed at the bottom layer;
    remaining countries are ordered by total citation volume.
    """
    target_name = COUNTRY_NAMES.get(target_country, target_country)

    top_countries = (
        spillover
        .group_by("citing_country")
        .agg(pl.col("n_citations").sum())
        .sort("n_citations", descending=True)
        .head(top_n)
        ["citing_country"].to_list()
    )

    if target_country in top_countries:
        top_countries.remove(target_country)
        ordered = [target_country] + top_countries
    else:
        ordered = top_countries

    df = (
        spillover
        .with_columns(
            pl.when(pl.col("citing_country").is_in(ordered))
            .then(pl.col("citing_country"))
            .otherwise(pl.lit("Other"))
            .alias("group")
        )
        .group_by([year_col, "group"])
        .agg(pl.col("n_citations").sum())
        .filter(
            (pl.col(year_col) >= year_range[0])
            & (pl.col(year_col) <= year_range[1])
        )
    )

    years = sorted(df[year_col].unique().to_list())
    groups = ordered + ["Other"]

    data: dict[str, list[int]] = {}
    for g in groups:
        sub = df.filter(pl.col("group") == g)
        yr_map = dict(
            zip(sub[year_col].to_list(), sub["n_citations"].to_list())
        )
        data[g] = [yr_map.get(y, 0) for y in years]

    labels = []
    for g in groups:
        name = COUNTRY_NAMES.get(g, g)
        if g == target_country:
            name += " (domestic)"
        labels.append(name)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.stackplot(
        years,
        *[data[g] for g in groups],
        labels=labels,
        colors=_STACKED_PALETTE[: len(groups)],
        alpha=0.85,
    )

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Number of citations", fontsize=12)
    ax.set_title(
        f"Who cites {target_name} green patents?",
        fontsize=13, fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(year_range)
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )

    plt.tight_layout()
    if output_path is None:
        output_path = (
            f"PATSTAT2025FALL/output/citation_spillover_{target_country.lower()}_composition.png"
        )
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")
    return output_path


# ============================================================================
# Export & summary helpers
# ============================================================================

def export_spillover_data(
    spillover_citing: pl.DataFrame,
    spillover_cited: pl.DataFrame,
    target_country: str = "US",
) -> tuple[str, str]:
    """Write spillover tables to CSV."""
    tc = target_country.lower()
    out1 = f"PATSTAT2025FALL/output/citation_spillover_{tc}_by_citing_year.csv"
    out2 = f"PATSTAT2025FALL/output/citation_spillover_{tc}_by_cited_year.csv"
    spillover_citing.write_csv(out1)
    spillover_cited.write_csv(out2)
    print(f"  Exported: {out1}")
    print(f"  Exported: {out2}")
    return out1, out2


def print_spillover_summary(
    spillover: pl.DataFrame,
    target_country: str = "US",
) -> None:
    """Print top citing countries and their citation shares."""
    target_name = COUNTRY_NAMES.get(target_country, target_country)
    total = spillover["n_citations"].sum()

    top = (
        spillover
        .group_by("citing_country")
        .agg(pl.col("n_citations").sum())
        .sort("n_citations", descending=True)
        .with_columns(
            (pl.col("n_citations") / total * 100)
            .round(2)
            .alias("share_pct")
        )
    )

    domestic = top.filter(pl.col("citing_country") == target_country)
    domestic_share = (
        domestic["share_pct"][0] if domestic.height > 0 else 0.0
    )

    print(f"\n{'='*60}")
    print(f"Spillover summary — who cites {target_name} green patents?")
    print(f"{'='*60}")
    print(f"  Total citations received  : {total:,}")
    print(f"  Domestic share ({target_country})      : {domestic_share:.1f}%")
    print(f"  Foreign share             : {100 - domestic_share:.1f}%")
    print(f"\n  Top 15 citing countries:")
    print(top.head(15))


# ============================================================================
# Main
# ============================================================================

def main():
    TARGET = "US"
    EXCLUDE = ["DE", "JP", "CN"]
    YEAR_RANGE = (1985, 2027)
    GREEN_PARQUET = "PATSTAT2025FALL/output/green_patent8526.parquet"
    CITATION_WINDOW = None  # set to 5 for 5-year window

    print("=" * 70)
    print(f" Citation Spillover Analysis — {COUNTRY_NAMES[TARGET]} Green Patents")
    print("=" * 70)

    # 1. Identify target-country green patent families
    print("\n[1/5] Identifying target families ...")
    family_countries = load_green_family_countries(GREEN_PARQUET)
    target_ids = identify_target_families(family_countries, TARGET)
    print(f"  {TARGET} green patent families: {target_ids.height:,}")

    # 2. Load citation edges to target families
    print("\n[2/5] Loading citation edges from TLS228 ...")
    edges = load_citations_to_target(target_ids, citation_window=CITATION_WINDOW)
    print(f"  Citation edges to {TARGET} patents: {edges.height:,}")

    # 3. Build citing-family → country lookup
    print("\n[3/5] Building country lookup for citing families ...")
    family_country = build_family_country_lookup()

    # 4. Build spillover tables (by citing year and by cited year)
    print("\n[4/5] Building spillover tables ...")
    sp_citing = build_spillover_table(edges, family_country, "citing_year")
    sp_cited = build_spillover_table(edges, family_country, "cited_year")
    export_spillover_data(sp_citing, sp_cited, TARGET)

    # 5. Counterfactual analysis & plots
    print("\n[5/5] Computing counterfactuals and plotting ...")

    # 5a. Counterfactual by citing year
    series_citing = compute_counterfactual_series(
        sp_citing, "citing_year", EXCLUDE, YEAR_RANGE,
    )
    plot_counterfactual(
        series_citing, "citing_year", TARGET, EXCLUDE, YEAR_RANGE,
        output_path=f"PATSTAT2025FALL/output/citation_spillover_{TARGET.lower()}_citing_year.png",
        title_suffix=" (by citing year)",
    )

    # 5b. Counterfactual by cited year
    series_cited = compute_counterfactual_series(
        sp_cited, "cited_year", EXCLUDE, YEAR_RANGE,
    )
    plot_counterfactual(
        series_cited, "cited_year", TARGET, EXCLUDE, YEAR_RANGE,
        output_path=f"PATSTAT2025FALL/output/citation_spillover_{TARGET.lower()}_cited_year.png",
        title_suffix=" (by cited year)",
    )

    # 5c. Stacked composition by citing year
    plot_composition_stacked(
        sp_citing, "citing_year", TARGET, year_range=YEAR_RANGE,
    )

    # Summary
    print_spillover_summary(sp_citing, TARGET)

    del edges, family_country, family_countries
    gc.collect()
    print("\nDone.")


if __name__ == "__main__":
    main()
