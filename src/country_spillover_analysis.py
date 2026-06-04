"""
Country-level PageRank and Knowledge Spillover Analysis

Aggregates PageRank scores to country level by year using fractional counting.
Constructs a country-to-country knowledge spillover (citation) matrix to divide
spillovers into domestic and international parts.
Produces tabular statistics and a global map visualization of spillover flows.
"""

import os
import warnings
import numpy as np
import polars as pl
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from shapely.geometry import Point

warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update({
    "text.usetex":    True,
    "font.family":    "serif",
    "font.serif":     ["Times New Roman", "Computer Modern Roman"],
    "font.size":      10,
    "axes.titlesize": 14,
    "figure.dpi":     300,
    "axes.grid":      False,
})

PROJ = "+proj=eqearth"

OUT = "PATSTAT2025FALL/output"
OUT_VIS = os.path.join(OUT, "vis")
os.makedirs(OUT_VIS, exist_ok=True)

# ── ISO 2→3 mapping ───────────────────────────────────────────────────────────
ISO2_TO_ISO3 = {
    "AF":"AFG","AL":"ALB","DZ":"DZA","AD":"AND","AO":"AGO","AG":"ATG","AR":"ARG",
    "AM":"ARM","AU":"AUS","AT":"AUT","AZ":"AZE","BS":"BHS","BH":"BHR","BD":"BGD",
    "BB":"BRB","BY":"BLR","BE":"BEL","BZ":"BLZ","BJ":"BEN","BT":"BTN","BO":"BOL",
    "BA":"BIH","BW":"BWA","BR":"BRA","BN":"BRN","BG":"BGR","BF":"BFA","BI":"BDI",
    "CV":"CPV","KH":"KHM","CM":"CMR","CA":"CAN","CF":"CAF","TD":"TCD","CL":"CHL",
    "CN":"CHN","CO":"COL","KM":"COM","CG":"COG","CD":"COD","CR":"CRI","CI":"CIV",
    "HR":"HRV","CU":"CUB","CY":"CYP","CZ":"CZE","DK":"DNK","DJ":"DJI","DM":"DMA",
    "DO":"DOM","EC":"ECU","EG":"EGY","SV":"SLV","GQ":"GNQ","ER":"ERI","EE":"EST",
    "SZ":"SWZ","ET":"ETH","FJ":"FJI","FI":"FIN","FR":"FRA","GA":"GAB","GM":"GMB",
    "GE":"GEO","DE":"DEU","GH":"GHA","GR":"GRC","GD":"GRD","GT":"GTM","GN":"GIN",
    "GW":"GNB","GY":"GUY","HT":"HTI","HN":"HND","HU":"HUN","IS":"ISL","IN":"IND",
    "ID":"IDN","IR":"IRN","IQ":"IRQ","IE":"IRL","IL":"ISR","IT":"ITA","JM":"JAM",
    "JP":"JPN","JO":"JOR","KZ":"KAZ","KE":"KEN","KI":"KIR","KP":"PRK","KR":"KOR",
    "KW":"KWT","KG":"KGZ","LA":"LAO","LV":"LVA","LB":"LBN","LS":"LSO","LR":"LBR",
    "LY":"LBY","LI":"LIE","LT":"LTU","LU":"LUX","MG":"MDG","MW":"MWI","MY":"MYS",
    "MV":"MDV","ML":"MLI","MT":"MLT","MH":"MHL","MR":"MRT","MU":"MUS","MX":"MEX",
    "FM":"FSM","MD":"MDA","MC":"MCO","MN":"MNG","ME":"MNE","MA":"MAR","MZ":"MOZ",
    "MM":"MMR","NA":"NAM","NR":"NRU","NP":"NPL","NL":"NLD","NZ":"NZL","NI":"NIC",
    "NE":"NER","NG":"NGA","MK":"MKD","NO":"NOR","OM":"OMN","PK":"PAK","PW":"PLW",
    "PA":"PAN","PG":"PNG","PY":"PRY","PE":"PER","PH":"PHL","PL":"POL","PT":"PRT",
    "QA":"QAT","RO":"ROU","RU":"RUS","RW":"RWA","KN":"KNA","LC":"LCA","VC":"VCT",
    "WS":"WSM","SM":"SMR","ST":"STP","SA":"SAU","SN":"SEN","RS":"SRB","SC":"SYC",
    "SL":"SLE","SG":"SGP","SK":"SVK","SI":"SVN","SB":"SLB","SO":"SOM","ZA":"ZAF",
    "SS":"SSD","ES":"ESP","LK":"LKA","SD":"SDN","SR":"SUR","SE":"SWE","CH":"CHE",
    "SY":"SYR","TJ":"TJK","TZ":"TZA","TH":"THA","TL":"TLS","TG":"TGO","TO":"TON",
    "TT":"TTO","TN":"TUN","TR":"TUR","TM":"TKM","TV":"TUV","UG":"UGA","UA":"UKR",
    "AE":"ARE","GB":"GBR","US":"USA","UY":"URY","UZ":"UZB","VU":"VUT","VE":"VEN",
    "VN":"VNM","YE":"YEM","ZM":"ZMB","ZW":"ZWE",
    "TW":"TWN",
    "DD":"DEU","SU":"RUS","CS":"CZE","YU":"SRB","AN":"NLD",
}
DROP_CODES = {"EP","WO","EA","AP","OA","GC","IB","BX"}


def build_country_list(df: pl.DataFrame, family_col: str = "docdb_family_id") -> pl.DataFrame:
    """Derive one row per (family, country) using person_ctry_code with single appln_auth fallback."""
    with_person = (
        df.select([family_col, "person_ctry_code"])
        .filter(pl.col("person_ctry_code").is_not_null() & (pl.col("person_ctry_code") != ""))
        .with_columns(pl.col("person_ctry_code").str.split(","))
        .explode("person_ctry_code")
        .with_columns(pl.col("person_ctry_code").str.strip_chars().alias("country"))
        .filter(pl.col("country") != "")
        .select([family_col, "country"])
    )
    
    # Use the pre-computed 1-to-1 family to country mapping for fallback
    fam_country = pl.read_parquet(f"{OUT}/docdb_family_country.parquet")
    
    without_person = (
        df.select([family_col, "person_ctry_code"])
        .filter(pl.col("person_ctry_code").is_null() | (pl.col("person_ctry_code") == ""))
        .join(fam_country, left_on=family_col, right_on="docdb_family_id", how="inner")
        .filter(
            pl.col("country").is_not_null() &
            (pl.col("country") != "") &
            (~pl.col("country").is_in(list(DROP_CODES)))
        )
        .select([family_col, "country"])
    )
    return pl.concat([with_person, without_person]).unique()


def get_fractional_family_countries() -> pl.DataFrame:
    """Build fractional country weights for all relevant families."""
    print("Loading green and neighbor patents to build fractional weights...")
    green = pl.read_parquet(f"{OUT}/green_patent8526.parquet",
                            columns=["appln_id", "appln_auth", "docdb_family_id", "person_ctry_code"])
    green_fc = build_country_list(green)

    nb_raw = pl.read_parquet(f"{OUT}/neighbor_families.parquet", columns=["docdb_family_id", "appln_auth"])
    nb_idx = pl.read_parquet(f"{OUT}/neighbor_index.parquet")
    nb_pers = pl.read_parquet(f"{OUT}/neighbor_persons_agg.parquet")

    nb_person_ctry = (
        nb_raw.select("docdb_family_id").unique()
        .join(nb_idx, on="docdb_family_id", how="left")
        .join(nb_pers, on="appln_id", how="left")
        .filter(pl.col("person_ctry_code").is_not_null() & (pl.col("person_ctry_code") != ""))
        .with_columns(pl.col("person_ctry_code").str.split(","))
        .explode("person_ctry_code")
        .with_columns(pl.col("person_ctry_code").str.strip_chars().alias("country"))
        .filter(pl.col("country") != "")
        .select(["docdb_family_id", "country"])
        .unique()
    )

    fam_country = pl.read_parquet(f"{OUT}/docdb_family_country.parquet")
    
    nb_has_person = nb_person_ctry.select("docdb_family_id").unique()
    nb_auth_fallback = (
        nb_raw.select("docdb_family_id").unique()
        .join(nb_has_person, on="docdb_family_id", how="anti")
        .join(fam_country, on="docdb_family_id", how="inner")
        .filter(
            pl.col("country").is_not_null() &
            (pl.col("country") != "") &
            (~pl.col("country").is_in(list(DROP_CODES)))
        )
        .select(["docdb_family_id", "country"])
        .unique()
    )
    nb_fc = pl.concat([nb_person_ctry, nb_auth_fallback]).unique()

    all_fc = pl.concat([green_fc, nb_fc]).unique()
    all_fc = all_fc.filter(
        pl.col("country").str.len_chars() == 2
    ).filter(~pl.col("country").is_in(list(DROP_CODES)))

    # Compute fractional weights
    n_ctry = all_fc.group_by("docdb_family_id").agg(pl.len().alias("n_countries"))
    weighted = (
        all_fc.join(n_ctry, on="docdb_family_id", how="left")
        .with_columns((1.0 / pl.col("n_countries")).alias("weight"))
    )
    
    # Map to ISO3 and handle historical
    df_pd = weighted.to_pandas()
    df_pd["iso3"] = df_pd["country"].map(ISO2_TO_ISO3)
    # Drop rows without valid iso3
    df_pd = df_pd.dropna(subset=["iso3"])
    
    # Return as polars DataFrame
    return pl.from_pandas(df_pd[["docdb_family_id", "iso3", "weight"]]).rename({"iso3": "country"})


def compute_metrics_by_country_year(fc_weights: pl.DataFrame):
    """Aggregate PageRank and HITS to country level by year (>= 1985)."""
    print("Computing PageRank and HITS by country-year...")
    pr = pl.read_parquet(f"{OUT}/net_5yr_pagerank.parquet")
    hits = pl.read_parquet(f"{OUT}/net_5yr_hits.parquet")
    fam_year = pl.read_parquet(f"{OUT}/docdb_family_year.parquet")
    
    # Merge metrics
    metrics = pr.join(hits.select(["docdb_family_id", "hub_score", "authority_score"]), 
                      on="docdb_family_id", how="left")
    
    # Merge family year
    metrics_year = metrics.join(fam_year, on="docdb_family_id", how="inner")
    
    # Filter by year >= 1985
    metrics_year = metrics_year.filter(pl.col("family_year") >= 1985)
    
    # Merge fractional countries
    metrics_ctry = metrics_year.join(fc_weights, on="docdb_family_id", how="inner")
    
    # Compute weighted metrics
    metrics_ctry = metrics_ctry.with_columns(
        (pl.col("pagerank") * pl.col("weight")).alias("frac_pagerank"),
        (pl.col("hub_score") * pl.col("weight")).alias("frac_hub"),
        (pl.col("authority_score") * pl.col("weight")).alias("frac_authority"),
    )
    
    # Aggregate by country and year
    country_year = (
        metrics_ctry.group_by(["country", "family_year"])
        .agg(
            pl.col("frac_pagerank").sum(),
            pl.col("frac_hub").sum(),
            pl.col("frac_authority").sum()
        )
        .sort(["country", "family_year"])
    )
    
    # Aggregate by country (total)
    country_total = (
        metrics_ctry.group_by("country")
        .agg(
            pl.col("frac_pagerank").sum(),
            pl.col("frac_hub").sum(),
            pl.col("frac_authority").sum()
        )
        .sort("frac_pagerank", descending=True)
    )
    
    country_year.write_csv(f"{OUT}/country_metrics_by_year.csv")
    country_total.write_csv(f"{OUT}/country_metrics_total.csv")
    print("  -> Saved country_metrics_by_year.csv and country_metrics_total.csv")
    
    return country_year, country_total


def compute_spillover_matrix(fc_weights: pl.DataFrame):
    """Construct fractional citation spillover matrix."""
    print("Computing Country-to-Country spillover matrix...")
    edges = pl.read_parquet(f"{OUT}/citation_edges_categorized.parquet")
    
    # Join citing family countries
    edges_citing = edges.join(
        fc_weights.rename({"docdb_family_id": "docdb_family_id", "country": "citing_country", "weight": "citing_weight"}),
        on="docdb_family_id", how="inner"
    )
    
    # Join cited family countries
    edges_cited = edges_citing.join(
        fc_weights.rename({"docdb_family_id": "cited_docdb_family_id", "country": "cited_country", "weight": "cited_weight"}),
        on="cited_docdb_family_id", how="inner"
    )
    
    # Edge weight is citing_weight * cited_weight
    edges_weighted = edges_cited.with_columns(
        (pl.col("citing_weight") * pl.col("cited_weight")).alias("flow_weight")
    )
    
    # Aggregate flows between countries
    spillover_matrix = (
        edges_weighted.group_by(["citing_country", "cited_country"])
        .agg(pl.col("flow_weight").sum())
        .sort(["citing_country", "cited_country"])
    )
    
    # Pivot to N x N matrix
    pivot = spillover_matrix.to_pandas().pivot_table(
        index="citing_country", columns="cited_country", values="flow_weight", fill_value=0
    )
    pivot.to_csv(f"{OUT}/country_spillover_matrix.csv")
    print("  -> Saved country_spillover_matrix.csv")
    
    return spillover_matrix


def compute_spillover_statistics(spillover_matrix: pl.DataFrame):
    """Calculate domestic, international inflow, and international outflow."""
    print("Computing domestic and international spillover statistics...")
    
    df = spillover_matrix.with_columns(
        pl.when(pl.col("citing_country") == pl.col("cited_country"))
        .then(pl.col("flow_weight"))
        .otherwise(0).alias("domestic_flow"),
        
        pl.when(pl.col("citing_country") != pl.col("cited_country"))
        .then(pl.col("flow_weight"))
        .otherwise(0).alias("intl_flow")
    )
    
    # Domestic (can group by either citing or cited, it's the same for domestic)
    domestic = (
        df.filter(pl.col("citing_country") == pl.col("cited_country"))
        .group_by(pl.col("citing_country").alias("country"))
        .agg(pl.col("domestic_flow").sum())
    )
    
    # International Outflow (citing -> other)
    outflow = (
        df.filter(pl.col("citing_country") != pl.col("cited_country"))
        .group_by(pl.col("citing_country").alias("country"))
        .agg(pl.col("intl_flow").sum().alias("intl_outflow"))
    )
    
    # International Inflow (other -> cited)
    inflow = (
        df.filter(pl.col("citing_country") != pl.col("cited_country"))
        .group_by(pl.col("cited_country").alias("country"))
        .agg(pl.col("intl_flow").sum().alias("intl_inflow"))
    )
    
    stats = (
        domestic.to_pandas()
        .merge(outflow.to_pandas(), on="country", how="outer")
        .merge(inflow.to_pandas(), on="country", how="outer")
        .fillna(0.0)
    )
    
    # Total generated knowledge (domestic + outflow) and Total utilized knowledge (domestic + inflow)
    stats["total_outgoing"] = stats["domestic_flow"] + stats["intl_outflow"]
    stats["total_incoming"] = stats["domestic_flow"] + stats["intl_inflow"]
    
    stats = stats.sort_values("total_incoming", ascending=False)
    
    # Back to polars to write_csv
    stats = pl.from_pandas(stats)

    stats.write_csv(f"{OUT}/country_spillover_statistics.csv")
    print("  -> Saved country_spillover_statistics.csv")
    
    return stats


def plot_global_spillover_map(spillover_matrix: pl.DataFrame, stats: pl.DataFrame, country_total: pl.DataFrame, top_n_edges: int = 150):
    """Plot global maps of citation-based knowledge spillovers (oranges-reds).

    MAP 1 — choropleth of citations received (domestic + international inflow,
            fractional) plus the strongest international citation flows drawn as
            directed arrows (citing -> cited country).
    MAP 2 — choropleth of HITS authority, i.e. how strongly a country is a
            source of knowledge in the green-patent citation network.
    """
    print("Generating global map visualization of knowledge spillover...")

    CMAP = "OrRd"          # oranges-reds, consistent with the choropleth maps
    NO_DATA = "#f0f0f0"     # light grey for countries with no flow data
    EDGE = "#bbbbbb"
    FLOW_COLOR = "#08306b"  # dark navy — high contrast on an orange-red base

    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    world = world[(world.pop_est > 0) & (world.name != "Antarctica")]
    world = world.to_crs(PROJ)

    # Prepare geometries and centroids
    world["centroid"] = world.geometry.centroid
    centroids = world.set_index("iso_a3")["centroid"].to_dict()

    # Merge spillover stats and HITS authority onto the basemap
    world = world.merge(stats.to_pandas(), left_on="iso_a3", right_on="country", how="left")
    world = world.merge(country_total.to_pandas(), left_on="iso_a3", right_on="country", how="left")
    world["total_incoming"] = world["total_incoming"].fillna(0)
    world["frac_authority"] = world["frac_authority"].fillna(0)
    world["frac_pagerank"] = world["frac_pagerank"].fillna(0)

    def choropleth(ax, col, label):
        """Grey no-data base + OrRd log-scaled choropleth for positive values."""
        ax.set_axis_off()
        world.plot(ax=ax, color=NO_DATA, edgecolor=EDGE, linewidth=0.3, zorder=1)
        sub = world[world[col] > 0]
        # Robust vmin: ignore near-zero outliers (which would otherwise stretch
        # the log scale over ~90 decades and wash the map out) by clamping the
        # dynamic range to ~4 decades / the 10th percentile.
        vmax = sub[col].max()
        vmin = max(sub[col].quantile(0.10), vmax * 1e-4)
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
        sub.plot(ax=ax, column=col, cmap=CMAP, norm=norm,
                 edgecolor=EDGE, linewidth=0.3, zorder=2)
        sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm); sm._A = []
        cax = ax.inset_axes([0.05, 0.20, 0.018, 0.32])
        cbar = plt.colorbar(sm, cax=cax, orientation="vertical")
        cbar.set_label(label, fontsize=9)
        cbar.ax.tick_params(labelsize=8)
        return norm

    def hub_labels(ax, col, k=8):
        """Annotate the k largest countries with their ISO-3 code."""
        for _, r in world[world[col] > 0].nlargest(k, col).iterrows():
            c = r["centroid"]
            ax.text(c.x, c.y, str(r["iso_a3"]), fontsize=7, ha="center", va="center",
                    color="#111111", zorder=6,
                    path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])

    # -----------------------------
    # MAP 1: Citations received + international citation flows
    # -----------------------------
    fig, ax = plt.subplots(figsize=(16, 9))
    choropleth(ax, "total_incoming",
               "Citations received (domestic + inflow, fractional)")

    df_edges = spillover_matrix.to_pandas()
    df_edges = df_edges[df_edges["citing_country"] != df_edges["cited_country"]]
    df_edges = df_edges.sort_values("flow_weight", ascending=False).head(top_n_edges)
    max_flow = df_edges["flow_weight"].max()

    for _, row in df_edges.iterrows():
        src, dst, weight = row["citing_country"], row["cited_country"], row["flow_weight"]
        if src in centroids and dst in centroids:
            p1, p2 = centroids[src], centroids[dst]
            frac = weight / max_flow
            ax.annotate("",
                        xy=(p2.x, p2.y), xycoords="data",
                        xytext=(p1.x, p1.y), textcoords="data",
                        arrowprops=dict(arrowstyle="-|>", color=FLOW_COLOR,
                                        shrinkA=4, shrinkB=4,
                                        connectionstyle="arc3,rad=0.2",
                                        alpha=min(0.85, 0.12 + 0.8 * frac),
                                        linewidth=0.4 + 3.0 * frac),
                        zorder=3)

    hub_labels(ax, "total_incoming")
    ax.plot([], [], color=FLOW_COLOR, lw=2.5,
            label=r"Top international citation flows (citing $\rightarrow$ cited)")
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.set_title("Global Knowledge Spillovers via Green Patent Citations",
                 fontsize=16, pad=15)
    plt.tight_layout()
    plt.savefig(f"{OUT_VIS}/global_spillover_map.png", bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  -> Saved {OUT_VIS}/global_spillover_map.png")

    # -----------------------------
    # MAP 2: HITS authority (knowledge-source strength)
    # -----------------------------
    fig, ax = plt.subplots(figsize=(16, 9))
    choropleth(ax, "frac_authority", "HITS authority score (fractional)")
    hub_labels(ax, "frac_authority")
    ax.set_title("Knowledge-Source Authority in the Green Patent Citation Network (HITS)",
                 fontsize=16, pad=15)
    plt.tight_layout()
    plt.savefig(f"{OUT_VIS}/global_spillover_hits_authority_map.png", bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  -> Saved {OUT_VIS}/global_spillover_hits_authority_map.png")

    # -----------------------------
    # MAP 3: PageRank (overall network influence, fractional by inventor country)
    # -----------------------------
    fig, ax = plt.subplots(figsize=(16, 9))
    choropleth(ax, "frac_pagerank", "PageRank centrality (fractional)")
    hub_labels(ax, "frac_pagerank")
    ax.set_title("Network Influence (PageRank) in the Green Patent Citation Network",
                 fontsize=16, pad=15)
    plt.tight_layout()
    plt.savefig(f"{OUT_VIS}/global_spillover_pagerank_map.png", bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  -> Saved {OUT_VIS}/global_spillover_pagerank_map.png")


def main():
    print("=" * 60)
    print(" Country-level PageRank and Spillover Analysis ")
    print("=" * 60)
    
    fc_weights = get_fractional_family_countries()
    country_year, country_total = compute_metrics_by_country_year(fc_weights)
    
    spillover_matrix = compute_spillover_matrix(fc_weights)
    stats = compute_spillover_statistics(spillover_matrix)
    
    plot_global_spillover_map(spillover_matrix, stats, country_total)
    
    print("\nAnalysis complete.")

if __name__ == "__main__":
    main()
