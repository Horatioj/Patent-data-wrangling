"""
Green patent world maps — four choropleth panels.

Maps produced (Equal Earth projection):
  MAP1 — Green patent APPLICATIONS by filing office (appln_auth)
  MAP2 — Green patent FAMILIES by inventor/applicant country
  MAP3 — High-influence (H) green patent families by inventor/applicant country
  MAP4 — Neighbor patent families by inventor/applicant country
  MAP0 — 2×2 composite panel

Design choices:
  - Only countries that collectively account for the top 99% of total patent
    volume are coloured; the remainder are treated as no-data (grey).
  - Reds palette, Jenks Natural Breaks, k=6 classes.
  - Vertical inset colorbar (lower-left), following patent_vis.ipynb style.
  - LaTeX-rendered text, seaborn-v0_8-paper base style.

Run from the project root:
    python src/green_patent_maps.py
"""

import os, warnings
import numpy as np
import polars as pl
import pandas as pd
import geopandas as gpd
import mapclassify
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

# ── Style (matches patent_vis.ipynb) ──────────────────────────────────────────
plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update({
    "text.usetex":     True,
    "font.family":     "serif",
    "font.serif":      ["Times New Roman", "Computer Modern Roman"],
    "font.size":       10,
    "axes.titlesize":  14,
    "figure.dpi":      300,
    "axes.grid":       False,
})

OUT   = "PATSTAT2025FALL/output/vis"
PROJ  = "+proj=eqearth"
K     = 6          # Jenks classes
CMAP  = "Reds"
P99   = 0.999        # cumulative volume threshold
NO_DATA_COLOR = "#f2f2f2"

os.makedirs(OUT, exist_ok=True)


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
    # Taiwan — included per study design
    "TW":"TWN",
    # Historical / dissolved states — merged to primary successor:
    #   DD (East Germany, →1990)  : merged into DEU; patent system absorbed by Germany
    #   SU (Soviet Union, →1991)  : merged into RUS; Russia inherited patent apparatus
    #                               (limitation: overlooks UA, KZ, etc.)
    #   CS (Czechoslovakia, →1993): merged into CZE; Czech R. held bulk of industry
    #   YU (Yugoslavia, →1992)    : merged into SRB; Serbia inherited federal institutions
    #                               (limitation: cannot allocate to HR, SI, BA, MK, ME)
    #   AN (Netherlands Antilles, →2010): merged into NLD; constitutional relationship
    # Combined these account for <0.1% of total green families.
    "DD":"DEU", "SU":"RUS", "CS":"CZE", "YU":"SRB", "AN":"NLD",
}
DROP_CODES = {"EP", "WO", "EA", "AP", "OA", "GC", "IB"}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Load data
# ═══════════════════════════════════════════════════════════════════════════════

# Maps 2 & 3: use the pre-built CSVs which derive country from person_ctry_code
# (multi-country inventor attribution, matching count_neighbor_per_year.py).
# docdb_family_country.parquet maps each family to ONE country only — that
# single-country approach would undercount countries like JP that appear
# heavily in multinational collaborations (1.85× difference for hi-influence).
print("Loading hi_patent_country_year_complete.csv ...")
hi_csv = pd.read_csv("PATSTAT2025FALL/output/hi_patent_country_year_complete.csv",
                     keep_default_na=False)

print("Loading neighbor_patent_country_year_complete.csv ...")
nb_csv = pd.read_csv("PATSTAT2025FALL/output/neighbor_patent_country_year_complete.csv",
                     keep_default_na=False)

# Map 2: green families — use quality rankings CSV whose num_patents is
# attributed by inventor country (person_ctry_code-based), not filing office.
# This is consistent with hi/nb CSVs' attribution methodology.
print("Loading oecd_patent_quality_country_rankings.csv (green family counts) ...")
q_csv = pd.read_csv("PATSTAT2025FALL/output/oecd_patent_quality_country_rankings.csv",
                    keep_default_na=False)

print("Loading index_by_total.csv ...")
idx_raw = pd.read_csv("ClimateDataAnalysis/out/index_by_total.csv",
                      keep_default_na=False)

print("All data loaded.\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Build country-level tables
# ═══════════════════════════════════════════════════════════════════════════════

def csv_to_ctry(df: pd.DataFrame, src_col: str, val_col: str,
                new_col: str) -> pd.DataFrame:
    """
    Aggregate a pre-built country-year CSV to totals per country.
    Historical codes (DD, SU, CS, YU, AN) are remapped via ISO2_TO_ISO3
    and their counts merged into the successor state.
    """
    agg = (df.groupby(src_col, as_index=False)[val_col]
             .sum()
             .rename(columns={src_col: "iso2", val_col: new_col}))
    agg["iso3"] = agg["iso2"].map(ISO2_TO_ISO3)
    # Merge historical successors: group by iso3 to combine DD+DE, SU+RU, etc.
    agg = (agg.dropna(subset=["iso3"])
              .groupby("iso3", as_index=False)
              .agg(iso2=("iso2", "first"), **{new_col: (new_col, "sum")}))
    return agg


# Map 1 — green families by inventor country
# Source: oecd_patent_quality_country_rankings.csv → 'countries' (2-letter),
# 'num_patents' = inventor-attributed green patent count (person_ctry_code based).
fam = (q_csv.rename(columns={"countries": "iso2", "num_patents": "green_fam"})
       [["iso2", "green_fam"]].copy())
fam["iso3"] = fam["iso2"].map(ISO2_TO_ISO3)
fam = (fam.dropna(subset=["iso3"])
          .groupby("iso3", as_index=False)
          .agg(iso2=("iso2", "first"), green_fam=("green_fam", "sum")))

# Map 2 — high-influence families (person_ctry_code, multi-country attribution)
hi_ctry = csv_to_ctry(hi_csv, "countries", "num_hi_patents", "hi_fam")

# Map 3 — neighbor families (person_ctry_code, multi-country attribution)
nb_ctry = csv_to_ctry(nb_csv, "countries", "num_neighbor_patents", "nb_fam")

# Map 4 — climate patent leadership index
# Remap 2-letter → 3-letter codes; merge historical successors
idx_df = idx_raw.rename(columns={"country": "iso2"}).copy()
idx_df["iso3"] = idx_df["iso2"].map(ISO2_TO_ISO3)
idx_df = (idx_df.dropna(subset=["iso3"])
          .groupby("iso3", as_index=False)
          .agg(
              iso2        = ("iso2",         "first"),
              num_patents = ("num_patents",  "sum"),
              index_score = ("index_score",  lambda x:
                             np.average(x, weights=idx_df.loc[x.index, "num_patents"])),
          ))

print("Tables ready.\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Top-99% volume filter
#    stats_idx['cumsum'] = stats_idx[col].cumsum() / stats_idx[col].sum()
#    Keep countries where cumsum <= P99 (plus the first one that crosses P99).
# ═══════════════════════════════════════════════════════════════════════════════
def top99_iso3(df: pd.DataFrame, col: str) -> set:
    """Return the set of iso3 codes whose cumulative share covers the top 99%.
    df must already have an 'iso3' column (historical codes merged upstream).
    """
    stats_idx = (df.dropna(subset=["iso3"])[["iso3", col]]
                   .groupby("iso3", as_index=False)[col].sum()
                   .sort_values(col, ascending=False)
                   .reset_index(drop=True))
    stats_idx["cumsum"] = stats_idx[col].cumsum() / stats_idx[col].sum()
    cutoff_idx = (stats_idx["cumsum"] >= P99).idxmax()
    included = stats_idx.loc[:cutoff_idx, "iso3"]
    print(f"  [{col}] top-{int(P99*100)}%: {len(included)} countries "
          f"(cumsum at cutoff = {stats_idx.loc[cutoff_idx,'cumsum']:.4f})")
    return set(included)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. World basemap
# ═══════════════════════════════════════════════════════════════════════════════
world_base = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
world_base  = world_base[(world_base.pop_est > 0) & (world_base.name != "Antarctica")]
world_base  = world_base.to_crs(PROJ)


def merge_world(df: pd.DataFrame, col: str, allowed_iso3: set) -> gpd.GeoDataFrame:
    """
    Left-join df onto world basemap by iso3.  Countries outside allowed_iso3
    (below the 99% volume threshold) are set to NaN (shown as grey).
    Historical-code merging is already done upstream in csv_to_ctry / apps building.
    """
    agg = df.dropna(subset=["iso3"])[["iso3", col]].copy()
    agg.loc[~agg["iso3"].isin(allowed_iso3), col] = np.nan
    return world_base.merge(agg, left_on="iso_a3", right_on="iso3", how="left")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. plot_choropleth — faithful to patent_vis.ipynb style
#    - #f2f2f2 background for all countries (including no-data)
#    - scheme='user_defined' with Jenks breaks
#    - vertical inset colorbar at lower-left [0.12, 0.08, 0.02, 0.35]
#    - integer-formatted tick labels
#    - no text on map face
# ═══════════════════════════════════════════════════════════════════════════════
def plot_choropleth(ax: plt.Axes,
                    world_proj: gpd.GeoDataFrame,
                    col: str,
                    cmap_name: str,
                    title: str | None = None,
                    k: int = K):
    """
    Choropleth following patent_vis.ipynb plot_choropleth() style.
    Countries with no data (NaN or zero) show in #f2f2f2.
    Only countries in the top-99% volume bracket are coloured.
    """
    ax.set_axis_off()

    # Full background: all countries in light grey first
    world_proj.plot(ax=ax, color=NO_DATA_COLOR,
                    edgecolor="#cccccc", linewidth=0.3, zorder=1)

    # Subset with valid, positive values
    data = world_proj.dropna(subset=[col])
    valid_mask = data[col] > 0
    valid = data[valid_mask][col]

    # Jenks Natural Breaks
    classifier = mapclassify.NaturalBreaks(valid, k=k)
    breaks = classifier.bins                          # k upper boundaries
    bounds = [float(valid.min())] + list(breaks)     # k+1 boundary values

    cmap_obj = plt.cm.get_cmap(cmap_name)
    norm = mcolors.BoundaryNorm(bounds, cmap_obj.N)

    # Draw choropleth (only rows with data)
    data[valid_mask].plot(
        ax=ax,
        column=col,
        cmap=cmap_name,
        scheme="user_defined",
        classification_kwds={"bins": breaks},
        edgecolor="#aaaaaa",
        linewidth=0.2,
        legend=False,
        zorder=2,
    )

    # Vertical inset colorbar — lower-left, same position as notebook
    cax = ax.inset_axes([0.12, 0.08, 0.02, 0.35])
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm._A = []
    cbar = plt.colorbar(sm, cax=cax, orientation="vertical", spacing="uniform")
    cbar.outline.set_linewidth(0.5)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_ticks(bounds)
    cbar.set_ticklabels(
        [rf"$\geq{int(bounds[0]):,}$"] +
        [rf"${int(b):,}$" for b in bounds[1:]],
        fontsize=7,
    )

    # No-data legend patch
    no_data_patch = mpatches.Patch(
        facecolor=NO_DATA_COLOR, edgecolor="#cccccc",
        linewidth=0.4, label=r"No data / below 99.9\% threshold",
    )
    ax.legend(
        handles=[no_data_patch],
        loc="lower right",
        fontsize=6.5,
        frameon=False,
        handlelength=1.0,
        handleheight=0.8,
    )
    if title:
        ax.set_title(title, pad=6, fontsize=11)


# ═══════════════════════════════════════════════════════════════════════════════
# 5b. plot_choropleth_rate — for proportions [0, 1]
#     Uses Jenks on raw values (no log), percentage tick labels.
# ═══════════════════════════════════════════════════════════════════════════════
def plot_choropleth_index(ax: plt.Axes,
                          world_proj: gpd.GeoDataFrame,
                          col: str,
                          cmap_name: str,
                          title: str | None = None,
                          k: int = K):
    """Choropleth for composite index scores; decimal colorbar labels."""
    ax.set_axis_off()
    world_proj.plot(ax=ax, color=NO_DATA_COLOR,
                    edgecolor="#cccccc", linewidth=0.3, zorder=1)

    data      = world_proj.dropna(subset=[col])
    valid_ser = data[col]
    valid     = valid_ser[valid_ser.notna()]

    classifier = mapclassify.NaturalBreaks(valid, k=k)
    breaks  = classifier.bins
    lo      = float(valid.min())
    bounds  = [lo] + list(breaks)

    cmap_obj = plt.cm.get_cmap(cmap_name)
    norm     = mcolors.BoundaryNorm(bounds, cmap_obj.N)

    data.plot(
        ax=ax, column=col, cmap=cmap_name,
        scheme="user_defined",
        classification_kwds={"bins": breaks},
        edgecolor="#aaaaaa", linewidth=0.2,
        legend=False, zorder=2,
    )

    cax = ax.inset_axes([0.12, 0.08, 0.02, 0.35])
    sm  = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm._A = []
    cbar = plt.colorbar(sm, cax=cax, orientation="vertical", spacing="uniform")
    cbar.outline.set_linewidth(0.5)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_ticks(bounds)
    cbar.set_ticklabels(
        [rf"$\geq{bounds[0]:.2f}$"] + [rf"${b:.2f}$" for b in bounds[1:]],
        fontsize=7,
    )

    no_data_patch = mpatches.Patch(
        facecolor=NO_DATA_COLOR, edgecolor="#cccccc",
        linewidth=0.4, label=r"No data / below 99.9\% threshold",
    )
    ax.legend(handles=[no_data_patch], loc="lower right",
              fontsize=6.5, frameon=False, handlelength=1.0, handleheight=0.8)
    if title:
        ax.set_title(title, pad=6, fontsize=11)


def plot_choropleth_rate(ax: plt.Axes,
                         world_proj: gpd.GeoDataFrame,
                         col: str,
                         cmap_name: str,
                         title: str | None = None,
                         k: int = K):
    """Choropleth for ratio/proportion columns; labels formatted as percentages."""
    ax.set_axis_off()
    world_proj.plot(ax=ax, color=NO_DATA_COLOR,
                    edgecolor="#cccccc", linewidth=0.3, zorder=1)

    data = world_proj.dropna(subset=[col])
    valid_mask = (data[col] > 0) & (data[col] <= 1)
    valid = data[valid_mask][col]


    classifier = mapclassify.NaturalBreaks(valid, k=k)
    breaks = classifier.bins
    bounds = [float(valid.min())] + list(breaks)

    cmap_obj = plt.cm.get_cmap(cmap_name)
    norm = mcolors.BoundaryNorm(bounds, cmap_obj.N)

    data[valid_mask].plot(
        ax=ax, column=col, cmap=cmap_name,
        scheme="user_defined",
        classification_kwds={"bins": breaks},
        edgecolor="#aaaaaa", linewidth=0.2,
        legend=False, zorder=2,
    )

    cax = ax.inset_axes([0.12, 0.08, 0.02, 0.35])
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm._A = []
    cbar = plt.colorbar(sm, cax=cax, orientation="vertical", spacing="uniform")
    cbar.outline.set_linewidth(0.5)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_ticks(bounds)
    cbar.set_ticklabels(
        [rf"$\geq{bounds[0]*100:.0f}\%$"] +
        [rf"${b*100:.0f}\%$" for b in bounds[1:]],
        fontsize=7,
    )

    no_data_patch = mpatches.Patch(
        facecolor=NO_DATA_COLOR, edgecolor="#cccccc",
        linewidth=0.4, label=r"No data / $<$10 green families",
    )
    ax.legend(handles=[no_data_patch], loc="lower right",
              fontsize=6.5, frameon=False, handlelength=1.0, handleheight=0.8)
    if title:
        ax.set_title(title, pad=6, fontsize=11)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Draw four individual maps
# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER = (r"\textit{Equal Earth projection} $\cdot$ "
#           r"PATSTAT 2025 Autumn $\cdot$ "
#           r"Reds / Jenks natural breaks ($k=6$) $\cdot$ "
#           r"Top 99\% volume countries coloured")


def make_single_map(gdf, col, fname, allowed):
    gdf_filtered = merge_world(
        # re-run merge so NaN is set correctly for this specific col
        gdf[["iso_a3", col]].rename(columns={"iso_a3": "iso3"})
        if "iso_a3" in gdf.columns
        else gdf,
        col,
        allowed,
    ) if False else gdf   # gdf already merged; re-use

    fig, ax = plt.subplots(figsize=(14, 7.5))
    plot_choropleth(ax, gdf_filtered, col, CMAP)
    # ax.text(0.99, -0.01, FOOTER, transform=ax.transAxes,
            # ha="right", va="top", fontsize=6.5, color="#777777")
    fig.tight_layout(rect=[0, 0.01, 1, 1])
    path = f"{OUT}/{fname}"
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved {fname}")


# --- Compute 99.9%-inclusion sets per metric
print("Computing top-99.9% country sets ...")
inc_fam  = top99_iso3(fam,      "green_fam")
inc_hi   = top99_iso3(hi_ctry,  "hi_fam")
inc_nb   = top99_iso3(nb_ctry,  "nb_fam")
inc_idx  = top99_iso3(idx_df,   "num_patents")

# --- Merge onto world GDF
world_fam  = merge_world(fam,      "green_fam",  inc_fam)
world_hi   = merge_world(hi_ctry,  "hi_fam",     inc_hi)
world_nb   = merge_world(nb_ctry,  "nb_fam",     inc_nb)

# Index map: filter to top-99.9% by num_patents, then display index_score
idx_display = idx_df[["iso3", "index_score", "num_patents"]].copy()
idx_display.loc[~idx_display["iso3"].isin(inc_idx), "index_score"] = np.nan
world_idx = world_base.merge(
    idx_display[["iso3", "index_score"]].dropna(subset=["index_score"]),
    left_on="iso_a3", right_on="iso3", how="left"
)

# Map titles
TITLE_FAM = r"Quantity of Climate Related Patent Families"
TITLE_HI  = r"Quantity of High Influential Climate Patent Families"
TITLE_NB  = r"Quantity of Non-Climate Neighbor Patent Families"
TITLE_IDX = r"Climate Patent Leadership Index"

MAPS = [
    (world_fam, "green_fam",    TITLE_FAM, "MAP1_green_families.png",    "choropleth"),
    (world_hi,  "hi_fam",       TITLE_HI,  "MAP2_hi_green_families.png", "choropleth"),
    (world_nb,  "nb_fam",       TITLE_NB,  "MAP3_neighbor_families.png", "choropleth"),
    (world_idx, "index_score",  TITLE_IDX, "MAP4_index_score.png",       "index"),
]

for gdf, col, title, fname, kind in MAPS:
    fig, ax = plt.subplots(figsize=(14, 7.5))
    if kind == "index":
        plot_choropleth_index(ax, gdf, col, CMAP, title)
    else:
        plot_choropleth(ax, gdf, col, CMAP, title)
    fig.suptitle(title, fontsize=13, y=1.002)
    fig.tight_layout(rect=[0, 0.01, 1, 1])
    fig.savefig(f"{OUT}/{fname}", bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved {fname}")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. 2×2 composite
# ═══════════════════════════════════════════════════════════════════════════════
print("\nDrawing 2x2 composite ...")

PANEL_LABELS = [
    r"\textbf{(a)} " + TITLE_FAM,
    r"\textbf{(b)} " + TITLE_HI,
    r"\textbf{(c)} " + TITLE_NB,
    r"\textbf{(d)} " + TITLE_IDX,
]

fig, axes = plt.subplots(2, 2, figsize=(20, 11),
                         gridspec_kw={"hspace": 0.14, "wspace": 0.04})
axes = axes.flatten()

for ax, (gdf, col, title, fname, kind), label in zip(axes, MAPS, PANEL_LABELS):
    if kind == "index":
        plot_choropleth_index(ax, gdf, col, CMAP, label)
    else:
        plot_choropleth(ax, gdf, col, CMAP, label)

# fig.suptitle(
#     r"\textbf{Climate Patent Geography} $\cdot$ PATSTAT 2025 Autumn "
#     r"$\cdot$ Reds $\cdot$ Jenks Natural Breaks ($k=6$) "
#     r"$\cdot$ Equal Earth $\cdot$ Top 99.9\% volume countries",
#     fontsize=11, y=1.003,
# )
fig.savefig(f"{OUT}/MAP0_composite_4panel.png", bbox_inches="tight", dpi=300)
plt.close()
print("Saved MAP0_composite_4panel.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. MAP5 — High-influence rate  (hi_fam / green_fam) by inventor country
#    Both numerator and denominator use full counting, same methodology.
#    Minimum 10 green families required to suppress noise from tiny countries.
# ═══════════════════════════════════════════════════════════════════════════════
print("\nBuilding hi-influence rate table ...")

# Join on iso3 so historical successors (DD+DE→DEU etc.) are already merged.
rate_df = (fam[["iso3", "green_fam"]]
    .merge(hi_ctry[["iso3", "hi_fam"]], on="iso3", how="left")
    .assign(hi_fam=lambda d: d["hi_fam"].fillna(0))
)
# Require at least 10 green families to plot a meaningful rate
rate_df.loc[rate_df["green_fam"] < 10, "hi_rate"] = np.nan
rate_df.loc[rate_df["green_fam"] >= 10, "hi_rate"] = (
    rate_df.loc[rate_df["green_fam"] >= 10, "hi_fam"] /
    rate_df.loc[rate_df["green_fam"] >= 10, "green_fam"]
)

# All countries with ≥10 green families are shown (no 99% filter — rate is
# not volume-weighted, so small countries are equally informative)
world_rate = world_base.merge(
    rate_df[["iso3", "hi_rate"]].dropna(subset=["hi_rate"]),
    left_on="iso_a3", right_on="iso3", how="left"
)

HI_RATE_TITLE = r"High-Influence Rate: Hi-Influence Families $\div$ Green Families"
fig5, ax5 = plt.subplots(figsize=(14, 7.5))
plot_choropleth_rate(
    ax5, world_rate, "hi_rate", CMAP, HI_RATE_TITLE,
    # r"\textbf{High-Influence Rate}: Hi-Influence Families $\div$ Green Families",
    # subtitle=(
    #     r"Share of green patent families meeting high-influence criteria "
    #     r"(top citations + family size / triadic) $\cdot$ "
    #     r"Countries with $\geq$10 green families shown $\cdot$ "
    #     r"Full inventor-country attribution"
    # ),
)
# FOOTER5 = (r"\textit{Equal Earth projection} $\cdot$ "
#            r"PATSTAT 2025 Autumn $\cdot$ "
#            r"Reds / Jenks natural breaks ($k=6$) $\cdot$ "
#            r"Quality dimension: intensity, not volume")
fig5.suptitle(HI_RATE_TITLE, fontsize=13, y=1.002)
fig5.tight_layout(rect=[0, 0.01, 1, 1])
fig5.savefig(f"{OUT}/MAP5_hi_rate.png", bbox_inches="tight", dpi=300)
plt.close()
print("Saved MAP5_hi_rate.png")

# Print top/bottom countries by hi-rate (min 50 green families)
rate_top = (rate_df[rate_df["green_fam"] >= 50]
    .sort_values("hi_rate", ascending=False)
    [["iso3", "green_fam", "hi_fam", "hi_rate"]])
print("\nHi-influence rate -- top & bottom 10 (min 50 green families):")
print(rate_top.head(10).to_string(index=False))
print("...")
print(rate_top.tail(10).to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Summary CSV
# ═══════════════════════════════════════════════════════════════════════════════
summary = (
    fam[["iso3", "green_fam"]]
    .merge(hi_ctry[["iso3", "hi_fam"]],              on="iso3", how="outer")
    .merge(nb_ctry[["iso3", "nb_fam"]],              on="iso3", how="outer")
    .merge(idx_df[["iso3", "index_score", "num_patents"]], on="iso3", how="outer")
    .rename(columns={"iso3": "Country"})
    .sort_values("green_fam", ascending=False)
)
summary.to_csv(f"{OUT}/country_summary.csv", index=False)

print("\n" + "=" * 70)
print("COUNTRY SUMMARY — top 25 by green families")
print("=" * 70)
print(summary.head(25).to_string(index=False))
print(f"\nCSV  -> {OUT}/country_summary.csv")
print(f"Maps -> {OUT}/MAP0-MAP5.png")
