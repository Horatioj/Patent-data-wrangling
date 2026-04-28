"""
Green patent world maps — fractional inventor-country counting.

Each patent family is credited to each of its inventor/applicant countries
with weight  w = 1 / n_countries  (fractional counting).  The global sum of
fractional counts equals the number of unique DOCDB families, unlike full
counting where the sum = n_families × avg_countries_per_family.

Maps produced (Equal Earth projection, Reds palette, Jenks k=6):
  FRAC1 — Green patent families        (fractional)
  FRAC2 — High-influence green families (fractional)
  FRAC3 — Neighbor patent families      (fractional)
  FRAC4 — Hi-influence rate             (FRAC2 / FRAC1)
  FRAC0 — 2×2 composite (FRAC1-FRAC3 + FRAC4)

Also saves country_summary_fractional.csv for comparison with full counting.

Run from the project root:
    python src/green_patent_maps_fractional.py
"""

import os, warnings
import numpy as np
import polars as pl
import pandas as pd
import geopandas as gpd
import mapclassify
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

# ── Style ─────────────────────────────────────────────────────────────────────
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

OUT  = "PATSTAT2025FALL/output/vis"
PROJ = "+proj=eqearth"
K    = 6
CMAP = "Reds"
P99  = 0.999          # cumulative volume threshold for colour inclusion
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
    #   DD → DEU  (East Germany absorbed into Germany after reunification)
    #   SU → RUS  (Russia inherited Soviet patent system)
    #   CS → CZE  (Czech R. inherited bulk of Czechoslovak industry)
    #   YU → SRB  (Serbia inherited Yugoslav federal institutions)
    #   AN → NLD  (Netherlands Antilles had constitutional link with Netherlands)
    "DD":"DEU","SU":"RUS","CS":"CZE","YU":"SRB","AN":"NLD",
}
DROP_CODES  = {"EP","WO","EA","AP","OA","GC","IB","BX"}
EXCLUDE_ISO2: set = set()   # Taiwan is now included


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Build fractional-weight country tables from raw parquets
# ═══════════════════════════════════════════════════════════════════════════════

def build_country_list(df: pl.DataFrame,
                       family_col: str = "docdb_family_id") -> pl.DataFrame:
    """
    From a green-patent-style DataFrame, derive one row per (family, country)
    using person_ctry_code (inventor) with appln_auth fallback.
    Returns a DataFrame with columns [family_col, 'country'].
    """
    # Inventor country (primary source)
    with_person = (
        df.select([family_col, "person_ctry_code"])
        .filter(pl.col("person_ctry_code").is_not_null() &
                (pl.col("person_ctry_code") != ""))
        .with_columns(pl.col("person_ctry_code").str.split(","))
        .explode("person_ctry_code")
        .with_columns(pl.col("person_ctry_code").str.strip_chars().alias("country"))
        .filter(pl.col("country") != "")
        .select([family_col, "country"])
    )
    # Fallback: filing office for applications without inventor data
    without_person = (
        df.select([family_col, "appln_auth", "person_ctry_code"])
        .filter(pl.col("person_ctry_code").is_null() |
                (pl.col("person_ctry_code") == ""))
        .filter(
            pl.col("appln_auth").is_not_null() &
            (pl.col("appln_auth") != "") &
            (~pl.col("appln_auth").is_in(list(DROP_CODES)))
        )
        .select([family_col, pl.col("appln_auth").alias("country")])
    )
    return pl.concat([with_person, without_person]).unique()


def fractional_counts(family_country: pl.DataFrame,
                      family_col: str = "docdb_family_id",
                      val_col: str = "frac_count") -> pd.DataFrame:
    """
    Given a (family_id, country) table, compute fractional weight = 1/n_countries
    per family, then sum by country.
    Historical codes (DD, SU, CS, YU, AN) are remapped via ISO2_TO_ISO3 and
    their fractional counts merged into the successor state.
    Returns pd.DataFrame[country, iso3, val_col].
    """
    n_ctry = (family_country
              .group_by(family_col)
              .agg(pl.len().alias("n_countries")))
    weighted = (family_country
                .join(n_ctry, on=family_col, how="left")
                .with_columns((1.0 / pl.col("n_countries")).alias("weight")))
    result = (weighted
              .group_by("country")
              .agg(pl.col("weight").sum().alias(val_col))
              .to_pandas())
    result["iso3"] = result["country"].map(ISO2_TO_ISO3)
    # Merge historical successors by grouping on iso3
    result = (result.dropna(subset=["iso3"])
                    .groupby("iso3", as_index=False)
                    .agg(country=("country", "first"), **{val_col: (val_col, "sum")}))
    return result


# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading green_patent8526.parquet ...")
green = pl.read_parquet("PATSTAT2025FALL/output/green_patent8526.parquet",
    columns=["appln_id", "appln_auth", "docdb_family_id",
             "person_ctry_code"])

print("Loading high_influence_green_patents.parquet ...")
hi_raw = pl.read_parquet("PATSTAT2025FALL/output/high_influence_green_patents.parquet",
    columns=["appln_id", "appln_auth", "docdb_family_id", "person_ctry_code"])

print("Loading neighbor_families.parquet, neighbor_index.parquet, neighbor_persons_agg.parquet ...")
nb_raw  = pl.read_parquet("PATSTAT2025FALL/output/neighbor_families.parquet",
    columns=["docdb_family_id", "appln_auth"])
nb_idx  = pl.read_parquet("PATSTAT2025FALL/output/neighbor_index.parquet")
nb_pers = pl.read_parquet("PATSTAT2025FALL/output/neighbor_persons_agg.parquet")

print("All data loaded.\n")

# ── Green families: fractional ────────────────────────────────────────────────
print("Building green family country lists ...")
green_fc = build_country_list(green)
# deduplicate to unique (family, country) pairs across all applications
green_fc = green_fc.unique()
# keep only valid 2-char codes, drop regional offices
green_fc = green_fc.filter(
    pl.col("country").str.len_chars() == 2
).filter(~pl.col("country").is_in(list(DROP_CODES)))

frac_green = fractional_counts(green_fc, val_col="frac_green_fam")
total_frac_green = frac_green["frac_green_fam"].sum()
print(f"  Fractional green family total: {total_frac_green:,.0f}  "
      f"(should be ~635,359 unique families)")

# ── Hi-influence families: fractional ─────────────────────────────────────────
print("Building hi-influence family country lists ...")
hi_fc = build_country_list(hi_raw)
hi_fc = hi_fc.unique().filter(
    pl.col("country").str.len_chars() == 2
).filter(~pl.col("country").is_in(list(DROP_CODES)))

frac_hi = fractional_counts(hi_fc, val_col="frac_hi_fam")
total_frac_hi = frac_hi["frac_hi_fam"].sum()
print(f"  Fractional hi-influence total: {total_frac_hi:,.0f}  "
      f"(should be ~304,858 unique hi families)")

# ── Neighbor families: fractional ─────────────────────────────────────────────
print("Building neighbor family country lists (join chain) ...")

# Step 1: get person_ctry_code per neighbor family via index → persons
nb_person_ctry = (
    nb_raw.select("docdb_family_id").unique()
    .join(nb_idx, on="docdb_family_id", how="left")
    .join(nb_pers, on="appln_id", how="left")
    .filter(pl.col("person_ctry_code").is_not_null() &
            (pl.col("person_ctry_code") != ""))
    .with_columns(pl.col("person_ctry_code").str.split(","))
    .explode("person_ctry_code")
    .with_columns(pl.col("person_ctry_code").str.strip_chars().alias("country"))
    .filter(pl.col("country") != "")
    .select(["docdb_family_id", "country"])
    .unique()
)

# Step 2: appln_auth fallback for families with no person data
nb_has_person = nb_person_ctry.select("docdb_family_id").unique()
nb_auth_fallback = (
    nb_raw
    .join(nb_has_person, on="docdb_family_id", how="anti")
    .with_columns(pl.col("appln_auth").str.split(","))
    .explode("appln_auth")
    .with_columns(pl.col("appln_auth").str.strip_chars().alias("country"))
    .filter(
        pl.col("country").is_not_null() &
        (pl.col("country") != "") &
        (~pl.col("country").is_in(list(DROP_CODES)))
    )
    .select(["docdb_family_id", "country"])
    .unique()
)

nb_fc = pl.concat([nb_person_ctry, nb_auth_fallback]).unique()
nb_fc = nb_fc.filter(
    pl.col("country").str.len_chars() == 2
).filter(~pl.col("country").is_in(list(DROP_CODES)))

frac_nb = fractional_counts(nb_fc, val_col="frac_nb_fam")
total_frac_nb = frac_nb["frac_nb_fam"].sum()
print(f"  Fractional neighbor total: {total_frac_nb:,.0f}  "
      f"(should be ~6,475,040 unique neighbor families)")

# ── Leadership index (same CSV as full-counting script) ────────────────────────
print("Loading index_by_total.csv ...")
idx_raw = pd.read_csv("ClimateDataAnalysis/out/index_by_total.csv",
                      keep_default_na=False)
idx_raw = idx_raw.rename(columns={"country": "iso2"})
idx_raw["iso3"] = idx_raw["iso2"].map(ISO2_TO_ISO3)
idx_df = (idx_raw.dropna(subset=["iso3"])
          .groupby("iso3", as_index=False)
          .agg(
              iso2        = ("iso2",         "first"),
              num_patents = ("num_patents",   "sum"),
              index_score = ("index_score",   lambda x:
                             np.average(x, weights=idx_raw.loc[x.index, "num_patents"])),
          ))


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  World basemap
# ═══════════════════════════════════════════════════════════════════════════════
world_base = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
world_base  = world_base[(world_base.pop_est > 0) & (world_base.name != "Antarctica")]
world_base  = world_base.to_crs(PROJ)


def top_pct_iso3(df: pd.DataFrame, col: str) -> set:
    """Return iso3 codes covering the top P99 cumulative share of *col*."""
    stats = (df.dropna(subset=["iso3"])[["iso3", col]]
               .groupby("iso3", as_index=False)[col].sum()
               .sort_values(col, ascending=False)
               .reset_index(drop=True))
    stats["cumsum"] = stats[col].cumsum() / stats[col].sum()
    cutoff = (stats["cumsum"] >= P99).idxmax()
    included = stats.loc[:cutoff, "iso3"]
    print(f"  [{col}] top-{P99*100:.1f}%: {len(included)} countries "
          f"(cumsum at cutoff = {stats.loc[cutoff,'cumsum']:.4f})")
    return set(included)


def merge_world_frac(df: pd.DataFrame, col: str,
                     allowed_iso3: set | None = None) -> gpd.GeoDataFrame:
    """Merge fractional counts onto the world basemap.
    Countries outside *allowed_iso3* are set to NaN (rendered grey)."""
    agg = (df.dropna(subset=["iso3"])
             .groupby("iso3", as_index=False)[col].sum())
    if allowed_iso3 is not None:
        agg.loc[~agg["iso3"].isin(allowed_iso3), col] = np.nan
    return world_base.merge(agg, left_on="iso_a3", right_on="iso3", how="left")


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Choropleth functions
# ═══════════════════════════════════════════════════════════════════════════════
def plot_frac_choropleth(ax: plt.Axes,
                         world_proj: gpd.GeoDataFrame,
                         col: str,
                         cmap_name: str,
                         title: str,
                         subtitle: str = "",
                         k: int = K):
    """Choropleth for fractional (non-integer) count columns.
    All countries with valid data are coloured; grey = no data.
    Jenks applied to raw values (no log) since fractional counts are
    smoother than integer counts.  Tick labels show rounded integers.
    """
    ax.set_axis_off()
    world_proj.plot(ax=ax, color=NO_DATA_COLOR,
                    edgecolor="#cccccc", linewidth=0.3, zorder=1)

    data = world_proj.dropna(subset=[col])
    valid_mask = data[col] > 0
    valid = data[valid_mask][col]

    if valid.empty:
        ax.set_title(title, fontsize=14, pad=8)
        return

    # Apply log10 transform for count maps (same approach as full-counting maps)
    log_valid = np.log10(valid + 1)
    classifier = mapclassify.NaturalBreaks(log_valid, k=k)
    log_breaks = classifier.bins
    log_bounds = [float(log_valid.min())] + list(log_breaks)
    raw_bounds  = [round(10**b - 1) for b in log_bounds]
    print(f"  [{col}] raw bounds: {raw_bounds}")

    cmap_obj = plt.cm.get_cmap(cmap_name)
    norm = mcolors.BoundaryNorm(log_bounds, cmap_obj.N)

    # Assign log values for plotting
    world_proj_log = world_proj.copy()
    world_proj_log[col + "_log"] = np.log10(world_proj_log[col].clip(lower=0) + 1)
    data_log = world_proj_log.dropna(subset=[col])
    data_log[valid_mask].plot(
        ax=ax, column=col + "_log", cmap=cmap_name,
        vmin=log_bounds[0], vmax=log_bounds[-1],
        edgecolor="#aaaaaa", linewidth=0.2, legend=False, zorder=2,
    )

    cax = ax.inset_axes([0.12, 0.08, 0.02, 0.35])
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm._A = []
    cbar = plt.colorbar(sm, cax=cax, orientation="vertical", spacing="uniform")
    cbar.outline.set_linewidth(0.5)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_ticks(log_bounds)
    cbar.set_ticklabels(
        [rf"$\geq{raw_bounds[0]:,}$"] +
        [rf"${raw_bounds[i]:,}$" for i in range(1, len(raw_bounds))],
        fontsize=7,
    )

    no_data_patch = mpatches.Patch(
        facecolor=NO_DATA_COLOR, edgecolor="#cccccc",
        linewidth=0.4, label=r"No data / below 99.9\% threshold",
    )
    ax.legend(handles=[no_data_patch], loc="lower right",
              fontsize=6.5, frameon=False, handlelength=1.0, handleheight=0.8)
    ax.set_title(title, fontsize=14, pad=8)
    if subtitle:
        ax.text(0.5, 1.003, subtitle, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=7.5, color="#444444")


def plot_frac_rate(ax: plt.Axes,
                   world_proj: gpd.GeoDataFrame,
                   col: str,
                   cmap_name: str,
                   title: str,
                   subtitle: str = "",
                   k: int = K):
    """Choropleth for hi-influence rate (proportion); percentage tick labels."""
    ax.set_axis_off()
    world_proj.plot(ax=ax, color=NO_DATA_COLOR,
                    edgecolor="#cccccc", linewidth=0.3, zorder=1)

    data = world_proj.dropna(subset=[col])
    valid_mask = (data[col] > 0) & (data[col] <= 1)
    valid = data[valid_mask][col]

    if valid.empty:
        ax.set_title(title, fontsize=14, pad=8)
        return

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
    ax.set_title(title, fontsize=14, pad=8)
    if subtitle:
        ax.text(0.5, 1.003, subtitle, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=7.5, color="#444444")


def plot_frac_choropleth_index(ax: plt.Axes,
                               world_proj: gpd.GeoDataFrame,
                               col: str,
                               cmap_name: str,
                               title: str,
                               k: int = K):
    """Choropleth for composite index scores; decimal colorbar labels."""
    ax.set_axis_off()
    world_proj.plot(ax=ax, color=NO_DATA_COLOR,
                    edgecolor="#cccccc", linewidth=0.3, zorder=1)

    data  = world_proj.dropna(subset=[col])
    valid = data[col]

    if valid.empty:
        ax.set_title(title, fontsize=14, pad=8)
        return

    classifier = mapclassify.NaturalBreaks(valid, k=k)
    breaks = classifier.bins
    bounds = [float(valid.min())] + list(breaks)

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
    ax.set_title(title, fontsize=14, pad=8)


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Build world GDFs
# ═══════════════════════════════════════════════════════════════════════════════
# ── Compute top-99.9% inclusion sets ──────────────────────────────────────────
print("Computing top-99.9% country sets ...")
inc_fg  = top_pct_iso3(frac_green, "frac_green_fam")
inc_fhi = top_pct_iso3(frac_hi,   "frac_hi_fam")
inc_fnb = top_pct_iso3(frac_nb,   "frac_nb_fam")
inc_idx = top_pct_iso3(idx_df,    "num_patents")

# ── Merge onto world GDF (NaN outside threshold → grey) ───────────────────────
world_fg  = merge_world_frac(frac_green, "frac_green_fam", inc_fg)
world_fhi = merge_world_frac(frac_hi,   "frac_hi_fam",    inc_fhi)
world_fnb = merge_world_frac(frac_nb,   "frac_nb_fam",    inc_fnb)

# Index map: filter by num_patents volume, then display index_score
idx_display = idx_df[["iso3", "index_score", "num_patents"]].copy()
idx_display.loc[~idx_display["iso3"].isin(inc_idx), "index_score"] = np.nan
world_fidx = world_base.merge(
    idx_display[["iso3", "index_score"]].dropna(subset=["index_score"]),
    left_on="iso_a3", right_on="iso3", how="left"
)

# Hi-influence rate (fractional) — kept as bonus FRAC5
rate_df = (frac_green[["iso3", "frac_green_fam"]]
    .merge(frac_hi[["iso3", "frac_hi_fam"]], on="iso3", how="left")
    .assign(frac_hi_fam=lambda d: d["frac_hi_fam"].fillna(0)))
rate_df["frac_hi_rate"] = np.where(
    rate_df["frac_green_fam"] >= 5,
    rate_df["frac_hi_fam"] / rate_df["frac_green_fam"],
    np.nan,
)
world_frate = world_base.merge(
    rate_df[["iso3", "frac_hi_rate"]].dropna(),
    left_on="iso_a3", right_on="iso3", how="left"
)

# ── Map titles ─────────────────────────────────────────────────────────────────
TITLE_FAM = r"Quantity of Climate Related Patent Families"
TITLE_HI  = r"Quantity of High Influential Climate Patent Families"
TITLE_NB  = r"Quantity of Non-Climate Neighbor Patent Families"
TITLE_IDX = r"Climate Patent Leadership Index"

MAPS = [
    (world_fg,   "frac_green_fam", TITLE_FAM, "FRAC1_green_families.png",    "count"),
    (world_fhi,  "frac_hi_fam",   TITLE_HI,  "FRAC2_hi_green_families.png", "count"),
    (world_fnb,  "frac_nb_fam",   TITLE_NB,  "FRAC3_neighbor_families.png", "count"),
    (world_fidx, "index_score",   TITLE_IDX, "FRAC4_index_score.png",       "index"),
]

# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Draw individual maps
# ═══════════════════════════════════════════════════════════════════════════════
for gdf, col, title, fname, kind in MAPS:
    fig, ax = plt.subplots(figsize=(14, 7.5))
    if kind == "index":
        plot_frac_choropleth_index(ax, gdf, col, CMAP, title)
    else:
        plot_frac_choropleth(ax, gdf, col, CMAP, title)
    fig.suptitle(title, fontsize=13, y=1.002)
    fig.tight_layout(rect=[0, 0.01, 1, 1])
    fig.savefig(f"{OUT}/{fname}", bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved {fname}")

# Bonus: hi-rate map (FRAC5)
HI_RATE_TITLE = r"High-Influence Rate (Fractional): Hi-Influence $\div$ Green Families"
fig_r, ax_r = plt.subplots(figsize=(14, 7.5))
plot_frac_rate(ax_r, world_frate, "frac_hi_rate", CMAP, HI_RATE_TITLE)
fig_r.suptitle(HI_RATE_TITLE, fontsize=13, y=1.002)
fig_r.tight_layout(rect=[0, 0.01, 1, 1])
fig_r.savefig(f"{OUT}/FRAC5_hi_rate.png", bbox_inches="tight", dpi=300)
plt.close()
print("Saved FRAC5_hi_rate.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  2×2 composite
# ═══════════════════════════════════════════════════════════════════════════════
print("\nDrawing 2x2 composite ...")

PANEL_LABELS = [
    r"\textbf{(a)} " + TITLE_FAM,
    r"\textbf{(b)} " + TITLE_HI,
    r"\textbf{(c)} " + TITLE_NB,
    r"\textbf{(d)} " + TITLE_IDX,
]

fig0, axes0 = plt.subplots(2, 2, figsize=(20, 11),
                            gridspec_kw={"hspace": 0.14, "wspace": 0.04})
axes0 = axes0.flatten()

for ax0, (gdf, col, title, fname, kind), label in zip(axes0, MAPS, PANEL_LABELS):
    if kind == "index":
        plot_frac_choropleth_index(ax0, gdf, col, CMAP, label)
    else:
        plot_frac_choropleth(ax0, gdf, col, CMAP, label)

# fig0.suptitle(
#     r"\textbf{Climate Patent Geography --- Fractional Inventor-Country Counting}"
#     r" $\cdot$ PATSTAT 2025 Autumn"
#     r" $\cdot$ Reds $\cdot$ Jenks ($k=6$) $\cdot$ Equal Earth"
#     r" $\cdot$ Top 99.9\% volume countries",
#     fontsize=11, y=1.003,
# )
fig0.savefig(f"{OUT}/FRAC0_composite_4panel.png", bbox_inches="tight", dpi=300)
plt.close()
print("Saved FRAC0_composite_4panel.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  Country summary CSV (fractional) + comparison table with full counting
# ═══════════════════════════════════════════════════════════════════════════════
frac_summary = (
    frac_green[["country","iso3","frac_green_fam"]].rename(
        columns={"country":"Country","frac_green_fam":"frac_green"})
    .merge(frac_hi[["iso3","frac_hi_fam"]].rename(columns={"frac_hi_fam":"frac_hi"}),
           on="iso3", how="outer")
    .merge(frac_nb[["iso3","frac_nb_fam"]].rename(columns={"frac_nb_fam":"frac_nb"}),
           on="iso3", how="outer")
    .merge(idx_df[["iso3","index_score","num_patents"]], on="iso3", how="outer")
    .sort_values("frac_green", ascending=False)
)
frac_summary[["frac_green","frac_hi","frac_nb"]] = (
    frac_summary[["frac_green","frac_hi","frac_nb"]].round(1))
frac_summary.to_csv(f"{OUT}/country_summary_fractional.csv", index=False)

focus = ["US","CN","JP","KR","DE","GB","FR","IN","AU","CH"]
print("\n" + "=" * 90)
print("FRACTIONAL vs FULL COUNTING — selected countries")
print("=" * 90)

q_full = pd.read_csv("PATSTAT2025FALL/output/oecd_patent_quality_country_rankings.csv",
                     keep_default_na=False)
hi_full = (pd.read_csv("PATSTAT2025FALL/output/hi_patent_country_year_complete.csv")
           .groupby("countries")["num_hi_patents"].sum().reset_index()
           .rename(columns={"countries":"Country","num_hi_patents":"full_hi"}))

comp = (frac_summary[frac_summary["Country"].isin(focus)]
    .merge(q_full[q_full["countries"].isin(focus)][["countries","num_patents"]]
           .rename(columns={"countries":"Country","num_patents":"full_green"}),
           on="Country", how="left")
    .merge(hi_full[hi_full["Country"].isin(focus)], on="Country", how="left")
    .fillna({"frac_green": 0, "frac_hi": 0, "full_green": 0, "full_hi": 0})
    .sort_values("frac_green", ascending=False))

comp["frac_rate"] = (comp["frac_hi"] / comp["frac_green"] * 100).round(1)
comp["full_rate"] = (comp["full_hi"] / comp["full_green"] * 100).round(1)

print(f"{'Country':>8}  {'frac_green':>11}  {'full_green':>10}  "
      f"{'frac_hi':>8}  {'full_hi':>8}  "
      f"{'frac_rate%':>10}  {'full_rate%':>10}")
print("-" * 90)
for _, r in comp.iterrows():
    print(f"{r['Country']:>8}  {r['frac_green']:>11,.0f}  {r['full_green']:>10,.0f}  "
          f"{r['frac_hi']:>8,.0f}  {r['full_hi']:>8,.0f}  "
          f"{r['frac_rate']:>9.1f}%  {r['full_rate']:>9.1f}%")

print(f"\nFractional CSV -> {OUT}/country_summary_fractional.csv")
print(f"Maps           -> {OUT}/FRAC0-FRAC5.png")
