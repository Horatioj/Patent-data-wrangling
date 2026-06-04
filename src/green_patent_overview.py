"""
Green patent descriptive overview figures (01–09), 1985–2025.

Updated to the inventor-country fractional basis used across this project:
  • Country-level charts (01 trend, 02 US–China, 03 top countries, 09 sector mix)
    use inventor-fractional green-family counts.
  • Composition charts (04 sector, 05 sector-over-time, 06 mitigation,
    07 applicant type, 08 citation distribution) are computed at the unique
    DOCDB-family level (each family counted once), not per application.

Country × year fractional counts come from the pre-computed panel
oecd_patent_quality_country_year_complete.csv; family attributes (sector,
mitigation, applicant type, citations) come from green_patent8526.parquet.

Run from the project root:
    python src/green_patent_overview.py
"""

import os
import numpy as np
import polars as pl
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.dpi": 150,
})

OUT = "PATSTAT2025FALL/output/vis"
DATA = "PATSTAT2025FALL/output"
os.makedirs(OUT, exist_ok=True)

YEAR_MIN, YEAR_MAX = 1985, 2025

C_GREEN, C_BLUE, C_AMBER, C_RED, C_GREY = "#2a9d8f", "#264653", "#e9c46a", "#e76f51", "#a8a8a8"
C_US, C_CN = "#4a90d9", "#e05252"

SECTOR_COLORS = {
    "Energy": "#2a9d8f", "Transportation": "#264653", "Manufacturing": "#e9c46a",
    "Buildings": "#e76f51", "ICT": "#6a4c93", "Waste management": "#a8dadc",
    "Agriculture": "#81b29a",
}
TOP_SECTORS = ["Energy", "Transportation", "Manufacturing", "Buildings",
               "ICT", "Waste management", "Agriculture"]

ENTITY_COLOR = {
    "Company": "#264653", "University": "#2a9d8f", "Gov/Non-profit": "#e9c46a",
    "Individual": "#e76f51", "Unknown/Other": "#a8a8a8",
}
ENTITY_ORDER = ["Company", "University", "Gov/Non-profit", "Individual", "Unknown/Other"]

FRAC_NOTE = "Inventor-country fractional counting"
years_all = np.arange(YEAR_MIN, YEAR_MAX + 1)


def classify_entity(psn_str):
    if not psn_str:
        return "Unknown/Other"
    s = psn_str.upper()
    if "COMPANY" in s:
        return "Company"
    if "UNIVERSITY" in s:
        return "University"
    if "GOV" in s:
        return "Gov/Non-profit"
    if "INDIVIDUAL" in s:
        return "Individual"
    return "Unknown/Other"


# ── Load inventor-fractional country-year panel (01, 02, 03) ──────────────────
print("Loading oecd_patent_quality_country_year_complete.csv …")
gq = (
    pl.read_csv(f"{DATA}/oecd_patent_quality_country_year_complete.csv")
    .select(["countries", "family_year", "inventor_frac_patents"])
    .filter((pl.col("family_year") >= YEAR_MIN) & (pl.col("family_year") <= YEAR_MAX))
    .to_pandas()
)
green_yr = gq.groupby("family_year")["inventor_frac_patents"].sum().reindex(years_all, fill_value=0.0)
ctry_yr = gq.pivot_table(index="family_year", columns="countries",
                         values="inventor_frac_patents", aggfunc="sum").reindex(years_all).fillna(0.0)

print("Loading docdb_family_year.parquet (all-family totals) …")
total_yr = (
    pl.scan_parquet(f"{DATA}/docdb_family_year.parquet")
    .filter((pl.col("family_year") >= YEAR_MIN) & (pl.col("family_year") <= YEAR_MAX))
    .group_by("family_year").agg(pl.len().alias("total"))
    .sort("family_year").collect().to_pandas().set_index("family_year")["total"]
    .reindex(years_all, fill_value=0)
)
green_pct = (green_yr / total_yr.replace(0, np.nan) * 100).fillna(0)

# ── Load green parquet → family-level table (04–09) ───────────────────────────
print("Loading green_patent8526.parquet (family attributes) …")
green = pl.read_parquet(
    f"{DATA}/green_patent8526.parquet",
    columns=["appln_filing_date", "sector", "mitigation_adaptation",
             "docdb_family_id", "nb_citing_docdb_fam", "psn_sector"],
).with_columns(
    pl.col("appln_filing_date").str.slice(0, 4).cast(pl.Int32).alias("yr")
)

gfam = (
    green
    .filter((pl.col("yr") >= YEAR_MIN) & (pl.col("yr") <= YEAR_MAX))
    .group_by("docdb_family_id")
    .agg([
        pl.col("yr").min().alias("yr"),
        pl.col("sector").drop_nulls().first().alias("sector"),
        pl.col("mitigation_adaptation").drop_nulls().first().alias("mitig"),
        pl.col("nb_citing_docdb_fam").max().alias("cites"),
        pl.col("psn_sector").drop_nulls().first().alias("psn"),
    ])
    .with_columns([
        pl.col("psn").map_elements(classify_entity, return_dtype=pl.Utf8).alias("entity"),
        pl.col("sector").str.split(",").list.first().str.strip_chars().alias("primary_sector"),
    ])
)
print(f"  Unique green families (1985-2025): {gfam.height:,}\n")


# ════════════════════════════════════════════════════════════════════════════
# 01 – Green family filings & green share over time
# ════════════════════════════════════════════════════════════════════════════
fig, ax1 = plt.subplots(figsize=(11, 5))
ax2 = ax1.twinx(); ax2.spines["top"].set_visible(False)
ax1.fill_between(years_all, green_yr.values / 1000, alpha=0.18, color=C_GREEN)
ax1.plot(years_all, green_yr.values / 1000, color=C_GREEN, lw=2.2,
         label="Green families, fractional (thousands)")
ax2.plot(years_all, green_pct.values, color=C_RED, lw=2, ls="--",
         label="Green share of all families (%)")
ax2.set_ylabel("Green share of all patent families (%)", color=C_RED, fontsize=10)
ax2.tick_params(axis="y", colors=C_RED)
ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f%%"))
ax1.set_xlabel("Earliest family filing year", fontsize=10)
ax1.set_ylabel("Green patent families, fractional (thousands)", color=C_GREEN, fontsize=10)
ax1.tick_params(axis="y", colors=C_GREEN)
ax1.set_xlim(YEAR_MIN, YEAR_MAX)
l1, lb1 = ax1.get_legend_handles_labels()
l2, lb2 = ax2.get_legend_handles_labels()
ax1.legend(l1 + l2, lb1 + lb2, loc="upper left", fontsize=9)
plt.title(f"Green Patent Families vs. All Families (1985–2025)\n({FRAC_NOTE})",
          fontsize=12, pad=12)
fig.tight_layout(); fig.savefig(f"{OUT}/01_green_filing_trend.png"); plt.close()
print("Saved 01_green_filing_trend.png")


# ════════════════════════════════════════════════════════════════════════════
# 02 – US vs China green family filings
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 5))
for c, lab, col, mk in [("US", "United States", C_US, "o"), ("CN", "China", C_CN, "s")]:
    v = ctry_yr[c].reindex(years_all).fillna(0).values if c in ctry_yr.columns else np.zeros(len(years_all))
    ax.plot(years_all, v / 1000, color=col, lw=2.2, marker=mk, markersize=4, label=lab)
    ax.fill_between(years_all, v / 1000, alpha=0.12, color=col)
ax.set_xlabel("Earliest family filing year", fontsize=10)
ax.set_ylabel("Green patent families, fractional (thousands)", fontsize=10)
ax.set_xlim(YEAR_MIN, YEAR_MAX)
ax.legend(fontsize=10)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}k"))
plt.title(f"US vs. China Green Patent Families (1985–2025)\n({FRAC_NOTE})",
          fontsize=12, pad=12)
fig.tight_layout(); fig.savefig(f"{OUT}/02_us_cn_comparison.png"); plt.close()
print("Saved 02_us_cn_comparison.png")


# ════════════════════════════════════════════════════════════════════════════
# 03 – Top-10 inventor countries (bar)
# ════════════════════════════════════════════════════════════════════════════
top10 = (gq.groupby("countries")["inventor_frac_patents"].sum()
         .sort_values(ascending=False).head(10))
fig, ax = plt.subplots(figsize=(10, 5))
order = top10.index[::-1]
colors = [C_US if a == "US" else C_CN if a == "CN" else C_GREY for a in order]
bars = ax.barh(list(order), (top10[order].values) / 1000, color=colors, height=0.65)
ax.set_xlabel("Green patent families, fractional (thousands)", fontsize=10)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}k"))
for bar, val in zip(bars, top10[order].values):
    ax.text(bar.get_width() + top10.max() / 1000 * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val/1000:.0f}k", va="center", fontsize=8.5, color="#444")
plt.title(f"Top-10 Inventor Countries — Green Patent Families (1985–2025)\n({FRAC_NOTE})",
          fontsize=12, pad=10)
fig.tight_layout(); fig.savefig(f"{OUT}/03_top10_authorities.png"); plt.close()
print("Saved 03_top10_authorities.png")


# ════════════════════════════════════════════════════════════════════════════
# 04 – Sector distribution (family-level, horizontal bar)
# ════════════════════════════════════════════════════════════════════════════
sec_tot = (
    gfam.with_columns(pl.col("sector").str.split(",")).explode("sector")
    .with_columns(pl.col("sector").str.strip_chars().alias("s"))
    .filter(pl.col("s").is_in(TOP_SECTORS))
    .group_by("s").agg(pl.col("docdb_family_id").n_unique().alias("count"))
    .to_pandas().set_index("s")["count"].reindex(TOP_SECTORS).fillna(0).sort_values()
)
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.barh(sec_tot.index, sec_tot.values / 1000,
        color=[SECTOR_COLORS.get(s, C_GREY) for s in sec_tot.index], height=0.65)
ax.set_xlabel("Green patent families (thousands)", fontsize=10)
for i, val in enumerate(sec_tot.values):
    ax.text(val / 1000 + sec_tot.max() / 1000 * 0.01, i, f"{val/1000:.0f}k",
            va="center", fontsize=8.5)
plt.title("Green Patent Families by Technology Sector (1985–2025)\n"
          "Unique families; a family may span multiple sectors", fontsize=12, pad=10)
fig.tight_layout(); fig.savefig(f"{OUT}/04_sector_distribution.png"); plt.close()
print("Saved 04_sector_distribution.png")


# ════════════════════════════════════════════════════════════════════════════
# 05 – Stacked area: sector composition over time (family-level)
# ════════════════════════════════════════════════════════════════════════════
sec_yr = (
    gfam.with_columns(pl.col("sector").str.split(",")).explode("sector")
    .with_columns(pl.col("sector").str.strip_chars().alias("s"))
    .filter(pl.col("s").is_in(TOP_SECTORS))
    .group_by(["yr", "s"]).agg(pl.col("docdb_family_id").n_unique().alias("count"))
    .to_pandas().pivot(index="yr", columns="s", values="count")
    .reindex(years_all).fillna(0)
)
fig, ax = plt.subplots(figsize=(12, 5))
present = [s for s in TOP_SECTORS if s in sec_yr.columns]
ax.stackplot(years_all, *[sec_yr[s].values / 1000 for s in present],
             labels=present, colors=[SECTOR_COLORS[s] for s in present], alpha=0.85)
ax.set_xlabel("Earliest family filing year", fontsize=10)
ax.set_ylabel("Green patent families (thousands)", fontsize=10)
ax.set_xlim(YEAR_MIN, YEAR_MAX)
ax.legend(loc="upper left", fontsize=8.5, ncol=2)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}k"))
plt.title("Green Patent Sector Composition Over Time (1985–2025)", fontsize=12, pad=10)
fig.tight_layout(); fig.savefig(f"{OUT}/05_sector_stacked_area.png"); plt.close()
print("Saved 05_sector_stacked_area.png")


# ════════════════════════════════════════════════════════════════════════════
# 06 – Mitigation vs Adaptation (family-level pie)
# ════════════════════════════════════════════════════════════════════════════
mit = (gfam.group_by("mitig").agg(pl.len().alias("count"))
       .sort("count", descending=True).to_pandas().dropna(subset=["mitig"]))
fig, ax = plt.subplots(figsize=(6.5, 5))
ax.pie(mit["count"], labels=mit["mitig"], colors=[C_GREEN, C_BLUE, C_AMBER][:len(mit)],
       autopct="%1.1f%%", startangle=140,
       wedgeprops={"linewidth": 1, "edgecolor": "white"}, textprops={"fontsize": 10})
plt.title("Mitigation vs. Adaptation Green Patent Families (1985–2025)", fontsize=12, pad=10)
fig.tight_layout(); fig.savefig(f"{OUT}/06_mitigation_adaptation.png"); plt.close()
print("Saved 06_mitigation_adaptation.png")


# ════════════════════════════════════════════════════════════════════════════
# 07 – Applicant type (family-level bar)
# ════════════════════════════════════════════════════════════════════════════
ent = (gfam.group_by("entity").agg(pl.len().alias("count"))
       .to_pandas().set_index("entity")["count"]
       .reindex(ENTITY_ORDER).fillna(0).sort_values())
fig, ax = plt.subplots(figsize=(7.5, 4))
ax.barh(ent.index, ent.values / 1000,
        color=[ENTITY_COLOR[e] for e in ent.index], height=0.6)
ax.set_xlabel("Green patent families (thousands)", fontsize=10)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}k"))
for i, val in enumerate(ent.values):
    ax.text(val / 1000 + ent.max() / 1000 * 0.01, i, f"{val/1000:.0f}k",
            va="center", fontsize=8.5)
plt.title("Green Patent Families by Applicant Type (1985–2025)", fontsize=12, pad=10)
fig.tight_layout(); fig.savefig(f"{OUT}/07_applicant_types.png"); plt.close()
print("Saved 07_applicant_types.png")


# ════════════════════════════════════════════════════════════════════════════
# 08 – Forward citation distribution (family-level)
# ════════════════════════════════════════════════════════════════════════════
bucket_order = ["0", "1–5", "6–10", "11–20", "21–50", "50+"]
cite = (
    gfam.with_columns(
        pl.when(pl.col("cites") == 0).then(pl.lit("0"))
        .when(pl.col("cites") <= 5).then(pl.lit("1–5"))
        .when(pl.col("cites") <= 10).then(pl.lit("6–10"))
        .when(pl.col("cites") <= 20).then(pl.lit("11–20"))
        .when(pl.col("cites") <= 50).then(pl.lit("21–50"))
        .otherwise(pl.lit("50+")).alias("bucket")
    )
    .group_by("bucket").agg(pl.len().alias("count"))
    .to_pandas().set_index("bucket")["count"].reindex(bucket_order).fillna(0)
)
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(cite.index, cite.values / 1000,
       color=[C_GREY if b == "0" else C_GREEN for b in cite.index], width=0.65)
ax.set_xlabel("Forward citations received (DOCDB families)", fontsize=10)
ax.set_ylabel("Green patent families (thousands)", fontsize=10)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}k"))
for i, val in enumerate(cite.values):
    ax.text(i, val / 1000 + cite.max() / 1000 * 0.01, f"{val/1000:.0f}k",
            ha="center", fontsize=8.5)
plt.title("Distribution of Forward Citations — Green Patent Families (1985–2025)",
          fontsize=12, pad=10)
fig.tight_layout(); fig.savefig(f"{OUT}/08_citation_distribution.png"); plt.close()
print("Saved 08_citation_distribution.png")


# ════════════════════════════════════════════════════════════════════════════
# 09 – US vs China sector mix (inventor-fractional, 2018–2025)
# ════════════════════════════════════════════════════════════════════════════
contrib = (
    pl.read_parquet(f"{DATA}/inventor_country_contrib_family.parquet")
    .filter(pl.col("country").is_in(["US", "CN"]))
    .select(["docdb_family_id", "country", "inventor_frac"])
)
mix = (
    gfam.filter((pl.col("yr") >= 2018) & (pl.col("primary_sector").is_in(TOP_SECTORS)))
    .select(["docdb_family_id", "primary_sector"])
    .join(contrib, on="docdb_family_id", how="inner")
    .group_by(["country", "primary_sector"])
    .agg(pl.col("inventor_frac").sum().alias("frac"))
    .to_pandas()
)
pivot = mix.pivot(index="primary_sector", columns="country", values="frac").reindex(TOP_SECTORS).fillna(0)
us_pct = pivot["US"] / pivot["US"].sum() * 100 if "US" in pivot else pd.Series(0, index=TOP_SECTORS)
cn_pct = pivot["CN"] / pivot["CN"].sum() * 100 if "CN" in pivot else pd.Series(0, index=TOP_SECTORS)
x = np.arange(len(TOP_SECTORS)); width = 0.38
fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(x - width / 2, us_pct.values, width, label="United States", color=C_US, alpha=0.85)
ax.bar(x + width / 2, cn_pct.values, width, label="China", color=C_CN, alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(TOP_SECTORS, fontsize=9, rotation=15, ha="right")
ax.set_ylabel("Share within country's green portfolio (%)", fontsize=10)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
ax.legend(fontsize=10)
plt.title(f"Green Patent Sector Mix — US vs. China (2018–2025)\n({FRAC_NOTE})",
          fontsize=12, pad=10)
fig.tight_layout(); fig.savefig(f"{OUT}/09_us_cn_sector_mix.png"); plt.close()
print("Saved 09_us_cn_sector_mix.png")

print(f"\nAll overview figures saved to: {OUT}/")
