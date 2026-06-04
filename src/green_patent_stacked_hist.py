"""
Stacked histogram: green patent family count by inventor country, 1985–2025.

Inventor-country fractional counting (consistent with the fractional maps and
green_patent_vis.py): each family is credited to its inventor countries with
weight inventor_frac (summing to 1 per family).  Country × year fractional
counts are read from oecd_patent_quality_country_year_complete.csv.

Shows US, JP, DE, KR, GB, FR as individual segments; all remaining inventor
countries are grouped into "Other".  Three policy event markers (Montreal
Protocol 1987, Kyoto Protocol 1997, Paris Agreement 2015) are drawn as vertical
lines.

Color palette: reds–oranges, consistent with the Reds choropleth maps.

Run from the project root:
    python src/green_patent_stacked_hist.py
"""

import os
import numpy as np
import polars as pl
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

matplotlib.rcParams.update({
    "font.family":        "DejaVu Sans",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.20,
    "grid.linestyle":     "--",
    "figure.dpi":         150,
})

OUT = "PATSTAT2025FALL/output/vis"
DATA = "PATSTAT2025FALL/output"
os.makedirs(OUT, exist_ok=True)

YEAR_START, YEAR_END = 1985, 2025

# Countries shown as individual segments (order = bottom to top of stack)
COUNTRIES = ["US", "JP", "DE", "KR", "GB", "FR"]

CTRY_LABEL = {
    "US": "United States",
    "JP": "Japan",
    "DE": "Germany",
    "KR": "South Korea",
    "GB": "United Kingdom",
    "FR": "France",
}

# Reds–oranges palette (dark red → bright orange → pale orange for "Other")
CTRY_COLOR = {
    "US": "#67000d",   # near-black dark red
    "JP": "#a50026",   # deep crimson
    "DE": "#d73027",   # vivid red
    "KR": "#f46d43",   # red-orange
    "GB": "#fdae61",   # amber
    "FR": "#fee08b",   # pale gold
}
OTHER_COLOR = "#f5f5dc"   # very light cream / off-white for "Other"

# ── Policy event markers ──────────────────────────────────────────────────────
EVENTS = {
    1987: "Montreal\nProtocol",
    1997: "Kyoto\nProtocol",
    2015: "Paris\nAgreement",
}

# ── Load inventor-fractional country-year panel ───────────────────────────────
print("Loading oecd_patent_quality_country_year_complete.csv …")
cy = (
    pl.read_csv(f"{DATA}/oecd_patent_quality_country_year_complete.csv")
    .select(["countries", "family_year", "inventor_frac_patents"])
    .filter((pl.col("family_year") >= YEAR_START) & (pl.col("family_year") <= YEAR_END))
)

years = np.arange(YEAR_START, YEAR_END + 1)

# Per-country fractional series
cy_pd = cy.to_pandas()
series = {}
for c in COUNTRIES:
    s = (
        cy_pd[cy_pd["countries"] == c]
        .set_index("family_year")["inventor_frac_patents"]
        .reindex(years, fill_value=0.0)
    )
    series[c] = s.values.astype(float)

# "Other" = global fractional total (all inventor countries) minus named segments
total_per_yr = (
    cy_pd.groupby("family_year")["inventor_frac_patents"].sum()
    .reindex(years, fill_value=0.0)
)
named_sum = sum(series[c] for c in COUNTRIES)
series["Other"] = (total_per_yr.values.astype(float) - named_sum).clip(min=0)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))

bar_colors  = [CTRY_COLOR[c] for c in COUNTRIES] + [OTHER_COLOR]
bar_labels  = [CTRY_LABEL[c] for c in COUNTRIES] + ["Other countries"]
bar_data    = [series[c]     for c in COUNTRIES] + [series["Other"]]

bottoms = np.zeros(len(years))
bars_handles = []
for data, color, label in zip(bar_data, bar_colors, bar_labels):
    b = ax.bar(
        years, data,
        bottom=bottoms,
        color=color,
        width=0.82,
        label=label,
        edgecolor="none",
    )
    bars_handles.append(b)
    bottoms += data

# ── Policy event vertical lines ───────────────────────────────────────────────
y_max = bottoms.max()
for yr, evt_label in EVENTS.items():
    ax.axvline(yr, color="#333333", lw=1.2, ls="--", zorder=5)
    ax.text(
        yr + 0.4,
        y_max * 0.97,
        evt_label,
        fontsize=8,
        color="#222222",
        va="top",
        ha="left",
        zorder=6,
    )

# ── Axes formatting ───────────────────────────────────────────────────────────
ax.set_xlim(YEAR_START - 0.7, YEAR_END + 0.7)
ax.set_ylim(0, y_max * 1.10)
ax.set_xlabel("Earliest family filing year", fontsize=11)
ax.set_ylabel("Green patent families, fractional", fontsize=11)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.set_xticks(np.arange(1985, 2026, 5))
ax.tick_params(axis="x", labelsize=9)
ax.tick_params(axis="y", labelsize=9)

# Legend — reverse order so stack reads top-to-bottom
handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles[::-1], labels[::-1],
    ncol=1,
    fontsize=8.5,
    loc="upper left",
    frameon=True,
    framealpha=0.85,
    edgecolor="#cccccc",
)

ax.set_title(
    "Green Patent Families by Inventor Country, 1985–2025\n"
    "Inventor-country fractional counting · families credited by inventor shares",
    fontsize=12,
    pad=10,
)

fig.tight_layout()
out_path = f"{OUT}/stacked_hist_green_families.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved: {out_path}")
