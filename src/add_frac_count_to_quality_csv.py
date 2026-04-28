"""
add_frac_count_to_quality_csv.py
=================================
Adds a ``frac_patents`` column to
  PATSTAT2025FALL/output/oecd_patent_quality_country_year_complete.csv

Method
------
``num_patents`` (existing) = full counting: each green patent family is
credited once to *every* inventor country it belongs to, so the column-sum
exceeds the number of unique families.

``frac_patents`` (new) = fractional counting: each family contributes
weight  w = 1 / n_countries  to each of its n inventor countries.
The column-sum across all countries for a given year equals the number of
unique green families in that year.

Data source
-----------
``patent_quality_family.parquet`` — the same family-level file used by
``count_pat_q_per_year.py``.  Its ``countries`` column already holds the
(deduplicated) list of 2-letter inventor/applicant country codes for each
DOCDB family.

Historical country codes (DD, SU, CS, YU, AN) that appear in the parquet
are remapped to their primary successor 2-letter codes before merging so
they accumulate into the modern country row in the CSV.

Run from the project root:
    python src/add_frac_count_to_quality_csv.py
"""

import polars as pl
import pandas as pd
import numpy as np

CSV_IN  = "PATSTAT2025FALL/output/oecd_patent_quality_country_year_complete.csv"
PQ_PATH = "PATSTAT2025FALL/output/patent_quality_family.parquet"
CSV_OUT = CSV_IN   # overwrite in place (backup kept below)

# Historical → successor 2-letter remapping
HIST_ISO2 = {"DD": "DE", "SU": "RU", "CS": "CZ", "YU": "RS", "AN": "NL"}

# ─────────────────────────────────────────────────────────────────────────────
print("Loading patent_quality_family.parquet ...")
pq = pl.read_parquet(PQ_PATH, columns=["docdb_family_id", "family_year", "countries"])
print(f"  {pq.height:,} family rows loaded")

# ─────────────────────────────────────────────────────────────────────────────
# Compute fractional count per (country 2-letter, family_year)
# ─────────────────────────────────────────────────────────────────────────────
frac = (
    pq
    # count countries per family (list length)
    .with_columns(
        pl.col("countries").list.len().alias("n_countries")
    )
    # explode to one row per (family, country)
    .explode("countries")
    .filter(
        pl.col("countries").is_not_null() &
        (pl.col("countries") != "")
    )
    # fractional weight
    .with_columns(
        (1.0 / pl.col("n_countries")).alias("weight")
    )
    # remap historical codes to successors
    .with_columns(
        pl.col("countries").replace(HIST_ISO2).alias("countries")
    )
    # sum weights per (country, year)
    .group_by(["countries", "family_year"])
    .agg(pl.col("weight").sum().alias("frac_patents"))
    .sort(["countries", "family_year"])
    .to_pandas()
)

total_frac = frac["frac_patents"].sum()
print(f"  Total fractional count (all countries × years): {total_frac:,.0f}")
print(f"  Unique families in parquet: {pq['docdb_family_id'].n_unique():,}")
print(f"  (Difference = families with no valid country attribution)")

# ─────────────────────────────────────────────────────────────────────────────
# Merge into CSV
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nLoading {CSV_IN} ...")
df = pd.read_csv(CSV_IN)
print(f"  {len(df):,} rows, {len(df.columns)} columns")

# backup before overwriting
backup = CSV_IN.replace(".csv", "_backup.csv")
df.to_csv(backup, index=False)
print(f"  Backup saved to {backup}")

# drop old frac_patents column if re-running
if "frac_patents" in df.columns:
    df = df.drop(columns=["frac_patents"])

df = df.merge(
    frac[["countries", "family_year", "frac_patents"]],
    on=["countries", "family_year"],
    how="left",
)
df["frac_patents"] = df["frac_patents"].fillna(0).round(4)

# reorder: put frac_patents right after num_patents
cols = list(df.columns)
np_idx = cols.index("num_patents")
cols.insert(np_idx + 1, cols.pop(cols.index("frac_patents")))
df = df[cols]

df.to_csv(CSV_OUT, index=False)
print(f"\nSaved {CSV_OUT}")
print(f"  Columns now: {list(df.columns)[:6]} ... (total {len(df.columns)})")

# ─────────────────────────────────────────────────────────────────────────────
# Sanity check: print top countries by total frac_patents vs num_patents
# ─────────────────────────────────────────────────────────────────────────────
YEAR_MIN, YEAR_MAX = 1985, 2025
summary = (
    df[(df["family_year"] >= YEAR_MIN) & (df["family_year"] <= YEAR_MAX)]
    .groupby("countries", as_index=False)
    .agg(full=("num_patents","sum"), frac=("frac_patents","sum"))
    .assign(frac_share=lambda d: (d["frac"] / d["full"]).round(3))
    .sort_values("full", ascending=False)
    .head(20)
)
print(f"\n{'Country':>10}  {'Full count':>12}  {'Frac count':>12}  {'Frac/Full':>10}")
print("-" * 52)
for _, r in summary.iterrows():
    print(f"{r['countries']:>10}  {r['full']:>12,.0f}  {r['frac']:>12,.1f}  {r['frac_share']:>10.3f}")
