"""
Save family-level diagnostics for green patent inventor/applicant attribution.

This script helps inspect cases where application-level rows have applicants or
generic person country codes but no inventor role/country rows. It does not
change the main pipeline; it exports family-level QA files for review.

Run from the project root:
    python src/save_green_patent_family_diagnostics.py
"""

from pathlib import Path

import polars as pl


OUT_DIR = Path("PATSTAT2025FALL/output")
GREEN_PATH = OUT_DIR / "green_patent8526.parquet"
INVENTOR_CONTRIB_PATH = OUT_DIR / "inventor_country_contrib_family.parquet"
APPLICANT_CONTRIB_PATH = OUT_DIR / "applicant_country_contrib_family.parquet"

OUT_PARQUET = OUT_DIR / "green_patent_family_diagnostics.parquet"
OUT_CSV = OUT_DIR / "green_patent_family_diagnostics_sample.csv"

INVALID_COUNTRY_CODES = {"", "0", "00", "null"}
REGIONAL_OFFICES = {"EP", "WO", "EA", "OA", "AP", "GC", "BX", "IB"}


def has_text(col: str) -> pl.Expr:
    return pl.col(col).is_not_null() & (pl.col(col).cast(pl.Utf8).str.strip_chars() != "")


def split_country_column(df: pl.DataFrame, source_col: str, out_col: str) -> pl.DataFrame:
    if source_col not in df.columns:
        return pl.DataFrame(schema={"docdb_family_id": pl.Int32, out_col: pl.List(pl.Utf8)})

    return (
        df.select(["docdb_family_id", source_col])
        .filter(has_text(source_col))
        .with_columns(pl.col(source_col).cast(pl.Utf8).str.split(","))
        .explode(source_col)
        .with_columns(pl.col(source_col).str.strip_chars().str.to_uppercase().alias("country"))
        .filter(
            has_text("country")
            & (pl.col("country").str.len_chars() == 2)
            & (~pl.col("country").is_in(list(INVALID_COUNTRY_CODES | REGIONAL_OFFICES)))
        )
        .group_by("docdb_family_id")
        .agg(pl.col("country").unique().sort().alias(out_col))
    )


def contrib_summary(path: Path, prefix: str) -> pl.DataFrame:
    country_col = f"{prefix}_countries_from_contrib"
    count_col = f"n_{prefix}s_total"
    if not path.exists():
        return pl.DataFrame(
            schema={
                "docdb_family_id": pl.Int32,
                country_col: pl.List(pl.Utf8),
                count_col: pl.Int64,
            }
        )

    return (
        pl.read_parquet(path)
        .with_columns(pl.col("country").str.strip_chars().str.to_uppercase())
        .filter(
            has_text("country")
            & (pl.col("country").str.len_chars() == 2)
            & (~pl.col("country").is_in(list(INVALID_COUNTRY_CODES | REGIONAL_OFFICES)))
        )
        .group_by("docdb_family_id")
        .agg([
            pl.col("country").unique().sort().alias(country_col),
            pl.col(count_col).max().alias(count_col),
        ])
    )


def main():
    if not GREEN_PATH.exists():
        raise FileNotFoundError(f"{GREEN_PATH} not found. Run src/load_classification.py first.")

    print(f"Loading {GREEN_PATH} ...")
    green = pl.read_parquet(GREEN_PATH)

    base = (
        green
        .with_columns(pl.col("earliest_filing_date").str.slice(0, 4).cast(pl.Int16).alias("family_year"))
        .group_by("docdb_family_id")
        .agg([
            pl.col("family_year").min(),
            pl.col("appln_id").n_unique().alias("n_green_applications"),
            pl.col("appln_id").cast(pl.Utf8).unique().sort().str.join(",").alias("appln_ids"),
            pl.col("appln_auth").drop_nulls().unique().sort().str.join(",").alias("appln_auths"),
            pl.col("appln_auth").drop_nulls().first().alias("first_appln_auth_in_green_file"),
            has_text("person_ctry_code").sum().alias("n_apps_with_person_ctry_code"),
            has_text("inventors").sum().alias("n_apps_with_inventors"),
            has_text("inventor_country_list").sum().alias("n_apps_with_inventor_country_list"),
            has_text("applicants").sum().alias("n_apps_with_applicants"),
            has_text("applicant_country_list").sum().alias("n_apps_with_applicant_country_list"),
            pl.col("appln_title").drop_nulls().first().alias("sample_title"),
        ])
    )

    person_country = split_country_column(green, "person_ctry_code", "person_countries_any_role")
    inventor_country_list = split_country_column(green, "inventor_country_list", "inventor_countries_from_rows")
    applicant_country_list = split_country_column(green, "applicant_country_list", "applicant_countries_from_rows")

    inventor_contrib = contrib_summary(INVENTOR_CONTRIB_PATH, "inventor")
    applicant_contrib = contrib_summary(APPLICANT_CONTRIB_PATH, "applicant")

    diagnostics = (
        base
        .join(person_country, on="docdb_family_id", how="left")
        .join(inventor_country_list, on="docdb_family_id", how="left")
        .join(applicant_country_list, on="docdb_family_id", how="left")
        .join(inventor_contrib, on="docdb_family_id", how="left")
        .join(applicant_contrib, on="docdb_family_id", how="left")
        .with_columns([
            (pl.col("n_apps_with_inventors") > 0).alias("has_inventor_names"),
            (pl.col("inventor_countries_from_rows").list.len().fill_null(0) > 0).alias("has_inventor_country_list"),
            (pl.col("inventor_countries_from_contrib").list.len().fill_null(0) > 0).alias("has_inventor_fractional_country"),
            (pl.col("person_countries_any_role").list.len().fill_null(0) > 0).alias("has_any_person_country"),
            (pl.col("applicant_countries_from_rows").list.len().fill_null(0) > 0).alias("has_applicant_country_list"),
            (pl.col("applicant_countries_from_contrib").list.len().fill_null(0) > 0).alias("has_applicant_fractional_country"),
            (
                has_text("first_appln_auth_in_green_file")
                & (~pl.col("first_appln_auth_in_green_file").is_in(list(REGIONAL_OFFICES)))
            ).alias("has_nonregional_appln_auth"),
        ])
        .with_columns([
            (
                pl.col("has_inventor_names")
                & ~pl.col("has_inventor_fractional_country")
                & pl.col("has_any_person_country")
            ).alias("inventor_country_missing_but_person_country_available"),
            (
                pl.col("has_inventor_names")
                & ~pl.col("has_inventor_fractional_country")
                & ~pl.col("has_any_person_country")
                & pl.col("has_nonregional_appln_auth")
            ).alias("inventor_country_missing_only_appln_auth_available"),
        ])
    )

    diagnostics.write_parquet(OUT_PARQUET, compression="zstd")
    print(f"Saved {OUT_PARQUET}")

    sample = (
        diagnostics
        .filter(
            pl.col("inventor_country_missing_but_person_country_available")
            | pl.col("inventor_country_missing_only_appln_auth_available")
        )
        .head(5000)
    )
    list_cols = [name for name, dtype in sample.schema.items() if isinstance(dtype, pl.List)]
    if list_cols:
        sample = sample.with_columns([
            pl.col(c).list.join(",").alias(c)
            for c in list_cols
        ])
    sample.write_csv(OUT_CSV)
    print(f"Saved sample {OUT_CSV}")

    summary = diagnostics.select([
        pl.len().alias("green_families"),
        pl.col("has_inventor_names").sum().alias("families_with_inventor_names"),
        pl.col("has_inventor_fractional_country").sum().alias("families_with_inventor_fractional_country"),
        pl.col("inventor_country_missing_but_person_country_available").sum().alias("missing_inventor_country_but_person_country_available"),
        pl.col("inventor_country_missing_only_appln_auth_available").sum().alias("missing_inventor_country_only_appln_auth_available"),
        pl.col("has_applicant_fractional_country").sum().alias("families_with_applicant_fractional_country"),
    ])
    print("Family-level attribution summary:")
    print(summary.to_dicts()[0])


if __name__ == "__main__":
    main()
