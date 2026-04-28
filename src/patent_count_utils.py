"""Shared utilities for building country × year patent count tables."""

import polars as pl
import pycountry

# Regional / international patent offices that don't map to a single country.
# Excluded when using appln_auth as country proxy.
REGIONAL_OFFICES = {"EP", "WO", "EA", "OA", "AP", "GC", "BX"}

# PATSTAT placeholder codes that are not real countries.
# "0" = unknown country, "XX" = not available.
INVALID_COUNTRY_CODES = {"0", "00", ""}


def convert_2digit_to_3digit(country_code: str):
    """Convert ISO 3166-1 alpha-2 to alpha-3. Returns None on failure."""
    try:
        country = pycountry.countries.get(alpha_2=country_code)
        return country.alpha_3 if country else None
    except Exception:
        return None


def build_complete_grid(
    df_country_year: pl.DataFrame,
    country_col: str,
    year_col: str,
    value_cols: list[str],
    year_range: tuple[int, int] = (1985, 2027),
    zero_fill_cols: list[str] | None = None,
) -> pl.DataFrame:
    """Create a full country × year grid, left-join actual data, and fill gaps.

    Parameters
    ----------
    df_country_year : DataFrame with at least *country_col*, *year_col*, and *value_cols*.
    country_col     : Name of the country column (2-digit codes).
    year_col        : Name of the year column.
    value_cols      : Numeric columns to keep (used only for documentation).
    year_range      : Inclusive (min_year, max_year).
    zero_fill_cols  : Columns to fill with 0 (typically counts).
                      Defaults to all *value_cols* for backward compatibility.
                      Columns in *value_cols* but not in *zero_fill_cols* stay null
                      (appropriate for averages / ratios where 0 ≠ "no data").
    """
    if zero_fill_cols is None:
        zero_fill_cols = value_cols

    all_countries = df_country_year[country_col].unique().sort()
    all_years = list(range(year_range[0], year_range[1] + 1))

    grid = pl.DataFrame({
        country_col: [c for c in all_countries for _ in all_years],
        year_col: all_years * len(all_countries),
    })

    df = (
        grid
        .join(df_country_year, on=[country_col, year_col], how="left")
        .with_columns([pl.col(c).fill_null(0) for c in zero_fill_cols])
        .sort([country_col, year_col])
    )

    return df


def add_country_3digit(df: pl.DataFrame, country_col: str) -> pl.DataFrame:
    """Append a ``country_code_3digit`` column converted from *country_col*."""
    return df.with_columns(
        pl.col(country_col)
        .map_elements(convert_2digit_to_3digit, return_dtype=pl.String)
        .alias("country_code_3digit")
    )


def print_summary(df: pl.DataFrame, count_col: str, year_col: str, label: str):
    """Print a short summary of the country-year table."""
    n_with = df.filter(pl.col(count_col) > 0).height
    n_without = df.filter(pl.col(count_col) == 0).height
    min_year = df[year_col].min()
    max_year = df[year_col].max()
    n_countries = df.filter(pl.col(count_col) > 0)["countries"].n_unique()
    print(f"\n{label}")
    print("=" * 70)
    print(f"  Total rows        : {df.height}")
    print(f"  With patents      : {n_with}")
    print(f"  Zero-filled       : {n_without}")
    print(f"  Year range        : {min_year} – {max_year}")
    print(f"  Countries w/ data : {n_countries}")
