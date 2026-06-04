import polars as pl

ids = [11281132, 40042300, 89131338]

df = pl.read_parquet("PATSTAT2025FALL/output/green_patent8526.parquet")

print(
    df.filter(pl.col("docdb_family_id").is_in(ids))
      .select([
          "docdb_family_id",
          "appln_id",
          "appln_auth",
          "appln_nr_original",
          "publn_nr",
          "earliest_pat_publn_id",
          "appln_title",
      ])
      .sort(["docdb_family_id", "appln_auth"])
)