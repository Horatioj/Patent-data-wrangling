import polars as pl
import gc
import re

# Define relevant columns once
tls201_columns = [
    "appln_id", "appln_auth", "appln_kind", "appln_filing_date", "receiving_office",
    "appln_nr_original", "ipr_type", "earliest_filing_date", "earliest_pat_publn_id",
    "granted", "docdb_family_id", "docdb_family_size", "nb_applicants", "nb_inventors", "nb_citing_docdb_fam"
]

file_paths = [
    "Z:/PATSTAT Global 2025 Autumn/tls201_appln_part01.csv",
    "Z:/PATSTAT Global 2025 Autumn/tls201_appln_part02.csv",
    "Z:/PATSTAT Global 2025 Autumn/tls201_appln_part03.csv"
]

scan_csv_dtypes = {
    "appln_nr": pl.String,
    "appln_nr_original": pl.String,
    "appln_id": pl.Int32,
    "docdb_family_id": pl.Int32,
    "docdb_family_size": pl.Int16,
    "nb_applicants": pl.Int16,
    "nb_inventors": pl.Int16,
    "nb_citing_docdb_fam": pl.Int32,
    "earliest_pat_publn_id": pl.Int32,
}

list_of_lazyframes = [
    pl.scan_csv(path, schema_overrides=scan_csv_dtypes, low_memory=True)
    for path in file_paths
]

tls201_filtered = (
    pl.concat(list_of_lazyframes, rechunk=False)
    .select(tls201_columns)
    .with_columns(
        pl.col("earliest_filing_date")
        .str.slice(0, 4)
        .cast(pl.Int16).alias("year")
    )
    .filter(
        (pl.col("year") >= 1985) &
        (pl.col("year") <= 2026) &
        (pl.col("granted") == "Y") &
        (pl.col("ipr_type").is_in(["PI", "UM"])) &
        (pl.col("docdb_family_size") >= 2)
    )
)

tls201_filtered.sink_parquet("PATSTAT2025FALL/output/tls201.parquet", compression="zstd")

tls201_index = pl.scan_parquet("PATSTAT2025FALL/output/tls201.parquet").select(["appln_id", "docdb_family_id"])
tls201_index.sink_parquet("PATSTAT2025FALL/output/tls201_index.parquet", compression="zstd")
del tls201_filtered
gc.collect()
# tls201.shape
# Out[6]: (14713755, 15)
# len(tls201['docdb_family_id'].unique())
# Out[8]: 5114258

# title
tls201_index = pl.scan_parquet("PATSTAT2025FALL/output/tls201_index.parquet")
tls201_filtered = pl.scan_parquet("PATSTAT2025FALL/output/tls201.parquet")

tls202_01 = pl.scan_csv(
    "Z:/PATSTAT Global 2025 Autumn/tls202_appln_title_part01.csv",
    schema_overrides={"appln_id": pl.Int32},
)
tls202_01_filtered = tls202_01.join(
    tls201_index.select("appln_id"), on="appln_id", how="semi"
)
tls_merge = tls201_filtered.join(tls202_01_filtered, on="appln_id", how="left")
tls_merge.sink_parquet("PATSTAT2025FALL/output/tls_merge.parquet", compression="zstd")

# --- Stage 1: save tls206 to parquet (CSV → compact parquet, only columns needed) ---
# PATSTAT CSVs encode missing values as literal "null" strings
patstat_null_values = ["", "null"]

tls206_agg_cols = [
    "person_id", "person_name", "person_ctry_code",
    "psn_id", "psn_sector", "han_id", "han_name"
]
tls206_files = [
    "Z:/PATSTAT Global 2025 Autumn/tls206_person_part01.csv",
    "Z:/PATSTAT Global 2025 Autumn/tls206_person_part02.csv"
]
pl.concat(
    [pl.scan_csv(f, schema_overrides={"person_id": pl.Int32},
                 null_values=patstat_null_values).select(tls206_agg_cols)
     for f in tls206_files],
    rechunk=False,
).sink_parquet("PATSTAT2025FALL/output/tls206.parquet", compression="zstd")
gc.collect()

# --- Stage 2: save filtered tls207 to parquet (semi-join shrinks it dramatically) ---
pl.scan_csv(
    "Z:/PATSTAT Global 2025 Autumn/tls207_pers_appln_part01.csv",
    schema_overrides={"appln_id": pl.Int32, "person_id": pl.Int32},
    null_values=patstat_null_values,
).join(
    tls201_index.select("appln_id"),
    on="appln_id",
    how="semi",
).sink_parquet("PATSTAT2025FALL/output/tls207_filtered.parquet", compression="zstd")
gc.collect()

# --- Stage 3: join tls207 x tls206 → save to parquet (streamed, never fully in RAM) ---
tls207_filtered = pl.scan_parquet("PATSTAT2025FALL/output/tls207_filtered.parquet")
tls206_df = pl.scan_parquet("PATSTAT2025FALL/output/tls206.parquet")
tls207_filtered.join(
    tls206_df, on="person_id", how="inner"
).sink_parquet("PATSTAT2025FALL/output/tls207_joined_persons.parquet", compression="zstd")
del tls207_filtered, tls206_df
gc.collect()

# --- Stage 4: aggregate per appln_id from the saved join result ---
cols_agg = [
    "person_name",
    "person_ctry_code",
    "person_id",
    "psn_id",
    "psn_sector",
    "han_id",
    "han_name"
]

# Normalize columns: cast to string, strip whitespace, turn blank/"null" into real null
clean_exprs = [
    pl.when(
        pl.col(c).is_null()
        | pl.col(c).cast(pl.Utf8).str.strip_chars().is_in(["", "null"])
    )
    .then(None)
    .otherwise(pl.col(c).cast(pl.Utf8).str.strip_chars())
    .alias(c)
    for c in cols_agg
]

agg_exprs = [
    pl.col(c)
      .drop_nulls()
      .unique()
      .sort()
      .str.join(",")
      .alias(c)
    for c in cols_agg
]

(
    pl.scan_parquet("PATSTAT2025FALL/output/tls207_joined_persons.parquet")
    .with_columns(clean_exprs)
    .group_by("appln_id")
    .agg(agg_exprs)
    .sink_parquet("PATSTAT2025FALL/output/persons_agg.parquet", compression="zstd", engine="streaming")
)
gc.collect()

# --- Stage 5: merge with tls_merge ---
pl.scan_parquet("PATSTAT2025FALL/output/tls_merge.parquet").join(
    pl.scan_parquet("PATSTAT2025FALL/output/persons_agg.parquet"), on="appln_id", how="left"
).sink_parquet("PATSTAT2025FALL/output/patent_title_inventor.parquet", compression="zstd")
gc.collect()
# joined_df.shape
# Publication # of claims tls211
path2 = [
    "Z:/PATSTAT Global 2025 Autumn\\tls211_pat_publn_part01.csv",
    "Z:/PATSTAT Global 2025 Autumn\\tls211_pat_publn_part02.csv",
]
scan_dtypes = {
    "publn_nr": pl.String,
    "publn_nr_original": pl.String,
    "pat_publn_id": pl.Int32,
    "publn_claims": pl.Int16,
}
tls211_scan = (
    pl.concat(
        [pl.scan_csv(p, schema_overrides=scan_dtypes) for p in path2],
        rechunk=False,
    )
    .select(["pat_publn_id", "publn_claims", "publn_nr"])
)
joined_df = pl.scan_parquet("PATSTAT2025FALL/output/patent_title_inventor.parquet")
joined_tls211 = joined_df.join(
    tls211_scan, left_on="earliest_pat_publn_id", right_on="pat_publn_id", how="left"
)
joined_tls211.sink_parquet("PATSTAT2025FALL/output/joined_tls211.parquet", compression="zstd")
del joined_df, tls211_scan
gc.collect()

# CPC at DOCDB FAM LEVEL -- use semi-join instead of Python set
tls201_index = pl.scan_parquet("PATSTAT2025FALL/output/tls201_index.parquet")

tls225_files = [
    "Z:/PATSTAT Global 2025 Autumn\\tls225_docdb_fam_cpc_part01.csv",
    "Z:/PATSTAT Global 2025 Autumn\\tls225_docdb_fam_cpc_part02.csv",
]
tls225 = (
    pl.concat(
        [pl.scan_csv(f, schema_overrides={"docdb_family_id": pl.Int32}) for f in tls225_files],
        rechunk=False,
    )
    .select(["docdb_family_id", "cpc_class_symbol"])
)

(
    tls225
    .join(tls201_index.select("docdb_family_id").unique(), on="docdb_family_id", how="semi")
    .group_by("docdb_family_id")
    .agg(pl.col("cpc_class_symbol").unique())
    .sink_parquet("PATSTAT2025FALL/output/tls225.parquet", compression="zstd", engine="streaming")
)
del tls225
gc.collect()

pl.scan_parquet("PATSTAT2025FALL/output/joined_tls211.parquet").join(
    pl.scan_parquet("PATSTAT2025FALL/output/tls225.parquet"), on="docdb_family_id", how="left"
).sink_parquet("PATSTAT2025FALL/output/joined_cpc_table.parquet", compression="zstd")
gc.collect()

# CPC at application level
cpc_cols = ["appln_id", "cpc_class_symbol"]
cpc_files = [
    "Z:/PATSTAT Global 2025 Autumn/tls224_appln_cpc_part01.csv",
    "Z:/PATSTAT Global 2025 Autumn/tls224_appln_cpc_part02.csv",
]
cpc_scan = (
    pl.concat(
        [pl.scan_csv(path, schema_overrides={"appln_id": pl.Int32}) for path in cpc_files],
        how="vertical_relaxed",
        rechunk=False,
    )
    .select(cpc_cols)
)
(
    cpc_scan
    .join(tls201_index.select("appln_id").unique(), on="appln_id", how="semi")
    .group_by("appln_id")
    .agg(pl.col("cpc_class_symbol").unique().alias("cpc"))
    .sink_parquet("PATSTAT2025FALL/output/agg_cpc.parquet", compression="zstd", engine="streaming")
)
gc.collect()

pl.scan_parquet("PATSTAT2025FALL/output/joined_cpc_table.parquet").join(
    pl.scan_parquet("PATSTAT2025FALL/output/agg_cpc.parquet"), on="appln_id", how="left"
).sink_parquet("PATSTAT2025FALL/output/joined_cpc_ipc_step1.parquet", compression="zstd")
gc.collect()

# IPC table
ipc_cols = ["appln_id", "ipc_class_symbol"]
ipc_files = [
    "Z:/PATSTAT Global 2025 Autumn/tls209_appln_ipc_part01.csv",
    "Z:/PATSTAT Global 2025 Autumn/tls209_appln_ipc_part02.csv",
]
ipc_scan = (
    pl.concat(
        [pl.scan_csv(path, schema_overrides={"appln_id": pl.Int32}) for path in ipc_files],
        how="vertical_relaxed",
        rechunk=False,
    )
    .select(ipc_cols)
)
(
    ipc_scan
    .join(tls201_index.select("appln_id").unique(), on="appln_id", how="semi")
    .group_by("appln_id")
    .agg(pl.col("ipc_class_symbol").unique().alias("ipc"))
    .sink_parquet("PATSTAT2025FALL/output/agg_ipc.parquet", compression="zstd", engine="streaming")
)
gc.collect()

pl.scan_parquet("PATSTAT2025FALL/output/joined_cpc_ipc_step1.parquet").join(
    pl.scan_parquet("PATSTAT2025FALL/output/agg_ipc.parquet"), on="appln_id", how="left"
).sink_parquet("PATSTAT2025FALL/output/class_tag_joined.parquet", compression="zstd")
class_tag_table = pl.read_parquet("PATSTAT2025FALL/output/class_tag_joined.parquet")
gc.collect()
class_tag_flag = class_tag_table.with_columns(
    (pl.col("cpc").is_null() & pl.col("ipc").is_null()).alias("both_null")
)
del class_tag_table
class_tag_flag.write_parquet("PATSTAT2025FALL/output/class_tag_table.parquet", compression="zstd")

class_tag_flag = class_tag_flag.filter(~pl.col("both_null"))

count_cpc_without_ipc = class_tag_flag.select(
    (pl.col("cpc").is_not_null() & pl.col("ipc").is_null()).sum().alias("count")
).item()

count_ipc_without_cpc = class_tag_flag.select(
    (pl.col("ipc").is_not_null() & pl.col("cpc").is_null()).sum().alias("count")
).item()

print(f"\nNumber of rows with 'cpc' but without 'ipc': {count_cpc_without_ipc}") # 108562
print(f"\nNumber of rows with 'ipc' but without 'cpc': {count_ipc_without_cpc}") # 246345

'''
patents without classification tags
1. PATSTAT GLOBAL 2023 SPRING EDITION uses only IPC-8 standard and keeps updating IPC1-7 tags. Some patents haven't been updated
will not have IPC tags
2. There are some patents do not have tags on Google Patent or EPO ESPACENET Patent search
3. Some patents do not have CPC tags
'''
# filtered_df = ((class_tag_flag.filter((pl.col("both_null")==True))
#  .select(['appln_id', 'appln_auth', 'appln_nr_original', 'earliest_pat_publn_id', 'ipr_type',
#           'docdb_family_size', 'nb_citing_docdb_fam', 'psn_sector',
#           'han_name', "earliest_filing_date", 'appln_title', 'cpc',
#           'cpc_class_symbol' , 'ipc', 'both_null']))
# )
# if isinstance(filtered_df, pl.LazyFrame):
#     filtered_df = filtered_df.collect()
# else:
#     filtered_df = filtered_df
#
# list_columns_to_stringify = ['cpc', 'cpc_class_symbol', 'ipc']
# delimiter = ";"
# conversion_expressions = []
#
# for col_name in list_columns_to_stringify:
#     # CORRECTED LINE: Remove the extra .dtype
#     if col_name in filtered_df.columns and isinstance(filtered_df.schema[col_name], pl.List):
#         conversion_expressions.append(
#             pl.when(pl.col(col_name).is_not_null())
#             .then(
#                 pl.col(col_name)
#                 .list.eval(pl.element().cast(pl.String)) # Cast elements to string
#                 .list.join(delimiter)                   # Join them
#             )
#             .otherwise(None) # Keep it null if the list itself was null
#             .alias(col_name)
#         )
#
# if conversion_expressions:
#     df_to_write = filtered_df.with_columns(conversion_expressions)
# else:
#     df_to_write = filtered_df
#
# output_csv_path = "filtered_data.csv"
# df_to_write.write_csv(output_csv_path)

'''
check cpc_class_symbol and cpc
two columns are identical. mismatched originate from different orders. maybe we can re-order to double check mismatched
'''
mismatched_rows = class_tag_flag.filter(pl.col('cpc_class_symbol') != pl.col('cpc'))
mismatched_rows = mismatched_rows.select(['appln_id', 'appln_auth', 'appln_nr_original', 'earliest_pat_publn_id', 'ipr_type',
           'docdb_family_size', 'nb_citing_docdb_fam', 'psn_sector',
           'han_name', "earliest_filing_date", 'appln_title', 'cpc',
           'cpc_class_symbol' , 'ipc', 'both_null'])
if isinstance(mismatched_rows, pl.LazyFrame):
    mismatched_rows = mismatched_rows.collect()
else:
    mismatched_rows = mismatched_rows

list_columns_to_stringify = ['cpc', 'cpc_class_symbol', 'ipc']
delimiter = ";"
conversion_expressions = []

for col_name in list_columns_to_stringify:
    if col_name in mismatched_rows.columns and isinstance(mismatched_rows.schema[col_name], pl.List):
        conversion_expressions.append(
            pl.when(pl.col(col_name).is_not_null())
            .then(
                pl.col(col_name)
                .list.eval(pl.element().cast(pl.String)) # Cast elements to string
                .list.join(delimiter)                   # Join them
            )
            .otherwise(None) # Keep it null if the list itself was null
            .alias(col_name)
        )

if conversion_expressions:
    df_to_write = mismatched_rows.with_columns(conversion_expressions).head(100)
else:
    df_to_write = mismatched_rows.head(100)
df_to_write.write_csv("PATSTAT2025FALL/output/not_match_cpc.csv")

# tls 902 / tls 229 IPC to NACE2 ONLY Manufactures
# appln_id to nace2
# tls229 = pl.read_csv("U:\\Climate_Innovation\\data\\data_PATSTAT_Global_2023_Spring_10\\tls229_part01.csv")
# tls229 = tls229.filter(pl.col("appln_id").is_in(appln_id_list)) # 14599453
# using NACE2 to classify sectors will lose 112,016 patents

# classification to sectors and green tags
# one column indicate sector and one column indicate TRUE/FALSE of green innovation
# combining CPC tags and WIPO GREEN INVENTORY
# to accelerate we can start from filtering green innovation first
# based on 'cpc' column, if it starts Y04S* Y02*, 'green' should be TRUE
# if 'cpc' column is None, based on 'ipc'. we classify it as 'green' if it appears in WIPO green inventory lists, ENVTECH_green_codes.csv
class_tag_flag_simple = class_tag_flag.select(pl.col(["appln_id", "cpc", "ipc"]))
green_df = pl.read_csv("ENVTECH_green_codes.csv")
green_codes_1 = green_df["Code"].to_list()
green_df_2 = pl.read_csv("IPC_green_codes.csv")
green_codes_2 = green_df_2["Code"].to_list()
green_codes = green_codes_1 + green_codes_2

# 2.a. Build an element‐wise check for a single CPC string:
cpc_element_check = (
    pl.element()
      .str.replace(" ", "")        # remove all spaces
      .str.to_uppercase()          # uppercase
      .str.starts_with("Y02")
    | pl.element()
      .str.replace(" ", "")
      .str.to_uppercase()
      .str.starts_with("Y04S")
)

# 2.b. Now wrap that inside `.list.eval(...)` to get a List[Bool] per row,
#     then `.list.any()` to see if any element was True.  But only if the list was nonempty.
cpc_green_expr = (
    (pl.col("cpc").list.len() > 0)
    & pl.col("cpc")
         .list.eval(cpc_element_check)  # yields List[Bool]
         .list.any()                     # True if any entry in that List[Bool] is True
)

ipc_green_expr = (
    # (1) only check IPC if CPC is empty (len=0) or null
    (pl.col("cpc").list.len() == 0) # comment this line, the # of green patents are 14086
    &
    # (2) for the IPC list, normalize each element and test membership in `green_set`
    pl.col("ipc")
      .list.eval(
          # normalize string, then is_in(...) expects a Python list-of-strings
          pl.element()
            .str.replace(" ", "")
            .str.to_uppercase()
            .is_in(green_codes)
      )
      .list.any()
)
result_df = (
    class_tag_flag_simple
      .lazy()
      .with_columns([
          cpc_green_expr.alias("green_from_cpc"),
          ipc_green_expr.alias("green_from_ipc"),
      ])
      .with_columns(
          (pl.col("green_from_cpc") | pl.col("green_from_ipc")).alias("green")
      )
      .collect()
)

n_total_green = result_df.filter(pl.col("green")).height
n_cpc_only    = result_df.filter(pl.col("green_from_cpc") & ~pl.col("green_from_ipc")).height
n_ipc_only    = result_df.filter(~pl.col("green_from_cpc") & pl.col("green_from_ipc")).height
n_both        = result_df.filter(pl.col("green_from_cpc") & pl.col("green_from_ipc")).height
print(f"Total green patents:        {n_total_green}")
print(f"  Green from CPC only:      {n_cpc_only}")
print(f"  Green from IPC only:      {n_ipc_only}  supplementary IPC codes contribution")
print(f"  Green from both:          {n_both}")


SECTOR_PATTERNS = {
    "Buildings": {
        "include": [
            re.compile(r"^Y02B"),                 # all Y02B/*
            re.compile(r"^Y02A30/(?!30)"),        # Y02A30/*  EXCEPT .../30
            re.compile(r"^Y04S20")                # Y04S20/*
        ],
        "exclude": []  # none
    },
    "CCS": {
        "include": [
            re.compile(r"^Y02C20/"),              # Y02C20/*
            re.compile(r"^Y02P40/18")             # Y02P40/18  (cement CCS)
        ],
        "exclude": []
    },
    "ICT": {
        "include": [
            re.compile(r"^Y02D"),                 # Y02D/*
            re.compile(r"^Y04S40"),               # Y04S40/*
            re.compile(r"^Y02A90/10")             # Y02A90/10
        ],
        "exclude": []
    },
    "Energy": {
        "include": [
            re.compile(r"^Y02E"),                 # Y02E/*
            re.compile(r"^Y04S10")                # Y04S10/*
        ],
        "exclude": []
    },
    "Manufacturing": {
        "include": [
            # Y02P10–Y02P90 except Y02P60
            re.compile(r"^Y02P(?:10|20|30|40|50|70|80|90)")
        ],
        "exclude": []
    },
    "Transportation": {
        "include": [
            re.compile(r"^Y02T"),                 # Y02T/*
            re.compile(r"^Y02A30/30"),            # Y02A30/30
            re.compile(r"^Y04S30")                # Y04S30/*
        ],
        "exclude": []
    },
    "Waste management": {
        "include": [
            re.compile(r"^Y02W"),                 # Y02W/*
            re.compile(r"^Y02A10/"),              # Y02A10/*
            re.compile(r"^Y02A20/")               # Y02A20/*
        ],
        "exclude": []
    },
    "Agriculture": {
        "include": [
            re.compile(r"^Y02P60/"),              # Y02P60/*
            re.compile(r"^Y02A40/")               # Y02A40/*
        ],
        "exclude": []
    }
}

green_only_df = result_df.filter(pl.col("green"))


def _norm(c: str) -> str:
    return c.replace(" ", "").upper()

def extract_sector_tags(cpc_codes: str) -> str:
    if cpc_codes == "":
        return ""
    sectors = set()
    for raw_code in cpc_codes.split(","):
        code = _norm(raw_code)
        for sector, patt_dict in SECTOR_PATTERNS.items():
            if sector in sectors:
                continue
            if any(p.match(code) for p in patt_dict["include"]) and not any(
                p.match(code) for p in patt_dict["exclude"]
            ):
                sectors.add(sector)
    return ",".join(sorted(sectors))


# 1. Convert the "cpc" column to a Python list of lists.
cpc_lists: list[list[str] | None] = green_only_df["cpc"].to_list()

# 2. Build a plain Python list of "sector" strings using your extract_sector_tags:
sector_values = []
for lst in cpc_lists:
    if lst is None or len(lst) == 0:
        sector_values.append("")
    else:
        joined = ",".join(lst)          # e.g. "Y02B10/00,Y04S20/10"
        sector_values.append(extract_sector_tags(joined))

# 3. Convert that Python list into a Polars Series of type Utf8:
sector_series = pl.Series("sector", sector_values, dtype=pl.Utf8)

# 4. Attach it back to green_only_df as a new column:
green_with_sector_df = green_only_df.with_columns(sector_series)

# Now green_with_sector_df has columns:
# ["appln_id", "cpc", "ipc", "green", "sector"]
print(green_with_sector_df.head(10)) #1353152/14711469

patent_df = class_tag_flag.join(green_with_sector_df, on="appln_id", how="left")
patent_df = patent_df.filter(pl.col("green"))

cpc_as_string = pl.col("cpc").list.join(" ").fill_null("")
patent_df = patent_df.with_columns(
    pl.when(cpc_as_string.str.contains(r"Y02A") & cpc_as_string.str.contains(r"Y02[BCDEPTW]|Y04S"))
    .then(pl.lit("adaptation, mitigation"))
    .when(cpc_as_string.str.contains(r"Y02A"))
    .then(pl.lit("adaptation"))
    .when(cpc_as_string.str.contains(r"Y02[BCDEPTW]|Y04S"))
    .then(pl.lit("mitigation"))
    .otherwise(None)
    .alias("mitigation_adaptation")
)

patent_df = patent_df.select([
    'appln_id', 'appln_auth', 'appln_kind', 'appln_filing_date',
    'receiving_office', 'appln_nr_original', 'ipr_type',
    'earliest_filing_date', 'earliest_pat_publn_id', 'granted',
    'docdb_family_id', 'docdb_family_size', 'nb_applicants',
    'nb_inventors', 'nb_citing_docdb_fam', 'appln_title_lg',
    'appln_title', 'person_name', 'person_ctry_code', 'person_id',
    'psn_id', 'psn_sector', 'han_id', 'han_name', 'publn_claims',
    'publn_nr', 'cpc_class_symbol', 'cpc', 'ipc', 'green', 'sector',
    'mitigation_adaptation',
])

def save2csv(df, name="green_patent.csv", list_columns_to_stringify=None, delimiter=";"):
    if list_columns_to_stringify is None:
        list_columns_to_stringify = ["cpc", "cpc_class_symbol", "ipc"]
    conversion_expressions = []
    for col_name in list_columns_to_stringify:
        if col_name in df.columns and isinstance(df.schema[col_name], pl.List):
            conversion_expressions.append(
                pl.when(pl.col(col_name).is_not_null())
                .then(
                    pl.col(col_name)
                    .list.eval(pl.element().cast(pl.String))
                    .list.join(delimiter)
                )
                .otherwise(None)
                .alias(col_name)
            )
    if conversion_expressions:
        df = df.with_columns(conversion_expressions)
    df.write_csv(name)
save2csv(patent_df, name="PATSTAT2025FALL/output/green_patent8526.csv")
patent_df.write_parquet("PATSTAT2025FALL/output/green_patent8526.parquet", compression="zstd")