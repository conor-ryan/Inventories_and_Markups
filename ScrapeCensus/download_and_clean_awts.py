from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup


AWTS_REPORT_URL = "https://www.census.gov/data/tables/2022/econ/awts/annual-reports.html"
RAW_DOWNLOAD_DIR = Path(__file__).resolve().parent / "raw_xlsx"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "CleanCensusFiles"

TABLE_LINK_PATTERN = re.compile(r"\bTable\s+([1-6])(?![\d.])\.", flags=re.IGNORECASE)
TABLE_FILE_PATTERN = re.compile(r"_table([1-6])\.xlsx$", flags=re.IGNORECASE)
YEAR_LABEL_PATTERN = re.compile(r"^\s*(\d{4})r?\s*$")
NUMERIC_PATTERN = re.compile(r"^(-?\d+(?:\.\d+)?)([A-Za-z]+)?$")


@dataclass
class TableMeta:
    table_number: int
    title: str
    download_url: str


def to_snake_case(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "column"


def parse_year_label(value: Any) -> int | None:
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        year = int(value)
        if 1900 <= year <= 2100:
            return year
        return None

    match = YEAR_LABEL_PATTERN.match(str(value))
    if not match:
        return None
    return int(match.group(1))


def parse_measure_value(value: Any) -> tuple[float | None, str | None, str | None]:
    if pd.isna(value):
        return None, None, None

    raw_text = str(value).strip()
    if not raw_text:
        return None, None, None

    cleaned = raw_text.replace(",", "")
    numeric_match = NUMERIC_PATTERN.match(cleaned)
    if numeric_match:
        estimate_value = float(numeric_match.group(1))
        estimate_flag = numeric_match.group(2)
        return estimate_value, estimate_flag, raw_text

    if cleaned.isalpha():
        return None, cleaned, raw_text

    return None, None, raw_text


def fetch_table_metadata(report_url: str) -> list[TableMeta]:
    response = requests.get(report_url, timeout=60)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    found: dict[int, TableMeta] = {}

    for anchor in soup.select("a[href$='.xlsx']"):
        href = anchor.get("href", "")
        url_match = TABLE_FILE_PATTERN.search(href)
        if not url_match:
            continue

        anchor_text = " ".join(anchor.get_text(" ", strip=True).split())
        match = TABLE_LINK_PATTERN.search(anchor_text)
        if match:
            table_number = int(match.group(1))
        else:
            table_number = int(url_match.group(1))

        if table_number in found:
            continue

        title = re.sub(r"\s*\[<[^\]]+\]\s*$", "", anchor_text)
        if not title:
            title = f"Table {table_number}"

        found[table_number] = TableMeta(
            table_number=table_number,
            title=title,
            download_url=urljoin(report_url, href),
        )

    missing = [str(i) for i in range(1, 7) if i not in found]
    if missing:
        raise RuntimeError(f"Could not find spreadsheet links for tables: {', '.join(missing)}")

    return [found[i] for i in range(1, 7)]


def download_excel_file(url: str, destination: Path) -> None:
    with requests.get(url, timeout=120, stream=True) as response:
        response.raise_for_status()
        with destination.open("wb") as file_obj:
            for chunk in response.iter_content(chunk_size=1024 * 256):
                if chunk:
                    file_obj.write(chunk)


def build_clean_column_names(header_values: list[Any], year_col_indices: list[int]) -> tuple[list[str], list[int]]:
    first_year_col = min(year_col_indices)
    id_col_indices = list(range(first_year_col))

    column_names: list[str] = []
    used_names: dict[str, int] = {}

    for col_idx in id_col_indices:
        cell_value = header_values[col_idx]
        if pd.isna(cell_value):
            base_name = f"id_column_{col_idx + 1}"
        else:
            base_name = to_snake_case(str(cell_value))

        if base_name in used_names:
            used_names[base_name] += 1
            name = f"{base_name}_{used_names[base_name]}"
        else:
            used_names[base_name] = 1
            name = base_name
        column_names.append(name)

    return column_names, id_col_indices


def transform_table_to_long(table_meta: TableMeta, excel_path: Path) -> pd.DataFrame:
    wide = pd.read_excel(excel_path, sheet_name=0, header=None, dtype=object)

    header_row_index = None
    for row_index in range(min(len(wide), 25)):
        row_values = wide.iloc[row_index].tolist()
        year_col_indices = [
            col_index for col_index, value in enumerate(row_values) if parse_year_label(value) is not None
        ]
        has_multiple_years = len(year_col_indices) >= 3
        has_identifier_before_years = bool(year_col_indices) and any(
            not pd.isna(value) and parse_year_label(value) is None for value in row_values[: min(year_col_indices)]
        )
        if has_multiple_years and has_identifier_before_years:
            header_row_index = row_index
            break

    if header_row_index is None:
        raise RuntimeError(f"Could not find header row in {excel_path.name}")

    header_values = wide.iloc[header_row_index].tolist()
    year_info: list[tuple[int, int]] = []
    for col_index, value in enumerate(header_values):
        year = parse_year_label(value)
        if year is not None:
            year_info.append((col_index, year))

    if not year_info:
        raise RuntimeError(f"Could not find year columns in {excel_path.name}")

    year_col_indices = [col for col, _ in year_info]
    year_labels = [year for _, year in year_info]
    id_column_names, id_col_indices = build_clean_column_names(header_values, year_col_indices)

    selected_cols = id_col_indices + year_col_indices
    subset = wide.iloc[header_row_index + 1 :, selected_cols].copy()
    subset.columns = id_column_names + [str(year) for year in year_labels]

    year_columns = [str(year) for year in year_labels]
    subset = subset.dropna(axis=0, how="all")
    subset = subset.dropna(subset=year_columns, how="all")

    for col_name in id_column_names:
        subset[col_name] = (
            subset[col_name]
            .astype("string")
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        subset[col_name] = subset[col_name].replace({"<NA>": pd.NA, "": pd.NA})

    long_df = subset.melt(
        id_vars=id_column_names,
        value_vars=year_columns,
        var_name="year",
        value_name="estimate_raw",
    )

    parsed_values = long_df["estimate_raw"].map(parse_measure_value)
    long_df[["estimate_value", "estimate_flag", "estimate_text"]] = pd.DataFrame(
        parsed_values.tolist(), index=long_df.index
    )

    long_df["year"] = pd.to_numeric(long_df["year"], errors="coerce").astype("Int64")
    long_df = long_df.dropna(subset=["year"])
    long_df["year"] = long_df["year"].astype(int)

    long_df = long_df[~(long_df["estimate_value"].isna() & long_df["estimate_flag"].isna() & long_df["estimate_text"].isna())]

    long_df.insert(0, "table_number", table_meta.table_number)
    long_df.insert(1, "table_title", table_meta.title)
    long_df.insert(2, "source_url", table_meta.download_url)

    preferred_columns = [
        "table_number",
        "table_title",
        "source_url",
        *id_column_names,
        "year",
        "estimate_value",
        "estimate_flag",
        "estimate_text",
    ]
    return long_df[preferred_columns]


def build_output_filename(table_meta: TableMeta) -> str:
    table_name_map = {
        1: "sales",
        2: "ecommerce_sales",
        3: "total_inventories",
        4: "purchases_and_gross_margins",
        5: "operating_expenses",
        6: "electronic_markets_agents_brokers",
    }
    table_label = table_name_map.get(table_meta.table_number, "awts")
    return f"table_{table_meta.table_number}_{table_label}.csv"


def write_combined_output(table_frames: list[pd.DataFrame], output_dir: Path) -> Path:
    combined = pd.concat(table_frames, ignore_index=True, sort=False)
    combined = combined.drop(columns=["source_url", "estimate_text"], errors="ignore")

    preferred_order = [
        "table_number",
        "table_title",
        "2012_naics_code",
        "data_item",
        "kind_of_business",
        "type_of_operation",
        "year",
        "estimate_value",
        "estimate_flag",
    ]
    ordered_columns = [col for col in preferred_order if col in combined.columns] + [
        col for col in combined.columns if col not in preferred_order
    ]
    combined = combined[ordered_columns]

    sort_columns = [
        col for col in ["table_number", "2012_naics_code", "data_item", "year"] if col in combined.columns
    ]
    if sort_columns:
        combined = combined.sort_values(sort_columns, kind="stable").reset_index(drop=True)

    combined_output_path = output_dir / "awts_tables_1_6_combined_long.csv"
    combined.to_csv(combined_output_path, index=False)
    return combined_output_path


def main() -> None:
    RAW_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    table_metadata = fetch_table_metadata(AWTS_REPORT_URL)
    processed_frames: list[pd.DataFrame] = []

    for table_meta in table_metadata:
        destination_xlsx = RAW_DOWNLOAD_DIR / f"table_{table_meta.table_number}.xlsx"
        download_excel_file(table_meta.download_url, destination_xlsx)

        long_format = transform_table_to_long(table_meta, destination_xlsx)
        processed_frames.append(long_format)
        output_csv = OUTPUT_DIR / build_output_filename(table_meta)
        long_format.to_csv(output_csv, index=False)
        print(f"Saved {output_csv}")

    combined_output_csv = write_combined_output(processed_frames, OUTPUT_DIR)
    print(f"Saved {combined_output_csv}")


if __name__ == "__main__":
    main()