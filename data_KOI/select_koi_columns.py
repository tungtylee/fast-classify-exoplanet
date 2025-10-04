"""Filter ``cumulative_KOI.csv`` based on the recommended column priority.

The script reads the column metadata in ``KOI_col_info.csv``, keeps the fields
whose priority suggestion is marked as high or medium-high, and writes a
trimmed copy of ``cumulative_KOI.csv`` containing only those columns.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable, Sequence

NAME_COLUMN_INDEX = 2
PRIORITY_COLUMN_INDEX = 8
DEFAULT_PRIORITIES = ("high", "medium_high")
PRIORITY_TRANSLATIONS = {
    "": "",
    "\u9ad8": "high",  # Chinese label for "high"
    "\u4e2d\u9ad8": "medium_high",  # Chinese label for "medium-high"
    "\u4e2d": "medium",  # Chinese label for "medium"
    "\u4f4e": "low",  # Chinese label for "low"
}
REQUIRED_COLUMNS = ("kepid", "kepler_name", "koi_disposition")


def _normalize_priority(value: str) -> str:
    """Map raw priority labels to lowercase English keywords."""

    normalized = value.replace("\u504f", "").strip()  # remove the modifier character used in Chinese labels
    if not normalized:
        return ""
    return PRIORITY_TRANSLATIONS.get(normalized, normalized.lower())


def load_priority_columns(info_path: Path, priorities: Sequence[str]) -> list[str]:
    """Read column names whose priority matches ``priorities``."""

    normalized_priorities = {_normalize_priority(p) for p in priorities}

    try:
        with info_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError("KOI column info file is missing a header row")

            try:
                name_key = reader.fieldnames[NAME_COLUMN_INDEX]
                priority_key = reader.fieldnames[PRIORITY_COLUMN_INDEX]
            except IndexError as exc:
                raise ValueError(
                    "KOI column info file does not match the expected layout."
                ) from exc

            selected: list[str] = []
            for idx, row in enumerate(reader, start=2):
                name = (row.get(name_key) or "").strip()
                if not name:
                    continue
                priority = _normalize_priority((row.get(priority_key) or "").strip())
                if priority and priority in normalized_priorities:
                    selected.append(name)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Column info file not found: {info_path}") from exc

    if not selected:
        raise ValueError(
            f"No columns with priority {priorities} were found in {info_path}."
        )

    return selected


def filter_cumulative_csv(
    cumulative_path: Path,
    output_path: Path,
    columns: Sequence[str],
) -> None:
    """Select the requested columns from ``cumulative_KOI.csv`` and write them."""

    try:
        with cumulative_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError("cumulative_KOI.csv is missing a header row")

            available_columns = reader.fieldnames

            desired_columns: list[str] = []
            desired_columns.extend(REQUIRED_COLUMNS)
            desired_columns.extend(columns)

            seen: set[str] = set()
            unique_columns = [
                col for col in desired_columns if not (col in seen or seen.add(col))
            ]

            missing = [col for col in unique_columns if col not in available_columns]
            if missing:
                print(
                    "Warning: the following columns are not present in cumulative_KOI.csv and will be skipped: "
                    + ", ".join(missing),
                    file=sys.stderr,
                )

            final_columns = [col for col in unique_columns if col in available_columns]
            if not final_columns:
                raise ValueError("None of the requested columns exist in cumulative_KOI.csv")

            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8", newline="") as out_handle:
                writer = csv.DictWriter(out_handle, fieldnames=final_columns)
                writer.writeheader()
                for row in reader:
                    writer.writerow({col: row.get(col, "") for col in final_columns})
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"KOI data file not found: {cumulative_path}") from exc


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--info",
        type=Path,
        default=Path(__file__).resolve().parent / "datas/KOI_col_info.csv",
        help="Path to KOI column info (default: data_KOI/datas/KOI_col_info.csv)",
    )
    parser.add_argument(
        "--cumulative",
        type=Path,
        default=Path(__file__).resolve().parent / "datas/cumulative_KOI.csv",
        help="Path to the cumulative_KOI.csv file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "datas/cumulative_KOI_filtered.csv",
        help="Destination CSV path",
    )
    parser.add_argument(
        "--priorities",
        nargs="*",
        default=list(DEFAULT_PRIORITIES),
        help="Priority labels to keep (default: high, medium_high)",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        columns = load_priority_columns(args.info, args.priorities)
        filter_cumulative_csv(args.cumulative, args.output, columns)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(
        "Column filtering finished:\n"
        f"- Source: {args.cumulative}\n"
        f"- Column info: {args.info}\n"
        f"- Saved as: {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
