"""依照欄位優先度整理 cumulative_KOI.csv。

此腳本會從 `KOI_col_info.csv` 讀取欄位清單，挑選 "優先程度建議"
為「高」或「中高」的欄位，從 cumulative_KOI.csv 擷取對應資料並另存 CSV。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable, Sequence

PRIORITY_COLUMN = "優先程度建議"
NAME_COLUMN = "欄位名稱"
DEFAULT_PRIORITIES = ("高", "中高")
REQUIRED_COLUMNS = ("kepid", "kepler_name")


def _normalize_priority(value: str) -> str:
    """移除常見修飾字以便比對，例如 `中偏高` → `中高`。"""

    return value.replace("偏", "").strip()


def load_priority_columns(info_path: Path, priorities: Sequence[str]) -> list[str]:
    """從 KOI 欄位資訊表讀取符合優先度的欄位名稱。"""

    normalized_priorities = {_normalize_priority(p) for p in priorities}

    try:
        with info_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError("KOI 欄位資訊表缺少表頭")

            selected: list[str] = []
            for idx, row in enumerate(reader, start=2):  # 起始列號扣除表頭
                name = (row.get(NAME_COLUMN) or "").strip()
                if not name:
                    continue
                priority = _normalize_priority((row.get(PRIORITY_COLUMN) or "").strip())
                if priority and priority in normalized_priorities:
                    selected.append(name)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"找不到欄位資訊檔案：{info_path}") from exc

    if not selected:
        raise ValueError(
            f"在 {info_path} 中未找到符合優先度 {priorities} 的欄位。"
        )

    return selected


def filter_cumulative_csv(
    cumulative_path: Path,
    output_path: Path,
    columns: Sequence[str],
) -> None:
    """從 cumulative_KOI.csv 擷取指定欄位並輸出。"""

    try:
        with cumulative_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError("cumulative_KOI.csv 缺少表頭")

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
                    "警告：以下欄位在 cumulative_KOI.csv 中不存在，將略過："
                    + ", ".join(missing),
                    file=sys.stderr,
                )

            final_columns = [col for col in unique_columns if col in available_columns]
            if not final_columns:
                raise ValueError("選定欄位皆不存在於 cumulative_KOI.csv")

            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8", newline="") as out_handle:
                writer = csv.DictWriter(out_handle, fieldnames=final_columns)
                writer.writeheader()
                for row in reader:
                    writer.writerow({col: row.get(col, "") for col in final_columns})
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"找不到 KOI 資料檔案：{cumulative_path}") from exc


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--info",
        type=Path,
        default=Path(__file__).resolve().parent / "datas/KOI_col_info.csv",
        help="KOI 欄位資訊檔案路徑 (預設：data_KOI/datas/KOI_col_info.csv)",
    )
    parser.add_argument(
        "--cumulative",
        type=Path,
        default=Path(__file__).resolve().parent / "datas/cumulative_KOI.csv",
        help="cumulative_KOI.csv 檔案路徑",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "datas/cumulative_KOI_filtered.csv",
        help="輸出 CSV 路徑",
    )
    parser.add_argument(
        "--priorities",
        nargs="*",
        default=list(DEFAULT_PRIORITIES),
        help="欲保留的優先程度（預設：高、中高）",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        columns = load_priority_columns(args.info, args.priorities)
        filter_cumulative_csv(args.cumulative, args.output, columns)
    except Exception as exc:
        print(f"錯誤：{exc}", file=sys.stderr)
        return 1

    print(
        "已完成欄位篩選：\n"
        f"- 來源：{args.cumulative}\n"
        f"- 欄位資訊：{args.info}\n"
        f"- 儲存為：{args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
