"""將 cumulative_KOI_filtered.csv 依 `koi_disposition` 分成訓練與測試資料集。

特別設計提供 CONFIRMED、CANDIDATE 視為正類，
FALSE POSITIVE 視為負類，並統計 NOT DISPOSITIONED。
同時維持相同 `kepid`（恆星 ID）的樣本不會跨越 train/test。
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

DEFAULT_INPUT = Path(__file__).resolve().parent / "datas/cumulative_KOI_filtered.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "datas"
GROUP_COLUMN = "kepid"
LABEL_COLUMN_INDEX = 2  # 第三個欄位 (0-based index)
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_SEED = 42
POSITIVE_LABELS = ("CONFIRMED", "CANDIDATE")
NEGATIVE_LABELS = ("FALSE POSITIVE",)
UNRESOLVED_LABEL = "NOT DISPOSITIONED"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="來源資料 (預設：data_KOI/datas/cumulative_KOI_filtered.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="輸出資料夾 (預設：data_KOI/datas)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=DEFAULT_TRAIN_RATIO,
        help="訓練資料比例 (0-1，預設 0.8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="亂數種子 (預設 42)",
    )
    parser.add_argument(
        "--train-name",
        default="train_set.csv",
        help="訓練資料輸出檔名 (預設 train_set.csv)",
    )
    parser.add_argument(
        "--test-name",
        default="test_set.csv",
        help="測試資料輸出檔名 (預設 test_set.csv)",
    )
    return parser.parse_args(argv)


def load_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError("來源檔案缺少表頭")
            rows = list(reader)
    except FileNotFoundError as exc:
        raise SystemExit(f"找不到來源檔案：{path}") from exc
    except Exception as exc:
        raise SystemExit(f"讀取 {path} 時發生錯誤：{exc}") from exc

    if not rows:
        raise SystemExit(f"來源資料為空：{path}")

    if LABEL_COLUMN_INDEX >= len(reader.fieldnames):
        raise SystemExit("來源資料第三欄不存在，請確認輸入檔格式。")

    label_col = reader.fieldnames[LABEL_COLUMN_INDEX]
    if label_col != "koi_disposition":
        print(
            f"警告：第三欄非 `koi_disposition`，實際欄位為 `{label_col}`，仍以此欄為標籤。"
        )

    return reader.fieldnames, rows


def split_by_group(
    rows: Iterable[dict[str, str]],
    group_column: str,
    train_ratio: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)

    for idx, row in enumerate(rows):
        group_key = row.get(group_column)
        if not group_key:
            group_key = f"__missing_{idx}"
        grouped[group_key].append(row)

    if not grouped:
        raise SystemExit("無法根據 `kepid` 分組，請確認欄位是否存在。")

    group_keys = list(grouped.keys())
    rng = random.Random(seed)
    rng.shuffle(group_keys)

    split_index = int(len(group_keys) * train_ratio)
    train_keys = set(group_keys[:split_index])

    train_rows: list[dict[str, str]] = []
    test_rows: list[dict[str, str]] = []

    for key, group_rows in grouped.items():
        if key in train_keys:
            train_rows.extend(group_rows)
        else:
            test_rows.extend(group_rows)

    return train_rows, test_rows


def class_balance(rows: Iterable[dict[str, str]], label_col: str) -> dict[str, float | int]:
    counts: dict[str, int] = defaultdict(int)
    total = 0
    for row in rows:
        label = row.get(label_col, "") or ""
        counts[label] += 1
        total += 1

    confirmed = counts.get("CONFIRMED", 0)
    candidate = counts.get("CANDIDATE", 0)
    false_positive = sum(counts.get(label, 0) for label in NEGATIVE_LABELS)
    not_dispositioned = counts.get(UNRESOLVED_LABEL, 0)

    positive = sum(counts.get(label, 0) for label in POSITIVE_LABELS)
    negative = false_positive
    classified = positive + negative

    positive_pct = (positive / classified * 100) if classified else 0.0
    negative_pct = (negative / classified * 100) if classified else 0.0
    others = total - (positive + negative + not_dispositioned)
    if others < 0:
        others = 0

    return {
        "total": total,
        "confirmed": confirmed,
        "candidate": candidate,
        "false_positive": false_positive,
        "not_dispositioned": not_dispositioned,
        "positive": positive,
        "negative": negative,
        "positive_pct": positive_pct,
        "negative_pct": negative_pct,
        "others": others,
    }


def print_balance(title: str, stats: dict[str, float | int]) -> None:
    print(f"{title}:")
    print(
        f"  - Positive (CONFIRMED+CANDIDATE): {stats['positive']} "
        f"({stats['positive_pct']:.2f}% of classified)"
    )
    print(
        f"  - Negative (FALSE POSITIVE):       {stats['negative']} "
        f"({stats['negative_pct']:.2f}% of classified)"
    )
    print(f"  - Not Dispositioned: {stats['not_dispositioned']}")
    print(f"  - Other labels:      {stats['others']}")
    print(
        "  - CONFIRMED: {confirmed} | CANDIDATE: {candidate} | FALSE POSITIVE: {fp}".format(
            confirmed=stats["confirmed"],
            candidate=stats["candidate"],
            fp=stats["false_positive"],
        )
    )
    print(f"  - Total rows: {stats['total']}")


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    fieldnames, rows = load_rows(args.input)
    label_col = fieldnames[LABEL_COLUMN_INDEX]

    train_rows, test_rows = split_by_group(
        rows=rows,
        group_column=GROUP_COLUMN,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )

    print("\n--- Data Split Summary ---")
    print(f"Total rows: {len(rows)}")
    print(f"Training rows: {len(train_rows)}")
    print(f"Test rows: {len(test_rows)}")

    print("\n--- Class Balance Analysis (CONFIRMED, CANDIDATE vs. FALSE POSITIVE, NOT DISPOSITIONED) ---")
    train_stats = class_balance(train_rows, label_col)
    test_stats = class_balance(test_rows, label_col)
    print_balance("Training Set", train_stats)
    print_balance("Test Set", test_stats)
    print("---------------------------------------------\n")

    output_dir = args.output_dir
    train_path = output_dir / args.train_name
    test_path = output_dir / args.test_name

    write_csv(train_path, fieldnames, train_rows)
    write_csv(test_path, fieldnames, test_rows)

    print("檔案已輸出：")
    print(f"- {train_path}")
    print(f"- {test_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
