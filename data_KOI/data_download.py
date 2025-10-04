"""下載 Kepler Cumulative KOI 資料表並存成 CSV。

此腳本透過 `kplr` 套件連線 MAST 服務端點，抓取最新的
Cumulative KOI 目錄資料並儲存為本地的 CSV 檔，供離線分析使用。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

try:
    import kplr
except ImportError as exc:
    raise SystemExit(
        "找不到 `kplr` 套件，請先執行 `pip install kplr` 再嘗試。"
    ) from exc


def _row_to_dict(row: object) -> Mapping[str, object]:
    """將 kplr 回傳的資料列轉成一般字典。"""

    if isinstance(row, Mapping):
        return dict(row)

    data = getattr(row, "data", None)
    if isinstance(data, Mapping):
        return dict(data)

    private = getattr(row, "_data", None)
    if isinstance(private, Mapping):
        return dict(private)

    if hasattr(row, "to_dict"):
        converted = row.to_dict()  # type: ignore[call-arg]
        if isinstance(converted, Mapping):
            return dict(converted)

    attrs = {
        key: value
        for key, value in getattr(row, "__dict__", {}).items()
        if not key.startswith("_") and not callable(value)
    }
    if attrs:
        return attrs

    raise TypeError(
        f"無法解析 kplr 資料列 (type={type(row)!r})，API 格式可能已變更。"
    )


def _normalize_results(result: object) -> Sequence[object]:
    """將 kplr 回傳的結果整理成序列。"""

    if result is None:
        return []

    if isinstance(result, Sequence) and not isinstance(result, (str, bytes, bytearray)):
        return list(result)

    if isinstance(result, Mapping):
        if "data" in result:
            data = result["data"]
            if isinstance(data, Mapping):
                return list(data.values())
            if isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray)):
                return list(data)
            return [data]
        return [result]

    data_attr = getattr(result, "data", None)
    if isinstance(data_attr, Sequence) and not isinstance(data_attr, (str, bytes, bytearray)):
        return list(data_attr)

    return [result]


def _call_with_strategies(client: kplr.API, query: Mapping[str, object]) -> Sequence[object]:
    """嘗試多種呼叫方式取得 cumulative KOI 資料。"""

    strategies = []

    def wrap(fn):
        strategies.append(fn)
        return fn

    @wrap
    def use_default() -> object:
        return client.kois(**query)

    @wrap
    def use_table_kw() -> object:
        return client.kois(table="cumulative", **query)

    @wrap
    def use_ea_request() -> object:
        params = dict(query)
        params.setdefault("format", "json")
        return client.ea_request("cumulative", "search", **params)

    errors: list[str] = []
    for caller in strategies:
        try:
            result = caller()
            normalized = _normalize_results(result)
        except TypeError as exc:
            errors.append(f"{caller.__name__}: {exc}")
            continue
        except Exception as exc:
            raise RuntimeError(f"kplr 下載 KOI 資料失敗：{exc}") from exc

        if normalized:
            return normalized

    detail = "; ".join(errors) or "無法取得任何資料"
    raise RuntimeError(f"kplr 下載 KOI 資料失敗：{detail}")


def download_cumulative_koi(columns: Iterable[str] | None = None) -> list[Mapping[str, object]]:
    """透過 kplr 下載 Cumulative KOI 資料。"""

    client = kplr.API()
    query: dict[str, object] = {}
    if columns:
        query["select"] = ",".join(columns)

    results = _call_with_strategies(client, query)

    rows = [_row_to_dict(row) for row in results]
    if not rows:
        raise RuntimeError("未從 kplr 取得任何 KOI 資料")

    return rows


def save_as_csv(rows: Iterable[Mapping[str, object]], output_path: Path) -> None:
    """將資料列寫入 CSV 檔。"""

    rows = list(rows)
    fieldnames: set[str] = set()
    for row in rows:
        fieldnames.update(row.keys())
    if not fieldnames:
        raise ValueError("KOI 資料缺少欄位，無法輸出 CSV")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=sorted(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "cumulative_koi.csv",
        help="輸出 CSV 路徑 (預設：data_KOI/cumulative_koi.csv)",
    )
    parser.add_argument(
        "--columns",
        nargs="*",
        help="可選欄位列表，若不指定則下載全部欄位",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        koi_rows = download_cumulative_koi(args.columns)
        save_as_csv(koi_rows, args.output)
    except Exception as exc:
        print(f"錯誤：{exc}", file=sys.stderr)
        return 1

    print(f"已將 Cumulative KOI 資料儲存至 {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
