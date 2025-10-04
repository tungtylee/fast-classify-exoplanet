"""Download the Kepler cumulative KOI catalog and store it as CSV.

The script queries the MAST archive via the ``kplr`` client, retrieves the
latest cumulative KOI table, and writes it locally so the dataset can be used
offline by the rest of the project.
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
        "The `kplr` package is required. Install it with `pip install kplr` before rerunning."
    ) from exc


def _row_to_dict(row: object) -> Mapping[str, object]:
    """Convert a ``kplr`` result row into a plain dictionary."""

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
        f"Unable to interpret kplr row (type={type(row)!r}); the API format may have changed."
    )


def _normalize_results(result: object) -> Sequence[object]:
    """Normalize a ``kplr`` response into an iterable of rows."""

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
    """Try several strategies to obtain the cumulative KOI catalog."""

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
            raise RuntimeError(f"Failed to retrieve KOI catalog via kplr: {exc}") from exc

        if normalized:
            return normalized

    detail = "; ".join(errors) or "No data was returned"
    raise RuntimeError(f"Failed to retrieve KOI catalog via kplr: {detail}")


def download_cumulative_koi(columns: Iterable[str] | None = None) -> list[Mapping[str, object]]:
    """Download the cumulative KOI catalog through the ``kplr`` client."""

    client = kplr.API()
    query: dict[str, object] = {}
    if columns:
        query["select"] = ",".join(columns)

    results = _call_with_strategies(client, query)

    rows = [_row_to_dict(row) for row in results]
    if not rows:
        raise RuntimeError("kplr returned an empty KOI catalog")

    return rows


def save_as_csv(rows: Iterable[Mapping[str, object]], output_path: Path) -> None:
    """Write KOI records into ``output_path`` as CSV."""

    rows = list(rows)
    fieldnames: set[str] = set()
    for row in rows:
        fieldnames.update(row.keys())
    if not fieldnames:
        raise ValueError("No columns discovered in KOI data; aborting CSV write")

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
        help="Destination CSV path (default: data_KOI/cumulative_koi.csv)",
    )
    parser.add_argument(
        "--columns",
        nargs="*",
        help="Optional list of KOI columns to request from the API.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        koi_rows = download_cumulative_koi(args.columns)
        save_as_csv(koi_rows, args.output)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Saved cumulative KOI catalog to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
