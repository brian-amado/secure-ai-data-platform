"""Download the official O*NET GWA-to-IWA-to-DWA reference CSV to Bronze."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
from pathlib import Path
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ONET_VERSION = "31.0"
SOURCE_URL = "https://www.onetcenter.org/dl_files/database/db_31_0_csv/gwas_to_iwas_to_dwas.csv"
DEFAULT_OUTPUT_DIR = Path("data/bronze/onet/31.0")
REQUIRED_COLUMNS = [
    "GWA Element ID",
    "GWA Element Name",
    "IWA Element ID",
    "IWA Element Name",
    "DWA Element ID",
    "DWA Element Name",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch and preserve the official O*NET 31.0 work-activity crosswalk."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Bronze output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    return parser.parse_args()


def inspect_csv(content: bytes) -> tuple[int, int]:
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ValueError("O*NET CSV could not be decoded as UTF-8.") from None

    reader = csv.DictReader(io.StringIO(text, newline=""))
    if reader.fieldnames != REQUIRED_COLUMNS:
        raise ValueError("O*NET CSV headers did not match the documented six-column schema.")

    row_count = 0
    iwa_ids: set[str] = set()
    for row in reader:
        row_count += 1
        if not row.get("IWA Element ID", "").strip() or not row.get("IWA Element Name", "").strip():
            raise ValueError("O*NET CSV contains a row with a missing IWA identifier or name.")
        iwa_ids.add(row["IWA Element ID"].strip())
    if row_count == 0:
        raise ValueError("O*NET CSV contains no data rows.")
    return row_count, len(iwa_ids)


def main() -> int:
    args = parse_args()
    request = Request(
        SOURCE_URL,
        headers={"User-Agent": "secure-ai-data-platform/0.1"},
    )
    try:
        with urlopen(request, timeout=30) as response:
            content = response.read()
    except HTTPError as error:
        print(f"O*NET returned HTTP {error.code}.", file=sys.stderr)
        return 1
    except (TimeoutError, URLError) as error:
        print(f"Could not reach the O*NET download: {error}", file=sys.stderr)
        return 1

    try:
        source_rows, unique_iwas = inspect_csv(content)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1

    digest = hashlib.sha256(content).hexdigest()
    output_path = args.output_dir / f"gwas_to_iwas_to_dwas_{digest[:12]}.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        if output_path.read_bytes() != content:
            print("Existing content-addressed Bronze file did not match downloaded bytes.", file=sys.stderr)
            return 1
        print("Identical O*NET source already exists; preserved the existing Bronze file.")
    else:
        temporary_path = output_path.with_suffix(".csv.tmp")
        temporary_path.write_bytes(content)
        temporary_path.replace(output_path)
        print("Saved the original O*NET CSV bytes to Bronze.")

    print(f"O*NET version: {ONET_VERSION}")
    print(f"Crosswalk rows: {source_rows}")
    print(f"Unique IWA identifiers: {unique_iwas}")
    print(f"SHA-256: {digest}")
    print(f"Bronze file: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
