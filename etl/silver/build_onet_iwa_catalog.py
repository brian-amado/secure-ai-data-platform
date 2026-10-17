"""Build a deduplicated IWA catalog plus the project-defined non-work category."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

DEFAULT_OUTPUT = Path("data/silver/onet/31.0/iwa_categories.csv")
ONET_VERSION = "31.0"
PROJECT_CATEGORY_VERSION = "1.0"
REQUIRED_COLUMNS = {
    "GWA Element ID",
    "GWA Element Name",
    "IWA Element ID",
    "IWA Element Name",
    "DWA Element ID",
    "DWA Element Name",
}
OUTPUT_COLUMNS = [
    "category_id",
    "category_name",
    "definition",
    "parent_gwa_id",
    "parent_gwa_name",
    "taxonomy_source",
    "taxonomy_version",
    "category_type",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create one row per O*NET IWA and add the project-defined OTHER category."
    )
    parser.add_argument("bronze_file", type=Path, help="Path to the O*NET Bronze CSV.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Silver catalog path (default: {DEFAULT_OUTPUT}).",
    )
    return parser.parse_args()


def read_iwas(path: Path) -> tuple[int, list[dict[str, str]]]:
    try:
        stream = path.open("r", encoding="utf-8-sig", newline="")
    except OSError as error:
        raise ValueError(f"Could not read O*NET Bronze input: {error}") from error

    iwas: dict[str, tuple[str, str, str]] = {}
    source_rows = 0
    with stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not REQUIRED_COLUMNS.issubset(reader.fieldnames):
            raise ValueError("O*NET Bronze input is missing required crosswalk columns.")
        for row in reader:
            source_rows += 1
            iwa_id = (row.get("IWA Element ID") or "").strip()
            iwa_name = (row.get("IWA Element Name") or "").strip()
            gwa_id = (row.get("GWA Element ID") or "").strip()
            gwa_name = (row.get("GWA Element Name") or "").strip()
            if not all((iwa_id, iwa_name, gwa_id, gwa_name)):
                raise ValueError("O*NET crosswalk has a row with missing IWA or parent GWA values.")
            value = (iwa_name, gwa_id, gwa_name)
            if iwa_id in iwas and iwas[iwa_id] != value:
                raise ValueError(f"IWA {iwa_id} maps to inconsistent names or GWA parents.")
            iwas[iwa_id] = value

    if source_rows == 0 or not iwas:
        raise ValueError("O*NET crosswalk contains no IWA records.")

    records = []
    for iwa_id, (iwa_name, gwa_id, gwa_name) in sorted(iwas.items()):
        records.append(
            {
                "category_id": f"IWA:{iwa_id}",
                "category_name": iwa_name,
                "definition": iwa_name,
                "parent_gwa_id": gwa_id,
                "parent_gwa_name": gwa_name,
                "taxonomy_source": "O*NET Work Activities",
                "taxonomy_version": ONET_VERSION,
                "category_type": "onet_iwa",
            }
        )
    records.append(
        {
            "category_id": "OTHER",
            "category_name": "Other",
            "definition": "Non-work-related conversation; not an O*NET work activity.",
            "parent_gwa_id": "",
            "parent_gwa_name": "",
            "taxonomy_source": "Project-defined",
            "taxonomy_version": PROJECT_CATEGORY_VERSION,
            "category_type": "non_work",
        }
    )
    return source_rows, records


def main() -> int:
    args = parse_args()
    try:
        source_rows, records = read_iwas(args.bronze_file)
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(records)
    temporary_path.replace(args.output)

    print(f"O*NET crosswalk rows read: {source_rows}")
    print(f"Unique IWA categories: {len(records) - 1}")
    print("Project-defined non-work categories: 1 (OTHER)")
    print(f"Category catalog: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
