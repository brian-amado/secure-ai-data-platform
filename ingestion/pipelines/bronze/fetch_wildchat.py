"""Fetch one page from the public WildChat dataset into the local Bronze layer."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


API_URL = "https://datasets-server.huggingface.co/rows"
DATASET = "allenai/WildChat-1M"
CONFIG = "default"
SPLIT = "train"
MAX_ROWS_PER_REQUEST = 100
DEFAULT_OUTPUT_DIR = Path("data/raw/bronze/wildchat-1m/train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch one page of WildChat rows into local Bronze storage."
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Zero-based row offset to request (default: 0).",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=100,
        help="Rows to request, from 1 to 100 (default: 100).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Local output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    args = parser.parse_args()

    if args.offset < 0:
        parser.error("--offset must be zero or greater")
    if not 1 <= args.length <= MAX_ROWS_PER_REQUEST:
        parser.error(f"--length must be between 1 and {MAX_ROWS_PER_REQUEST}")

    return args


def main() -> int:
    args = parse_args()
    query = urlencode(
        {
            "dataset": DATASET,
            "config": CONFIG,
            "split": SPLIT,
            "offset": args.offset,
            "length": args.length,
        }
    )
    request = Request(
        f"{API_URL}?{query}",
        headers={"User-Agent": "secure-ai-data-platform/0.1"},
    )

    try:
        with urlopen(request, timeout=30) as response:
            raw_response = response.read()
    except HTTPError as error:
        print(f"Hugging Face API returned HTTP {error.code}.", file=sys.stderr)
        return 1
    except (TimeoutError, URLError) as error:
        print(f"Could not reach the Hugging Face API: {error}", file=sys.stderr)
        return 1

    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError:
        print("Hugging Face API returned invalid JSON.", file=sys.stderr)
        return 1

    rows = payload.get("rows")
    if not isinstance(rows, list):
        print("Hugging Face API response did not contain a rows list.", file=sys.stderr)
        return 1
    if len(rows) > args.length:
        print("Hugging Face API returned more rows than requested.", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fetched_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_path = args.output_dir / (
        f"rows_offset_{args.offset:06d}_{fetched_at}.json"
    )
    output_path.write_bytes(raw_response)

    print(f"Fetched {len(rows)} rows from {DATASET} ({SPLIT}, offset {args.offset}).")
    print(f"Saved raw API response to {output_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
