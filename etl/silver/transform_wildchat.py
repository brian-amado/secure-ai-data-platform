"""Validate and minimize a Bronze WildChat response into Silver Parquet."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_OUTPUT_DIR = Path("data/silver/wildchat-1m/train")
DATASET = "allenai/WildChat-1M"
SCHEMA = pa.schema([
    ("conversation_id", pa.string()),
    ("source_row_index", pa.int64()),
    ("dataset", pa.string()),
    ("model", pa.string()),
    ("timestamp", pa.string()),
    ("turn_count", pa.int64()),
    ("language", pa.string()),
    ("toxic", pa.bool_()),
    ("redacted", pa.bool_()),
    ("messages", pa.list_(pa.struct([
        ("role", pa.string()),
        ("content", pa.string()),
        ("openai_moderation", pa.string()),
        ("detoxify_moderation", pa.string()),
    ]))),
])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bronze_file", type=Path, help="Path to a saved Bronze API response JSON file.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def validate_record(item: Any) -> tuple[dict[str, Any] | None, list[str]]:
    """Return a minimized record, or validation errors without source text."""
    errors: list[str] = []
    if not isinstance(item, dict):
        return None, ["row wrapper is not an object"]
    row_index = item.get("row_idx")
    if type(row_index) is not int or row_index < 0:
        errors.append("row_idx is missing or invalid")
    source = item.get("row")
    if not isinstance(source, dict):
        return None, errors + ["row is not an object"]
    conversation_hash = source.get("conversation_hash")
    if not isinstance(conversation_hash, str) or not conversation_hash.strip():
        errors.append("conversation_hash is missing or invalid")
    for field in ("model", "timestamp", "language"):
        value = source.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{field} is missing or invalid")
    turn_count = source.get("turn")
    if type(turn_count) is not int or turn_count < 0:
        errors.append("turn is missing or invalid")
    for field in ("toxic", "redacted"):
        if type(source.get(field)) is not bool:
            errors.append(f"{field} is missing or invalid")
    source_messages = source.get("conversation")
    if not isinstance(source_messages, list) or not source_messages:
        errors.append("conversation is missing or empty")
        source_messages = []
    for message in source_messages:
        if not isinstance(message, dict):
            errors.append("conversation contains a non-object message")
        elif not isinstance(message.get("role"), str) or not message["role"].strip():
            errors.append("conversation contains a message with an invalid role")
        elif not isinstance(message.get("content"), str):
            errors.append("conversation contains a message with invalid content")
    moderation = [source.get("openai_moderation"), source.get("detoxify_moderation")]
    for field, values in zip(("openai_moderation", "detoxify_moderation"), moderation):
        if not isinstance(values, list):
            errors.append(f"{field} is missing or invalid")
        elif len(values) != len(source_messages) or any(not isinstance(value, dict) for value in values):
            errors.append(f"{field} does not align with conversation messages")
    if errors:
        return None, errors
    openai_results, detoxify_results = moderation
    messages = [
        {
            "role": message["role"],
            "content": message["content"],
            # Moderation schemas can evolve; encode each object as JSON in a stable Parquet string field.
            "openai_moderation": json.dumps(openai_results[index], ensure_ascii=False),
            "detoxify_moderation": json.dumps(detoxify_results[index], ensure_ascii=False),
        }
        for index, message in enumerate(source_messages)
    ]
    # Deliberately omit hashed_ip, request headers, country, and state.
    return {
        "conversation_id": conversation_hash,
        "source_row_index": row_index,
        "dataset": DATASET,
        "model": source["model"],
        "timestamp": source["timestamp"],
        "turn_count": turn_count,
        "language": source["language"],
        "toxic": source["toxic"],
        "redacted": source["redacted"],
        "messages": messages,
    }, []


def main() -> int:
    args = parse_args()
    try:
        payload = json.loads(args.bronze_file.read_text(encoding="utf-8"))
    except OSError as error:
        print(f"Could not read Bronze input: {error}", file=sys.stderr)
        return 1
    except json.JSONDecodeError:
        print("Bronze input is not valid JSON.", file=sys.stderr)
        return 1
    if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list):
        print("Bronze input must contain a rows list.", file=sys.stderr)
        return 1
    records: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    for item in payload["rows"]:
        normalized, errors = validate_record(item)
        if normalized is None:
            rejected.update(errors)
        else:
            records.append(normalized)
    if not records:
        print("No valid conversations found; no Silver output was written.", file=sys.stderr)
        for reason, count in sorted(rejected.items()):
            print(f"  {reason}: {count}", file=sys.stderr)
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"silver_{args.bronze_file.stem}.parquet"
    temporary = output.with_suffix(".parquet.tmp")
    pq.write_table(pa.Table.from_pylist(records, schema=SCHEMA), temporary, compression="snappy")
    temporary.replace(output)
    print(f"Bronze rows received: {len(payload['rows'])}")
    print(f"Silver conversations written: {len(records)}")
    print(f"Rows rejected: {len(payload['rows']) - len(records)}")
    print(f"Silver Parquet: {output}")
    print("Removed from Silver: hashed IP, request headers, country, and state.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
