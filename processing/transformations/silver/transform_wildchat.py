"""Validate and minimize a Bronze WildChat response into conversation-level Silver JSONL."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

DEFAULT_OUTPUT_DIR = Path("data/processed/silver/wildchat-1m/train")
DATASET = "allenai/WildChat-1M"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a Bronze WildChat response and write minimized conversation records."
    )
    parser.add_argument("bronze_file", type=Path, help="Path to a saved Bronze API response JSON file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Silver output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    return parser.parse_args()


def validate_record(item: Any) -> tuple[dict[str, Any] | None, list[str]]:
    """Return a minimized conversation record, or validation errors without source text."""
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

    for index, message in enumerate(source_messages):
        if not isinstance(message, dict):
            errors.append("conversation contains a non-object message")
            continue
        if not isinstance(message.get("role"), str) or not message["role"].strip():
            errors.append("conversation contains a message with an invalid role")
        if not isinstance(message.get("content"), str):
            errors.append("conversation contains a message with invalid content")

    openai_results = source.get("openai_moderation")
    detoxify_results = source.get("detoxify_moderation")
    for field, results in (
        ("openai_moderation", openai_results),
        ("detoxify_moderation", detoxify_results),
    ):
        if not isinstance(results, list):
            errors.append(f"{field} is missing or invalid")
        elif len(results) != len(source_messages):
            errors.append(f"{field} does not align with conversation messages")
        elif any(not isinstance(result, dict) for result in results):
            errors.append(f"{field} contains an invalid result")

    if errors:
        return None, errors

    messages = []
    for index, message in enumerate(source_messages):
        messages.append(
            {
                "role": message["role"],
                "content": message["content"],
                "openai_moderation": openai_results[index],
                "detoxify_moderation": detoxify_results[index],
            }
        )

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

    valid_records: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    for item in payload["rows"]:
        normalized, errors = validate_record(item)
        if normalized is not None:
            valid_records.append(normalized)
        else:
            rejection_counts.update(errors)

    if not valid_records:
        print("No valid conversations found; no Silver output was written.", file=sys.stderr)
        print(f"Rows rejected: {len(payload['rows'])}", file=sys.stderr)
        for reason, count in sorted(rejection_counts.items()):
            print(f"  {reason}: {count}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"silver_{args.bronze_file.stem}.jsonl"
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        for record in valid_records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary_path.replace(output_path)

    rejected_count = len(payload["rows"]) - len(valid_records)
    print(f"Bronze rows received: {len(payload['rows'])}")
    print(f"Silver conversations written: {len(valid_records)}")
    print(f"Rows rejected: {rejected_count}")
    if rejection_counts:
        print("Validation issues:")
        for reason, count in sorted(rejection_counts.items()):
            print(f"  {reason}: {count}")
    print(f"Silver output: {output_path}")
    print("Removed from Silver: hashed IP, request headers, country, and state.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
