"""Bulk-download WildChat's published Parquet shards into one Bronze file."""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path
import sys

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

DATASET = "allenai/WildChat-1M"
SPLIT = "train"
DEFAULT_OUTPUT_DIR = Path("data/bronze/wildchat-1m/train")
DOWNLOAD_WORKERS = 8
BATCH_SIZE = 5_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_date = datetime.now(timezone.utc).strftime("%Y%m%d")
    output = args.output_dir / f"wildchat_train_{run_date}.parquet"
    temporary = output.with_suffix(".parquet.tmp")
    writer = None
    output_schema = None
    row_count = 0

    try:
        snapshot = Path(snapshot_download(
            repo_id=DATASET,
            repo_type="dataset",
            allow_patterns=[f"data/{SPLIT}-*.parquet"],
            max_workers=DOWNLOAD_WORKERS,
        ))
        shards = sorted((snapshot / "data").glob(f"{SPLIT}-*.parquet"))
        if not shards:
            raise RuntimeError("Hugging Face returned no WildChat train Parquet shards.")

        for shard in shards:
            parquet = pq.ParquetFile(shard)
            if writer is None:
                output_schema = parquet.schema_arrow
                writer = pq.ParquetWriter(temporary, output_schema, compression="snappy")
            elif not parquet.schema_arrow.equals(output_schema):
                raise ValueError(f"Parquet schema differs in shard {shard.name}.")
            for batch in parquet.iter_batches(batch_size=BATCH_SIZE):
                writer.write_batch(batch)
                row_count += batch.num_rows
        writer.close()
        writer = None
        if row_count == 0:
            raise ValueError("Downloaded WildChat train split was empty.")
        temporary.replace(output)
        # Remove obsolete per-page checkpoints left by the former /rows ingestion.
        shutil.rmtree(args.output_dir / ".staging", ignore_errors=True)
    except Exception as error:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)
        print(f"WildChat bulk Bronze download failed: {error}", file=sys.stderr)
        return 1

    print(f"Dataset: {DATASET} ({SPLIT})")
    print(f"Dataset revision: {snapshot.name}")
    print(f"Parquet source shards: {len(shards)}")
    print(f"Rows downloaded: {row_count}")
    print(f"Bronze Parquet: {output}")
    print(f"Bronze file for downstream notebook steps: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
