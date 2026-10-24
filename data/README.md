# Dataset source profile

## Source

- Name: WildChat-1M
- Publisher: Allen Institute for AI (AI2)
- Location: https://huggingface.co/datasets/allenai/WildChat-1M
- Format: Parquet
- License listed by publisher: ODC-BY
- Dataset page reports approximately 838,000 rows in the train split (checked 2026-10-05)

## Purpose

Use the dataset to develop a pipeline that can categorize AI usage, report aggregate usage, and support a future RAG interface for questions about those reports.

To pull usage catgorization/taxonomy.

## Initial development scope

- Bronze ingests the full published snapshot; keep classification samples small while validating costs and category quality.
- Keep local samples under `data/bronze/`, `data/silver/`, and `data/gold/`; these data folders are excluded from Git. CSV inspection exports for all three Gold tables are under the ignored `data/exports/` directory.
- Before processing, review fields and decide which are needed.
- WildChat Bronze is one consolidated Parquet file per UTC-day run. Source shards are downloaded through the Hugging Face Hub cache and combined in batches.
- Gold filenames use the Silver input batch name, so rerunning classification for the same Silver file overwrites its three Parquet tables, three CSV exports, and review JSONL. Association rows retain the exact classification time in `classified_at`.
- Notebook Step 2 bulk-downloads the published WildChat Parquet shards through the Hugging Face Hub client and consolidates them into one full Bronze Parquet file. The Hub client uses concurrent, resumable shard downloads. A same-day rerun replaces the Bronze batch file.
- From the repository root, run `python etl/bronze/fetch_wildchat.py` to download and consolidate the train split.
- Silver transforms the complete Bronze batch into one Parquet file, streaming records to avoid loading the full dataset into memory. Gold writes one full conversation table plus the category and conversation-category tables for that Silver batch. The notebook currently classifies one conversation by default; the rest are marked pending until classification is run for them.

## Data handling notes

The dataset contains public conversation text and metadata that is treated as sensitive, including location, hashed IP, browser headers, timestamps, and moderation labels. 

The first version should focus on validated, minimized data and aggregate reports. Raw conversation text is out of scope for the initial RAG index.

## Reference taxonomy: O*NET work activities

- Source: [O*NET 31.0 GWAs to IWAs to DWAs](https://www.onetcenter.org/dictionary/31.0/csv/gwas_to_iwas_to_dwas.html)
- Bronze: preserve the downloaded CSV under `data/bronze/onet/31.0/`.
- Silver: deduplicate the crosswalk to one record per IWA and add the project-defined `OTHER` non-work category under `data/silver/onet/31.0/`.
- WildChat Silver: write one validated conversation per Parquet row, including nested messages; omit hashed IP, request headers, country, and state.
- Gold: publish `conversations`, `categories`, and `conversation_categories` as separate Parquet tables, plus CSV inspection copies of each under `data/exports/`. Keep uncertain and failed cases in a separate review JSONL. Outputs are keyed by the Silver input batch, so repeating the same batch replaces those outputs; `classified_at` records when associations were produced. The notebook can calculate distinct classified-conversation counts by category directly from these Gold tables; it omits zero-count categories from the adjustable top-N view and does not persist a duplicate summary table.
- The data layer folders are excluded from Git.
