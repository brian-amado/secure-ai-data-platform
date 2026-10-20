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

- Start with a small sample, about 1,000 conversations, for pipeline development.
- Keep local samples under `data/bronze/`, `data/silver/`, and `data/gold/`; these data folders are excluded from Git. CSV inspection exports for all three Gold tables are under the ignored `data/exports/` directory.
- Before processing, review fields and decide which are needed.

## Data handling notes

The dataset contains public conversation text and metadata that is treated as sensitive, including location, hashed IP, browser headers, timestamps, and moderation labels. 

The first version should focus on validated, minimized data and aggregate reports. Raw conversation text is out of scope for the initial RAG index.

## Reference taxonomy: O*NET work activities

- Source: [O*NET 31.0 GWAs to IWAs to DWAs](https://www.onetcenter.org/dictionary/31.0/csv/gwas_to_iwas_to_dwas.html)
- Bronze: preserve the downloaded CSV under `data/bronze/onet/31.0/`.
- Silver: deduplicate the crosswalk to one record per IWA and add the project-defined `OTHER` non-work category under `data/silver/onet/31.0/`.
- WildChat Silver: write one validated conversation per Parquet row, including nested messages; omit hashed IP, request headers, country, and state.
- Gold: publish `conversations`, `categories`, and `conversation_categories` as separate Parquet tables, plus CSV inspection copies of each under `data/exports/`. Keep uncertain and failed cases in a separate review JSONL. The notebook can calculate distinct classified-conversation counts by category directly from these Gold tables; it omits zero-count categories from the adjustable top-N view and does not persist a duplicate summary table.
- The data layer folders are excluded from Git.
