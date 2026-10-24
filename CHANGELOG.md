# Project Progress

## October 2026

- **Oct 1** — Created and committed the initial repository structure.
- **Oct 2** — Added the platform problem statement to the README.
- **Oct 3** — Documented the high-level Mermaid architecture, including the Bronze/Silver/Gold flow.
- **Oct 5** — Selected WildChat converstaion dataset and scoped conversation-level usage categorization and reporting.
- **Oct 6** — Set up the Python environment and Jupyter workflow; added the Hugging Face Bronze fetch script.
- **Oct 7** — Successfully fetched a 100-row sample into local Bronze storage; added safe schema and conversation inspection to the notebook.
- **Oct 8** — Added Silver validation and minimization for WildChat conversations, with a notebook step to run it.
- **Oct 10** — Simplified local data paths to `data/bronze/` and `data/silver/`; documented the conversation-category analytics model.
- **Oct 13** — Added O*NET 31.0 Bronze ingestion and a deduplicated IWA catalog with a project-defined `OTHER` category.
- **Oct 14** — Added a capped, explicit-run OpenAI classifier that writes Gold conversation-category associations and a review queue.
- **Oct 15** — Switched classification to the OpenAI Decisions API, simplified Gold processing, and added multi-IWA scoring with safe error diagnostics.
- **Oct 16** — Changed WildChat Silver to Parquet and modeled Gold as conversation, category, and conversation-category Parquet tables, with CSV exports for all three. Consolidated pipeline scripts under `etl/` and aligned the problem statement with conversation usage analytics.
- **Oct 20** — Added an adjustable top-N notebook view of distinct category counts calculated directly from the three Gold tables; zero-count categories are hidden.
- **Oct 22** — Set the classifier to evaluate all 332 IWAs plus `OTHER` in one Decisions API request per conversation; made identical WildChat Bronze requests overwrite their same-day file; updated the notebook and architecture/data docs.
- **Oct 23** — Changed WildChat Bronze ingestion to bulk-download the published Parquet shards and consolidate them into one full Bronze Parquet file; Silver streams the batch into one Parquet file, and Gold writes the full conversation table plus category tables with same-batch overwrite.
