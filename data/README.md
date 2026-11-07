# Dataset and Local Data Notes

## WildChat

- **Dataset:** [WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M)
- **Publisher:** Allen Institute for AI (AI2)
- **Format:** Parquet
- **Publisher-listed license:** ODC-BY
- **Snapshot:** train split, approximately 838,000 conversation rows

WildChat is public research data, not enterprise telemetry. It has no dependable employee, team, or organization identifiers and does not provide token or spend measures.

## O*NET taxonomy

- **Source:** [O*NET 31.0 GWAs to IWAs to DWAs](https://www.onetcenter.org/dictionary/31.0/csv/gwas_to_iwas_to_dwas.html)
- **Bronze:** source crosswalk CSV under <code>data/bronze/onet/31.0/</code>
- **Silver:** one row per unique IWA with its name and parent GWA metadata; <code>OTHER</code> is a project-defined category

The source crosswalk has 2,087 rows because it includes DWA-level mappings. The classifier catalog deduplicates to 332 IWAs plus <code>OTHER</code>.

## Local data flow

- WildChat Bronze bulk-downloads the published Parquet shards and consolidates them into one local Parquet batch.
- WildChat Silver validates and minimizes conversations, retaining nested message content for classification while omitting fields such as hashed IP, request headers, country, and state.
- Gold writes three tables—<code>conversations</code>, <code>categories</code>, and <code>conversation_categories</code>—as Parquet and as CSV inspection exports under <code>data/exports/</code>.
- Notebook Step 7 defaults to a reproducible random sample of 1,000 conversations (seed 42). The full conversation table includes the remaining rows as pending. The current classification run is partial; see [Power BI interpretation notes](../reports/powerbi/README.md).
- Re-running the same source batch replaces its matching outputs. Classification checkpoints are local SQLite files; review records are separate JSONL files.

Local datasets, checkpoints, review records, and exports are excluded from Git and are not included in a clone. Conversation content can still contain sensitive information; do not publish raw text.

## Local download

Create the environment and install dependencies as described in the main [README](../README.md). From the repository root, run:

    python etl/bronze/fetch_wildchat.py

This downloads and consolidates the train split. Keep classification samples small while validating quality and cost.
