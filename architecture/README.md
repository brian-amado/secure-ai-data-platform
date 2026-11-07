# Architecture and Data Model

The diagram shows the implemented local pipeline and planned consumer path. Azure services and deployment remain future work. See [system-architecture.md](system-architecture.md) for design notes.

## System flow

~~~mermaid
flowchart TD
    WildChat["WildChat published Parquet dataset"] --> WB["BRONZE<br/>Consolidated source snapshot"]
    ONET["O*NET 31.0 crosswalk CSV"] --> OB["BRONZE<br/>Source taxonomy file"]
    WB --> WS["SILVER<br/>Validated, minimized conversations"]
    OB --> OS["SILVER<br/>One row per IWA"]
    WS --> GOLD["GOLD<br/>Conversation, category, and association tables"]
    OS --> GOLD
    GOLD --> PBI["Power BI aggregate reporting"]
    GOLD -. Future .-> AIRAG["Future: approved content → permission-aware AI/RAG → cited answers"]
    Controls["Security across the platform<br/>Least privilege · protected credentials · encryption · audit and monitoring"]
~~~

## Conversation-category model

The reporting question is: **How many distinct conversations involved each category?** A conversation can involve several activities, so the relationship is many-to-many.

~~~mermaid
erDiagram
    CONVERSATION ||--o{ CONVERSATION_CATEGORY : has
    CATEGORY ||--o{ CONVERSATION_CATEGORY : labels

    CONVERSATION {
        string conversation_id PK
        string conversation_hash
        datetime timestamp
        string model
        string language
        int turn_count
        string classification_status
    }

    CATEGORY {
        string category_id PK
        string taxonomy_version PK
        string name
        string definition
        string taxonomy_source
    }

    CONVERSATION_CATEGORY {
        string conversation_id PK, FK
        string category_id PK, FK
        string taxonomy_version PK, FK
        string classification_method
        float confidence
    }
~~~

- **Conversation:** one row per source instance. <code>conversation_id</code> derives from the first turn’s <code>turn_identifier</code> and is unique across this snapshot’s 837,989 conversation rows. <code>conversation_hash</code> can repeat and is not a key. The source timestamp is the last-turn time.
- **Category:** one row per category and taxonomy version. O*NET contributes unique IWAs with parent GWA metadata; <code>OTHER</code> is project-defined.
- **Conversation category:** one row per unique conversation/category assignment, keyed by <code>(conversation_id, category_id, taxonomy_version)</code>. Confidence is a model-reported signal, not a calibrated probability.

Count distinct conversation IDs per category, not bridge rows. A conversation with several categories contributes once to each; counts can overlap and sum to more than the conversation total. This measures category presence per conversation, not request frequency.

## Current pipeline

- **Bronze:** bulk-download WildChat’s published Parquet shards into one local Parquet batch. Preserve the O*NET crosswalk as its source CSV. A same-day WildChat rerun replaces that batch; historical source snapshots are not retained.
- **Silver:** validate and minimize WildChat records into one Parquet file, retaining nested messages for classification. Reduce the O*NET DWA-to-IWA-to-GWA crosswalk to one row per unique IWA.
- **Gold:** write the full conversation table plus category and conversation-category tables for the Silver batch, as Parquet and CSV. The notebook classifies a reproducible random sample of up to 1,000 conversations by default; remaining rows stay pending. Re-running the batch replaces its outputs.
- **Classification:** evaluate 332 IWAs and <code>OTHER</code> in one OpenAI Decisions API request per conversation, using bounded parallelism. Completed results are checkpointed locally in SQLite for compatible resume. Failed or unresolved records are kept separate from <code>OTHER</code>.
- **Reporting:** Power BI uses the three Gold CSVs. Report visuals and limitations are documented in [reports/powerbi](../reports/powerbi/README.md).

The taxonomy is the unique Intermediate Work Activities (IWAs) from the [O*NET 31.0 GWA-to-IWA-to-DWA crosswalk](https://www.onetcenter.org/dictionary/31.0/csv/gwas_to_iwas_to_dwas.html). Its 2,087 rows include DWA-level mappings; Silver removes DWA detail and retains each unique IWA and parent GWA. <code>OTHER</code> represents non-work conversations and is not part of O*NET.
