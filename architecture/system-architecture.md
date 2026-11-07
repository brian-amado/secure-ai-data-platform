# System Architecture

This diagram describes the implemented local pipeline and its future consumers. Azure services and deployment remain open.

~~~mermaid
flowchart TD
    WildChat["WildChat published Parquet dataset"] --> WB["BRONZE<br/>Consolidated source snapshot"]
    ONET["O*NET 31.0 crosswalk CSV"] --> OB["BRONZE<br/>Source taxonomy file"]
    WB --> WS["SILVER<br/>Validate, minimize, derive conversation_id from first turn_identifier"]
    OB --> OS["SILVER<br/>Deduplicate to one row per IWA"]
    WS --> GOLD["GOLD<br/>Conversation, category, and conversation-category tables"]
    OS --> GOLD
    GOLD --> Analytics["Power BI aggregate reporting"]
    GOLD -. Future .-> AIRAG["Future: approved content → permission-aware AI/RAG → cited answers"]
    Controls["Security across the platform<br/>Identity and least privilege · protected credentials · encryption · audit and monitoring"]
~~~

## Design notes

- WildChat Bronze bulk-downloads the published Parquet shards with the Hugging Face Hub client and consolidates them into one local batch. O*NET Bronze preserves its crosswalk CSV. Local data folders are excluded from Git.
- WildChat Silver derives <code>conversation_id</code> from the first turn’s <code>turn_identifier</code>, unique per conversation row in the current 837,989-row snapshot. <code>conversation_hash</code> is a non-unique content fingerprint, not a key.
- The source conversation timestamp is the last-turn time. The Power BI report labels trends by date of last activity.
- Gold keeps conversation, category, and conversation-category tables separate; reports count distinct conversation IDs.
- The notebook defaults to a reproducible random sample of 1,000 conversations with seed 42. The full conversation table can include all Silver records while unclassified rows remain pending.
- Gold classification checkpoints completed results in local SQLite without storing conversation text or API keys. A compatible rerun skips completed conversations.
- Azure hosting and permission-aware, citation-grounded AI/RAG are design directions, not deployed capabilities.
