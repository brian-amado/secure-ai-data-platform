# Secure AI & Data Platform

A portfolio project for building a secure enterprise data and AI platform on Azure.

The project will grow through documented steps across data ingestion and processing, AI and retrieval, Azure infrastructure, and security controls.

## Problem Statement

Organizations are adopting generative AI, but leaders often lack a clear, privacy-conscious view of the kinds of work it supports. Conversation data is messy, sensitive, and difficult to summarize consistently, while individual messages may contain details that do not belong in routine reporting.

This project uses the public WildChat dataset to demonstrate a secure data pipeline that validates and minimizes conversation records, classifies conversations into work-activity categories, and produces aggregate usage data. The first reports focus on how many conversations involve each category; a conversation may map to multiple categories. The platform is designed to grow toward Azure security controls and a permission-aware, citation-grounded interface for asking questions about those reports. Raw message text is used for classification, but excluded from the reporting tables.


## Repository Map

- `architecture/` — system and data-flow design
- `infrastructure/terraform/` — infrastructure as code
- `etl/bronze/` — source ingestion and immutable landing
- `etl/silver/` — validation, minimization, and standardization
- `etl/gold/` — classification and report-ready tables
- `ai/` — AI and retrieval components
- `security/` — identity and security controls
- `monitoring/` — operational visibility
- `tests/` — automated checks


## Current Status

Local prototype in progress: WildChat Bronze ingestion, Silver Parquet preparation, and Gold conversation-category modeling. Azure deployment and the reporting/RAG experience are future steps. See [CHANGELOG.md](CHANGELOG.md) for progress.

## Local Development

Requires Python 3.13. From the repository root, create and activate the virtual environment:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
```

Install the notebook tools and launch the walkthrough from the repository root:

```bash
python -m pip install -r requirements-dev.txt
python -m notebook notebooks/wildchat_pipeline.ipynb
```
