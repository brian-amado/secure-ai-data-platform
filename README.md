# Secure AI & Data Platform

A portfolio project for building a secure enterprise data and AI platform on Azure.

The project will grow through documented steps across data ingestion and processing, AI and retrieval, Azure infrastructure, and security controls.

## Problem Statement

Organizations want to use AI with internal data, but data can be scattered across sources, inconsistently prepared, and subject to access controls. A basic RAG demo may retrieve useful information without showing how data quality, authorization, and auditability are handled.

This project will demonstrate an Azure-centered data platform that ingests and validates data, applies security and governance controls, and develops toward AI answers grounded in authorized sources with citations. Early examples will use public or synthetic data.


## Repository Map

- `architecture/` — system and data-flow design
- `infrastructure/terraform/` — infrastructure as code
- `ingestion/pipelines/` — data ingestion
- `processing/transformations/` — data processing
- `ai/` — AI and retrieval components
- `security/` — identity and security controls
- `monitoring/` — operational visibility
- `tests/` — automated checks


## Current Status

Repository foundation. See [CHANGELOG.md](CHANGELOG.md) for progress.

## Local Development

Requires Python 3.13. From the repository root, create and activate the virtual environment:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
```
