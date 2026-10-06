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
- Keep the sample in a local location excluded from Git.
- Before processing, review fields and decide which are needed.

## Data handling notes

The dataset contains public conversation text and metadata that is treated as sensitive, including location, hashed IP, browser headers, timestamps, and moderation labels. 

The first version should focus on validated, minimized data and aggregate reports. Raw conversation text is out of scope for the initial RAG index.


