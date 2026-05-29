# UK Public Procurement RAG System

This project implements a system for querying UK public procurement data using natural language. It processes the Find a Tender Service (FTS) dataset from the Open Contracting Data Registry.

## Overview

The system handles the deeply nested OCDS JSON schema by normalizing it into a flat, contract-level dataset. The data is indexed into Elasticsearch using a Hybrid Search approach (BM25 for keyword matching and k-NN vector search for semantic similarity). Dense embeddings are generated locally using the `all-MiniLM-L6-v2` model from `sentence-transformers`.

## Prerequisites

- Python 3.10+
- Docker & Docker Compose

## Setup Instructions

1. **Download Data**: Place the FTS OCDS data (`ocds.jsonl`) into the `data/` directory.
2. **Start Elasticsearch**:
   ```bash
   docker compose up -d
   ```
3. **Install Dependencies**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

## Usage

### 1. Ingestion
To parse the raw JSONL, flatten the contracts, generate vector embeddings, and bulk index them into Elasticsearch, run:
```bash
python src/ingest.py
```
This handles deduplication and maps nested supplier, buyer, and category data to individual contract records.

### 2. Exploration & Data Science
We provide two notebooks to help understand the data:
- `notebooks/raw_eda.ipynb`: A raw JSON parsing and basic exploratory data analysis (EDA) using pure pandas/matplotlib without connecting to the database.
- `notebooks/explore.ipynb`: Connects to Elasticsearch to run aggregations (top buyers, procurement methods) and tests the base Semantic Search queries used for RAG operations.
