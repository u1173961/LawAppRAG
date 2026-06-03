# LawAppRAG

Local retrieval-augmented generation prototype for UK criminal law information.

The project ingests public legal and justice-system sources into JSON files, builds a FAISS vector index over those sources, and uses a local LM Studio chat-completions endpoint to answer questions with retrieved citations.

This is a legal information tool, not a legal advice tool. Outputs should be treated as general information and checked against the cited sources.

## Project Layout

```text
.
├── build_index.py       # rebuilds rag_index from sources
├── ingest_sources.py    # fetches URLs from urls.txt into sources
├── rag_chat.py          # interactive RAG chat using LM Studio
├── update_index.py      # appends newly ingested sources to an existing index
├── urls.txt             # seed URLs, optionally prefixed by jurisdiction
├── sources/             # ingested JSON source documents
└── rag_index/           # FAISS index and docs.jsonl metadata
```

## Requirements

Python 3 is required. Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

`rag_chat.py` expects LM Studio or another OpenAI-compatible local server at:

```text
http://localhost:1234/v1
```

The default chat model name is set in `rag_chat.py` as `qwen2.5-7b-instruct-1m`.

## Usage

Ingest sources listed in `urls.txt`:

```bash
python3 ingest_sources.py
```

Rebuild the vector index from scratch:

```bash
python3 build_index.py
```

Add newly ingested source documents to an existing index:

```bash
python3 update_index.py
```

Run the interactive chat:

```bash
python3 rag_chat.py
```

## URL Format

`urls.txt` accepts one URL per line. A line can optionally start with one of these jurisdictions:

```text
england_wales
scotland
northern_ireland
uk_wide
```

Example:

```text
england_wales https://www.gov.uk/arrested-your-rights
uk_wide https://www.legislation.gov.uk/ukpga/1971/38
```

Lines without a jurisdiction default to `uk_wide`.

## Index Notes

The `sources/` directory is intentionally kept in this repository as a sample corpus (as are contents of urls.txt) so the project is usable without starting from an empty ingest run. You can add to it with `ingest_sources.py`, then update or rebuild the index.

`build_index.py` writes:

```text
rag_index/faiss.index
rag_index/docs.jsonl
```

`rag_chat.py` checks that the number of FAISS vectors matches the number of rows in `docs.jsonl`. If it reports an index mismatch, run a full rebuild:

```bash
python3 build_index.py
```

## Safety Boundary

The chat prompt is intentionally framed as public-facing UK criminal legal information. It should:

- avoid legal advice
- ask clarifying questions when jurisdiction matters
- ground claims in retrieved sources
- return citations for source checking
