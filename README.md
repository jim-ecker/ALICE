# ALICE — AI Leveraged Information Capture and Exploration

ALICE ingests research papers from NASA's Technical Reports Server (NTRS), builds a knowledge graph of entities and relationships using an LLM, and lets you query that graph through a natural language chat interface.

**Pipeline:**
```
NTRS → Docling (PDF parsing) → LLM (NER + relation extraction) → KuzuDB knowledge graph → RAG chat
```

---

## Components

### Core (`core/`)
Shared library used by all services. Handles document chunking, LLM triple extraction (subject → relation → object), the KuzuDB graph store, embedding index, and trust scoring. Every retrieved triple carries a composite trust score derived from ingest certainty, embedding relevance, provenance count, and optional LLM grounding.

### Ingestion CLI (`ingest`)
Downloads papers from NTRS by keyword or NASA center, chunks them with Docling, and extracts knowledge graph triples with the LLM. Operates on a standalone graph database — useful for exploratory ingestion separate from the chat service.

### Chat service (`alice chat`)
Builds an embedding index over ingested chunks, retrieves relevant context at query time, and serves a FastAPI REST API with a browser UI. Knowledge Retention mode lets you switch between the general graph and virtual expert databases.

### Virtual Experts (`alice experts`)
Each virtual expert is a NASA researcher with their own isolated knowledge graph built from their NTRS publications. Experts have a configurable personality prompt that shapes their responses. An interactive TUI handles expert creation, ingestion, and management.

---

## Prerequisites

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- **Embeddings:** [Ollama](https://ollama.com) running locally with `nomic-embed-text` pulled (required on both Mac and Linux)

```bash
ollama pull nomic-embed-text
```

---

## Installation

### Mac (Apple Silicon — MLX)

```bash
git clone <repo-url>
cd ALICE_Reworked
uv sync --extra mlx
```

The `[llm]` section in `alice.toml` is pre-configured for MLX. Models are downloaded automatically by `mlx-lm` on first use.

### Linux (Ollama)

```bash
git clone <repo-url>
cd ALICE_Reworked
uv sync
```

Edit `alice.toml` and replace the `[llm]` block with the Ollama configuration:

```toml
[llm]
backend = "openai-compatible"
model = "qwen2.5:14b"
base_url = "http://localhost:11434/v1"
workers = 8
```

Then pull the model:

```bash
ollama pull qwen2.5:14b
```

### Linux (vLLM)

```bash
uv sync --extra vllm
```

Edit `alice.toml`:

```toml
[llm]
backend = "openai-compatible"
model = "your-model-id"
base_url = "http://localhost:8000/v1"
workers = 8
```

---

## Usage

### 1. Ingest documents for chat

Run one or more ingest commands to build the knowledge graph:

```bash
alice chat ingest "autonomous systems" --max-docs 30
alice chat ingest "human machine teaming" --max-docs 20
alice chat ingest "explainable AI" --max-docs 20
```

Each call downloads PDFs from NTRS, chunks them, extracts triples, and updates the embedding index. Calls accumulate — existing data is not overwritten.

You can also scope results to a NASA center:

```bash
alice chat ingest "robotics" --max-docs 20 --location langley
```

### 2. Start the web UI

```bash
alice chat serve
```

Open [http://127.0.0.1:8766](http://127.0.0.1:8766). The UI has two modes:

- **Chat** — query the general knowledge graph
- **Knowledge Retention** — switch to a virtual expert's isolated knowledge graph

### 3. Manage virtual experts

```bash
alice experts
```

This opens an interactive TUI where you can:

- Create a new expert (name, personality prompt, max docs) — triggers NTRS ingestion scoped to that author automatically
- Re-ingest or add aliases to pull additional publications
- Refresh expertise areas (pulled from NTRS subject categories)
- Edit personality, reset database, or delete an expert

### 4. Advanced: standalone ingestion CLI

For building a separate knowledge graph (not used by the chat service):

```bash
# Download and chunk papers
ingest download "space robotics" --max-docs 20 --db-path alice.db

# Extract triples with the LLM
ingest extract --db-path alice.db
```

---

## Configuration

All settings are in `alice.toml`. Key sections:

| Section | Controls |
|---|---|
| `[llm]` | Backend, model, and worker count for triple extraction |
| `[chat_llm]` | Model used for chat responses (can differ from extraction model) |
| `[embeddings]` | Ollama base URL and embedding model |
| `[chat]` | Server host/port, retrieval parameters, context window |
| `[scoring]` | Trust signal weights and optional LLM grounding pass |
