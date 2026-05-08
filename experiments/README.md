# ALICE Experiment Workbench

Reproducible query-benchmark harness for comparing retrieval profiles.

## Files

```
experiments/
  run_query_matrix.py       — run a question set across profiles, no ground truth needed
  run_dataset_suite.py      — run QA datasets with EM/F1 scoring
  _core.py                  — shared session setup, query execution, metrics
  configs/
    profiles.toml           — retrieval profile definitions
    questions.json          — question bank for run_query_matrix
    dataset_suite.toml      — dataset configuration for run_dataset_suite
  data/
    local_eval.jsonl        — local ground-truth QA pairs
  results/                  — run outputs (gitignored)
```

## Quick Start

**Query matrix** (no ground truth — compare retrieval quality across profiles):
```bash
uv run python experiments/run_query_matrix.py
```

**Dataset suite** (EM/F1 scoring against ground-truth answers):
```bash
uv run python experiments/run_dataset_suite.py
```

**Run against a specific expert DB:**
```bash
uv run python experiments/run_query_matrix.py \
  --db-path services/experts/data/natalia_alexandrov/natalia_alexandrov.db \
  --embeddings-path services/experts/data/natalia_alexandrov/natalia_alexandrov.embeddings.npz
```

**Filter to specific profiles:**
```bash
uv run python experiments/run_query_matrix.py --profile base --profile high_recall
```

**Filter to specific dataset + profile:**
```bash
uv run python experiments/run_dataset_suite.py --dataset local_eval --profile base
```

## Output Artifacts

Results are written to `experiments/results/<run_type>_<timestamp>/`:

| File | Contents |
|------|----------|
| `manifest.json` | Run metadata: git HEAD, DB path, config paths |
| `summary.json` | Aggregate metrics across all profiles |
| `<profile>.json` | Full per-profile report (query matrix) |
| `<profile>.jsonl` | One JSON line per question (query matrix) |
| `<dataset>__<profile>.json` | Full per-combination report (dataset suite) |
| `<dataset>__<profile>.jsonl` | One JSON line per example (dataset suite) |

## Metrics

### Per-query (all runs)
| Metric | Description |
|--------|-------------|
| `runtime_seconds` | Wall-clock latency for retrieve + prompt + LLM |
| `chunks_retrieved` | Chunks returned by retriever |
| `facts_retrieved` | Trust bundles after rerank/dedup |
| `avg_composite_trust` | Mean composite trust score across facts |
| `avg_ingest_certainty` | Mean LLM extraction confidence |
| `avg_relevance_score` | Mean embedding cosine similarity |
| `citations_count` | Unique Fact_N references in the response |

### Additional (dataset suite only)
| Metric | Description |
|--------|-------------|
| `answer_em` | Exact match against gold answers (normalised) |
| `answer_f1` | Token-level F1 against gold answers |

## Profiles

Profiles are defined in `configs/profiles.toml`. Tunable parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `top_k_chunks` | 15 | ANN index query size |
| `max_context_chunks` | 20 | Max chunks sent to LLM |
| `entity_hop_depth` | 2 | Graph traversal depth |
| `max_hop2_entities` | 20 | Max entities at hop 2 |
| `min_composite_trust` | 0.0 | Post-retrieval trust filter |
| `ingest_certainty_weight` | 0.4 | Scoring weight |
| `relevance_weight` | 0.4 | Scoring weight |
| `provenance_weight` | 0.2 | Scoring weight |

## Adding Questions / Datasets

- **Query matrix questions**: edit `configs/questions.json`
- **Local eval pairs**: add JSONL lines to `data/local_eval.jsonl` (fields: `id`, `question`, `answers`)
- **Hugging Face datasets**: set `enabled = true` in the relevant `[[datasets]]` block in `dataset_suite.toml`, then `uv add datasets`
