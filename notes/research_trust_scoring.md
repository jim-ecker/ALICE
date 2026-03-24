# ALICE Trust-Grounded Retrieval: System Design & Research Agenda

**Date:** 2026-03-19
**Status:** Initial implementation complete, experimental

---

## 1. Background and Motivation

ALICE is a document ingestion pipeline that builds a knowledge graph from scientific literature. Documents are chunked, chunks are embedded for semantic search, and an LLM extracts named entity triples (subject → relation → object) from each chunk. The triples are stored in a Kuzu graph database with a `certainty_score` field representing the LLM's self-reported extraction confidence.

The chat service allows users to query this graph in natural language. The retrieval pipeline is:

```
query → embed → top-k chunk lookup → triples from those chunks → LLM answer
```

Two observations motivated the trust scoring work:

1. **The `certainty_score` is uncalibrated.** It is a single-sample, self-reported confidence from the extracting LLM. There is no guarantee it correlates with actual factual accuracy. LLMs tend toward overconfidence and the score conflates two distinct things: extraction confidence ("did I parse this correctly?") and grounding quality ("is this claim actually in the source text?").

2. **Retrieval operates at the chunk level, not the triple level.** A chunk scores high because it is semantically similar to the query. Its triples inherit that score by proximity, not by their own relevance. A chunk about a robotics workshop may be top-k for a propulsion query, and its triples—about workshop logistics—contaminate the context with noise the LLM cannot always suppress.

The core research question is: **how do ingest-time extraction uncertainty and retrieval-time triple relevance interact, and can their combination produce more trustworthy, evidence-grounded responses?**

---

## 2. What Was Built

### 2.1 `core/scoring/` — Trust Signal Layer

A new package sits between retrieval and prompt construction. It is decoupled from both: `retrieval.py` returns raw graph data; `scoring/` applies trust signals; the prompt builder receives the scored result.

#### `TrustBundle` (the core dataclass)

```python
@dataclass
class TrustBundle:
    triple: RetrievedTriple
    ingest_certainty: float        # from RELATES_TO.certainty_score
    relevance_score: float | None  # cosine(query_vec, embed("s rel o"))
    grounding_score: float | None  # LLM: does source text entail this triple?
    provenance_count: int          # distinct chunks that yielded same (s, r, o)
    composite_trust: float         # weighted combination
```

Each field answers a distinct epistemic question:

| Signal | Question | Source |
|---|---|---|
| `ingest_certainty` | How confident was the extractor? | LLM self-report at ingest |
| `relevance_score` | Is this triple pertinent to the query? | Embedding cosine at query time |
| `grounding_score` | Does the source text actually support this? | LLM entailment pass (opt-in) |
| `provenance_count` | How many independent sources corroborate it? | Graph edge count |
| `composite_trust` | Weighted combination of active signals | `WeightedCompositeScorer` |

#### `EmbeddingRelevanceScorer`

Embeds each retrieved triple as the string `"{subject} {relation} {object_}"` using the same embedding model as the chunk index, then computes cosine similarity against the query vector. This is fast (one batched embedding call) and leverages the same semantic space as chunk retrieval, so the scores are directly comparable.

Key property: this score is **query-conditional**. The same triple will score differently for different queries, which is exactly what we want.

#### `ProvenanceScorer`

Queries Kuzu for the count of `RELATES_TO` edges with identical `(subject, relation, object_)` across all chunks. Because `write_triple` uses `CREATE` (not `MERGE`), each independent extraction from a different chunk creates a new edge. A triple appearing in 5 chunks independently is far more credible than one that appeared once.

Provenance is normalised to `[0, 1]` as `min(count - 1, 4) / 4` — baseline is 1 source (score 0), maximum is 5+ sources (score 1.0).

#### `GroundingScorer`

The most expensive signal. For each triple, it prompts the LLM:

> "On a scale of 0.0–1.0, how well does the following source text support the claim: '{subject} {relation} {object_}'?"

This is an NLI (natural language inference) pass that checks whether the extraction was grounded. It is opt-in (`grounding_enabled = true` in `alice.toml`) because it costs one LLM call per triple per query, which adds significant latency.

#### `WeightedCompositeScorer`

Combines active signals into a single composite:

```
composite = Σ(weight_i × signal_i) / Σ(weight_i)
```

Weights are configured in `alice.toml [scoring]`. Signals not yet computed (e.g. grounding when disabled) are excluded from both numerator and denominator, so the composite is always a proper weighted average of what is actually measured.

An optional `relevance_filter_top_k` setting discards triples below the top-k by composite score before they reach the prompt. This controls context noise at the cost of potentially dropping low-relevance-but-true facts.

#### `ScoredRetrievalResult.rerank()`

After scoring, the result is passed through `rerank()` before reaching the prompt builder. This does two things:

1. **Triple deduplication** — triples with identical `(subject, relation, object)` are collapsed to the single highest-scoring bundle. Without this, the same fact extracted from multiple chunks appears multiple times in the prompt, inflating its apparent weight.

2. **Chunk reranking** — chunks are re-sorted by the best composite trust score among their triples (chunks with no surviving triples score 0.0). This means the prompt builder's `max_context_chunks` limit preferentially keeps chunks whose triples are most trustworthy, not just most semantically similar to the query.

`rerank()` is called automatically by the `Retriever` on every query — it is not optional.

### 2.2 Pipeline Integration

The retrieval pipeline now returns `ScoredRetrievalResult` instead of `RetrievalResult`:

```
query → embed → top-k chunk lookup → raw triples → TrustBundles → prompt → LLM
```

The `Retriever` accepts a `TripleScorer` at construction, making it easy to swap scoring strategies without touching retrieval logic. The `WeightedCompositeScorer` is the default; a `NullScorer` could be written that passes through raw triples for ablation.

### 2.3 Prompt Integration

The system prompt now instructs the LLM to prefer high-trust triples. The context injection includes the composite trust score and all active signals for each triple:

```
- NASA Langley Research Center (ORGANIZATION) --[is located in]--> Hampton (LOCATION)
  [composite=91%, ingest=100%, rel=87%, prov=3×]  [chunk 3]
```

This gives the LLM explicit signal about which facts to weight more heavily, rather than presenting all triples as equally credible.

### 2.4 API and UI

Every triple in the API response now carries the full `TrustBundle` breakdown:

```json
{
  "subject": "NASA Langley Research Center",
  "relation": "is located in",
  "object_": "Hampton",
  "ingest_certainty": 1.0,
  "relevance_score": 0.87,
  "grounding_score": null,
  "provenance_count": 3,
  "composite_trust": 0.91
}
```

The chat UI renders per-signal coloured bars on each triple card, making the breakdown immediately visible. Divergence between `ingest_certainty` and `relevance_score` is the most experimentally interesting case to watch.

---

## 3. Current Limitations

### 2.5 `ingest_certainty` Dampening Transform

Because `ingest_certainty` is systematically overconfident (see §3.1), a configurable dampening transform is applied to the raw value before it enters the weighted composite. The transform is:

```
effective = min(raw, cap) ^ exponent
```

Two parameters, both in `alice.toml [scoring]`:

| Parameter | Default | Effect |
|---|---|---|
| `ingest_certainty_cap` | `1.0` | Clips the raw score. Set to `0.9` to prevent the model from ever claiming ≥90% certainty. |
| `ingest_certainty_exponent` | `1.0` | Power applied after capping. Exponent > 1 pulls high scores down faster (e.g. `0.9^2 = 0.81`). |

Defaults of `(1.0, 1.0)` are identity — no dampening unless opted in. The `TrustBundle.ingest_certainty` field and the API/UI display the **post-transform** value so the composite derivation is fully transparent.

Recommended starting point: `cap = 0.9, exponent = 2.0`. This maps a raw 1.0 → 0.81, 0.8 → 0.64, 0.5 → 0.25 — meaningfully separating high-confidence from moderate-confidence extractions that the LLM previously scored near-identically. This is a lightweight proxy for Platt/temperature scaling that requires no labelled calibration data.

---

### 3.1 `ingest_certainty` is self-reported and uncalibrated

The score comes from the same LLM call that extracted the triple. The LLM has no external reference to check against. There is no reason to expect this score to be calibrated — i.e., that 80% confident triples are correct 80% of the time. It likely over-reports confidence near 1.0.

### 3.2 Triple embedding is a crude proxy for semantics

Embedding `"subject relation object_"` captures surface-level semantic similarity, but relation strings are often highly variable (`"is located in"`, `"located in"`, `"is in"`, `"resides in"` are all present for the same fact in the current graph). This means relevance scores for logically identical triples can vary significantly depending on how the extractor phrased the relation.

### 3.3 Provenance undercounts due to entity fragmentation

Entity names are stored as extracted, without normalisation. `"NASA Langley Research Center"`, `"NASA Langley"`, `"LaRC"`, and `"Langley Research Center"` are four distinct entities in the graph, each with their own provenance counts. True corroboration is higher than measured.

### 3.4 Grounding is a closed-loop LLM check

The grounding scorer asks the same class of model that performed extraction whether the extraction is correct. This is better than nothing but is not independent verification. A model that consistently misreads a particular type of claim will score its own incorrect extractions highly.

### 3.5 Composite trust treats signals as independent

The weighted sum assumes signals are uncorrelated, which is not true. `ingest_certainty` and `grounding_score` are both LLM-sourced and will be correlated (confident extractions tend to be better-grounded). A proper calibration study would measure these correlations.

---

## 4. Research Agenda

### 4.1 Near-term: empirical characterisation

**Calibration study**
Select a sample of triples with known ground truth (manually annotated or from a benchmark). Plot reliability diagrams for `ingest_certainty` against actual accuracy. If the curve is far from the diagonal, the score needs recalibration (e.g. Platt scaling or temperature scaling).

**Signal correlation analysis**
For a fixed query set, collect `(ingest_certainty, relevance_score, provenance_count, grounding_score)` for all retrieved triples. Compute pairwise correlations. High correlation between ingest and grounding would suggest the expensive grounding pass adds little new information. High correlation between relevance and composite_trust would suggest relevance dominates.

**Divergence cases**
Identify triples where `ingest_certainty` is high but `relevance_score` is low, and vice versa. These are the most informative cases. High-certainty, low-relevance triples may represent facts the extractor was confident about but that are off-topic. Low-certainty, high-relevance triples may represent noisy extractions about the right subject — worth investigating whether they are useful despite low confidence.

### 4.2 Medium-term: signal improvements

**Multi-sample consistency for `ingest_certainty`**
Run triple extraction N times (with temperature > 0) and measure agreement across samples. A triple that appears in K/N samples is empirically more reliable than self-reported confidence. This requires storing multiple extractions per chunk — a schema addition (`extraction_run_id` on the edge) and a new ingest mode that samples repeatedly.

**Entity normalisation for provenance**
Before counting provenance, resolve entity variants to canonical forms (e.g. using embedding similarity or a dedicated NER coreference model). `"NASA Langley"` and `"NASA Langley Research Center"` should contribute to the same provenance count.

**Relation normalisation**
Cluster relation strings by embedding similarity. Triples with the same (subject, relation-cluster, object) but different surface-form relations should be treated as corroborating each other.

**Cross-document grounding**
Instead of checking whether the source chunk entails the triple, check whether any chunk in the graph entails it. This decouples grounding from the single extraction event and makes it a graph-wide signal.

### 4.3 Longer-term: trust-grounded responses

**Claim-triple alignment**
After the LLM generates an answer, run a verification pass that asks: "Which triple in the context supports each sentence of this answer?" Sentences with no supporting triple are flagged as potentially ungrounded. This turns trust scoring from a retrieval filter into a post-generation audit.

**Evidence chains**
Instead of flat triple lists, traverse the graph from retrieved entities outward (1–2 hops) to find connecting paths. A multi-hop chain `A → B → C` where each edge has a trust score produces a path-level trust: `trust(A→B→C) = f(trust(A→B), trust(B→C))`. Natural choices for `f` include the minimum (weakest-link) or the geometric mean.

**Response trust score**
Aggregate triple-level trust across all citations in a response to produce a single response-level trust estimate. This could be displayed in the UI (e.g. a confidence indicator on the assistant message) and logged per conversation to track how trust varies with query type.

**Comparative scoring**
Run the same query with different weight configurations and compare the LLM answers. A Jupyter notebook that queries the API with multiple scoring configs and presents the responses side-by-side would make this experimentally tractable without restarting the server.

### 4.4 Infrastructure for iteration

The current system is structured to support rapid experimentation:

- **Config-driven weights**: Change `alice.toml [scoring]` weights and restart — no code changes.
- **Pluggable scorers**: `Retriever` accepts any `TripleScorer`. Write a new scorer implementing the ABC and pass it in `server.py`.
- **Full signal exposure in API**: All signals are in the JSON response, so a Jupyter notebook can pull them without modifying server code.
- **Grounding on/off**: Toggle `grounding_enabled` to compare response quality with and without the entailment pass.

A natural next step for the notebook layer is a `TrustAnalyser` class that wraps the API, fires a batch of queries, collects `TrustBundle` data into a DataFrame, and produces calibration curves and divergence plots.

---

## 5. Open Questions

1. **Does higher composite trust correlate with user-perceived answer quality?** This requires a human evaluation loop.
2. **At what relevance threshold does injecting a triple hurt more than help?** The `relevance_filter_top_k` parameter controls this, but its optimal value is unknown.
3. **Is the grounding pass worth its latency cost for this domain?** Scientific literature tends toward precise claims, so extraction quality may already be high enough that grounding adds noise rather than signal.
4. **How does provenance interact with document diversity?** A triple corroborated by 5 sections of the same document is very different from one corroborated by 5 independent papers. The current scorer cannot distinguish these.
5. **Can the trust signals be used to guide further ingestion?** Low-provenance, high-relevance triples in a query response indicate a knowledge gap that targeted ingestion could fill.
