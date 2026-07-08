# Measurable Trust Signals (v3) — Design

*2026-07-06 · Phase 1 of the calibrated-trust roadmap*

## 1. Motivation

The v2 ingest confidence (`core/llm/ingest_confidence.py`) blends deterministic
evidence-alignment checks with the LLM's *verbalized* certainty float using
hand-picked weights (0.45/0.25/0.20/0.10). Two problems, both already flagged in
`research_trust_scoring.md` §3.1:

1. Verbalized certainty is poorly calibrated — models cluster at 0.8–0.9
   regardless of correctness, so the 0.10 raw-certainty term carries little
   information and the composite number has no probabilistic semantics.
2. The lexical checks are semantically blind. *"X was **not** developed at
   Langley"* passes exact evidence alignment and full entity anchoring for the
   triple `(X, developed_at, Langley)`; only the hedge regex (may/might/…)
   dampens anything, and negation isn't in it.

Phase 1 replaces "the LLM decides" with three **measurable** signals, each with
defined probabilistic semantics, persisted per triple alongside the untouched
v2 score. Later phases turn these measurements into calibrated probabilities
and formal guarantees (§6).

## 2. Signal definitions

### 2.1 P(True) — verifier posterior (`ptrue_score`)

A second LLM pass per unique triple: system prompt fixes a one-word Yes/No
answer; the user message is the evidence sentence plus *"Does the text state or
imply: ⟨subject⟩ ⟨relation⟩ ⟨object⟩?"* (Kadavath et al. 2022: models are far
better calibrated on this discrimination task than on verbalized numbers.)

Two estimators, recorded in `ptrue_method`:

- **`logprob`** — with token logprobs available (vLLM), take the first output
  position whose token is a Yes/No variant and normalize the mass:
  `P(Yes) = Σexp(lp_yes) / (Σexp(lp_yes) + Σexp(lp_no))` over the top-10
  alternatives at that position. `ptrue_samples = 0`.
- **`vote`** — without logprobs (Ollama's OpenAI-compatible endpoint ignores
  `logprobs`; MLX not wired for them), sample the answer m times at
  temperature 1.0 and report the Beta(1,1) posterior mean over the parseable
  votes: `(v + 1) / (m + 2)`. Unparseable samples are dropped from m;
  `ptrue_samples` records the parseable count. The full posterior is
  Beta(v+1, m−v+1). Reasoning models (gpt-oss) burn hidden reasoning tokens
  before the one-word answer reaches `content`, so the vote budget defaults to
  `ptrue_vote_max_tokens = 256`; too small a budget yields empty content and no
  votes (verified against Ollama gpt-oss, 2026-07-06).

Backend capability is probed once per process (`VLLMBackend._logprobs_supported`)
and the fallback is automatic.

### 2.2 NLI entailment (`nli_entailment_score`, `nli_contradiction_score`)

A local MNLI cross-encoder (default `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`,
trained on MNLI+FEVER+ANLI so robust to hard negation) scores
premise = *the sentence covering the aligned evidence span*, hypothesis = the
verbalized triple `"{s} {r} {o}."`. Both the entailment and the contradiction
softmax probabilities are stored — contradiction is the negation/misattribution
detector the v2 lexical checks cannot provide, and Phase 4 uses it as disbelief
evidence. This is an *independent measurement instrument*: a separate model
family, so the extractor is not grading its own homework.

Requires `uv sync --extra trust` (transformers/torch/sentencepiece). If the
extra is missing or the model fails to load, the signal degrades to NULL and
never blocks ingest. The model is a process-wide singleton behind a lock
(`core/trust/nli.py`), warmed up in `init_worker`.

### 2.3 Self-consistency (`self_consistency_freq`, `_count`, `_runs`)

Re-run the whole-chunk extraction k extra times at temperature 0.7 (reruns are
parse-only — no scoring, no dedup) and count in how many of the `runs = k+1`
total runs each triple reappears, matching on the canonicalized key
(lowercase + whitespace-collapsed subject/relation/object, the same
normalization the evidence aligner uses). `freq = count/runs` is a Monte Carlo
estimate of the model's extraction stability for that fact; the exact posterior
is **Beta(count+1, runs−count+1)**, derivable from the two stored integers.
This is the triple-level analogue of semantic-entropy hallucination detection
(Farquhar et al., Nature 2024). Off by default: (k+1)× extraction cost.

## 3. Schema, versioning, A/B

Nine additive `RELATES_TO` columns (NULL on legacy rows):

| column | type | semantics |
|---|---|---|
| `ptrue_score` | DOUBLE | verifier P(triple entailed by evidence) |
| `ptrue_method` | STRING | "logprob" \| "vote" |
| `ptrue_samples` | INT64 | parseable votes (0 for logprob) |
| `nli_entailment_score` | DOUBLE | NLI P(entailment) |
| `nli_contradiction_score` | DOUBLE | NLI P(contradiction) |
| `nli_model` | STRING | model id that produced the NLI scores |
| `self_consistency_freq` | DOUBLE | count/runs |
| `self_consistency_count` | INT64 | runs containing the triple (incl. base) |
| `self_consistency_runs` | INT64 | total runs = k+1 |

- `certainty_score` (v2) and the `min_ingest_confidence` admission gate are
  **unchanged** — v3 signals are observational until Phase 3.
- Rows extracted with `[trust] enabled = true` get
  `confidence_version = "v3_signals"`. **Backfilled** rows keep their original
  version; presence of non-NULL signal columns is the A/B discriminator.
- Backfill (`scripts/rescore_trust_signals.py`) populates P(True)+NLI on
  existing edges from their stored evidence spans. Self-consistency cannot be
  backfilled — it requires re-extraction.

## 4. Configuration (`[trust]` in alice.toml)

| key | default | notes |
|---|---|---|
| `enabled` | `false` | master switch; false = pipeline identical to v2 |
| `ptrue_enabled` | `true` | one ≤8-token LLM call per unique triple |
| `ptrue_fallback_samples` | `5` | m votes when no logprobs |
| `ptrue_temperature` | `1.0` | vote-path sampling temperature |
| `ptrue_vote_max_tokens` | `256` | vote budget; reasoning models need the headroom |
| `nli_enabled` | `true` | needs the `trust` extra; NULL otherwise |
| `nli_model` | DeBERTa-v3-base-mnli-fever-anli | any HF MNLI checkpoint |
| `nli_device` | `auto` | mps → cuda → cpu |
| `nli_max_premise_chars` | `1000` | premise truncation |
| `self_consistency_enabled` | `false` | (k+1)× extraction cost |
| `self_consistency_k` | `3` | extra runs |
| `self_consistency_temperature` | `0.7` | rerun sampling temperature |

## 5. Cost model

LLM calls per chunk = `1 (base extraction) + k (consistency reruns)
+ |unique triples| × (1 logprob call or m vote calls)`. NLI is local and
per-triple cheap (one cross-encoder pass). All signal computation runs inside
the existing extraction worker threads, so `[llm] workers` still bounds total
LLM concurrency. A future optimization: run self-consistency only for triples
whose P(True)/NLI land in an ambiguous middle band.

## 6. Roadmap (Phases 2–4)

- **Phase 2 — Calibration (implemented 2026-07-06).** A calibration map is a
  statement about a distribution, and it does not survive domain shift — so the
  deployed calibrator must be fitted on labels from ALICE's own deployment
  distribution (NTRS documents through ALICE's pipeline), **not** on public
  benchmarks. RAGTruth/HotpotQA keep three supporting roles: cheap development
  data for this machinery, external-validity evidence (signal *ranking* should
  transfer; raw calibration need not), and known-ground-truth stress tests.
  The Phase 3 conformal guarantee is conditional on exchangeability between
  calibration and production data, which benchmark calibration would violate
  outright.

  Implementation:
  - `scripts/mint_calibration_labels.py` — corpus-agnostic: stratified
    (round-robin by document) triple sample from any ALICE DB; each labeled
    Supported/NotSupported/Unclear by an LLM judge (`core/trust/judge.py`,
    `[trust] judge_*` config — a different model than the extractor);
    `--audit N` interactively collects human labels on a uniform random
    subsample. Checkpointed and resumable.
  - **PPI** (`core/trust/ppi.py`, Angelopoulos et al. 2023): the judge labels
    everything cheaply, the human audit corrects its bias —
    `θ̂ = mean(judge_all) − mean(judge − human on audit)` with a valid normal
    CI. The correctness estimate is anchored to human ground truth, not to
    another LLM's opinion; Cohen's kappa reports judge–human agreement.
  - `scripts/fit_trust_calibrator.py` + `core/trust/calibration.py` — K-fold
    out-of-fold logistic regression over the signal vector [ptrue, nli_entail,
    nli_contra, consistency_freq, alignment, anchor, scope, raw_certainty]
    (+ per-group missingness indicators, mean imputation from the training
    set), then isotonic calibration on the OOF scores. ECE / Brier and the
    reliability table are reported on a stratified holdout half of the OOF
    scores (isotonic evaluated in-sample is ~0 ECE by construction and would
    overstate calibration); emits `calibrator.json`
    (version `v4_calibrated`) which `apply_calibrator` evaluates per triple
    and Phase 3 consumes. Then "trust = 0.83" *means* ~83% of such triples are
    correct on held-out in-domain data. (`research_trust_scoring.md` §4.1 made
    concrete.)
  - Suggested scale: 500–2000 judged triples, 200–300 audited. Stratify the
    sample across the real corpus; recalibrate (rolling audit) as the corpus
    drifts. Mondrian/group-conditional strata (relation type, document family)
    are a Phase 3 refinement.
- **Phase 3 — Conformal admission.** Replace the fixed 0.6 gate with conformal
  risk control on the calibrated score: admitted-triple error rate ≤ α with a
  finite-sample, distribution-free guarantee under exchangeability.
- **Phase 4 — Evidential fusion at retrieval.** Subjective-logic Beta opinions
  replace the provenance count ramp and the +0.20 entity boost: each supporting
  edge contributes evidence (r from calibrated support, s from contradiction),
  cumulative fusion across sources, discounting along multi-hop paths;
  (belief, disbelief, uncertainty) surfaced in citations.

## 7. Risks

| risk | mitigation |
|---|---|
| Ollama ignores `logprobs` | one-time probe → cached → vote fallback |
| reasoning-model preamble before Yes/No | scan all token positions / regex whole completion; unparseable votes dropped |
| torch on Apple Silicon | mps auto-detect; optional extra; failure → NULL, never blocks ingest |
| throughput | everything opt-in; per-signal toggles; reruns parse-only |
| MLX sampling API drift | `make_sampler` pinned by mlx-lm ≥ 0.21; base-class fallback degrades to near-identical samples (freq → 1.0) |
| Kuzu migration on live DBs | additive try/except ALTERs (existing pattern); NULL-safe reads |
