# Trust UQ — Handoff for continuing the calibration effort

*Written 2026-07-08. Audience: a Claude Code instance (or human) picking this up
on another machine (Mac Studio M3 Ultra, 512 GB). Read
`notes/trust_uq_design.md` first for the theory; this file is operational
state: what exists, what's running, what's next, and the decisions already made.*

## 1. What this is

ALICE is replacing LLM self-reported extraction certainty with a defensible,
calibrated trust architecture. Four phases (design doc §6):

1. **v3 measurable signals at ingest** — DONE. P(True) verification, local NLI
   entailment/contradiction, self-consistency across k rerun extractions.
2. **Calibration** — MACHINERY DONE, **the in-domain calibration run is the
   current effort**. LLM-judge + human-audit labels, PPI bias correction,
   logistic + isotonic calibrator.
3. **Conformal admission** — NOT STARTED. Will replace the fixed
   `min_ingest_confidence = 0.6` gate using `calibrator.json` + held-out labels.
4. **Evidential fusion at retrieval** — NOT STARTED (subjective-logic Beta
   opinions replacing the provenance ramp and the `+0.20` entity boost in
   `core/scoring/composite.py`).

Owner's core requirement (do not regress this): no hand-picked weights and no
LLM self-report as a load-bearing signal. Every trust number must be a
certificate, a calibrated probability, or a statistic with a valid interval.
Jim also flagged that public benchmarks (RAGTruth/HotpotQA) are
distribution-mismatched for calibration — the deployed calibrator is fitted
ONLY on in-domain NTRS labels; benchmarks are for development and
external-validity reporting.

## 2. Code map (all on branch `trust-uq`)

| path | what |
|---|---|
| `core/trust/config.py` | `TrustConfig` ← `[trust]` in alice.toml; master `enabled=false` = byte-identical v2 pipeline |
| `core/trust/ptrue.py` | P(True): logprob estimator or Beta-vote fallback `(v+1)/(m+2)` |
| `core/trust/nli.py` | DeBERTa-v3 MNLI singleton (lazy, thread-safe, degrades to NULL without the `trust` extra) |
| `core/trust/consistency.py` | canonical triple keys + count-across-runs (Beta(count+1, runs−count+1)) |
| `core/trust/signals.py` | `compute_evidence_signals` — shared by ingest + backfill |
| `core/trust/judge.py` | calibration judge (openai-compatible or anthropic), Supported/NotSupported/Unclear |
| `core/trust/calibration.py` | feature matrix (mean-impute + missingness indicators), K-fold OOF logistic + isotonic, ECE/Brier/AUROC, `calibrator.json` artifact |
| `core/trust/ppi.py` | prediction-powered inference (judge bias corrected by human audit) + Cohen's kappa |
| `scripts/rescore_trust_signals.py` | backfill P(True)+NLI onto existing edges; resumable (NULL filter); `--model/--backend/--base-url` override the verifier |
| `scripts/mint_calibration_labels.py` | stratified triple sample from any DB → judge labels → `labels.jsonl`; `--audit N` = interactive human labeling |
| `scripts/fit_trust_calibrator.py` | labels.jsonl → `calibrator.json` + `calibration_report.md` (holdout metrics + PPI section) |
| `tests/test_trust_*.py` | 25 tests across signals/persistence/calibration (force-added; `tests/` is gitignored in this repo) |

Extraction wiring: `generate_triples(..., trust_cfg=)` in
`core/llm/generate_triple.py`; worker plumb in `core/llm/worker.py` +
`services/ingest/service.py:extract()`. Persistence: nine additive columns on
`RELATES_TO` (`kuzu_store.py` — migration runs automatically on first
`KuzuStore` open). Backends: `score_yes_no()` / `sample_n()` added to
`core/llm/base.py` (vLLM/Ollama impl in `vllm.py`, MLX `sample_n` in `mlx.py`).

`confidence_version`: `"v2_evidence_hybrid"` = legacy; `"v3_signals"` = signals
computed at ingest. Backfilled rows keep v2 version — non-NULL signal columns
are the discriminator.

## 3. Current state of the calibration run (as of 2026-07-08)

- **Corpus**: ~30 NTRS docs on autonomous systems, **2,894 chunks**, ingested
  via `ingest download ... --db-path <calibration db>` on the MacBook Pro
  (96 GB). The DB is **not in git** (`*.db` ignored) — it lives wherever
  `--db-path` pointed (likely `calibration.db` in the repo root on the MacBook).
- **Extraction is running** on the MacBook: `ingest extract --db-path <db>
  --min-ingest-confidence 0` with gpt-oss:latest via local Ollama.
  Measured ~63 s/chunk (workers=4, ptrue disabled) ≈ 50 h total.
  Extraction is resumable: chunks are stamped `extracted_at`; Ctrl-C between
  chunks is safe; restart continues with `unextracted_only`.
- **Live `alice.toml` on the MacBook** (gitignored — recreate on the Studio):

  ```toml
  [llm]
  backend = "openai-compatible"
  model = "gpt-oss:latest"
  base_url = "http://localhost:11434/v1"
  workers = 4        # raise to 8 on the Studio; also OLLAMA_NUM_PARALLEL=8

  [trust]
  enabled = true
  self_consistency_enabled = true
  self_consistency_k = 3
  judge_model = "llama3.3:70b"
  ptrue_enabled = false     # deliberately OFF at ingest — see decision D3
  ```

  (Other sections — `[chat_llm]`, `[scoring]`, `[ingest]`, `[chat]`,
  `[embeddings]` — as in `alice.toml.example`.)

## 4. Decisions already made (with reasons — don't silently reverse)

- **D1 — Extractor is gpt-oss:latest and must stay so for this corpus.** Prod
  `chat.db` was extracted by gpt-oss; the calibrator is only valid for the
  extraction distribution it's fitted on. Do NOT re-extract parts of the
  calibration corpus with a different model.
- **D2 — Admission gate is 0 for the calibration DB only** (CLI flag
  `--min-ingest-confidence 0`), so sub-0.6 borderline candidates are in the
  labeled pool — Phase 3's admission threshold must be fitted on candidates,
  not only on v2-admitted triples. Never set 0 in the shared `[ingest]` section.
- **D3 — P(True) is decoupled from ingest** (`ptrue_enabled = false`): gpt-oss
  gets no logprobs through Ollama, so ingest-time P(True) costs 5 vote calls ×
  hidden reasoning per triple. Instead, after extraction, backfill P(True)
  uniformly with **qwen2.5:14b** (Ollama DOES return logprobs for it → single
  cheap call/edge, ~1 s):
  `uv run python3 scripts/rescore_trust_signals.py --db <db> --only ptrue --model qwen2.5:14b --no-backup`
  Use the SAME verifier on prod's backfill so P(True) is one instrument
  everywhere (`ptrue_method`/`nli_model` columns record provenance).
- **D4 — Judge is llama3.3:70b** (≠ extractor; no self-grading). Judge bias is
  corrected by the human audit via PPI — judge accuracy does not need to be
  perfect, it needs to be *audited*.
- **D5 — Same model on different hardware is fine** (weights+sampling define
  the distribution, not the silicon). Moving the job MacBook→Studio is safe.

## 5. Remaining workflow (in order)

```bash
# 0. every step below: repo on branch trust-uq, `uv sync --extra trust --extra mlx`
# 1. finish extraction (resumable; see §3 for the running command)
# 2. P(True) backfill with the uniform verifier (D3)          [~1-3 h]
uv run python3 scripts/rescore_trust_signals.py --db <calibration db> --only ptrue --model qwen2.5:14b --no-backup
# 3. judge-label a stratified sample                          [llama3.3:70b, hours]
uv run python3 scripts/mint_calibration_labels.py --db <calibration db> --sample 1000
# 4. human audit ~200-300 rows (Jim does this; resumable)
uv run python3 scripts/mint_calibration_labels.py --audit 250
# 5. fit + read the report (holdout ECE/Brier/AUROC, reliability table, PPI CI)
uv run python3 scripts/fit_trust_calibrator.py
# 6. optionally backfill prod chat.db (non-destructive; service stopped):
uv run python3 scripts/rescore_trust_signals.py --only nli                          # ~3 min
uv run python3 scripts/rescore_trust_signals.py --only ptrue --model qwen2.5:14b    # ~30 min
```

Then Phase 3: conformal risk control choosing the admission threshold on
`calibrator.json` scores, with a guarantee α on admitted-triple error rate,
using a held-out slice of the same labels.

## 6. Running on the Mac Studio

Two options; option A was recommended:

- **A (no migration): point the MacBook at the Studio's Ollama.** On the
  Studio: `OLLAMA_HOST=0.0.0.0 OLLAMA_NUM_PARALLEL=8 ollama serve`, pull
  `gpt-oss:latest`. On the MacBook: stop extract, set
  `base_url = "http://<studio>.local:11434/v1"` and `workers = 8`, restart the
  same extract command. DB and NLI stay on the MacBook.
- **B (full migration): check out branch `trust-uq` on the Studio**, `uv sync
  --extra trust --extra mlx`, recreate `alice.toml` (§3 — it's gitignored),
  pull `gpt-oss:latest` / `qwen2.5:14b` / `llama3.3:70b` in Ollama, **stop
  extraction on the MacBook first**, copy the calibration `.db` file over
  (never copy a Kuzu DB with a live writer), then rerun the same
  `ingest extract` command — it resumes from the chunk stamps.

## 7. Testing & verification

```bash
uv run pytest tests/ -q --deselect tests/test_ingest_offsets.py::test_search_returns_offset_after_last_result_examined
```
47 pass as of handoff. The deselected test is a PRE-EXISTING failure (its
NTRS-API monkeypatch doesn't take effect; appears to hit the live network) —
not related to this work. `tests/` and `notes/` are gitignored repo-wide; the
trust test files and these two docs were force-added so they travel with the
branch.

Signal sanity-checks used during development (all passed):
- NLI catches negation: premise "battery does **not** power controller" →
  contradiction > entailment (`test_nli_scores_negation`).
- Self-consistency discriminates where v2 doesn't: stable fact freq 1.0 vs
  0.33 for unstable ones at identical v2 certainty (e2e run, design doc §2.3).
- Calibration metrics are honest: ECE/Brier reported on a stratified holdout —
  isotonic evaluated in-sample is ~0 ECE by construction (this bug was found
  and fixed; don't reintroduce it).
- PPI recovers a synthetically biased judge: θ inside CI with 60-row audit.

## 8. Known gotchas

- gpt-oss via Ollama returns **no logprobs** and burns hidden reasoning tokens;
  any Yes/No probe needs `max_tokens` headroom (`ptrue_vote_max_tokens = 256`)
  and content may be empty if truncated. qwen2.5:14b DOES get logprobs.
- Kuzu is **single-writer, single-process**: never run two things against the
  same `.db`, never copy it while a writer is open. `KuzuStore` (not raw
  `kuzu.Database`) must be used to open DBs so column migrations run.
- `safe_db` (default in the backfill script) **rolls back on interrupt** —
  correct for prod safety, wrong for long runs you might Ctrl-C; use
  `--no-backup` there (updates are idempotent).
- The chat UI/service is untouched by all of this: retrieval reads the new
  columns but only Phase 4 will use them for scoring.
