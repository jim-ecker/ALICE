"""Generate study/config/response_bank.json for the ALICE human evaluation study.

Queries each registered agent (virtual experts + generic LLM baseline) against
all 9 canonical questions and writes pre-generated responses for static-bank mode.

Prerequisites:
  - Ollama running with the model from [chat_llm] in alice.toml
  - All virtual expert DBs ingested via `alice experts`

Usage:
  uv run python scripts/generate_response_bank.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from core.embeddings.client import EmbeddingsClient
from core.llm.factory import create_backend
from services.chat.config import load_chat_config
from services.experiment.live import query_commercial_llm, query_expert, query_generic_llm
from services.experiment.session import load_agents, load_questions

OUTPUT_PATH = _REPO_ROOT / "study" / "config" / "response_bank.json"


def main() -> None:
    print("Loading config from alice.toml…")
    chat_cfg, embed_cfg, scoring_cfg, llm_cfg = load_chat_config(_REPO_ROOT)

    if llm_cfg is None:
        print("ERROR: No [chat_llm] section found in alice.toml.", file=sys.stderr)
        sys.exit(1)

    print(f"  LLM backend: {llm_cfg.backend} / model: {llm_cfg.model or '(auto)'}")
    print(f"  Embed model: {embed_cfg.model}")

    embed_client = EmbeddingsClient(embed_cfg)
    llm = create_backend(llm_cfg)

    questions = load_questions()
    agents = load_agents()

    if not questions:
        print("ERROR: No questions found in study/config/questions.json", file=sys.stderr)
        sys.exit(1)
    if not agents:
        print("ERROR: No agents found in study/config/agents.json", file=sys.stderr)
        sys.exit(1)

    virtual_experts = [a for a in agents if a.get("class") == "virtual_expert"]
    commercial = [a for a in agents if a.get("class") == "commercial_llm"]
    generics = [a for a in agents if a.get("class") == "generic_llm"]
    print(f"\n{len(questions)} questions × {len(agents)} agents = {len(questions) * len(agents)} responses to generate")
    print(f"  Virtual experts: {[a['label'] for a in virtual_experts]}")
    print(f"  Commercial LLMs: {[a['label'] for a in commercial]}")
    print(f"  Generic agents:  {[a['label'] for a in generics]}\n")

    # Warn about missing API keys before starting
    import os
    for a in commercial:
        env = a.get("api_key_env", "")
        if env and not os.environ.get(env):
            print(f"  WARNING: ${env} not set — {a['label']} responses will fail")

    responses: list[dict] = []
    errors: list[str] = []

    for agent in agents:
        agent_id = agent["agent_id"]
        label = agent["label"]
        is_expert = agent.get("class") == "virtual_expert"

        for q in questions:
            q_id = q["q_id"]
            question_text = q["canonical"]
            tag = f"{q_id} | {agent_id}"
            print(f"  {tag} … ", end="", flush=True)

            try:
                if is_expert:
                    text = query_expert(
                        expert_slug=agent["expert_slug"],
                        question=question_text,
                        cfg=chat_cfg,
                        embed_client=embed_client,
                        llm=llm,
                        scoring_cfg=scoring_cfg,
                    )
                elif agent.get("class") == "commercial_llm":
                    text = query_commercial_llm(agent, question_text, chat_cfg.max_tokens)
                else:
                    text = query_generic_llm(question_text, llm, chat_cfg.max_tokens)

                if not text or not text.strip():
                    errors.append(f"{tag}: empty response")
                    print("EMPTY")
                elif text.startswith("[Error"):
                    errors.append(f"{tag}: {text[:80]}")
                    print("ERROR")
                else:
                    print("done")

            except Exception as exc:
                text = f"[Error: {exc}]"
                errors.append(f"{tag}: {exc}")
                print(f"EXCEPTION: {exc}")

            responses.append({
                "q_id": q_id,
                "agent_id": agent_id,
                "response_text": text,
            })

    bank = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": llm_cfg.model or "unknown",
        "responses": responses,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(bank, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'─' * 60}")
    print(f"Wrote {len(responses)} responses to {OUTPUT_PATH.relative_to(_REPO_ROOT)}")
    if errors:
        print(f"\nWARNING: {len(errors)} problem(s):")
        for e in errors:
            print(f"  ✗ {e}")
        print("\nReview and re-run or manually fill in empty entries before starting the study.")
    else:
        print("All responses generated successfully. Review before sending to participants.")


if __name__ == "__main__":
    main()
