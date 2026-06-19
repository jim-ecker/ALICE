from __future__ import annotations

from core.scoring.base import ScoredRetrievalResult
from core.graph.chat_store import MessageRecord

_CITATION_RULES = """\
CITATION RULES:
1. Answer from the retrieved context and knowledge graph facts provided. Both the "## Retrieved Context" passages and the "## Knowledge Graph Facts" triples are from the knowledge graph.
2. For every sentence in your answer that uses information from a Fact_N triple, write (Fact_N) immediately after that sentence — not at the end of a paragraph, not at the end of the response, but right after the sentence. Multiple facts for one sentence: (Fact_1, Fact_4).
   Example of correct citation: "The project focuses on trust and certification. (Fact_2) It is funded by NASA. (Fact_5, Fact_8)"
   Example of WRONG citation: "...trust and certification. It is funded by NASA. (Fact_2, Fact_5, Fact_8)"
3. If a claim comes from a "## Retrieved Context" passage, state it without any citation label. Do NOT write "(Retrieved Context #N)" or any similar reference — only Fact_N citations are permitted.\
"""

_SYSTEM_PROMPT = """\
You are ALICE, an AI research assistant grounded exclusively in a knowledge graph built from ingested scientific literature.

""" + _CITATION_RULES


def _persona_framing(strength: float) -> tuple[str, bool]:
    """Return (instruction_phrase, include_query_reminder) for a given strength.

    instruction_phrase is appended after the persona text in the system prompt.
    include_query_reminder controls whether the per-query persona nudge is added.
    """
    if strength <= 0.0:
        return "", False
    elif strength <= 0.30:
        return "Let this subtly inform your tone.", False
    elif strength <= 0.60:
        return "Let this come through in your tone and word choice.", True
    elif strength <= 0.90:
        return "Let this come through strongly in your tone and word choice.", True
    else:
        return "Fully embody this persona in every response.", True


def build_prompt(
    query: str,
    retrieval: ScoredRetrievalResult,
    history: list[MessageRecord],
    *,
    history_turns: int = 10,
    max_context_chunks: int = 20,
    expert_name: str | None = None,
    expert_persona: str | None = None,
    expert_persona_strength: float = 1.0,
) -> tuple[list[dict[str, str]], dict[int, str]]:
    """Build the LLM message list for a chat turn.

    Returns (messages, fact_index_to_chunk_id) where fact_index_to_chunk_id maps
    the 1-based Fact_n index used in the prompt to chunk_id.
    """
    messages: list[dict[str, str]] = []
    index_to_chunk_id: dict[int, str] = {}
    fact_index_to_chunk_id: dict[int, str] = {}

    # 1. System message
    phrase, include_reminder = _persona_framing(expert_persona_strength)
    if expert_name and expert_persona and expert_persona_strength > 0.0:
        system_content = (
            f"You are {expert_name}, a NASA researcher and subject matter expert "
            f"answering questions grounded in a knowledge graph built from your published research.\n"
            f"{expert_persona}\n"
            f"{phrase}\n\n"
            + _CITATION_RULES
        )
    elif expert_persona and expert_persona_strength > 0.0:
        system_content = expert_persona + f"\n{phrase}\n\n" + _CITATION_RULES
    else:
        system_content = _SYSTEM_PROMPT
    messages.append({"role": "system", "content": system_content})

    # 2. Context injection
    if retrieval.chunks:
        context_lines: list[str] = ["## Retrieved Context\n"]
        chunk_index: dict[str, int] = {}
        for i, chunk in enumerate(retrieval.chunks[:max_context_chunks], start=1):
            chunk_index[chunk.chunk_id] = i
            index_to_chunk_id[i] = chunk.chunk_id
            page_str = f"p.{chunk.page_number}" if chunk.page_number is not None else "p.?"
            heading_str = f'§"{chunk.section_heading}"' if chunk.section_heading else ""
            source_parts = [chunk.document_title or "Unknown", page_str]
            if heading_str:
                source_parts.append(heading_str)
            context_lines.append(f"Source: {', '.join(source_parts)}")
            context_lines.append(f"    {chunk.content}\n")

        if retrieval.trust_bundles:
            context_lines.append("## Knowledge Graph Facts\n")
            for enum_idx, bundle in enumerate(retrieval.trust_bundles, start=1):
                t = bundle.triple
                fact_id = t.fact_id if t.fact_id is not None else enum_idx
                fact_index_to_chunk_id[fact_id] = t.chunk_id

                # Build trust signal summary
                signals = [f"composite={bundle.composite_trust:.0%}"]
                signals.append(f"ingest={bundle.ingest_certainty:.0%}")
                if bundle.relevance_score is not None:
                    signals.append(f"rel={bundle.relevance_score:.0%}")
                if bundle.provenance_count > 1:
                    signals.append(f"prov={bundle.provenance_count}x")
                if bundle.grounding_score is not None:
                    signals.append(f"gnd={bundle.grounding_score:.0%}")

                context_lines.append(
                    f"Fact_{fact_id}: {t.subject} ({t.subject_type})"
                    f" --[{t.relation}]--> "
                    f"{t.object_} ({t.object_type})"
                    f"  [{', '.join(signals)}]"
                )

        context_lines.append(
            "\nCITATION FORMAT — follow this exactly:\n"
            "Write (Fact_N) immediately after every sentence that uses a Fact_N triple above.\n"
            "Example: 'The system uses neural networks. (Fact_3) It was developed at Langley. (Fact_7, Fact_9)'\n"
            "Do NOT group citations at the end of a paragraph. Do NOT omit citations for claims from the facts list."
        )
        messages.append({"role": "user", "content": "\n".join(context_lines)})
        if expert_name:
            prefill = "Got it. I'll draw on those facts and cite them inline as (Fact_N) after each sentence that uses one."
        else:
            prefill = "Understood. I will answer from the retrieved context and cite each Fact_N inline immediately after the sentence that uses it."
        messages.append({"role": "assistant", "content": prefill})

    # 3. Recent conversation history (last N turns = 2*N messages)
    recent = history[-(history_turns * 2):] if history else []
    for msg in recent:
        messages.append({"role": msg.role, "content": msg.content})

    # 4. Current query (with persona reminder injected at generation point for expert mode)
    if expert_name and expert_persona and include_reminder:
        query_content = (
            f"[Persona reminder: You are {expert_name}. {expert_persona} "
            f"{phrase}]\n\n{query}"
        )
    else:
        query_content = query
    messages.append({"role": "user", "content": query_content})

    return messages, fact_index_to_chunk_id
