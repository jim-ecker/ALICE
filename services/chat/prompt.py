from __future__ import annotations

from core.scoring.base import ScoredRetrievalResult
from core.graph.chat_store import MessageRecord

_CITATION_RULES = """\
STRICT RULES — you must follow these without exception:
1. Prefer to answer from the retrieved context chunks and knowledge graph facts. You MUST cite every claim drawn from the knowledge graph using the exact Fact_N label provided, e.g. (Fact_1) or (Fact_3, Fact_7). Never omit citations for knowledge-graph-grounded claims, even in long or multi-part answers.
2. If the knowledge graph does not contain enough information to answer the question, you may draw on your general training knowledge to answer — but you MUST begin your response with this exact warning: "⚠️ The knowledge graph does not contain information to answer this question. The following answer is based on general knowledge and should be verified for accuracy." Do not use fact labels for claims sourced from general knowledge.
3. If some parts of a question are answerable from the knowledge graph and others are not, answer the supported parts with fact citations, then answer the unsupported parts from general knowledge with the warning above.
4. Prefer knowledge graph facts with higher composite trust scores — they are more reliably grounded in the source material.\
"""

_SYSTEM_PROMPT = """\
You are ALICE, an AI research assistant grounded exclusively in a knowledge graph built from ingested scientific literature.

""" + _CITATION_RULES


def build_prompt(
    query: str,
    retrieval: ScoredRetrievalResult,
    history: list[MessageRecord],
    *,
    history_turns: int = 10,
    max_context_chunks: int = 20,
    expert_name: str | None = None,
    expert_persona: str | None = None,
) -> tuple[list[dict[str, str]], dict[int, str]]:
    """Build the LLM message list for a chat turn.

    Returns (messages, fact_index_to_chunk_id) where fact_index_to_chunk_id maps
    the 1-based Fact_n index used in the prompt to chunk_id.
    """
    messages: list[dict[str, str]] = []
    index_to_chunk_id: dict[int, str] = {}
    fact_index_to_chunk_id: dict[int, str] = {}

    # 1. System message
    if expert_name and expert_persona:
        system_content = (
            f"You are {expert_name}, a NASA researcher and subject matter expert "
            f"answering questions grounded in a knowledge graph built from your published research.\n"
            f"{expert_persona}\n\n"
            + _CITATION_RULES
        )
    elif expert_persona:
        system_content = expert_persona + "\n\n" + _CITATION_RULES
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
            context_lines.append(f"[{i}] Source: {', '.join(source_parts)}")
            context_lines.append(f"    {chunk.content}\n")

        if retrieval.trust_bundles:
            context_lines.append("## Knowledge Graph Facts\n")
            for fact_idx, bundle in enumerate(retrieval.trust_bundles, start=1):
                t = bundle.triple
                fact_index_to_chunk_id[fact_idx] = t.chunk_id

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
                    f"Fact_{fact_idx}: {t.subject} ({t.subject_type})"
                    f" --[{t.relation}]--> "
                    f"{t.object_} ({t.object_type})"
                    f"  [{', '.join(signals)}]"
                )

        context_lines.append(
            "\nIMPORTANT: When answering, cite every claim that comes from the above facts "
            "using its exact label, e.g. (Fact_1). Only use Fact_N labels that appear above."
        )
        messages.append({"role": "user", "content": "\n".join(context_lines)})
        if expert_name:
            prefill = f"Got it. I'll draw on those facts and cite them with Fact_N labels as I go."
        else:
            prefill = "Understood. I have reviewed the retrieved context and knowledge graph facts. I will cite every knowledge-graph-grounded claim using the exact Fact_N labels provided above (e.g., Fact_1, Fact_2). I'm ready to answer."
        messages.append({"role": "assistant", "content": prefill})

    # 3. Recent conversation history (last N turns = 2*N messages)
    recent = history[-(history_turns * 2):] if history else []
    for msg in recent:
        messages.append({"role": msg.role, "content": msg.content})

    # 4. Current query (with persona reminder injected at generation point for expert mode)
    if expert_name and expert_persona:
        query_content = (
            f"[Persona reminder: You are {expert_name}. {expert_persona} "
            f"Let this come through in your tone and word choice.]\n\n{query}"
        )
    else:
        query_content = query
    messages.append({"role": "user", "content": query_content})

    return messages, fact_index_to_chunk_id
