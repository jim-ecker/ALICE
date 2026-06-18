# How ALICE Finds Answers: Sheaf Harmonic Interpolation

## The Problem We're Solving

When you ask ALICE a question, it needs to find the right information from a large knowledge base. The challenge is that the answer to a complex question often isn't stored in a single place — it's scattered across many documents, connected by relationships. For example, "What are the failure modes of the propulsion system on the X-57?" requires linking propulsion system properties, historical test data, and failure reports that may live in completely different documents.

Earlier retrieval methods work like a search engine: find documents that contain similar words to the question, then hope the answer is in there. ALICE's TPPR method improved on this by following chains of relationships in the knowledge graph — but it treats all paths as equally trustworthy and struggles with multi-hop reasoning (e.g., A relates to B, which relates to C, therefore A relates to C).

Sheaf Harmonic Interpolation (SHI) is ALICE's next-generation approach to this problem.

---

## The Core Idea: Spreading Information Like Electricity

Imagine the knowledge graph as an electrical circuit. Each **entity** (a person, system, concept, or event) is a node in the circuit. Each **relationship** between entities is a wire connecting two nodes. The **strength of evidence** behind each relationship — how confident the system was when it extracted that fact — determines how well that wire conducts electricity.

When you ask ALICE a question, the algorithm does the following:

1. **Anchors the question** — it identifies the key entities in your question (e.g., "X-57," "propulsion system") and treats them as a power source plugged into the circuit.

2. **Lets the signal flow** — like electricity, the query signal spreads outward through the knowledge graph, flowing more strongly along high-confidence relationships and more weakly along uncertain ones.

3. **Finds where the signal lands** — after the signal has propagated, the parts of the graph that "lit up" most strongly are the ones most relevant to your question. Those are the documents ALICE retrieves.

This is fundamentally different from keyword search or simple path-following. The signal finds the most *consistent* and *trustworthy* path through the entire graph simultaneously, rather than checking one path at a time.

---

## What Makes It Smarter Than Before

### It Respects Relationship Types

Not all connections are the same. "X-57 **is a type of** aircraft" is a different kind of relationship than "X-57 **was tested at** Armstrong Flight Research Center." SHI uses the *type* of each relationship to transform the signal as it travels — so the algorithm understands that following a "is a component of" link should propagate information differently than following a "contradicts" link. Earlier methods treated all relationships as interchangeable.

### It Handles Multi-Hop Questions Naturally

SHI is designed to reason across chains of relationships in a single mathematical operation, rather than hopping from node to node one step at a time. This is why it's particularly strong on compositional questions — questions whose answer requires combining facts from multiple sources.

### It Knows When It Doesn't Know

SHI produces an **abstention score** as a byproduct of its math. If the query signal can't flow smoothly through the graph — because the relevant facts simply aren't there, or the available evidence is inconsistent — the energy required to fit the question to the graph is high. ALICE uses this signal to say "I don't have enough information to answer this confidently" rather than hallucinating an answer. This is the missing-context warning you see in the interface.

### It Is Grounded in Trust Scores

Every relationship in the knowledge graph was extracted from a source document with an associated confidence score. SHI uses those scores as the "conductance" of each wire in the circuit — high-confidence facts conduct the signal strongly, low-confidence facts conduct it weakly. The retrieved documents are therefore ranked not just by relevance but by the trustworthiness of the evidence chain connecting them to your question.


---

## Phase 0 Status

The version currently deployed uses **identity maps** — meaning all relationship types are treated as equivalent conductors, and the algorithm reduces to a mathematically proven generalization of the PageRank method used by Google. This establishes the baseline and proves the plumbing works.

In Phase 1, the algorithm will be given relationship-specific transformation rules (either learned from data or derived from pretrained models), which is when the multi-hop compositional reasoning advantage becomes fully active.
