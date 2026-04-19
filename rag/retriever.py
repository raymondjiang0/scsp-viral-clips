from typing import Optional

from sentence_transformers import SentenceTransformer

from rag.indexer import _EMBED_MODEL, get_collection

_embedder = None  # type: Optional[SentenceTransformer]


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(_EMBED_MODEL)
    return _embedder


def retrieve(query: str, n_results: int = 8) -> str:
    col = get_collection()
    if col.count() == 0:
        return _fallback_context()

    embedder = _get_embedder()
    embedding = embedder.encode([query])[0].tolist()

    results = col.query(query_embeddings=[embedding], n_results=min(n_results, col.count()))

    docs = results.get("documents", [[]])[0]
    sources = [m.get("source", "") for m in results.get("metadatas", [[]])[0]]

    sections = []
    for doc, src in zip(docs, sources):
        label = "Brand Guide" if "brand_guide" in src else "SCSP Website" if "scsp.ai" in src else "Feedback"
        sections.append(f"[{label}]\n{doc.strip()}")

    return "\n\n---\n\n".join(sections)


def get_brand_context() -> str:
    """Retrieve broad brand context for the analysis prompt."""
    query = "SCSP brand mission audience tone national security AI competition"
    return retrieve(query, n_results=10)


def _fallback_context() -> str:
    return """SCSP (Special Competitive Studies Project) is a nonpartisan think tank focused on ensuring the United States maintains competitive advantage in technology — especially artificial intelligence — against China and other strategic rivals. SCSP works at the intersection of national security, technology policy, and government-industry collaboration. Its audience spans senior government officials, defense leaders, tech executives, and policy professionals. Content should be credible, urgent, and direct — communicating the stakes of the technology competition without jargon."""
