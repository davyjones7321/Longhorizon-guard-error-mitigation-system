"""Lightweight local semantic and lexical concept index for LongHorizon Guard.

Provides zero-dependency TF-IDF vector matching for task concepts and goal strings
with sub-millisecond retrieval latency and zero external API calls.
"""

import math
import re
from typing import Any, Dict, List, Optional, Set


def _tokenize(text: str) -> List[str]:
    """Tokenize and normalize text into alphanumeric lower-case terms."""
    return [w for w in re.findall(r"\b[a-z0-9_]{2,}\b", text.lower())]


class LocalConceptIndex:
    """Local vector index for semantic concept retrieval using TF-IDF cosine similarity."""

    def __init__(self) -> None:
        self._documents: Dict[str, str] = {}
        self._doc_tokens: Dict[str, List[str]] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._doc_freq: Dict[str, int] = {}
        self._num_docs: int = 0

    def add_concept(
        self,
        concept_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Index a concept or task text with optional metadata."""
        tokens = _tokenize(text)
        if not tokens:
            return

        # If updating an existing doc, remove prior frequency contributions
        if concept_id in self._documents:
            old_unique = set(self._doc_tokens[concept_id])
            for t in old_unique:
                self._doc_freq[t] = max(0, self._doc_freq.get(t, 1) - 1)
        else:
            self._num_docs += 1

        self._documents[concept_id] = text
        self._doc_tokens[concept_id] = tokens
        self._metadata[concept_id] = dict(metadata or {})

        unique_tokens: Set[str] = set(tokens)
        for t in unique_tokens:
            self._doc_freq[t] = self._doc_freq.get(t, 0) + 1

    def search(
        self,
        query_text: str,
        top_k: int = 3,
        min_similarity: float = 0.10,
    ) -> List[Dict[str, Any]]:
        """Search indexed concepts using TF-IDF cosine similarity."""
        query_tokens = _tokenize(query_text)
        if not query_tokens or self._num_docs == 0:
            return []

        # Calculate query term weights
        q_term_counts: Dict[str, int] = {}
        for t in query_tokens:
            q_term_counts[t] = q_term_counts.get(t, 0) + 1

        q_vec: Dict[str, float] = {}
        q_norm_sq = 0.0
        for t, count in q_term_counts.items():
            df = self._doc_freq.get(t, 0)
            idf = math.log((self._num_docs + 1.0) / (df + 1.0)) + 1.0
            weight = count * idf
            q_vec[t] = weight
            q_norm_sq += weight * weight

        q_norm = math.sqrt(q_norm_sq)
        if q_norm == 0.0:
            return []

        results = []
        for doc_id, tokens in self._doc_tokens.items():
            d_counts: Dict[str, int] = {}
            for t in tokens:
                d_counts[t] = d_counts.get(t, 0) + 1

            dot_product = 0.0
            d_norm_sq = 0.0
            for t, count in d_counts.items():
                df = self._doc_freq.get(t, 0)
                idf = math.log((self._num_docs + 1.0) / (df + 1.0)) + 1.0
                d_weight = count * idf
                d_norm_sq += d_weight * d_weight
                if t in q_vec:
                    dot_product += q_vec[t] * d_weight

            d_norm = math.sqrt(d_norm_sq)
            if d_norm == 0.0:
                continue

            similarity = dot_product / (q_norm * d_norm)
            if similarity >= min_similarity:
                results.append({
                    "concept_id": doc_id,
                    "text": self._documents[doc_id],
                    "similarity": round(similarity, 4),
                    "metadata": self._metadata[doc_id],
                })

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize index state to dictionary."""
        return {
            "documents": dict(self._documents),
            "metadata": dict(self._metadata),
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        """Load index from dictionary."""
        self._documents.clear()
        self._doc_tokens.clear()
        self._metadata.clear()
        self._doc_freq.clear()
        self._num_docs = 0

        docs = data.get("documents", {})
        meta = data.get("metadata", {})
        for cid, text in docs.items():
            self.add_concept(cid, text, metadata=meta.get(cid))
