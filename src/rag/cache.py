import time
import hashlib
import math
from typing import Any, Dict, List, Optional, Tuple, Union

class SemanticRAGCache:
    """
    High-performance in-memory L1 hybrid semantic cache for cleanroom RAG retrieval (Comp 24).
    Implements a 2-stage lookup hierarchy:
      - Stage 1 (Fast-Path Facet Match): Sub-millisecond exact match on discrete physical
        failure facets (wafer_class, die_class, location, shape, orientation, chamber).
      - Stage 2 (Semantic / Vector Fallback): Cosine similarity search over description
        embeddings for candidates pre-filtered by wafer and die class.
    Fully backwards-compatible with standard query strings.
    """
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._hits = 0
        self._misses = 0
        self._stage1_hits = 0
        self._stage2_hits = 0

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Computes cosine similarity between two numeric vectors."""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        return dot / (norm1 * norm2)

    def _hash_key(self, query: str, top_k: int) -> str:
        raw_key = f"{query.strip().lower()}_top{top_k}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def _facet_key(
        self,
        wafer_class: str,
        die_class: str,
        location: str = "",
        shape: str = "",
        orientation: str = "",
        chamber: str = "",
        top_k: int = 3
    ) -> str:
        parts = [
            wafer_class.strip().lower(),
            die_class.strip().lower(),
            location.strip().lower(),
            shape.strip().lower(),
            orientation.strip().lower(),
            chamber.strip().lower(),
            str(top_k)
        ]
        raw_key = ":".join(parts)
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def get(
        self,
        query: Union[str, Dict[str, Any]],
        top_k: int = 3,
        min_similarity: float = 0.88
    ) -> Optional[Any]:
        """
        Retrieves cached FMEA data using either a query string or a structured facet dictionary.
        Executes Stage 1 (exact facet match), falling back to Stage 2 (vector similarity).
        """
        # Case A: String Query (Backwards-compatible)
        if isinstance(query, str):
            key = self._hash_key(query, top_k)
            if key in self._cache:
                entry = self._cache[key]
                if time.time() - entry["timestamp"] < self.ttl_seconds:
                    self._hits += 1
                    self._stage1_hits += 1
                    return entry["data"]
                else:
                    del self._cache[key]
            self._misses += 1
            return None

        # Case B: Structured Facet Query
        wafer_class = query.get("wafer_class", "")
        die_class = query.get("die_class", "")
        pattern = query.get("pattern", {})
        location = pattern.get("location", query.get("location", ""))
        shape = pattern.get("shape", query.get("shape", ""))
        orientation = pattern.get("orientation", query.get("orientation", ""))
        chamber = query.get("chamber", "")
        query_embedding = query.get("descr_embedding", None)

        # Stage 1: Fast-Path Sub-Millisecond Exact Facet Match
        facet_k = self._facet_key(wafer_class, die_class, location, shape, orientation, chamber, top_k)
        if facet_k in self._cache:
            entry = self._cache[facet_k]
            if time.time() - entry["timestamp"] < self.ttl_seconds:
                self._hits += 1
                self._stage1_hits += 1
                return entry["data"]
            else:
                del self._cache[facet_k]

        # Stage 2: Semantic Vector Fallback (Nearest Neighbor over Description Embeddings)
        if query_embedding:
            best_entry = None
            best_score = -1.0
            now = time.time()
            clean_wafer = wafer_class.strip().lower()
            clean_die = die_class.strip().lower()

            for k, entry in list(self._cache.items()):
                if now - entry["timestamp"] >= self.ttl_seconds:
                    del self._cache[k]
                    continue

                # Pre-filter candidates: must match top-level failure classes
                cached_wafer = entry.get("wafer_class", "").strip().lower()
                cached_die = entry.get("die_class", "").strip().lower()
                if cached_wafer and cached_die and (cached_wafer != clean_wafer or cached_die != clean_die):
                    continue

                cached_emb = entry.get("descr_embedding")
                if cached_emb:
                    sim = self._cosine_similarity(query_embedding, cached_emb)
                    if sim > best_score:
                        best_score = sim
                        best_entry = entry

            if best_entry is not None and best_score >= min_similarity:
                self._hits += 1
                self._stage2_hits += 1
                return best_entry["data"]

        self._misses += 1
        return None

    def set(
        self,
        query: Union[str, Dict[str, Any]],
        top_k: int,
        data: Any,
        descr_embedding: Optional[List[float]] = None
    ) -> None:
        """
        Stores retrieved FMEA data into the L1 cache.
        Supports both raw query strings and structured facet payloads.
        """
        if len(self._cache) >= self.max_size:
            # Evict oldest entry (FIFO/LRU-style eviction)
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k]["timestamp"])
            del self._cache[oldest_key]

        now = time.time()
        if isinstance(query, str):
            key = self._hash_key(query, top_k)
            self._cache[key] = {
                "data": data,
                "timestamp": now,
                "type": "string_query",
                "descr_embedding": descr_embedding
            }
            return

        # Structured facet insertion
        wafer_class = query.get("wafer_class", "")
        die_class = query.get("die_class", "")
        pattern = query.get("pattern", {})
        location = pattern.get("location", query.get("location", ""))
        shape = pattern.get("shape", query.get("shape", ""))
        orientation = pattern.get("orientation", query.get("orientation", ""))
        chamber = query.get("chamber", "")
        embedding = descr_embedding or query.get("descr_embedding", None)

        facet_k = self._facet_key(wafer_class, die_class, location, shape, orientation, chamber, top_k)
        self._cache[facet_k] = {
            "data": data,
            "timestamp": now,
            "type": "structured_facet",
            "wafer_class": wafer_class,
            "die_class": die_class,
            "pattern": pattern,
            "location": location,
            "shape": shape,
            "orientation": orientation,
            "chamber": chamber,
            "description_text": pattern.get("description", query.get("description_text", "")),
            "descr_embedding": embedding,
            "top_k": top_k
        }

    def clear(self) -> None:
        self._cache.clear()
        self._hits = 0
        self._misses = 0
        self._stage1_hits = 0
        self._stage2_hits = 0

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return (self._hits / total) if total > 0 else 0.0

    @property
    def metrics(self) -> Dict[str, Any]:
        return {
            "cache_size": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "stage1_facet_hits": self._stage1_hits,
            "stage2_vector_hits": self._stage2_hits,
            "hit_rate": round(self.hit_rate, 4)
        }

