import os
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from config.settings import settings
from src.rag.indexer import FMEAIndexer
from src.rag.cache import SemanticRAGCache

logger = logging.getLogger("MS-ADC.VertexVectorSearch")

class VertexVectorSearchClient:
    """
    Enterprise Google Cloud Vertex AI Vector Search Client.
    Generates text embeddings via `text-embedding-004` and queries the deployed Vertex AI Index Endpoint (Comp 24).
    """
    def __init__(
        self,
        project_id: Optional[str] = None,
        project_number: str = "705987267327",
        location: str = "us-central1",
        index_endpoint_id: Optional[str] = None,
        public_endpoint_domain: Optional[str] = None,
        deployed_index_id: str = "fmea_playbooks_deployed",
        corpus_dir: Optional[Path] = None,
        cache: Optional[SemanticRAGCache] = None
    ):
        self.project_id = project_id or os.environ.get("PROJECT_ID", "kshitiz-gemma3")
        self.project_number = project_number
        self.location = location
        self.index_endpoint_id = index_endpoint_id or os.environ.get("VERTEX_INDEX_ENDPOINT_ID", "2893710095143665664")
        self.public_endpoint_domain = public_endpoint_domain or os.environ.get(
            "VERTEX_ENDPOINT_DOMAIN", "1444198386.us-central1-705987267327.vdb.vertexai.goog"
        )
        self.deployed_index_id = deployed_index_id
        self.corpus_dir = corpus_dir or (Path(__file__).parent.parent.parent / "data" / "fmea_corpus")
        self.indexer = FMEAIndexer(self.corpus_dir)
        self.chunks = self.indexer.load_and_chunk_corpus()
        self.chunk_map = {f"{c.doc_id}_{i+1}": c.to_dict() for i, c in enumerate(self.chunks)}
        self.cache = cache or SemanticRAGCache()

    def embed_text(self, text: str) -> List[float]:
        """
        Generates 768-dimensional dense vector embeddings using Google Cloud Vertex AI text-embedding-004.
        """
        try:
            from google import genai
            client = genai.Client(vertexai=True, project=self.project_id, location=self.location)
            resp = client.models.embed_content(
                model="text-embedding-004",
                contents=text
            )
            return resp.embeddings[0].values
        except Exception as e:
            logger.warning(f"Vertex AI text-embedding-004 failed, using deterministic hash vector: {e}")
            import hashlib
            h = hashlib.sha256(text.encode("utf-8")).digest()
            return [(b / 255.0) for b in h[:32]] + [0.0] * (768 - 32)

    def search(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """
        Queries the deployed Vertex AI Vector Search Index for nearest neighbor FMEA chunks.
        Falls back seamlessly to local semantic cache and keyword scoring if the remote endpoint is warming up.
        """
        # 1. Semantic Cache check
        cached_result = self.cache.get(query, top_k)
        if cached_result is not None:
            return cached_result

        # 2. Query live deployed Vertex AI Index Endpoint via public REST API
        query_vector = self.embed_text(query)
        try:
            import google.auth
            from google.auth.transport.requests import Request
            import urllib.request

            credentials, _ = google.auth.default()
            credentials.refresh(Request())

            url = (
                f"https://{self.public_endpoint_domain}/v1/projects/{self.project_number}/"
                f"locations/{self.location}/indexEndpoints/{self.index_endpoint_id}:findNeighbors"
            )
            payload = {
                "deployedIndexId": self.deployed_index_id,
                "queries": [
                    {
                        "datapoint": {
                            "datapointId": "query_vector",
                            "featureVector": query_vector
                        },
                        "neighborCount": top_k
                    }
                ]
            }

            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {credentials.token}",
                    "Content-Type": "application/json"
                }
            )

            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                results = []
                nearest = data.get("nearestNeighbors", [{}])[0].get("neighbors", [])
                for n in nearest:
                    dp_id = n.get("datapoint", {}).get("datapointId", "")
                    distance = float(n.get("distance", 0.0))
                    chunk_dict = dict(self.chunk_map.get(dp_id, {}))
                    if chunk_dict:
                        chunk_dict["similarity_score"] = round(max(0.0, 1.0 - distance), 4)
                        results.append(chunk_dict)

                if results:
                    self.cache.set(query, top_k, results)
                    return results
        except Exception as e:
            logger.error(f"Vertex Vector Search live ScaNN query failed: {e}")
            if os.getenv("ENVIRONMENT") == "production":
                raise RuntimeError(f"Vertex Vector Search Index Endpoint query failed: {e}") from e

        # 3. Offline/Unit-test fallback for non-production environments
        from src.rag.fmea_retriever import FMEARetriever
        fallback_retriever = FMEARetriever(corpus_dir=self.corpus_dir, cache=self.cache)
        results = fallback_retriever.retrieve(query, top_k=top_k)
        return results
