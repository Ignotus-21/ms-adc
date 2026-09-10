import unittest
from pathlib import Path
from src.rag.indexer import FMEAIndexer
from src.rag.fmea_retriever import FMEARetriever
from src.rag.cache import SemanticRAGCache

CORPUS_DIR = Path(__file__).parent.parent.parent / "data" / "fmea_corpus"

class TestFMEARAGKnowledgeEngine(unittest.TestCase):

    def setUp(self):
        self.corpus_dir = CORPUS_DIR
        self.cache = SemanticRAGCache(max_size=50, ttl_seconds=60)
        self.retriever = FMEARetriever(corpus_dir=self.corpus_dir, cache=self.cache)

    def test_fmea_corpus_files_exist(self):
        expected_files = [
            "300mm_rie_plasma_etch_fmea.md",
            "300mm_photolithography_fmea.md",
            "300mm_cmp_planarization_fmea.md"
        ]
        for fname in expected_files:
            path = self.corpus_dir / fname
            self.assertTrue(path.exists(), f"Corpus file missing: {fname}")
            self.assertGreater(path.stat().st_size, 200, f"File {fname} too small")

    def test_indexer_chunks_documents(self):
        indexer = FMEAIndexer(self.corpus_dir)
        chunks = indexer.load_and_chunk_corpus()
        self.assertGreaterEqual(len(chunks), 6, "Should create at least 6 distinct section chunks")
        
        doc_ids = {c.doc_id for c in chunks}
        self.assertIn("FMEA-SOP-ETCH-300-CH3", doc_ids)
        self.assertIn("FMEA-SOP-LITHO-300-SC2", doc_ids)
        self.assertIn("FMEA-SOP-CMP-300-PL1", doc_ids)

    def test_retriever_etch_center_short_query(self):
        query = "Center defect pattern with micro bridging short in 300mm RIE Chamber 3"
        results = self.retriever.retrieve(query, top_k=2)
        
        self.assertGreater(len(results), 0, "Retriever should return matching results")
        top_match = results[0]
        self.assertEqual(top_match["doc_id"], "FMEA-SOP-ETCH-300-CH3")
        self.assertIn("Center", top_match["failure_classes"])
        self.assertGreaterEqual(top_match["similarity_score"], 0.5)

    def test_retriever_litho_scratch_query(self):
        query = "Linear scratch streaks across dies on scanner track"
        results = self.retriever.retrieve(query, top_k=2)
        
        self.assertGreater(len(results), 0)
        top_match = results[0]
        self.assertEqual(top_match["doc_id"], "FMEA-SOP-LITHO-300-SC2")
        self.assertIn("Scratch", top_match["failure_classes"])

    def test_semantic_cache_hit_and_latency_reduction(self):
        query = "Backside Helium leakage and RF match calibration in Chamber 3"
        
        # 1st call: Cache Miss
        self.cache.clear()
        res1 = self.retriever.retrieve(query, top_k=1)
        self.assertEqual(self.cache.metrics["misses"], 1)
        self.assertEqual(self.cache.metrics["hits"], 0)

        # 2nd call: Cache Hit
        res2 = self.retriever.retrieve(query, top_k=1)
        self.assertEqual(self.cache.metrics["hits"], 1)
        self.assertEqual(self.cache.metrics["hit_rate"], 0.5)
        self.assertEqual(res1, res2)

    def test_cache_eviction_and_ttl(self):
        tiny_cache = SemanticRAGCache(max_size=2, ttl_seconds=1)
        tiny_retriever = FMEARetriever(corpus_dir=self.corpus_dir, cache=tiny_cache)
        
        tiny_retriever.retrieve("query 1", top_k=1)
        tiny_retriever.retrieve("query 2", top_k=1)
        self.assertEqual(tiny_cache.metrics["cache_size"], 2)
        
        # Adding 3rd should evict oldest
        tiny_retriever.retrieve("query 3", top_k=1)
        self.assertEqual(tiny_cache.metrics["cache_size"], 2)

    def test_hybrid_l1_stage1_exact_facet_hit(self):
        cache = SemanticRAGCache(max_size=10, ttl_seconds=60)
        query_payload = {
            "wafer_class": "ring_defect",
            "die_class": "bridge_defect",
            "pattern": {
                "location": "edge",
                "shape": "elongated",
                "orientation": "vertical",
                "distribution": "clustered",
                "description": "A very long vertical defect pattern near outer edge"
            },
            "chamber": "300mm_RIE_Etch_Chamber_3"
        }
        mock_playbook = [{"doc_id": "FMEA-SOP-ETCH-300-CH3", "section": "3.2"}]

        # 1st: Miss & Set
        self.assertIsNone(cache.get(query_payload, top_k=2))
        cache.set(query_payload, top_k=2, data=mock_playbook)

        # 2nd: Stage 1 Exact Facet Hit
        cached = cache.get(query_payload, top_k=2)
        self.assertEqual(cached, mock_playbook)
        self.assertEqual(cache.metrics["stage1_facet_hits"], 1)
        self.assertEqual(cache.metrics["stage2_vector_hits"], 0)

    def test_hybrid_l1_stage2_vector_similarity_hit(self):
        cache = SemanticRAGCache(max_size=10, ttl_seconds=60)
        base_vector = [0.12, 0.45, 0.88, 0.05, 0.33]
        # Very close vector (high cosine similarity > 0.99)
        similar_vector = [0.121, 0.449, 0.882, 0.049, 0.331]

        cache_entry = {
            "wafer_class": "ring_defect",
            "die_class": "bridge_defect",
            "pattern": {
                "location": "edge",
                "shape": "elongated",
                "orientation": "vertical",
                "description": "A very long vertical defect pattern near outer edge"
            },
            "descr_embedding": base_vector,
            "chamber": "300mm_RIE_Etch_Chamber_3"
        }
        mock_playbook = [{"doc_id": "FMEA-SOP-ETCH-300-CH3", "fix": "Replace focus ring"}]
        cache.set(cache_entry, top_k=2, data=mock_playbook)

        # Incoming query has slightly different orientation/description, but same classes and high vector similarity
        incoming_query = {
            "wafer_class": "ring_defect",
            "die_class": "bridge_defect",
            "pattern": {
                "location": "edge",
                "shape": "elongated",
                "orientation": "radial_vertical",  # slight variation misses Stage 1
                "description": "Perimeter vertical bridging streak"
            },
            "descr_embedding": similar_vector,
            "chamber": "300mm_RIE_Etch_Chamber_3"
        }

        # Should fall back to Stage 2 vector similarity and hit!
        result = cache.get(incoming_query, top_k=2, min_similarity=0.95)
        self.assertEqual(result, mock_playbook)
        self.assertEqual(cache.metrics["stage2_vector_hits"], 1)

    def test_hybrid_l1_prefilter_rejection(self):
        cache = SemanticRAGCache(max_size=10, ttl_seconds=60)
        vec = [1.0, 0.0, 0.0]
        cache.set({
            "wafer_class": "center",
            "die_class": "short",
            "descr_embedding": vec
        }, top_k=1, data="Etch Playbook")

        # Different wafer class must NOT match even if vector is identical (pre-filter safety)
        miss = cache.get({
            "wafer_class": "scratch",
            "die_class": "short",
            "descr_embedding": vec
        }, top_k=1, min_similarity=0.5)
        self.assertIsNone(miss)

if __name__ == "__main__":
    unittest.main()

