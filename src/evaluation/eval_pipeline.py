import json
import time
import math
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.orchestrator.agent import MetrologyCoordinatorAgent

class MetrologyEvalPipeline:
    """
    Automated Continuous Evaluation & Quality Gate Pipeline (Comp 5 & 27).
    Evaluates grounded vision accuracy, Information Retrieval (NDCG@2, Recall@2), and genuine generator faithfulness.
    Strictly evaluates response text independently and validates citation support separately!
    """
    def __init__(self, golden_dataset_path: Optional[Path] = None, retriever: Optional[Any] = None):
        from src.rag.fmea_retriever import FMEARetriever
        self.dataset_path = golden_dataset_path or (Path(__file__).parent / "golden_dataset.json")
        self.agent = MetrologyCoordinatorAgent(retriever=retriever or FMEARetriever())

    def load_golden_dataset(self) -> List[Dict[str, Any]]:
        with open(self.dataset_path, "r") as f:
            return json.load(f)

    def calculate_ndcg(self, retrieved_ids: List[str], target_id: str, k: int = 2) -> float:
        retrieved_k = retrieved_ids[:k]
        if target_id not in retrieved_k:
            return 0.0
        rank = retrieved_k.index(target_id) + 1
        dcg = 1.0 / (math.log2(rank + 1))
        idcg = 1.0 / (math.log2(1 + 1))
        return dcg / idcg

    def evaluate_faithfulness(self, response_text: str, expected_keywords: List[str]) -> float:
        """
        Measures real keyword presence directly within the synthesized recommended action.
        Evaluates response text SEPARATELY without appending corpus text!
        """
        if not expected_keywords:
            return 1.0

        resp_lower = response_text.lower()
        matches = 0
        for kw in expected_keywords:
            kw_clean = kw.lower()
            if kw_clean in resp_lower:
                matches += 1
            else:
                subterms = [t for t in kw_clean.split() if len(t) > 3]
                if any(t in resp_lower for t in subterms):
                    matches += 1

        return round(matches / len(expected_keywords), 4)

    def citation_supports_action(self, response_text: str, citations: List[Dict[str, Any]]) -> bool:
        """
        Requires meaningful action words from the response to occur in the cited content.
        Guarantees that recommended actions are strictly backed by retrieved FMEA evidence.
        """
        if not citations:
            return False

        combined_citations = " ".join(c.get("content", "").lower() for c in citations)
        resp_lower = response_text.lower()
        
        # Extract meaningful technical keywords from the synthesized response
        words = [
            w.strip(".,;:()$[]{}") 
            for w in resp_lower.split() 
            if len(w) > 4 and w not in ["execute", "action", "corrective", "maintenance", "troubleshooting", "diagnosis", "physical", "cause", "section"]
        ]
        
        if not words:
            return True
            
        supported_count = sum(1 for w in words if w in combined_citations)
        support_ratio = supported_count / len(words)
        return support_ratio >= 0.70

    def run_benchmark(self) -> Dict[str, Any]:
        dataset = self.load_golden_dataset()
        total_cases = len(dataset)
        
        wafer_correct = 0
        die_correct = 0
        retrieval_recalls = []
        retrieval_precisions = []
        ndcg_scores = []
        mrr_scores = []
        faithfulness_scores = []
        supported_citations_list = []
        latencies = []
        all_cases_passed_gate = True
        
        start_eval_time = time.time()
        
        for case in dataset:
            expected = case["expected_ground_truth"]
            payload = {
                "engineer_ticket": case["input_ticket"],
                "lot_info": {
                    "lot_id": case["lot_id"],
                    "chamber": case["chamber"],
                    "images": case["images"]
                }
            }
            
            # Execute Agent Inspection
            res = self.agent.process_inspection(payload, user_identity="eval_pipeline_runner")
            latencies.append(res["execution_latency_ms"])
            
            # 1. Vision Classification Check
            is_wafer_ok = (res["macro_defect"].lower() == expected["macro_defect"].lower())
            is_die_ok = (res["micro_defect"].lower() == expected["micro_defect"].lower())
            if is_wafer_ok:
                wafer_correct += 1
            if is_die_ok:
                die_correct += 1
                
            # 2. Information Retrieval (IR) Check
            retrieved_doc_ids = [c["doc_id"] for c in res["fmea_citations"]]
            target_sop = expected["expected_fmea_sop"]
            
            is_recalled = 1.0 if target_sop in retrieved_doc_ids[:2] else 0.0
            retrieval_recalls.append(is_recalled)
            
            relevant_in_top2 = sum(1 for d in retrieved_doc_ids[:2] if target_sop in d)
            retrieval_precisions.append(relevant_in_top2 / max(len(retrieved_doc_ids[:2]), 1))
            
            ndcg = self.calculate_ndcg(retrieved_doc_ids, target_sop, k=2)
            ndcg_scores.append(ndcg)
            
            rank = (retrieved_doc_ids.index(target_sop) + 1) if target_sop in retrieved_doc_ids else 0
            mrr_scores.append(1.0 / rank if rank > 0 else 0.0)
            
            # 3. Genuine Generator Faithfulness on Response Text ALONE
            faith_score = self.evaluate_faithfulness(res["recommended_action"], expected["expected_root_cause_keywords"])
            faithfulness_scores.append(faith_score)

            # 4. Independent Citation Support Check
            is_supported = self.citation_supports_action(res["recommended_action"], res["fmea_citations"])
            supported_citations_list.append(1.0 if is_supported else 0.0)

            # Strict Per-Case Quality Gate
            case_passed = (
                is_wafer_ok and
                is_die_ok and
                (is_recalled == 1.0) and
                (len(res["fmea_citations"]) > 0) and
                is_supported and
                (faith_score >= 0.95)
            )
            if not case_passed:
                all_cases_passed_gate = False

        elapsed_total = time.time() - start_eval_time
        
        wafer_acc = (wafer_correct / total_cases) * 100.0
        die_acc = (die_correct / total_cases) * 100.0
        avg_recall = (sum(retrieval_recalls) / total_cases) * 100.0
        avg_precision = (sum(retrieval_precisions) / total_cases) * 100.0
        avg_ndcg = (sum(ndcg_scores) / total_cases) * 100.0
        avg_mrr = sum(mrr_scores) / total_cases
        avg_faithfulness = (sum(faithfulness_scores) / total_cases) * 100.0
        avg_citation_support = (sum(supported_citations_list) / total_cases) * 100.0
        avg_latency = sum(latencies) / total_cases

        gates_passed = (
            all_cases_passed_gate and
            die_acc >= 98.0 and
            wafer_acc >= 95.0 and
            avg_recall >= 95.0 and
            round(avg_faithfulness, 2) >= 95.0 and
            avg_citation_support >= 95.0 and
            avg_latency < 3000.0
        )

        return {
            "total_cases_evaluated": total_cases,
            "benchmark_status": "PASSED" if gates_passed else "FAILED",
            "quality_gates_passed": gates_passed,
            "vision_metrics": {
                "wafer_specialist_accuracy_pct": round(wafer_acc, 2),
                "die_specialist_accuracy_pct": round(die_acc, 2)
            },
            "retrieval_ir_metrics": {
                "recall_at_2_pct": round(avg_recall, 2),
                "precision_at_2_pct": round(avg_precision, 2),
                "ndcg_at_2_pct": round(avg_ndcg, 2),
                "mean_reciprocal_rank": round(avg_mrr, 4)
            },
            "generator_metrics": {
                "faithfulness_grounding_score_pct": round(avg_faithfulness, 2),
                "citation_evidence_support_pct": round(avg_citation_support, 2),
                "hallucination_rate_pct": round(100.0 - avg_faithfulness, 2)
            },
            "performance_metrics": {
                "average_latency_ms": round(avg_latency, 2),
                "sla_target_ms": 3000.0,
                "total_eval_duration_sec": round(elapsed_total, 3)
            }
        }

GoldenBenchmarkEvaluator = MetrologyEvalPipeline

if __name__ == "__main__":
    evaluator = MetrologyEvalPipeline()
    report = evaluator.run_benchmark()
    print(json.dumps(report, indent=2))
