import os
import sys

# Ensure repository root is on Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.orchestrator.agent import MetrologyCoordinatorAgent

print("=" * 60)
print("MS-ADC: INITIALIZING LIVE MULTI-AGENT INFERENCE PIPELINE")
print("=" * 60)

agent = MetrologyCoordinatorAgent()

payload = {
    "lot_id": "LOT-LIVE-882",
    "chamber": "300mm_RIE_Etch_Chamber_3",
    "engineer_ticket": "Center radial excursion observed on 300mm wafer batch. Investigate micro-bridging and RF plasma bias.",
    "images": [
        "data/test_images/LOT-123_wafer_map.bmp",
        "data/test_images/LOT-123_die_micrograph.bmp"
    ]
}

print("\nExecuting live inspection...")
result = agent.process_inspection(payload, user_identity="kkshitiz@google.com")

macro_conf = result.get("macro_confidence", 0.0)
micro_conf = result.get("micro_confidence", 0.0)
tool_trace = result.get("tool_call_trace", [])
die_obs = tool_trace[1].get("observation", {}) if len(tool_trace) > 1 else {}
probs = die_obs.get("all_probabilities", {})

print("\n" + "=" * 60)
print("LIVE INFERENCE RESULTS (ZERO HARDCODING)")
print("=" * 60)
print(f"Inspection ID:            {result.get('inspection_id')}")
print(f"Macro Wafer Defect:       {result.get('macro_defect')} (Confidence: {macro_conf:.4f})")
print(f"Micro Die Defect:         {result.get('micro_defect')} (Confidence: {micro_conf:.4f})")
print("\nNeural Network Softmax Distribution (Aditya's DINOv2 ViT-L/14):")
for defect_class, probability in probs.items():
    print(f"  • {defect_class:<18}: {probability:.4f} ({probability*100:.2f}%)")

print(f"\nPrimary FMEA Citation:    {result.get('primary_fmea_citation')}")
print(f"Circuit Breaker Status:   {result.get('circuit_breaker_status')}")
print(f"Execution Latency:        {result.get('execution_latency_ms')} ms")
print("\nSynthesized Corrective Maintenance Action:")
print(result.get("recommended_action", ""))
print("=" * 60)
