import os
from src.orchestrator.agent import MetrologyCoordinatorAgent

# 1. Provide the environment variable to point to our newly downloaded v4 model weights
os.environ["DIE_VFM_WEIGHTS"] = "models/v4.0.0-dinov2-vitl-518/die_vfm_head.pt"

# 2. Initialize the Multi-Agent Coordinator
agent = MetrologyCoordinatorAgent()

# 3. Define a new inspection request with real image files
payload = {
    "engineer_ticket": "Yield drop detected on Etch Chamber 3 with center shorts after gate-1 etch.",
    "lot_info": {
        "lot_id": "LOT-2026-X99",
        "chamber": "300mm_RIE_Etch_Chamber_3",
        "recipe_id": "RECIPE-ETCH-774",
        "images": [
            "data/pcb_dataset/PCB_DATASET/images/Short/04_short_02.jpg"
        ]
    }
}

# 4. Process the inspection
result = agent.process_inspection(payload, user_identity="cleanroom_process_engineer")

# 5. View results
print("\n" + "="*50)
print("INFERENCE RESULTS:")
print(f"Wafer Defect:      {result['macro_defect']} (conf={result['macro_confidence']})")
print(f"Die Defect:        {result['micro_defect']} (conf={result['micro_confidence']})")
print(f"FMEA Citation:     {result['fmea_citations'][0]['doc_id'] if result.get('fmea_citations') else 'None'}")
print(f"Corrective Action: {result['recommended_action']}")
print(f"Latency:           {result['execution_latency_ms']} ms")
print("="*50 + "\n")
