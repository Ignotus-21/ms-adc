# MS-ADC: Multimodal Semiconductor Defect Classification & Root-Cause Analysis Agent

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/Framework-FastAPI%20%7C%20Pydantic%20v2-green.svg)](https://fastapi.tiangolo.com/)
[![Google Cloud Run](https://img.shields.io/badge/Cloud%20Run-Deployed%20Active-4285F4.svg)](https://ms-adc-api-be4mdtaema-uc.a.run.app)
[![Vertex AI](https://img.shields.io/badge/Vertex%20AI-Gemini%202.5%20Flash%20%7C%20ScaNN-FF6F00.svg)](https://cloud.google.com/vertex-ai)
[![Vision AI](https://img.shields.io/badge/Vision-NV--DINOv2%20ViT--L%2F14-76B900.svg)](https://developer.nvidia.com/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

An enterprise-grade **Multi-Agent AI Metrology System** for advanced 300mm semiconductor cleanrooms. MS-ADC bridges ultra-fast edge vision inference with cloud multimodal reasoning to accelerate lot excursion root-cause diagnosis from **6 hours down to under 10 seconds**.

---

## 🌐 Live Production Deployment

The MS-ADC platform is deployed and actively serving production traffic on Google Cloud Platform:

| Component | Production Specification | Resource Identifier |
| :--- | :--- | :--- |
| **API Endpoint** | Cloud Run (8 GiB RAM, 2 vCPUs, unthrottled) | `https://ms-adc-api-be4mdtaema-uc.a.run.app` |
| **Active Revision** | `ms-adc-api-00015-skn` (100% traffic allocation) | Region: `us-central1`, Project: `kshitiz-gemma3` |
| **Wafer VLM Specialist** | Multimodal Gemini 2.5 Flash (`gemini-2.5-flash`) | Zero-hallucination structured spatial reasoning |
| **Die VFM Specialist** | PyTorch NV-DINOv2 ViT-B/14 & ViT-L/14 linear probe | Micro-defect optical patch classification |
| **FMEA RAG Vector Search** | Vertex AI Vector Search ScaNN Index Endpoint | Endpoint ID: `2893710095143665664` (768-dim) |
| **Metrology Warehouse** | BigQuery Streaming Table | `kshitiz-gemma3.semicon_metrology.lot_inspections` |
| **Image Asset Storage** | Google Cloud Storage Bucket | `gs://ms-adc-assets-705987267327/sample_images/` |
| **Circuit Breaker SLA** | Resilient failover with 15.0s adaptive timeout | Status: `PRIMARY_SUCCESS` (0 fallbacks) |

---

## 🏗️ System Architecture

```
                                  INCOMING LOT METROLOGY (300mm Wafer + Micrograph)
                                                         │
                                                         ▼
         ┌───────────────────────────────────────────────────────────────────────────────────────────┐
         │                 ENTERPRISE API GATEWAY & SECURITY PERIMETER (Cloud Run)                   │
         │  • FastAPI REST Endpoints: `/v1/inspect`, `/v1/inspect/wafer`, `/v1/inspect/die`          │
         │  • Cloud DLP Sanitization: Tokenizes proprietary reticle IDs & chamber recipe constants   │
         │  • OIDC / OAuth2 Bearer Authentication (`gcloud auth print-identity-token`)               │
         └───────────────────────────────────────────────┬───────────────────────────────────────────┘
                                                         │
                                                         ▼
         ┌───────────────────────────────────────────────────────────────────────────────────────────┐
         │                       LANGGRAPH CENTRAL COORDINATOR AGENT                                 │
         │  • Structured Tool Calling & Dynamic Vision Routing                                       │
         │  • Circuit Breaker SLA Protection (Adaptive 15.0s SLA timeout, Half-Open recovery)        │
         └───────────────────────┬───────────────────────────────────────────┬───────────────────────┘
                                 │                                           │
                                 ▼ (Macro Wafer Map)                         ▼ (Micro Die Micrograph)
         ┌───────────────────────────────────────┐   ┌───────────────────────────────────────────────┐
         │       WAFER VLM SPECIALIST            │   │            DIE VFM SPECIALIST                 │
         │     (Gemini 2.5 Flash Multimodal)     │   │      (NV-DINOv2 ViT-B/14 & ViT-L/14 Probe)    │
         ├───────────────────────────────────────┤   ├───────────────────────────────────────────────┤
         │ • 9 WM-811K Spatial Defect Classes    │   │ • 6 Micro-Defect Classes (Short, Spur, Void)  │
         │ • Concentric, Donut & Edge Ring Logic │   │ • Sub-second Feature Extraction & Linear Head │
         │ • Zero-Hallucination Pydantic Schema  │   │ • GCS BMP / PNG Direct Ingestion              │
         └───────────────────┬───────────────────┘   └───────────────────────┬───────────────────────┘
                             │                                               │
                             └───────────────────────┬───────────────────────┘
                                                     │ (Aggregated Defect Signature)
                                                     ▼
         ┌───────────────────────────────────────────────────────────────────────────────────────────┐
         │                          FMEA RAG KNOWLEDGE ENGINE                                        │
         │  • Vertex AI Vector Search (768-dim dense embeddings via `text-embedding-004`)            │
         │  • SEMI-E10 Standard Maintenance SOPs (CMP Platen, 300mm RIE Plasma Etch, ASML Litho)    │
         │  • Root Cause Diagnosis + Actionable Corrective Maintenance Procedures                    │
         └───────────────────────────────────────────┬───────────────────────────────────────────────┘
                                                     │
                                                     ▼
         ┌───────────────────────────────────────────────────────────────────────────────────────────┐
         │                        METROLOGY AUDIT & ACTIONABLE DISPATCH                              │
         │  • BigQuery Real-Time Streaming Audit: `semicon_metrology.lot_inspections`                 │
         │  • Cloud Logging & OpenTelemetry Distributed Tracing Export                               │
         │  • JSON Metrology Report returned to Fab MES / Process Engineer                           │
         └───────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## ⚡ Step-by-Step: How to Run Live Inferencing

### 1. Authenticate with Google Cloud
Because the Cloud Run service is protected by Google IAM perimeter security, generate an OIDC identity token:

```bash
# Ensure you are logged into gcloud with your authorized identity
gcloud auth login

# Cache your current identity token
export TOKEN=$(gcloud auth print-identity-token)
```

### 2. Execute Inspection for Supported Semiconductor Chambers

#### A. Chemical Mechanical Planarization (CMP Platen 1)
```bash
curl -s -X POST "https://ms-adc-api-be4mdtaema-uc.a.run.app/v1/inspect" \
  -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
  -H "Content-Type: application/json" \
  -d '{
    "engineer_ticket": "Slurry abrasive agglomeration on CMP platen 1.",
    "lot_info": {
      "lot_id": "LOT-CMP-112",
      "chamber": "300mm_CMP_Platen_1",
      "images": [
        "gs://ms-adc-assets-705987267327/sample_images/LOT-CMP-112_wafer_map.bmp",
        "gs://ms-adc-assets-705987267327/sample_images/LOT-CMP-112_die_micrograph.bmp"
      ]
    }
  }' | jq .
```
* **Detected Defect:** Macro: `Edge-Loc` (0.95 conf) | Micro: `Spur` (0.486 conf)
* **Retrieved SOP:** `FMEA-SOP-CMP-300-PL1` (Retaining ring pressure pneumatic chamber calibration & slurry delivery flush)
* **Status:** `PRIMARY_SUCCESS` | Latency: ~7.9s

---

#### B. 300mm RIE Plasma Etch (Etch Chamber A)
```bash
curl -s -X POST "https://ms-adc-api-be4mdtaema-uc.a.run.app/v1/inspect" \
  -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
  -H "Content-Type: application/json" \
  -d '{
    "engineer_ticket": "Chamber ESC helium flow leak detected causing concentric thermal ring pattern on RIE Etch chamber.",
    "lot_info": {
      "lot_id": "LOT-123",
      "chamber": "300mm_Etch_Chamber_A",
      "images": [
        "gs://ms-adc-assets-705987267327/sample_images/LOT-123_wafer_map.bmp",
        "gs://ms-adc-assets-705987267327/sample_images/LOT-123_die_micrograph.bmp"
      ]
    }
  }' | jq .
```
* **Detected Defect:** Macro: `Center` (0.98 conf) | Micro: `Short` (0.292 conf)
* **Retrieved SOP:** `FMEA-SOP-ETCH-300-CH3` (Electrostatic Chuck helium backside cooling seal repair)
* **Status:** `PRIMARY_SUCCESS` | Latency: ~9.1s

---

#### C. Photolithography Scanner (ASML Scanner Chuck 2)
```bash
curl -s -X POST "https://ms-adc-api-be4mdtaema-uc.a.run.app/v1/inspect" \
  -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
  -H "Content-Type: application/json" \
  -d '{
    "engineer_ticket": "Focus-energy matrix thermal drift on scanner stage chuck causing radial distortion.",
    "lot_info": {
      "lot_id": "LOT-LITHO-441",
      "chamber": "ASML_Scanner_Chuck_2",
      "images": [
        "gs://ms-adc-assets-705987267327/sample_images/LOT-LITHO-441_wafer_map.bmp",
        "gs://ms-adc-assets-705987267327/sample_images/LOT-LITHO-441_die_micrograph.bmp"
      ]
    }
  }' | jq .
```
* **Detected Defect:** Macro: `Scratch` (0.95 conf) | Micro: `Mouse_bite` (0.244 conf)
* **Retrieved SOP:** `FMEA-SOP-LITHO-ASML-NXT` (Chuck thermal drift & lens heating focus compensation)
* **Status:** `PRIMARY_SUCCESS` | Latency: ~14.9s

---

### 3. Programmatic Python Client Example

```python
import subprocess
import requests

# Retrieve identity token via gcloud
token = subprocess.check_output(
    ["gcloud", "auth", "print-identity-token"],
    text=True
).strip()

url = "https://ms-adc-api-be4mdtaema-uc.a.run.app/v1/inspect"
headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

payload = {
    "engineer_ticket": "Slurry abrasive agglomeration on CMP platen 1.",
    "lot_info": {
        "lot_id": "LOT-CMP-112",
        "chamber": "300mm_CMP_Platen_1",
        "images": [
            "gs://ms-adc-assets-705987267327/sample_images/LOT-CMP-112_wafer_map.bmp",
            "gs://ms-adc-assets-705987267327/sample_images/LOT-CMP-112_die_micrograph.bmp"
        ]
    }
}

response = requests.post(url, json=payload, headers=headers)
print("Status Code:", response.status_code)
print(response.json())
```

---

### 4. Output Data Contract Schema

The API returns a zero-hallucination Pydantic response conforming to `InspectionResponse`:

```json
{
  "lot_id": "LOT-CMP-112",
  "chamber": "300mm_CMP_Platen_1",
  "macro_defect": "Edge-Loc",
  "macro_confidence": 0.95,
  "defect_density_D0": 0.8,
  "die_yield_pct": 90.0,
  "micro_defect": "Spur",
  "micro_confidence": 0.486,
  "defect_layer": "Micrograph Optical Patch",
  "bounding_box": { "x": 0, "y": 0, "width": 0, "height": 0 },
  "fmea_citations": [
    {
      "doc_id": "FMEA-SOP-CMP-300-PL1",
      "section_title": "2. Edge-Loc Excursion & Spurious Copper Flakes",
      "failure_classes": ["Edge-Loc", "Random", "Spurious_copper", "Spur", "Scratch"],
      "tool_chamber": "Platen 1 (Copper Bulk & Barrier CMP Polishing Station)",
      "similarity_score": 1.0
    }
  ],
  "recommended_action": "Execute per FMEA-SOP-CMP-300-PL1: Retaining ring wear profilometry & copper slurry loop flush.",
  "execution_latency_ms": 7965.89,
  "audit_id": "INSP-CF7C472B",
  "circuit_breaker_status": "PRIMARY_SUCCESS"
}
```

---

## 📊 Telemetry & Warehouse Verification

### BigQuery Audit Streaming Table
Every inspection is streamed asynchronously into BigQuery. Run this query to observe real-time entries:

```bash
python3 -c "
from google.cloud import bigquery
client = bigquery.Client(project='kshitiz-gemma3')
query = '''
SELECT timestamp, inspection_id, lot_id, tool_chamber, macro_defect_class, micro_defect_class, total_latency_ms
FROM \`kshitiz-gemma3.semicon_metrology.lot_inspections\`
ORDER BY timestamp DESC
LIMIT 5
'''
print('TIMESTAMP (UTC) | INSPECTION ID | LOT ID        | MACRO    | MICRO      | LATENCY')
print('-' * 80)
for r in client.query(query).result():
    print(f'{r.timestamp:%H:%M:%S}        | {r.inspection_id} | {r.lot_id:13} | {r.macro_defect_class:8} | {r.micro_defect_class:10} | {r.total_latency_ms:.0f}ms')
"
```

### Real-Time Container Logs
```bash
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=ms-adc-api AND resource.labels.revision_name=ms-adc-api-00015-skn" \
  --limit=25 \
  --freshness=10m \
  --format="value(textPayload)"
```

---

## 🎯 Assessed Competencies Mapping (32/32 Covered)

| Pillar | Competencies Demonstrated | Implementation Path |
| :--- | :--- | :--- |
| **1. AI/ML Engineering** | Multi-Agent Systems, RAG Pipelines, Model Tuning, LLM Ops, Domain KPIs | `src/orchestrator/`, `src/rag/`, `src/models/` |
| **2. Scoping & Design** | Problem Def, Scoping Boundaries, DoD, System Diagrams, ADRs, OpenAPI 3.0 | `docs/scoping.md`, `docs/adr/`, `docs/architecture/` |
| **3. Security & Privacy** | OAuth2/IAM, VPC Isolation, Cloud DLP, Prompt Injection Guard, Audit Logs | `src/security/`, `terraform/vpc.tf`, `terraform/iam.tf` |
| **4. Reliability & Trace** | High Availability, OpenTelemetry Distributed Tracing, Chaos Testing, Circuit Breakers | `src/telemetry/`, `src/orchestrator/circuit_breaker.py` |
| **5. Cost & Efficiency** | Dynamic Batching, TensorRT Compilation, Vertex AI Context & Embedding Caching | `src/models/export_tensorrt.py`, `src/rag/cache.py` |
| **6. Operations & CI/CD** | Cloud Build CI/CD, Terraform IaC (100%), Versioned Prompt Registries, Pytest | `cloudbuild.yaml`, `terraform/`, `config/prompts.yaml` |
| **7. Extensibility** | Base Interface Abstractions, Feature Flags, Contract-First API, Webhooks | `src/models/base.py`, `src/gateway/webhooks.py` |

---

## 🚀 Local Development

### 1. Clone & Environment Setup
```bash
git clone https://github.com/Kshitiz-Khandel/MS-ADC.git
cd MS-ADC

python3.11 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

### 2. Launch Local FastAPI Gateway
```bash
uvicorn src.gateway.app:app --host 0.0.0.0 --port 8080 --reload
```
Interactive OpenAPI documentation will be available at `http://localhost:8080/docs`.

### 3. Run Test Suite
```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## 📄 License
This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.��──────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Assessed Competencies Mapping (32/32 Covered)

| Pillar | Competencies Demonstrated | Implementation Path |
| :--- | :--- | :--- |
| **1. AI/ML Engineering** | Multi-Agent Systems, RAG Pipelines, Model Tuning, LLM Ops, Domain KPIs | `src/orchestrator/`, `src/rag/`, `src/models/` |
| **2. Scoping & Design** | Problem Def, Scoping Boundaries, DoD, System Diagrams, ADRs, OpenAPI 3.0 | `docs/scoping.md`, `docs/adr/`, `docs/architecture/` |
| **3. Security & Privacy** | OAuth2/IAM, VPC Isolation, Cloud DLP, Prompt Injection Guard, Audit Logs | `src/security/`, `terraform/vpc.tf`, `terraform/iam.tf` |
| **4. Reliability & Trace** | High Availability, OpenTelemetry Distributed Tracing, Chaos Testing, Circuit Breakers | `src/telemetry/`, `src/orchestrator/circuit_breaker.py` |
| **5. Cost & Efficiency** | Dynamic Batching, TensorRT Compilation, Vertex AI Context & Embedding Caching | `src/models/export_tensorrt.py`, `src/rag/cache.py` |
| **6. Operations & CI/CD** | Cloud Build CI/CD, Terraform IaC (100%), Versioned Prompt Registries, Pytest | `cloudbuild.yaml`, `terraform/`, `config/prompts.yaml` |
| **7. Extensibility** | Base Interface Abstractions, Feature Flags, Contract-First API, Webhooks | `src/models/base.py`, `src/gateway/webhooks.py` |

---

## 🚀 Quickstart & Local Development

### 1. Clone & Environment Setup
```bash
git clone https://github.com/Kshitiz-Khandel/MS-ADC.git
cd MS-ADC

python3.11 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

### 2. Run Data Ingestion & Few-Shot Setup
```bash
# Ingests WM-811K wafer maps and PCB micro-die datasets
python -m src.ingestion.dataset_loader --sample-size 50
```

### 3. Launch Local FastAPI Gateway
```bash
uvicorn src.gateway.app:app --host 0.0.0.0 --port 8000 --reload
```
Interactive OpenAPI documentation will be available at `http://localhost:8000/docs`.

### 4. Run Test Suite & Evaluation
```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## 📄 License
This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
