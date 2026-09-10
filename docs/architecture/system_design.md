# MS-ADC System Design & Architecture Specification

**Document Version:** 2.0.0  
**Author:** Aditya Sharma (Lead AI/ML Engineer)  
**Co-Author:** Kshitiz Khandelwal (Lead AI Architect)  
**Competencies Demonstrated:** Comp 9 (System Design Artifacts)  

---

## 1. High-Level Linear Architecture Overview

MS-ADC is an enterprise multi-agent metrology platform for automated defect classification (ADC) and root-cause analysis (RCA) in 300mm cleanrooms. The architecture follows a strict, end-to-end linear pipeline:

```mermaid
graph TD
    %% 1. Ingestion Layer
    subgraph S1 ["1. Cleanroom Ingestion Layer"]
        Client["Fab MES & Cleanroom Process Engineer<br/><i>(ASML Scanners, KLA Tracks, Engineer UI)</i>"]
    end

    %% 2. Edge Gateway Layer
    subgraph S2 ["2. Enterprise Edge Perimeter (Google Cloud Apigee)"]
        Apigee["Apigee Edge Gateway (MS-ADC-Metrology-v1)<br/>• Spike Arrest (120 req/min) & Chamber Quotas<br/>• Tool API Key Validation<br/>• Google Cloud OIDC Bearer Token Minting"]
    end

    %% 3. Ingress Security Layer
    subgraph S3 ["3. AI Security Ingress (Vertex AI Model Armor)"]
        ModelArmorIn["Model Armor Ingress Filter<br/>• Prompt Injection & Jailbreak Defense<br/>• Proprietary Recipe & Secret Redaction"]
    end

    %% 4. Agent Orchestration Layer
    subgraph S4 ["4. Central Coordinator Agent (Google ADK 2.0)"]
        ADK["Metrology Coordinator Agent<br/>• Gemini Pro Reasoning Engine<br/>• Tool-Calling Dispatch & Circuit Breaker SLA"]
    end

    %% 5. Multimodal Vision Layer
    subgraph S5 ["5. Multimodal Vision Specialist Tier"]
        direction LR
        WaferVLM["Wafer VLM Specialist<br/><i>(Gemini Pro / Spatial Matrix)</i><br/>Macro: Center, Ring, Edge, Scratch"]
        DieVFM["Die VFM Specialist<br/><i>(DINOv2 ViT-L/14 Linear Probe &lt;50ms)</i><br/>Micro: Short, Void, Spur, Bridge"]
    end

    %% 6. Knowledge & RAG Layer
    subgraph S6 ["6. Root-Cause RAG & Hybrid Caching Tier"]
        RAG["L1 Semantic Cache & Vertex AI Vector Search<br/>• Fast-Path In-Memory L1 Cache (&lt;1ms)<br/>• ScaNN 768-dim SEMI-E10 FMEA Index (~350ms)"]
    end

    %% 7. Egress Guardrail Layer
    subgraph S7 ["7. AI Egress Security (Vertex AI Model Armor)"]
        ModelArmorOut["Model Armor Egress Filter<br/>• Manufacturing IP Exfiltration Defense<br/>• Actionable Maintenance SOP Validation"]
    end

    %% 8. Presentation & Analytics Layer
    subgraph S8 ["8. Output Delivery & Compliance Warehouse"]
        Output["Standardized /v1 Diagnostic JSON Report<br/><i>(Returned to Cleanroom MES / Process Engineer)</i>"]
        BQ[("BigQuery Metrology Warehouse<br/><i>(Async Audit Log & SEMI Telemetry)</i>")]
    end

    %% Straight-Down Linear Pipeline Flow
    Client -->|POST /v1/inspect| Apigee
    Apigee -->|HTTPS + OIDC Bearer Token| ModelArmorIn
    ModelArmorIn -->|Sanitized Cleanroom Payload| ADK
    ADK -->|Dispatch Wafer Map| WaferVLM
    ADK -->|Dispatch Die Micrograph| DieVFM
    WaferVLM -->|Macro Spatial Defect JSON| ADK
    DieVFM -->|Micro Defect Code & Probabilities| ADK
    ADK -->|Defect Signatures & Query| RAG
    RAG -->|Grounded FMEA Playbook| ADK
    ADK -->|Diagnostic Plan| ModelArmorOut
    ModelArmorOut -->|Validated Report| Output
    ADK -.->|Async Audit Streaming| BQ
```

---

## 2. Sequence Flow Specification

The interaction sequence below details the zero-hallucination execution loop for an end-to-end inspection request:

```mermaid
sequenceDiagram
    autonumber
    actor Tool as Fab MES / Cleanroom Tool
    participant Apigee as Apigee Edge Gateway
    participant Gateway as Cloud Run FastAPI Gateway
    participant Armor as Vertex AI Model Armor
    participant Agent as Google ADK Coordinator
    participant WaferVLM as Wafer VLM Specialist
    participant DieVFM as Die VFM Specialist
    participant Cache as L1 Hybrid Semantic Cache
    participant RAG as Vertex AI Vector Search
    participant BQ as BigQuery Warehouse

    Tool->>Apigee: POST /v1/inspect (lot_id, wafer_img, die_img, ticket)
    Apigee->>Apigee: Verify Fab API Key, Enforce Spike Arrest (120 req/min)
    Apigee->>Apigee: Mint Google Cloud OIDC Bearer Token & Inject Header Tags
    Apigee->>Gateway: Forward Request (Private VPC Ingress)
    Gateway->>Armor: Sanitize Ingress (Prompt Injection & Secret Detection)
    Armor-->>Gateway: Validated Cleanroom Payload
    Gateway->>Agent: Dispatch Payload to ADK Coordinator

    par Parallel Multimodal Vision Analysis
        Agent->>WaferVLM: Classify Spatial Wafer Map
        WaferVLM-->>Agent: {"macro_defect": "Center", "confidence": 0.962}
        Agent->>DieVFM: Classify Optical Micrograph Patch (<50ms)
        DieVFM-->>Agent: {"micro_defect": "Short", "confidence": 0.982}
    end

    Agent->>Cache: Query L1 Semantic Cache (wafer_class, die_class, spatial pattern)
    alt Cache Hit (<1ms)
        Cache-->>Agent: Return Cached FMEA Maintenance Playbook
    else Cache Miss (~350ms)
        Cache->>RAG: ScaNN Approximate Nearest Neighbor Search
        RAG-->>Cache: Top-1 Grounded FMEA SOP (FMEA-SOP-ETCH-300-CH3)
        Cache-->>Agent: Grounded SOP Playbook Chunks
    end

    Agent->>Armor: Validate Egress (Prevent IP / Configuration Leakage)
    Armor-->>Agent: Egress Clearance Confirmed
    Agent->>BQ: Async Stream Audit Log (Lot ID, Defect Classes, FMEA Citation, Latency)
    Agent-->>Gateway: Standardized InspectionResponse JSON
    Gateway-->>Tool: Render Diagnostic Report with Actionable SOP Corrective Actions
```

---

## 3. Data Schema & Contracts

### A. Metrology Warehouse Schema (`BigQuery`)
```sql
CREATE TABLE `semicon_metrology.lot_inspections` (
    inspection_id STRING NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    lot_id STRING NOT NULL,
    wafer_id STRING NOT NULL,
    tool_chamber STRING NOT NULL,
    macro_defect_class STRING NOT NULL,
    macro_confidence FLOAT64 NOT NULL,
    micro_defect_class STRING,
    micro_confidence FLOAT64,
    fmea_citation_id STRING NOT NULL,
    actionable_sop STRING NOT NULL,
    severity STRING NOT NULL,
    cache_hit BOOLEAN NOT NULL,
    total_latency_ms FLOAT64 NOT NULL
);
```

### B. REST API Programmatic Contracts (`Pydantic / OpenAPI`)
* **`POST /v1/inspect/wafer`**: Ingests 2D wafer-bin maps (or `gs://` URIs) for macro spatial defect classification.
* **`POST /v1/inspect/die`**: Ingests high-resolution optical/e-beam micrographs for $<50\text{ms}$ sub-micron die-level classification.
* **`POST /v1/inspect`**: Composite endpoint executing end-to-end multi-agent inspection, semantic caching, and FMEA root-cause playbook generation.
* **`GET /healthz`**: Liveness and readiness container health probe utilized by Cloud Run autoscaling.
