# MS-ADC System Design & Architecture Specification

**Document Version:** 2.0.0  
**Author:** Aditya Sharma (Lead AI/ML Engineer)  
**Co-Author:** Kshitiz Khandelwal (Lead AI Architect)  
**Competencies Demonstrated:** Comp 9 (System Design Artifacts)  

---

## 1. High-Level Architecture Overview

MS-ADC is an enterprise multi-agent metrology platform engineered for automated defect classification (ADC) and root-cause analysis (RCA) in 300mm semiconductor cleanrooms. It bridges high-throughput edge vision models (Wafer VLM and Die VFM) coordinated by an autonomous Google Agent Development Kit (ADK 2.0) orchestrator, safeguarded by Google Cloud Apigee and Vertex AI Model Armor, and grounded in SEMI-E10 FMEA playbooks via an in-memory hybrid semantic cache and Vertex AI Vector Search.

```mermaid
flowchart TD
    subgraph ClientLayer ["Client & Cleanroom Ingestion Layer"]
        MES["Fab MES / Metrology Tool Track (ASML, KLA, Lam)"]
        Engineer["Cleanroom Process Engineer Web UI / CLI"]
    end

    subgraph EdgePerimeter ["Enterprise Edge API Perimeter (Google Cloud Apigee)"]
        ApigeeProxy["Apigee Edge Proxy (MS-ADC-Metrology-v1)"]
        SpikeArrest["Spike Arrest (120 req/min) & Chamber Quota"]
        ToolAuth["Fab API Key Validation & Google OIDC Token Minting"]
    end

    subgraph SecurityPerimeter ["Inference Gateway & Security Perimeter (Cloud Run)"]
        FastAPIGateway["FastAPI Gateway (/v1/inspect)"]
        ModelArmorIngress["Vertex AI Model Armor (Ingress Prompt Guard)"]
        ModelArmorEgress["Vertex AI Model Armor (Egress IP Exfiltration Guard)"]
    end

    subgraph AgentOrchestration ["Autonomous Agent Tier (Google ADK 2.0)"]
        Coordinator["Metrology Coordinator Agent (Gemini 2.5 Flash / ADK)"]
        CircuitBreaker["Circuit Breaker (Closed, Open, Half-Open SLA)"]
        ToolRouter{"Dynamic Tool-Calling Router"}
    end

    subgraph VisionSpecialists ["Multimodal Vision Tier"]
        WaferVLM["Wafer VLM Specialist\n(Gemini 2.5 Flash / Spatial Matrix Analysis)"]
        DieVFM["Die VFM Specialist\n(PyTorch DINOv2 ViT-L/14 Linear Probe <50ms)"]
    end

    subgraph KnowledgeTier ["RAG Knowledge & Data Warehouse Tier"]
        L1Cache[("In-Memory Hybrid Semantic Cache (L1)\n(<1ms Sub-Millisecond Hit)")]
        VectorDB[("Vertex AI Vector Search\n(ScaNN SEMI-E10 FMEA Index)")]
        BQ[("BigQuery Metrology Warehouse\n(semicon_metrology.lot_inspections)")]
    end

    MES -->|SECS/GEM or HTTP| ApigeeProxy
    Engineer -->|REST /v1/inspect| ApigeeProxy
    ApigeeProxy --> SpikeArrest --> ToolAuth
    ToolAuth -->|HTTPS + Google OIDC Bearer| FastAPIGateway

    FastAPIGateway --> ModelArmorIngress
    ModelArmorIngress -->|Sanitized Cleanroom Payload| Coordinator

    Coordinator --> CircuitBreaker --> ToolRouter
    ToolRouter -->|Wafer Map Matrix| WaferVLM
    ToolRouter -->|Die Micrograph Patch| DieVFM
    WaferVLM -->|Macro Defect JSON| Coordinator
    DieVFM -->|Micro Defect Code & Probabilities| Coordinator

    Coordinator -->|Defect Facets & Query| L1Cache
    L1Cache -.->|Cache Miss| VectorDB
    VectorDB -.->|Cited FMEA SOP Chunks| L1Cache
    L1Cache -->|Resolved Maintenance Playbook| Coordinator

    Coordinator --> ModelArmorEgress
    ModelArmorEgress -->|Validated Response| FastAPIGateway
    Coordinator -->|Async Compliance Stream| BQ
    FastAPIGateway -->|Standardized /v1 JSON Report| Engineer
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
