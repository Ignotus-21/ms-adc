import os
from enum import Enum
from typing import List, Optional, Dict, Any

class ChamberType(str, Enum):
    RIE_ETCH = "300mm_RIE_Etch_Chamber_3"
    LITHO_SCANNER = "300mm_Immersion_Litho_Track_2"
    CMP_PLATEN = "300mm_CMP_Platen_1"
    UNKNOWN = "unknown_chamber"

class MacroDefectType(str, Enum):
    CENTER = "Center"
    DONUT = "Donut"
    EDGE_RING = "Edge-Ring"
    EDGE_LOC = "Edge-Loc"
    SCRATCH = "Scratch"
    RANDOM = "Random"
    UNKNOWN = "Unknown"

class MicroDefectType(str, Enum):
    SHORT = "Short"
    OPEN = "Open_circuit"
    PARTICLE = "Particle"
    VOID = "Void"
    LINE_COLLAPSE = "Line_collapse"
    SPURIOUS_COPPER = "Spurious_copper"
    UNKNOWN = "Unknown"

class ValidationError(ValueError):
    pass

try:
    from pydantic import BaseModel, Field, field_validator

    class HealthCheckResponse(BaseModel):
        status: str
        environment: str
        timestamp: str
        version: str

    class LotInfo(BaseModel):
        lot_id: str = Field(..., description="Manufacturing lot ID")
        chamber: str = Field(..., description="Tool chamber ID")
        recipe_id: Optional[str] = Field(None, description="Chemical recipe ID")
        images: List[str] = Field(
            default_factory=list,
            description="List of GCS image URIs (gs://), local paths, or base64 strings containing wafer maps and SEM die micrographs."
        )

        @field_validator("lot_id", "chamber")
        @classmethod
        def check_non_empty(cls, v):
            if v is None or not str(v).strip():
                raise ValidationError("Field cannot be empty or None")
            return v

    class CompositeInspectionRequest(BaseModel):
        engineer_ticket: str = Field(..., description="Engineering problem description")
        lot_info: LotInfo
        metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

        @field_validator("engineer_ticket")
        @classmethod
        def check_ticket(cls, v):
            if v is None or not str(v).strip():
                raise ValidationError("engineer_ticket cannot be empty or None")
            return v

    class WaferInspectionRequest(BaseModel):
        lot_id: str
        chamber: str
        image_uri: str
        engineer_ticket: Optional[str] = ""

    class DieInspectionRequest(BaseModel):
        lot_id: str
        chamber: str
        image_uri: str
        engineer_ticket: Optional[str] = ""

    class InspectionResponse(BaseModel):
        lot_id: str
        chamber: str
        macro_defect: str
        macro_confidence: float
        defect_density_D0: float
        die_yield_pct: float
        micro_defect: str
        micro_confidence: float
        defect_layer: Optional[str] = None
        bounding_box: Optional[Dict[str, Any]] = None
        fmea_citations: List[Dict[str, Any]] = Field(default_factory=list)
        recommended_action: str
        execution_latency_ms: float
        audit_id: str
        circuit_breaker_status: str
        model_armor_ingress: Optional[Dict[str, Any]] = None
        model_armor_egress: Optional[Dict[str, Any]] = None
        tool_call_trace: Optional[List[Dict[str, Any]]] = None
        agent_framework: Optional[str] = None

    class WaferInspectionResponse(BaseModel):
        lot_id: str
        chamber: str
        macro_defect: str
        macro_confidence: float
        defect_density_D0: float
        die_yield_pct: float
        spatial_cluster_evidence: Dict[str, Any]
        pattern_description: str
        execution_latency_ms: float

    class DieInspectionResponse(BaseModel):
        lot_id: str
        chamber: str
        micro_defect: str
        micro_confidence: float
        defect_layer: str
        structural_damage: str
        bounding_box: Dict[str, Any]
        defect_area_nm2: float
        execution_latency_ms: float

    InspectionRequest = CompositeInspectionRequest

except ImportError:
    class HealthCheckResponse:
        def __init__(self, status: str, environment: str, timestamp: str, version: str):
            self.status = status
            self.environment = environment
            self.timestamp = timestamp
            self.version = version

    class LotInfo:
        def __init__(self, lot_id: str, chamber: str, recipe_id: Optional[str] = None, images: Optional[List[str]] = None):
            if not lot_id or not str(lot_id).strip():
                raise ValidationError("lot_id cannot be empty or None")
            if not chamber or not str(chamber).strip():
                raise ValidationError("chamber cannot be empty or None")
            self.lot_id = lot_id
            self.chamber = chamber
            self.recipe_id = recipe_id
            self.images = images if images is not None else []

    class CompositeInspectionRequest:
        def __init__(self, engineer_ticket: str, lot_info: Any, metadata: Optional[Dict[str, Any]] = None, lot_id: Optional[str] = None, chamber: Optional[str] = None, recipe_id: Optional[str] = None, images: Optional[List[str]] = None):
            if not engineer_ticket or not str(engineer_ticket).strip():
                raise ValidationError("engineer_ticket cannot be empty or None")
            if isinstance(lot_info, dict):
                lot_info = LotInfo(**lot_info)
            elif not lot_info:
                if lot_id and chamber:
                    lot_info = LotInfo(lot_id=lot_id, chamber=chamber, recipe_id=recipe_id, images=images or [])
                else:
                    raise ValidationError("lot_info is required")
            self.engineer_ticket = engineer_ticket
            self.lot_info = lot_info
            self.metadata = metadata or {}

        def model_dump(self):
            return {
                "engineer_ticket": self.engineer_ticket,
                "lot_info": {
                    "lot_id": self.lot_info.lot_id,
                    "chamber": self.lot_info.chamber,
                    "recipe_id": self.lot_info.recipe_id,
                    "images": self.lot_info.images
                },
                "metadata": self.metadata
            }

    class WaferInspectionRequest:
        def __init__(self, lot_id: str, chamber: str, image_uri: str, engineer_ticket: str = ""):
            if not lot_id or not chamber or not image_uri:
                raise ValidationError("Required fields cannot be empty")
            self.lot_id = lot_id
            self.chamber = chamber
            self.image_uri = image_uri
            self.engineer_ticket = engineer_ticket

        def model_dump(self):
            return {
                "lot_id": self.lot_id,
                "chamber": self.chamber,
                "image_uri": self.image_uri,
                "engineer_ticket": self.engineer_ticket
            }

    class DieInspectionRequest:
        def __init__(self, lot_id: str, chamber: str, image_uri: str, engineer_ticket: str = ""):
            if not lot_id or not chamber or not image_uri:
                raise ValidationError("Required fields cannot be empty")
            self.lot_id = lot_id
            self.chamber = chamber
            self.image_uri = image_uri
            self.engineer_ticket = engineer_ticket

        def model_dump(self):
            return {
                "lot_id": self.lot_id,
                "chamber": self.chamber,
                "image_uri": self.image_uri,
                "engineer_ticket": self.engineer_ticket
            }

    InspectionRequest = CompositeInspectionRequest

    class WaferInspectionResponse:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class DieInspectionResponse:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class InspectionResponse:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
