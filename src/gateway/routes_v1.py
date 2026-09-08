from typing import Dict, Any, Optional

try:
    from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
except ImportError:
    class BackgroundTasks:
        def add_task(self, func, *args, **kwargs):
            try:
                func(*args, **kwargs)
            except Exception:
                pass

    class APIRouter:
        def __init__(self, prefix: str = "", tags: Optional[list] = None):
            self.prefix = prefix
            self.tags = tags or []
            self.routes = {}

        def post(self, path: str, response_model: Any = None):
            def decorator(func):
                self.routes[("POST", self.prefix + path)] = func
                return func
            return decorator

        def get(self, path: str, response_model: Any = None):
            def decorator(func):
                self.routes[("GET", self.prefix + path)] = func
                return func
            return decorator

    def Depends(dependency):
        return dependency

    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: str):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    class Status:
        HTTP_400_BAD_REQUEST = 400
        HTTP_401_UNAUTHORIZED = 401
        HTTP_503_SERVICE_UNAVAILABLE = 503

    status = Status()

from src.gateway.schemas import (
    CompositeInspectionRequest,
    WaferInspectionRequest,
    DieInspectionRequest,
    InspectionResponse,
    WaferInspectionResponse,
    DieInspectionResponse
)
from src.gateway.auth import verify_cleanroom_token
from src.orchestrator.agent import MetrologyCoordinatorAgent

router = APIRouter(prefix="/v1", tags=["Metrology Inspection v1"])
agent = MetrologyCoordinatorAgent()

@router.post("/inspect", response_model=InspectionResponse)
async def composite_inspection(
    request: CompositeInspectionRequest,
    background_tasks: BackgroundTasks,
    principal: str = Depends(verify_cleanroom_token)
):
    """
    Full hierarchical multimodal inspection: Macro Wafer Map + Micro Die Patch + Grounded FMEA RAG Retrieval.
    """
    try:
        data = request.model_dump() if hasattr(request, "model_dump") else (request.dict() if hasattr(request, "dict") else dict(request))
        return agent.process_inspection(data, user_identity=principal, background_tasks=background_tasks)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

@router.post("/inspect/wafer", response_model=WaferInspectionResponse)
async def inspect_wafer(
    request: WaferInspectionRequest,
    background_tasks: BackgroundTasks,
    principal: str = Depends(verify_cleanroom_token)
):
    """
    Inspects 300mm macro wafer-bin maps for spatial defect pattern classification (WM-811K).
    """
    try:
        data = request.model_dump() if hasattr(request, "model_dump") else (request.dict() if hasattr(request, "dict") else dict(request))
        return agent.inspect_wafer_only(data, user_identity=principal, background_tasks=background_tasks)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

@router.post("/inspect/die", response_model=DieInspectionResponse)
async def inspect_die(
    request: DieInspectionRequest,
    background_tasks: BackgroundTasks,
    principal: str = Depends(verify_cleanroom_token)
):
    """
    Inspects high-magnification sub-micron SEM die micrographs for micro-defect classification (NV-DINOv2).
    """
    try:
        data = request.model_dump() if hasattr(request, "model_dump") else (request.dict() if hasattr(request, "dict") else dict(request))
        return agent.inspect_die_only(data, user_identity=principal, background_tasks=background_tasks)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
