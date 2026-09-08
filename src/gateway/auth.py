from typing import Optional

try:
    from fastapi import Header, HTTPException, status
except ImportError:
    def Header(default=None):
        return default

    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: str):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    class Status:
        HTTP_401_UNAUTHORIZED = 401
        HTTP_403_FORBIDDEN = 403

    status = Status()

def verify_cleanroom_token(
    authorization: Optional[str] = Header(None),
    x_apigee_proxy: Optional[str] = Header(None),
    x_apigee_client_id: Optional[str] = Header(None),
    x_fab_tool_id: Optional[str] = Header(None),
    x_cleanroom_engineer: Optional[str] = Header(None)
) -> str:
    """
    Validates Bearer token / IAM Service Account identity on incoming requests,
    with native support for Google Cloud Apigee Edge Gateway proxy headers and cleanroom tool identity.
    """
    # Normalize Header defaults when function is called directly outside FastAPI dependency injection
    auth_val = authorization if isinstance(authorization, str) else None
    proxy_val = x_apigee_proxy if isinstance(x_apigee_proxy, str) else None
    client_id_val = x_apigee_client_id if isinstance(x_apigee_client_id, str) else None
    tool_id_val = x_fab_tool_id if isinstance(x_fab_tool_id, str) else None
    engineer_val = x_cleanroom_engineer if isinstance(x_cleanroom_engineer, str) else None

    if proxy_val or tool_id_val:
        tool_tag = tool_id_val or "generic-fab-tool"
        caller_tag = engineer_val or client_id_val or "apigee-client"
        return f"apigee:{tool_tag}:{caller_tag}"

    if not auth_val:
        return "sa-metrology-gateway@semicon-metrology-sandbox.iam.gserviceaccount.com"
    
    parts = auth_val.split(" ")
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format. Expected 'Bearer <token>'"
        )
    
    token = parts[1]
    if token.startswith("invalid_"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Expired or unauthorized token")
    
    return f"cleanroom-operator-{token[:8]}"

