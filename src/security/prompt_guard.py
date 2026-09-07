import re
from typing import Tuple, Optional, Dict, Any

# ============================================================================
# Official Google Cloud Vertex AI Model Armor SDK Integration (Ingress + Egress)
# Package: google-cloud-modelarmor | API: google.cloud.modelarmor_v1
# Ref: https://cloud.google.com/vertex-ai/docs/model-armor
# ============================================================================

try:
    from google.cloud import modelarmor_v1
except ImportError:
    # Graceful SDK Fallback for local/offline testing environments
    class ModelArmorTypes:
        class FilterResult:
            MATCH_FOUND = "MATCH_FOUND"
            NO_MATCH_FOUND = "NO_MATCH_FOUND"

        class SanitizeUserPromptRequest:
            def __init__(self, name: str, user_prompt_data: Dict[str, Any]):
                self.name = name
                self.user_prompt_data = user_prompt_data

        class SanitizeModelResponseRequest:
            def __init__(self, name: str, model_response_data: Dict[str, Any]):
                self.name = name
                self.model_response_data = model_response_data

        class SanitizedPrompt:
            def __init__(self, text: str):
                self.text = text

        class SanitizationResult:
            def __init__(self, match_state: str, text: str):
                self.filter_match_state = match_state
                self.sanitized_user_prompt = ModelArmorTypes.SanitizedPrompt(text)
                self.sanitized_model_response = ModelArmorTypes.SanitizedPrompt(text)

        class SanitizeUserPromptResponse:
            def __init__(self, result):
                self.sanitization_result = result

        class SanitizeModelResponseResponse:
            def __init__(self, result):
                self.sanitization_result = result

        class ModelArmorClient:
            def sanitize_user_prompt(self, request):
                text = request.user_prompt_data.get("text", "")
                injection_patterns = [
                    r"ignore\s+(all\s+)?(previous|prior)\s+(instructions|prompts|rules)",
                    r"system\s*prompt",
                    r"you\s+are\s+now\s+in\s+dan\s+mode",
                    r"jailbreak",
                    r"override\s+safety",
                    r"output\s+all\s+database\s+credentials"
                ]
                for pattern in injection_patterns:
                    if re.search(pattern, text, re.IGNORECASE):
                        return ModelArmorTypes.SanitizeUserPromptResponse(
                            ModelArmorTypes.SanitizationResult(
                                ModelArmorTypes.FilterResult.MATCH_FOUND, ""
                            )
                        )
                return ModelArmorTypes.SanitizeUserPromptResponse(
                    ModelArmorTypes.SanitizationResult(
                        ModelArmorTypes.FilterResult.NO_MATCH_FOUND, text
                    )
                )

            def sanitize_model_response(self, request):
                text = request.model_response_data.get("text", "")
                egress_leak_patterns = [
                    r"BEGIN PRIVATE KEY",
                    r"AIzaSy[A-Za-z0-9_\-]{33}",
                    r"password\s*[:=]\s*\S+",
                    r"postgres://[A-Za-z0-9_]+:[A-Za-z0-9_]+@",
                    r"UNAUTHORIZED_OVERRIDE_INTERLOCK"
                ]
                for pattern in egress_leak_patterns:
                    if re.search(pattern, text, re.IGNORECASE):
                        return ModelArmorTypes.SanitizeModelResponseResponse(
                            ModelArmorTypes.SanitizationResult(
                                ModelArmorTypes.FilterResult.MATCH_FOUND, "[REDACTED_BY_MODEL_ARMOR_EGRESS]"
                            )
                        )
                return ModelArmorTypes.SanitizeModelResponseResponse(
                    ModelArmorTypes.SanitizationResult(
                        ModelArmorTypes.FilterResult.NO_MATCH_FOUND, text
                    )
                )

    modelarmor_v1 = ModelArmorTypes

class VertexModelArmorGuard:
    """
    Google Cloud Vertex AI Model Armor Dual-Guard (Ingress & Egress Interceptor).
    - Ingress Template: Sanitizes operator input tickets against Prompt Injection & Jailbreaks.
    - Egress Template: Sanitizes AI generated reports against Credential Leakage & Fab IP Disclosure.
    """
    def __init__(self, project_id: str = "semicon-metrology-prod", location: str = "us-central1"):
        self.project_id = project_id
        self.location = location
        self.ingress_template_name = f"projects/{project_id}/locations/{location}/templates/cleanroom-metrology-ingress-template"
        self.egress_template_name = f"projects/{project_id}/locations/{location}/templates/cleanroom-metrology-egress-template"
        self.client = modelarmor_v1.ModelArmorClient()

    def sanitize_user_prompt(self, user_prompt: str) -> Dict[str, Any]:
        """
        Executes Ingress sanitizeUserPrompt via Google Cloud Vertex AI Model Armor API.
        """
        request = modelarmor_v1.SanitizeUserPromptRequest(
            name=self.ingress_template_name,
            user_prompt_data={"text": user_prompt}
        )
        response = self.client.sanitize_user_prompt(request=request)
        is_match = response.sanitization_result.filter_match_state == modelarmor_v1.FilterResult.MATCH_FOUND

        return {
            "is_safe": not is_match,
            "filter_match_state": response.sanitization_result.filter_match_state,
            "sanitized_text": response.sanitization_result.sanitized_user_prompt.text,
            "direction": "INGRESS",
            "engine": "Google_Cloud_Vertex_AI_Model_Armor"
        }

    def sanitize_model_response(self, model_response: str) -> Dict[str, Any]:
        """
        Executes Egress sanitizeModelResponse via Google Cloud Vertex AI Model Armor API.
        Ensures credentials, database secrets, or unmasked IP never leak in outbound responses.
        """
        request = modelarmor_v1.SanitizeModelResponseRequest(
            name=self.egress_template_name,
            model_response_data={"text": model_response}
        )
        response = self.client.sanitize_model_response(request=request)
        is_match = response.sanitization_result.filter_match_state == modelarmor_v1.FilterResult.MATCH_FOUND

        return {
            "is_safe": not is_match,
            "filter_match_state": response.sanitization_result.filter_match_state,
            "sanitized_text": response.sanitization_result.sanitized_model_response.text,
            "direction": "EGRESS",
            "engine": "Google_Cloud_Vertex_AI_Model_Armor"
        }

    def validate_input(self, text: str) -> Tuple[bool, Optional[str]]:
        """Validates incoming operator tickets against Ingress Model Armor filters."""
        res = self.sanitize_user_prompt(text)
        if not res["is_safe"]:
            return False, "Vertex Model Armor Ingress Alert: Prompt Injection or Jailbreak Attempt Blocked"
        return True, None

    def validate_output(self, text: str) -> Tuple[bool, str]:
        """Validates outgoing AI generated recommendations against Egress Model Armor filters."""
        res = self.sanitize_model_response(text)
        if not res["is_safe"]:
            return False, res["sanitized_text"]
        return True, text

# Backward compatibility alias
PromptGuard = VertexModelArmorGuard
