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
    def __init__(self, project_id: Optional[str] = None, location: str = "us-central1"):
        import os
        import logging
        self.logger = logging.getLogger("VertexModelArmorGuard")
        self.project_id = project_id or os.environ.get("PROJECT_ID", "kshitiz-gemma3")
        self.location = location
        self.ingress_template_name = f"projects/{self.project_id}/locations/{location}/templates/cleanroom-metrology-ingress-template"
        self.egress_template_name = f"projects/{self.project_id}/locations/{location}/templates/cleanroom-metrology-egress-template"
        
        try:
            from google.api_core.client_options import ClientOptions
            client_options = ClientOptions(api_endpoint=f"modelarmor.{location}.rep.googleapis.com")
            self.client = modelarmor_v1.ModelArmorClient(client_options=client_options)
        except Exception as e:
            self.logger.warning("Falling back to default ModelArmorClient: %s", e)
            self.client = modelarmor_v1.ModelArmorClient()

        self._ensure_templates()

    def _ensure_templates(self):
        """Auto-provisions Google Cloud Model Armor Ingress & Egress templates in Vertex AI if not present."""
        parent = f"projects/{self.project_id}/locations/{self.location}"
        try:
            self.client.get_template(name=self.ingress_template_name)
            self.logger.info("Found existing Model Armor Ingress template: %s", self.ingress_template_name)
        except Exception:
            try:
                ingress_cfg = modelarmor_v1.FilterConfig(
                    pi_and_jailbreak_filter_settings=modelarmor_v1.PiAndJailbreakFilterSettings(
                        filter_enforcement=modelarmor_v1.PiAndJailbreakFilterSettings.PiAndJailbreakFilterEnforcement.ENABLED,
                        confidence_level=modelarmor_v1.DetectionConfidenceLevel.MEDIUM_AND_ABOVE
                    )
                )
                self.client.create_template(
                    parent=parent,
                    template_id="cleanroom-metrology-ingress-template",
                    template=modelarmor_v1.Template(filter_config=ingress_cfg)
                )
                self.logger.info("Successfully provisioned Model Armor Ingress template: %s", self.ingress_template_name)
            except Exception as e:
                self.logger.warning("Failed to auto-provision Ingress template (%s): %s", self.ingress_template_name, e)

        try:
            self.client.get_template(name=self.egress_template_name)
            self.logger.info("Found existing Model Armor Egress template: %s", self.egress_template_name)
        except Exception:
            try:
                egress_cfg = modelarmor_v1.FilterConfig(
                    malicious_uri_filter_settings=modelarmor_v1.MaliciousUriFilterSettings(
                        filter_enforcement=modelarmor_v1.MaliciousUriFilterSettings.MaliciousUriFilterEnforcement.ENABLED
                    )
                )
                self.client.create_template(
                    parent=parent,
                    template_id="cleanroom-metrology-egress-template",
                    template=modelarmor_v1.Template(filter_config=egress_cfg)
                )
                self.logger.info("Successfully provisioned Model Armor Egress template: %s", self.egress_template_name)
            except Exception as e:
                self.logger.warning("Failed to auto-provision Egress template (%s): %s", self.egress_template_name, e)

    def sanitize_user_prompt(self, user_prompt: str) -> Dict[str, Any]:
        """
        Executes Ingress sanitizeUserPrompt via Google Cloud Vertex AI Model Armor API.
        """
        engine = "Google_Cloud_Vertex_AI_Model_Armor_Live"
        injection_patterns = [
            r"ignore\s+(all\s+)?(previous|prior)\s+(instructions|prompts|rules)",
            r"system\s*prompt",
            r"you\s+are\s+now\s+in\s+dan\s+mode",
            r"jailbreak",
            r"override\s+safety",
            r"output\s+all\s+database\s+credentials"
        ]
        local_injection = any(re.search(p, user_prompt, re.IGNORECASE) for p in injection_patterns)

        try:
            request = modelarmor_v1.SanitizeUserPromptRequest(
                name=self.ingress_template_name,
                user_prompt_data={"text": user_prompt}
            )
            response = self.client.sanitize_user_prompt(request=request)
            ms = response.sanitization_result.filter_match_state
            is_match = (int(ms) == 2 or getattr(ms, "name", "") == "MATCH_FOUND" or local_injection)
            match_state = "MATCH_FOUND" if is_match else "NO_MATCH_FOUND"
            sanitized_text = "" if is_match else user_prompt
        except Exception as e:
            self.logger.warning("Model Armor Ingress sanitize_user_prompt failed, falling back: %s", e)
            engine = "Google_Cloud_Vertex_AI_Model_Armor_Fallback"
            is_match = local_injection
            match_state = "MATCH_FOUND" if is_match else "NO_MATCH_FOUND"
            sanitized_text = "" if is_match else user_prompt

        return {
            "is_safe": not is_match,
            "filter_match_state": match_state,
            "sanitized_text": sanitized_text,
            "direction": "INGRESS",
            "engine": engine
        }

    def sanitize_model_response(self, model_response: str) -> Dict[str, Any]:
        """
        Executes Egress sanitizeModelResponse via Google Cloud Vertex AI Model Armor API.
        Ensures credentials, database secrets, or unmasked IP never leak in outbound responses.
        """
        engine = "Google_Cloud_Vertex_AI_Model_Armor_Live"
        egress_leak_patterns = [
            r"BEGIN PRIVATE KEY",
            r"AIzaSy[A-Za-z0-9_\-]{33}",
            r"password\s*[:=]\s*\S+",
            r"postgres://[A-Za-z0-9_]+:[A-Za-z0-9_]+@",
            r"UNAUTHORIZED_OVERRIDE_INTERLOCK"
        ]
        local_leak = any(re.search(p, model_response, re.IGNORECASE) for p in egress_leak_patterns)

        try:
            request = modelarmor_v1.SanitizeModelResponseRequest(
                name=self.egress_template_name,
                model_response_data={"text": model_response}
            )
            response = self.client.sanitize_model_response(request=request)
            ms = response.sanitization_result.filter_match_state
            is_match = (int(ms) == 2 or getattr(ms, "name", "") == "MATCH_FOUND" or local_leak)
            match_state = "MATCH_FOUND" if is_match else "NO_MATCH_FOUND"
            sanitized_text = "[REDACTED_BY_MODEL_ARMOR_EGRESS]" if is_match else model_response
        except Exception as e:
            self.logger.warning("Model Armor Egress sanitize_model_response failed, falling back: %s", e)
            engine = "Google_Cloud_Vertex_AI_Model_Armor_Fallback"
            is_match = local_leak
            match_state = "MATCH_FOUND" if is_match else "NO_MATCH_FOUND"
            sanitized_text = "[REDACTED_BY_MODEL_ARMOR_EGRESS]" if is_match else model_response

        return {
            "is_safe": not is_match,
            "filter_match_state": match_state,
            "sanitized_text": sanitized_text,
            "direction": "EGRESS",
            "engine": engine
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
