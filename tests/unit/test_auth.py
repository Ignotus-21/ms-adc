import unittest
from src.gateway.auth import verify_cleanroom_token

class TestCleanroomAuthSuite(unittest.TestCase):
    def test_default_service_account_fallback(self):
        principal = verify_cleanroom_token()
        self.assertEqual(principal, "sa-metrology-gateway@semicon-metrology-sandbox.iam.gserviceaccount.com")

    def test_valid_bearer_token(self):
        principal = verify_cleanroom_token(authorization="Bearer ya29.a0AfH6SM...")
        self.assertEqual(principal, "cleanroom-operator-ya29.a0A")

    def test_invalid_bearer_format_raises_401(self):
        with self.assertRaises(Exception) as ctx:
            verify_cleanroom_token(authorization="Basic dXNlcjpwYXNz")
        self.assertTrue("401" in str(ctx.exception) or "Invalid authorization" in str(ctx.exception))

    def test_expired_or_invalid_token_raises_403(self):
        with self.assertRaises(Exception) as ctx:
            verify_cleanroom_token(authorization="Bearer invalid_token_123")
        self.assertTrue("403" in str(ctx.exception) or "Expired or unauthorized" in str(ctx.exception))

    def test_apigee_proxy_headers(self):
        principal = verify_cleanroom_token(
            x_apigee_proxy="MS-ADC-Metrology-v1",
            x_apigee_client_id="client-asml-twinscan-01",
            x_fab_tool_id="ASML-EXE-5000",
            x_cleanroom_engineer="eng_kshitiz"
        )
        self.assertEqual(principal, "apigee:ASML-EXE-5000:eng_kshitiz")

    def test_apigee_proxy_without_engineer_header(self):
        principal = verify_cleanroom_token(
            x_apigee_proxy="MS-ADC-Metrology-v1",
            x_apigee_client_id="client-lam-etch-ch3",
            x_fab_tool_id="LAM-RIE-300"
        )
        self.assertEqual(principal, "apigee:LAM-RIE-300:client-lam-etch-ch3")

if __name__ == "__main__":
    unittest.main()
