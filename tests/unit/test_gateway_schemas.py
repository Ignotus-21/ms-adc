import unittest
from src.gateway.schemas import (
    InspectionRequest,
    LotInfo,
    ChamberType,
    MacroDefectType,
    MicroDefectType,
    InspectionResponse,
    ValidationError
)

class TestGatewaySchemasSuite(unittest.TestCase):
    def test_valid_inspection_request_parsing(self):
        payload = {
            "engineer_ticket": "Lot-123 failed metal-1 resistance test after Etch Chamber 3.",
            "lot_info": {
                "lot_id": "LOT-123",
                "chamber": "300mm_RIE_Etch_Chamber_3",
                "recipe_id": "RECIPE-OXIDE-ETCH-994",
                "images": [
                    "gs://semicon-raw-bucket/LOT-123/img_01.png",
                    "gs://semicon-raw-bucket/LOT-123/img_02.png"
                ]
            },
            "metadata": {"shift": "Day", "operator": "op_44"}
        }
        req = InspectionRequest(**payload)
        self.assertEqual(req.lot_info.lot_id, "LOT-123")
        self.assertEqual(req.lot_info.chamber, "300mm_RIE_Etch_Chamber_3")
        self.assertEqual(len(req.lot_info.images), 2)
        self.assertEqual(req.metadata["shift"], "Day")

    def test_missing_required_fields_raises_validation_error(self):
        # Missing engineer_ticket
        with self.assertRaises(Exception):
            InspectionRequest(
                engineer_ticket=None,
                lot_info={
                    "lot_id": "LOT-123",
                    "chamber": "300mm_RIE_Etch_Chamber_3"
                }
            )

        # Missing lot_id in lot_info
        with self.assertRaises(Exception):
            InspectionRequest(
                engineer_ticket="Test ticket",
                lot_info={
                    "lot_id": None,
                    "chamber": "300mm_RIE_Etch_Chamber_3"
                }
            )

    def test_empty_and_single_image_lists(self):
        req_empty = InspectionRequest(
            engineer_ticket="Test ticket with zero images",
            lot_info={"lot_id": "LOT-001", "chamber": "300mm_CMP_Platen_1", "images": []}
        )
        self.assertEqual(len(req_empty.lot_info.images), 0)

        req_single = InspectionRequest(
            engineer_ticket="Test ticket with 1 image",
            lot_info={"lot_id": "LOT-002", "chamber": "300mm_CMP_Platen_1", "images": ["gs://bucket/single.png"]}
        )
        self.assertEqual(len(req_single.lot_info.images), 1)

    def test_inspection_response_schema_validation(self):
        resp_data = {
            "lot_id": "LOT-123",
            "chamber": "300mm_RIE_Etch_Chamber_3",
            "macro_defect": "Center",
            "macro_confidence": 0.965,
            "defect_density_D0": 0.42,
            "die_yield_pct": 58.0,
            "micro_defect": "Short",
            "micro_confidence": 0.982,
            "fmea_citations": [{"doc_id": "FMEA-SOP-01", "section_title": "RF Diagnostics"}],
            "recommended_action": "Check RF Match Capacitor",
            "execution_latency_ms": 42.5,
            "audit_id": "INSP-A1B2C3D4",
            "circuit_breaker_status": "PRIMARY_SUCCESS"
        }
        resp = InspectionResponse(**resp_data)
        self.assertEqual(resp.audit_id, "INSP-A1B2C3D4")
        self.assertEqual(resp.macro_defect, "Center")
        self.assertEqual(resp.circuit_breaker_status, "PRIMARY_SUCCESS")

if __name__ == "__main__":
    unittest.main()
