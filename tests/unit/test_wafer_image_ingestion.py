import unittest
import os
import tempfile
import math

from src.ingestion.image_utils import write_bmp, read_image_pixels
from src.models.wafer_vlm import WaferVLMClassifier

class TestWaferImageIngestion(unittest.TestCase):
    def setUp(self):
        os.environ["USE_GEMINI_VLM"] = "false"
        self.classifier = WaferVLMClassifier()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_img_path = os.path.join(self.temp_dir.name, "unit_test_wafer.bmp")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_real_image_pixel_decoding_to_matrix(self):
        # 1. Create a synthetic test image with known pixel coordinates
        size = 52
        radius = 26.0
        pixels = [[(10, 10, 10) for _ in range(size)] for _ in range(size)]
        
        red_pixels = [(26, 26), (26, 27), (26, 28), (27, 26), (28, 26)]
        for y in range(size):
            for x in range(size):
                dist = math.sqrt((x - radius)**2 + (y - radius)**2) / radius
                if dist <= 1.0:
                    pixels[y][x] = (40, 180, 40) # Green pass
                    
        for y, x in red_pixels:
            pixels[y][x] = (255, 0, 0) # Red fail

        write_bmp(self.test_img_path, pixels)
        self.assertTrue(os.path.exists(self.test_img_path))

        # 2. Ingest real image file and decode to 0/1/2 matrix
        matrix, meta = self.classifier.load_wafer_matrix(self.test_img_path)
        self.assertEqual(meta["source_type"], "DECODED_IMAGE_FILE")
        self.assertEqual(len(matrix), 52)
        self.assertEqual(len(matrix[0]), 52)

        # 3. Assert decoded matrix matches pixel values
        for y, x in red_pixels:
            self.assertEqual(matrix[y][x], 2)

        failing_count = sum(row.count(2) for row in matrix)
        self.assertEqual(failing_count, len(red_pixels))

        # 4. Classify and assert computed spatial metrology
        res = self.classifier.classify("300mm_RIE_Etch_Chamber_3", self.test_img_path)
        self.assertEqual(res["failing_die_count"], len(red_pixels))
        self.assertEqual(res["macro_defect"], "Center")
        self.assertGreater(res["macro_confidence"], 0.90)

if __name__ == "__main__":
    unittest.main()
