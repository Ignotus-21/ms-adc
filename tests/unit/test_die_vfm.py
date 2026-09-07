import unittest
import tempfile
import os
from PIL import Image
from src.models.die_vfm import DieVFMClassifier, DIE_DEFECT_CLASSES
from src.models.export_tensorrt import TensorRTExporter


class TestDieVFM(unittest.TestCase):
    def setUp(self):
        self.classifier = DieVFMClassifier(num_classes=6, embedding_dim=512)
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_predict_logits_shape(self):
        img = Image.new("RGB", (224, 224), color=(100, 150, 200))
        logits = self.classifier.predict_logits(img)
        self.assertEqual(len(logits), 6)

    def test_classify_patch(self):
        img = Image.new("RGB", (224, 224), color=(50, 80, 120))
        pred = self.classifier.classify_patch(img)
        self.assertIn("predicted_class", pred)
        self.assertIn(pred["predicted_class"], DIE_DEFECT_CLASSES)
        self.assertGreaterEqual(pred["confidence"], 0.0)
        self.assertLessEqual(pred["confidence"], 1.0)
        self.assertEqual(len(pred["all_probabilities"]), 6)

    def test_extract_features(self):
        img = Image.new("RGB", (224, 224), color=(20, 40, 60))
        feats = self.classifier.extract_features(img)
        self.assertEqual(len(feats), 512)

    def test_save_and_load_checkpoint(self):
        ckpt_path = os.path.join(self.temp_dir.name, "test_checkpoint.pt")
        saved_path = self.classifier.save_checkpoint(ckpt_path, epoch=5, val_accuracy=98.5)
        self.assertTrue(os.path.exists(saved_path))

        new_classifier = DieVFMClassifier(num_classes=6, embedding_dim=512)
        meta = new_classifier.load_checkpoint(saved_path)
        self.assertEqual(meta.get("epoch"), 5)
        self.assertEqual(meta.get("val_accuracy"), 98.5)

    def test_tensorrt_exporter(self):
        exporter = TensorRTExporter(target_precision="FP16", max_batch_size=16)
        onnx_file = os.path.join(self.temp_dir.name, "model.onnx")
        engine_file = os.path.join(self.temp_dir.name, "model.engine")

        res_onnx = exporter.export_onnx(onnx_file)
        self.assertEqual(res_onnx["status"], "ONNX_EXPORT_FAILED")
        self.assertFalse(os.path.exists(onnx_file))

        res_trt = exporter.build_tensorrt_engine(onnx_file, engine_file)
        self.assertEqual(res_trt["status"], "TENSORRT_ENGINE_NOT_BUILT")
        self.assertFalse(os.path.exists(engine_file))
        self.assertIn("benchmarks", res_trt)


if __name__ == "__main__":
    unittest.main()
