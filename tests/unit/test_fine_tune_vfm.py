import unittest
import tempfile
import os
import json
from PIL import Image
from pathlib import Path

from src.models.fine_tune_vfm import run_training_pipeline
from src.utils.metrics import SemiconductorYieldCalculator


class TestFineTuneVFM(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.models_dir = os.path.join(self.temp_dir.name, "models")
        self.data_dir = os.path.join(self.temp_dir.name, "pcb_data")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_yield_calculator(self):
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 0, 1, 1]
        classes = ["c0", "c1", "c2"]
        metrics = SemiconductorYieldCalculator.calculate_classification_metrics(y_true, y_pred, classes)
        self.assertAlmostEqual(metrics["accuracy"], 5/6 * 100.0, places=1)
        self.assertIn("classes", metrics)
        report = SemiconductorYieldCalculator.format_classification_report(metrics)
        self.assertIn("Overall Accuracy", report)

    def test_run_training_pipeline_rejects_missing_dataset(self):
        with self.assertRaisesRegex(ValueError, "Real training requires"):
            run_training_pipeline(
                version="v1.0.0-test",
                epochs=2,
                output_dir=self.models_dir,
                data_dir_path=self.data_dir
            )


if __name__ == "__main__":
    unittest.main()
