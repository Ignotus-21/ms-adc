import unittest
import tempfile
import os
import glob
from pathlib import Path

from src.models.fine_tune_vfm import run_training_pipeline


class TestExperimentTracking(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.models_dir = os.path.join(self.temp_dir.name, "models")
        self.data_dir = os.path.join(self.temp_dir.name, "pcb_data")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_tensorboard_and_mlflow_integration_requires_real_data(self):
        with self.assertRaisesRegex(ValueError, "Real training requires"):
            run_training_pipeline(
                version="v1.0.0-track-test",
                epochs=2,
                output_dir=self.models_dir,
                data_dir_path=self.data_dir,
                use_tracking=True
            )


if __name__ == "__main__":
    unittest.main()
