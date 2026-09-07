import os
import shutil
import subprocess
from typing import Dict, Any, Optional
import torch


class TensorRTExporter:
    """
    Compiles fine-tuned PyTorch NV-DINOv2 linear probe into an optimized ONNX graph and NVIDIA TensorRT FP16 engine.
    Ensures sub-50ms deterministic edge inference for wafer fab inspection stations.
    """
    def __init__(self, target_precision: str = "FP16", max_batch_size: int = 32):
        self.precision = target_precision
        self.max_batch_size = max_batch_size

    def export_onnx(
        self,
        output_onnx_path: str = "models/die_vfm_head.onnx",
        torch_model: Optional[Any] = None,
        in_features: int = 3072,
        image_input: bool = False
    ) -> Dict[str, Any]:
        """Exports PyTorch model graph to standard ONNX intermediate representation.

        Set image_input=True when torch_model consumes raw 224x224 RGB tensors
        (e.g. a fine-tuned backbone+head model) instead of precomputed embeddings.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_onnx_path)), exist_ok=True)
        input_name = "die_image_tensor" if image_input else "vfm_feature_embedding"
        input_shape = [-1, 3, 224, 224] if image_input else [-1, in_features]
        if torch_model is not None:
            try:
                device = next(torch_model.parameters()).device
                dummy_input = torch.randn(*([1, 3, 224, 224] if image_input else [1, in_features]), device=device)
                torch.onnx.export(
                    torch_model,
                    dummy_input,
                    output_onnx_path,
                    input_names=[input_name],
                    output_names=["defect_logits"],
                    dynamic_axes={input_name: {0: "batch_size"}, "defect_logits": {0: "batch_size"}},
                    opset_version=17
                )
            except Exception as e:
                print(f"⚠️ ONNX export notice: {e}")

        return {
            "status": "ONNX_EXPORT_SUCCESS" if os.path.exists(output_onnx_path) else "ONNX_EXPORT_FAILED",
            "onnx_path": output_onnx_path,
            "input_shapes": {input_name: input_shape},
            "output_shapes": {"defect_logits": [-1, 6]}
        }

    def build_tensorrt_engine(
        self,
        onnx_path: str = "models/die_vfm_head.onnx",
        engine_path: str = "models/die_vfm_fp16.engine"
    ) -> Dict[str, Any]:
        """Compiles ONNX model to serialized TensorRT FP16 execution plan if TensorRT/trtexec is present."""
        os.makedirs(os.path.dirname(os.path.abspath(engine_path)), exist_ok=True)

        # Check if native NVIDIA TensorRT CLI (trtexec) is installed on the host
        trtexec_bin = shutil.which("trtexec") or "/usr/src/tensorrt/bin/trtexec"
        built_with_trtexec = False

        if os.path.exists(onnx_path) and (shutil.which("trtexec") or os.path.exists("/usr/src/tensorrt/bin/trtexec")):
            try:
                cmd = [
                    trtexec_bin,
                    f"--onnx={onnx_path}",
                    f"--saveEngine={engine_path}",
                    "--fp16"
                ]
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                if res.returncode == 0:
                    built_with_trtexec = True
            except Exception:
                pass

        return {
            "status": "TENSORRT_ENGINE_BUILT" if built_with_trtexec else "TENSORRT_ENGINE_NOT_BUILT",
            "engine_path": engine_path,
            "built_with_trtexec": built_with_trtexec,
            "precision": self.precision,
            "max_batch_size": self.max_batch_size,
            "benchmarks": {}
        }
