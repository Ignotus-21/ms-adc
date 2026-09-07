import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def run_cmd(cmd_list, check=True):
    print(f"📡 Executing: {' '.join(cmd_list)}", flush=True)
    res = subprocess.run(cmd_list, capture_output=True, text=True)
    if res.returncode != 0:
        # Always surface failures (e.g. IAM permission errors) even when check=False,
        # so a failed upload can never look identical to a successful one.
        print(f"❌ Command failed (exit {res.returncode}): {res.stderr.strip()}", flush=True)
        if check:
            raise RuntimeError(f"Command failed: {' '.join(cmd_list)}\n{res.stderr.strip()}")
    elif res.stdout.strip():
        print(f"   {res.stdout.strip()}", flush=True)
    return res


def upload_to_gcs(
    bucket_target: str = None,
    version: str = "v1.0.0",
    upload_dataset: bool = True
):
    """
    Uploads MS-ADC models, safetensors, evaluation graphs, metrics, logs,
    and datasets to GCS bucket organized into modular cleanroom directories.
    """
    print("=" * 80)
    print(f"☁️ MS-ADC Google Cloud Storage (GCS) Sync Engine")
    bucket_target = bucket_target or os.environ.get("GCS_ASSETS_BUCKET")
    if not bucket_target:
        raise ValueError("No GCS bucket provided: pass --bucket or set GCS_ASSETS_BUCKET in .env")
    print(f"🎯 Target Destination: {bucket_target}")
    print("=" * 80)

    bucket_target = bucket_target.rstrip("/")

    # 1. Upload Model Artifacts (PT, SafeTensors, TensorRT Engine, Metrics, Plots)
    model_dir = f"models/{version}"
    if os.path.exists(model_dir):
        print(f"\n[1/4] Uploading Versioned Model Artifacts ({version})...")
        gcs_models_dest = f"{bucket_target}/models/{version}/"
        run_cmd(["gcloud", "storage", "cp", "-r", f"{model_dir}/*", gcs_models_dest], check=False)
    else:
        print(f"⚠️ Model directory {model_dir} not found locally. Run training first.")

    # 2. Upload Evaluation Reports & Visual Metric Heatmaps
    print(f"\n[2/4] Uploading Yield Reports & Visual Graphs...")
    gcs_reports_dest = f"{bucket_target}/reports/{version}/"
    for report_file in ["confusion_matrix.png", "precision_recall_f1.png", "training_loss_curve.png", "metrics.json"]:
        local_f = os.path.join(model_dir, report_file)
        if os.path.exists(local_f):
            run_cmd(["gcloud", "storage", "cp", local_f, f"{gcs_reports_dest}{report_file}"], check=False)

    # 3. Upload Experiment Tracking Logs (TensorBoard & MLflow)
    print(f"\n[3/4] Uploading Experiment Tracking Logs (TensorBoard & MLflow)...")
    if os.path.exists("runs"):
        run_cmd(["gcloud", "storage", "cp", "-r", "runs/*", f"{bucket_target}/logs/tensorboard/"], check=False)
    if os.path.exists("mlflow.db"):
        run_cmd(["gcloud", "storage", "cp", "mlflow.db", f"{bucket_target}/logs/mlflow/mlflow.db"], check=False)

    # 4. Package and Upload Dataset (PCB Micrograph Defect Corpus)
    if upload_dataset:
        print(f"\n[4/4] Packaging & Uploading Training Dataset...")
        data_src = "data/pcb_dataset"
        data_zip = "data/pcb_dataset.zip"
        if os.path.exists(data_src) and not os.path.exists(data_zip):
            print(f"   Creating archive: {data_zip}...")
            with zipfile.ZipFile(data_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(data_src):
                    for file in files:
                        full_path = os.path.join(root, file)
                        rel_path = os.path.relpath(full_path, "data")
                        zipf.write(full_path, rel_path)
        if os.path.exists(data_zip):
            run_cmd(["gcloud", "storage", "cp", data_zip, f"{bucket_target}/datasets/pcb_dataset.zip"], check=False)
            print(f"   ✅ Dataset archive uploaded to: {bucket_target}/datasets/pcb_dataset.zip")
        else:
            print("   ℹ️ No local dataset archive found.")

    print("\n" + "=" * 80)
    print(f"🎉 GCS Upload Process Completed!")
    print(f"📁 Directory Layout on {bucket_target}/:")
    print(f"   • {bucket_target}/models/{version}/ (die_vfm_head.pt, die_vfm_head.safetensors, die_vfm_fp16.engine)")
    print(f"   • {bucket_target}/reports/{version}/ (confusion_matrix.png, metrics.json, loss_curves)")
    print(f"   • {bucket_target}/logs/ (TensorBoard runs & MLflow database)")
    print(f"   • {bucket_target}/datasets/ (pcb_dataset.zip)")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload MS-ADC artifacts and datasets to Google Cloud Storage")
    parser.add_argument("--bucket", type=str, default=None, help="GCS target destination URI (defaults to $GCS_ASSETS_BUCKET from .env)")
    parser.add_argument("--version", type=str, default="v1.0.0", help="Model version tag")
    parser.add_argument("--skip-dataset", action="store_true", help="Skip uploading the large dataset zip")

    args = parser.parse_args()
    upload_to_gcs(
        bucket_target=args.bucket,
        version=args.version,
        upload_dataset=(not args.skip_dataset)
    )
