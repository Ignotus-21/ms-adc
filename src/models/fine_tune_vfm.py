import argparse
import os
import sys
import time
import json
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image

from src.ingestion.dataset_loader import PCBDefectDatasetLoader, DIE_DEFECT_CLASSES
from src.ingestion.augmentor import MetrologyAugmentor
from src.models.die_vfm import build_linear_probe, DieVFMModel
from src.models.export_tensorrt import TensorRTExporter
from src.utils.metrics import SemiconductorYieldCalculator


class MicrographDataset(torch.utils.data.Dataset):
    """Loads defect micrographs on demand for end-to-end backbone fine-tuning."""
    def __init__(self, split_dict: Dict[str, List[Path]], loader: PCBDefectDatasetLoader, transform: Any):
        self.samples: List[Tuple[Path, int]] = [
            (path, cls_idx)
            for cls_idx, cls_name in enumerate(DIE_DEFECT_CLASSES)
            for path in split_dict.get(cls_name, [])
        ]
        self.loader = loader
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        image = self.loader.load_and_preprocess_image(path)
        return self.transform(image), label


@torch.no_grad()
def evaluate_full_model(model: nn.Module, dataset: MicrographDataset, device: torch.device, batch_size: int) -> Tuple[List[int], List[int], float]:
    """Runs a full forward pass over a dataset and returns predictions plus average loss."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    y_true: List[int] = []
    y_pred: List[int] = []
    total_loss = 0.0
    total_count = 0
    for bx, by in loader:
        bx, by = bx.to(device), by.to(device)
        logits = model(bx)
        total_loss += nn.functional.cross_entropy(logits, by, reduction="sum").item()
        total_count += by.size(0)
        y_true.extend(by.cpu().tolist())
        y_pred.extend(logits.argmax(1).cpu().tolist())
    avg_loss = total_loss / max(1, total_count)
    return y_true, y_pred, avg_loss


def extract_dataset_features(
    backbone: nn.Module,
    split_dict: Dict[str, List[Path]],
    loader: PCBDefectDatasetLoader,
    augmentor: MetrologyAugmentor,
    eval_transform: Any,
    device: torch.device,
    num_aug: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extracts 768-dimensional visual token representations from frozen DINOv2 ViT-B/14.
    Supports cleanroom optical augmentations (orthogonal rotations, axial flips) for support set.
    """
    backbone.eval()
    features_list = []
    labels_list = []

    for cls_idx, cls_name in enumerate(DIE_DEFECT_CLASSES):
        paths = split_dict.get(cls_name, [])
        for p in paths:
            img = loader.load_and_preprocess_image(p)
            imgs = [img]
            for _ in range(num_aug):
                imgs.append(augmentor.augment_pil_image(img))

            for item in imgs:
                tensor = eval_transform(item).unsqueeze(0).to(device)
                with torch.no_grad():
                    f = backbone(tensor)
                features_list.append(f.cpu())
                labels_list.append(cls_idx)

    if not features_list:
        return torch.empty(0), torch.empty(0, dtype=torch.long)

    return torch.cat(features_list, dim=0), torch.tensor(labels_list, dtype=torch.long)


def fine_tune_vfm(
    version: str = "v1.0.0",
    data_path: str = "data/pcb_dataset",
    output_dir: str = "models",
    backbone_name: str = "dinov2_vitb14",
    epochs: int = 25,
    k_shot: int = 10,
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    num_aug: int = 7,
    unfreeze_blocks: int = 0,
    backbone_lr: float = 1e-5
) -> Dict[str, Any]:
    """
    Executes real few-shot training of the Vision Foundation Model (DINOv2 ViT-B/14).
    When unfreeze_blocks=0 (default), trains a linear probe on frozen DINOv2 features
    (fast: features are extracted once). When unfreeze_blocks>0, jointly fine-tunes the
    final N transformer blocks end-to-end with a lower backbone learning rate.
    Evaluates on the real held-out test split, exports SafeTensors & ONNX artifacts,
    and logs telemetry to MLflow.
    """
    start_time = time.time()
    version_output_dir = os.path.join(output_dir, version)
    os.makedirs(version_output_dir, exist_ok=True)

    print("=" * 88)
    print(f"🔬 MS-ADC: Vision Foundation Model (NV-DINOv2 / {backbone_name}) Pipeline [{version}]")
    print(f"🎯 Target DoD: Die-Level Defect Classification on Real Micrographs")
    print("=" * 88)

    # 1. Hardware Detection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("⚙️ Compute Device: Apple Silicon GPU (MPS Acceleration)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"⚙️ Compute Device: NVIDIA GPU ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print("⚙️ Compute Device: CPU")

    # 2. Tracking Setup
    tb_logdir = f"runs/{version}"
    tb_writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir=tb_logdir)
        print(f"📈 TensorBoard Logging enabled: {tb_logdir}")
    except Exception:
        pass

    mlflow_active = False
    try:
        import mlflow
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("ms-adc-die-vfm")
        mlflow.start_run(run_name=f"vfm-{version}")
        mlflow.log_params({
            "version": version,
            "backbone": f"NV-DINOv2 ({backbone_name})",
            "epochs": epochs,
            "k_shot": k_shot,
            "lr": learning_rate,
            "batch_size": batch_size,
            "num_augmentations": num_aug,
            "unfreeze_blocks": unfreeze_blocks,
            "backbone_lr": backbone_lr,
            "device": str(device)
        })
        mlflow_active = True
        print(f"📋 MLflow Experiment Tracking active in sqlite:///mlflow.db: ms-adc-die-vfm (run: vfm-{version})")
    except Exception:
        pass

    # 3. Ingest Dataset & Partition Few-Shot Splits
    print(f"\n[1/5] Ingesting real defect micrographs from: {data_path}")
    loader = PCBDefectDatasetLoader(data_dir=Path(data_path))
    discovered = loader.discover_image_files()
    total_found = sum(len(v) for v in discovered.values())
    print(f"      Discovered {total_found} optical patches across {len(DIE_DEFECT_CLASSES)} defect classes:")
    for cls_name in DIE_DEFECT_CLASSES:
        print(f"      • {cls_name:<18}: {len(discovered.get(cls_name, []))} samples")

    train_split, val_split, test_split = loader.get_stratified_split(k_shot_train=k_shot, val_ratio=0.2)
    print(f"\n[2/5] Partitioned Dataset:")
    print(f"      • Train Support Set (K={k_shot}) : {sum(len(v) for v in train_split.values())} samples")
    print(f"      • Validation Set (20%)   : {sum(len(v) for v in val_split.values())} samples")
    print(f"      • Held-out Test Set        : {sum(len(v) for v in test_split.values())} samples")

    insufficient_classes = [
        class_name for class_name in DIE_DEFECT_CLASSES
        if not train_split[class_name] or not val_split[class_name] or not test_split[class_name]
    ]
    if total_found == 0 or insufficient_classes:
        raise ValueError(
            "Real training requires at least one train, validation, and test image per class; "
            f"insufficient classes: {', '.join(insufficient_classes or DIE_DEFECT_CLASSES)}"
        )

    # 4. Load Frozen DINOv2 Backbone
    print(f"\n[3/5] Loading Vision Foundation Model ({backbone_name})...")
    embed_dim = 768
    epoch_history: List[Dict[str, Any]] = []

    if unfreeze_blocks == 0:
        # --- Frozen-backbone linear probe: extract features once (fast) ---
        try:
            backbone = torch.hub.load("facebookresearch/dinov2", backbone_name)
        except Exception:
            backbone = torch.hub.load("facebookresearch/dinov2", backbone_name, source="local")

        backbone.to(device)
        backbone.eval()
        for param in backbone.parameters():
            param.requires_grad = False

        embed_dim = getattr(backbone, "embed_dim", 768)
        augmentor = MetrologyAugmentor(target_size=224)
        eval_transform = augmentor.get_torch_eval_transform()

        print(f"      Extracting real {embed_dim}-dim representations from defect images...")
        train_x, train_y = extract_dataset_features(backbone, train_split, loader, augmentor, eval_transform, device, num_aug=num_aug)
        val_x, val_y = extract_dataset_features(backbone, val_split, loader, augmentor, eval_transform, device, num_aug=0)
        test_x, test_y = extract_dataset_features(backbone, test_split, loader, augmentor, eval_transform, device, num_aug=0)

        print(f"      • Train Features Tensor: {train_x.shape}")
        print(f"      • Val Features Tensor  : {val_x.shape}")
        print(f"      • Test Features Tensor : {test_x.shape}")

        # 5. Build Few-Shot Linear Probe Head
        probe_head = build_linear_probe(embed_dim, len(DIE_DEFECT_CLASSES)).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(probe_head.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        train_dataset = torch.utils.data.TensorDataset(train_x.to(device), train_y.to(device))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        print(f"\n[4/5] Starting Multi-Epoch Linear Probe Training ({epochs} epochs)...")
        print("-" * 88)
        print(f"{'Epoch':<10} | {'Train Loss':<12} | {'Train Acc':<11} | {'Val Loss':<11} | {'Val Acc':<10} | {'Status'}")
        print("-" * 88)

        best_val_acc = 0.0
        best_epoch = 1
        best_head_state = None

        for epoch in range(1, epochs + 1):
            probe_head.train()
            running_loss = 0.0
            train_correct = 0
            train_total = 0

            for bx, by in train_loader:
                optimizer.zero_grad()
                logits = probe_head(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * bx.size(0)
                train_correct += logits.argmax(1).eq(by).sum().item()
                train_total += bx.size(0)

            scheduler.step()

            train_loss = running_loss / max(1, train_total)
            train_acc = (train_correct / max(1, train_total)) * 100.0

            # Evaluate on validation split
            probe_head.eval()
            with torch.no_grad():
                val_logits = probe_head(val_x.to(device))
                val_loss = criterion(val_logits, val_y.to(device)).item()
                val_acc = (val_logits.argmax(1).eq(val_y.to(device)).sum().item() / max(1, len(val_y))) * 100.0

            status_flag = ""
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                best_head_state = {k: v.cpu().clone() for k, v in probe_head.state_dict().items()}
                status_flag = "⭐ Best Model"

            epoch_history.append({
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_acc": round(train_acc, 2),
                "val_loss": round(val_loss, 4),
                "val_acc": round(val_acc, 2),
                "lr": round(scheduler.get_last_lr()[0], 6)
            })

            print(f"Epoch [{epoch:02d}/{epochs:02d}] | {train_loss:<12.4f} | {train_acc:<5.1f}     % | {val_loss:<11.4f} | {val_acc:<5.1f}    % | {status_flag}")

            if tb_writer:
                tb_writer.add_scalar("Loss/Train", train_loss, epoch)
                tb_writer.add_scalar("Loss/Validation", val_loss, epoch)
                tb_writer.add_scalar("Accuracy/Train", train_acc, epoch)
                tb_writer.add_scalar("Accuracy/Validation", val_acc, epoch)

            if mlflow_active:
                try:
                    import mlflow
                    mlflow.log_metrics({
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "train_acc": train_acc,
                        "val_acc": val_acc
                    }, step=epoch)
                except Exception:
                    pass

        print("-" * 88)
        print(f"✅ Training completed! Best Validation Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")

        if best_head_state is not None:
            probe_head.load_state_dict({k: v.to(device) for k, v in best_head_state.items()})

        # 6. Evaluate on Held-Out Test Set (Real unseen micrographs)
        print(f"\n[5/5] Evaluating on Held-Out Test Set ({len(test_y)} unseen micrographs)...", flush=True)
        probe_head.eval()
        with torch.no_grad():
            test_logits = probe_head(test_x.to(device))
            test_preds = test_logits.argmax(1).cpu().tolist()
            test_y_list = test_y.tolist()

        model_for_export: nn.Module = probe_head
        export_image_input = False
        checkpoint_payload = {
            "epoch": best_epoch,
            "val_accuracy": best_val_acc,
            "backbone": backbone_name,
            "embed_dim": embed_dim,
            "unfreeze_blocks": 0,
            "head_state_dict": probe_head.state_dict(),
            "classes": DIE_DEFECT_CLASSES
        }

    else:
        # --- Real fine-tuning: unfreeze the final N transformer blocks ---
        model = DieVFMModel(backbone_name=backbone_name, num_classes=len(DIE_DEFECT_CLASSES), unfreeze_blocks=unfreeze_blocks).to(device)
        embed_dim = model.embed_dim
        augmentor = MetrologyAugmentor(target_size=224)
        train_transform = augmentor.get_torch_train_transform()
        eval_transform = augmentor.get_torch_eval_transform()

        train_dataset = MicrographDataset(train_split, loader, train_transform)
        val_dataset = MicrographDataset(val_split, loader, eval_transform)
        test_dataset = MicrographDataset(test_split, loader, eval_transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        print(f"      • Train Images: {len(train_dataset)} | Val Images: {len(val_dataset)} | Test Images: {len(test_dataset)}")
        print(f"      • Unfreezing final {unfreeze_blocks} DINOv2 transformer block(s) (backbone_lr={backbone_lr})")

        criterion = nn.CrossEntropyLoss()
        unfrozen_backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
        optimizer = optim.AdamW([
            {"params": model.head.parameters(), "lr": learning_rate},
            {"params": unfrozen_backbone_params, "lr": backbone_lr}
        ], weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        print(f"\n[4/5] Starting Multi-Epoch VFM Fine-Tuning ({epochs} epochs)...")
        print("-" * 88)
        print(f"{'Epoch':<10} | {'Train Loss':<12} | {'Train Acc':<11} | {'Val Loss':<11} | {'Val Acc':<10} | {'Status'}")
        print("-" * 88)

        best_val_acc = 0.0
        best_epoch = 1
        best_model_state = None

        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            train_correct = 0
            train_total = 0

            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                logits = model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * bx.size(0)
                train_correct += logits.argmax(1).eq(by).sum().item()
                train_total += bx.size(0)

            scheduler.step()

            train_loss = running_loss / max(1, train_total)
            train_acc = (train_correct / max(1, train_total)) * 100.0

            _, val_preds, val_loss = evaluate_full_model(model, val_dataset, device, batch_size)
            val_labels = [label for _, label in val_dataset.samples]
            val_acc = (sum(1 for t, p in zip(val_labels, val_preds) if t == p) / max(1, len(val_labels))) * 100.0

            status_flag = ""
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                status_flag = "⭐ Best Model"

            epoch_history.append({
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_acc": round(train_acc, 2),
                "val_loss": round(val_loss, 4),
                "val_acc": round(val_acc, 2),
                "lr": round(scheduler.get_last_lr()[0], 6)
            })

            print(f"Epoch [{epoch:02d}/{epochs:02d}] | {train_loss:<12.4f} | {train_acc:<5.1f}     % | {val_loss:<11.4f} | {val_acc:<5.1f}    % | {status_flag}")

            if tb_writer:
                tb_writer.add_scalar("Loss/Train", train_loss, epoch)
                tb_writer.add_scalar("Loss/Validation", val_loss, epoch)
                tb_writer.add_scalar("Accuracy/Train", train_acc, epoch)
                tb_writer.add_scalar("Accuracy/Validation", val_acc, epoch)

            if mlflow_active:
                try:
                    import mlflow
                    mlflow.log_metrics({
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "train_acc": train_acc,
                        "val_acc": val_acc
                    }, step=epoch)
                except Exception:
                    pass

        print("-" * 88)
        print(f"✅ Training completed! Best Validation Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")

        if best_model_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

        # 6. Evaluate on Held-Out Test Set (Real unseen micrographs)
        print(f"\n[5/5] Evaluating on Held-Out Test Set ({len(test_dataset)} unseen micrographs)...", flush=True)
        test_y_list, test_preds, _ = evaluate_full_model(model, test_dataset, device, batch_size)

        model_for_export = model
        export_image_input = True
        checkpoint_payload = {
            "epoch": best_epoch,
            "val_accuracy": best_val_acc,
            "backbone": backbone_name,
            "embed_dim": embed_dim,
            "unfreeze_blocks": unfreeze_blocks,
            "model_state_dict": model.state_dict(),
            "head_state_dict": model.head.state_dict(),
            "classes": DIE_DEFECT_CLASSES
        }

    test_metrics = SemiconductorYieldCalculator.calculate_classification_metrics(
        y_true=test_y_list,
        y_pred=test_preds,
        class_names=DIE_DEFECT_CLASSES
    )

    print("\n" + SemiconductorYieldCalculator.format_classification_report(test_metrics), flush=True)

    # 7. Save Real Checkpoints, SafeTensors, and ONNX
    head_pt_path = os.path.join(version_output_dir, "die_vfm_head.pt")
    head_safetensors_path = os.path.join(version_output_dir, "die_vfm_head.safetensors")
    best_ckpt_path = os.path.join(version_output_dir, "checkpoint_best.pt")

    checkpoint_payload["test_accuracy"] = test_metrics["accuracy"]
    torch.save(checkpoint_payload, best_ckpt_path)
    shutil.copy2(best_ckpt_path, head_pt_path)

    try:
        from safetensors.torch import save_file
        head_state = checkpoint_payload["head_state_dict"]
        save_file({f"head_{k}": v.contiguous() for k, v in head_state.items()}, head_safetensors_path)
    except Exception:
        pass

    # Export ONNX representation for TensorRT compilation
    exporter = TensorRTExporter(target_precision="FP16", max_batch_size=32)
    onnx_path = os.path.join(version_output_dir, "die_vfm_head.onnx")
    exporter.export_onnx(onnx_path, torch_model=model_for_export, in_features=embed_dim, image_input=export_image_input)
    engine_path = os.path.join(version_output_dir, "die_vfm_fp16.engine")
    trt_meta = exporter.build_tensorrt_engine(onnx_path, engine_path)

    # 8. Save Visual Evaluation Charts
    cm_path = os.path.join(version_output_dir, "confusion_matrix.png")
    curve_path = os.path.join(version_output_dir, "training_loss_curve.png")
    prf1_path = os.path.join(version_output_dir, "precision_recall_f1.png")

    SemiconductorYieldCalculator.save_confusion_matrix_plot(
        cm=test_metrics["confusion_matrix"],
        class_names=DIE_DEFECT_CLASSES,
        output_path=cm_path
    )
    SemiconductorYieldCalculator.save_loss_accuracy_curves(
        history=epoch_history,
        output_path=curve_path
    )
    SemiconductorYieldCalculator.save_precision_recall_f1_chart(
        class_metrics=test_metrics["classes"],
        output_path=prf1_path
    )

    # 9. Save Structured Metrics JSON
    metrics_json_path = os.path.join(version_output_dir, "metrics.json")
    metrics_payload = {
        "version": version,
        "backbone": f"NV-DINOv2 / {backbone_name}",
        "unfreeze_blocks": unfreeze_blocks,
        "accuracy": test_metrics["accuracy"],
        "loss": epoch_history[-1]["val_loss"],
        "macro_f1": test_metrics["macro_f1"],
        "macro_precision": test_metrics["macro_precision"],
        "macro_recall": test_metrics["macro_recall"],
        "epochs": epochs,
        "k_shot": k_shot,
        "best_epoch": best_epoch,
        "test_samples": len(test_y_list),
        "tensorrt_benchmarks": trt_meta.get("benchmarks", {}),
        "definition_of_done_met": (test_metrics["accuracy"] >= 95.0),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
    }

    with open(metrics_json_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)

    # Mirroring to models/ root
    shutil.copy2(head_pt_path, os.path.join(output_dir, "die_vfm_head.pt"))
    if os.path.exists(head_safetensors_path):
        shutil.copy2(head_safetensors_path, os.path.join(output_dir, "die_vfm_head.safetensors"))
    shutil.copy2(metrics_json_path, os.path.join(output_dir, "metrics.json"))
    shutil.copy2(curve_path, os.path.join(output_dir, "training_loss_curve.png"))

    if tb_writer:
        tb_writer.flush()
        tb_writer.close()

    if mlflow_active:
        try:
            import mlflow
            mlflow.log_metrics({
                "final_test_accuracy": test_metrics["accuracy"],
                "final_macro_f1": test_metrics["macro_f1"]
            })
            mlflow.log_artifact(cm_path)
            mlflow.log_artifact(curve_path)
            mlflow.log_artifact(metrics_json_path)
            mlflow.end_run()
        except Exception:
            pass

    elapsed = time.time() - start_time
    print("=" * 88)
    print(f"🏆 Final Verification: Real DINOv2 Test Accuracy = {test_metrics['accuracy']:.2f}%")
    print(f"📦 Versioned Artifacts saved in: {version_output_dir}/")
    print(f"   • {head_pt_path}")
    print(f"   • {head_safetensors_path}")
    print(f"   • {metrics_json_path}")
    print(f"   • {curve_path}")
    print(f"⏱️ Total Execution Time: {elapsed:.2f}s")
    print("=" * 88)

    test_metrics["version"] = version
    test_metrics["version_output_dir"] = version_output_dir
    return test_metrics


def run_training_pipeline(
    version: str = "v1.0.0",
    epochs: int = 25,
    k_shot: int = 10,
    learning_rate: float = 1e-3,
    output_dir: str = "models",
    data_dir_path: str = "data/pcb_dataset",
    use_tracking: bool = True,
    unfreeze_blocks: int = 0,
    backbone_lr: float = 1e-5
) -> Dict[str, Any]:
    """Compatibility entrypoint for running the end-to-end VFM training pipeline."""
    return fine_tune_vfm(
        version=version,
        data_path=data_dir_path,
        output_dir=output_dir,
        epochs=epochs,
        k_shot=k_shot,
        learning_rate=learning_rate,
        unfreeze_blocks=unfreeze_blocks,
        backbone_lr=backbone_lr
    )


def run_progression_experiment():
    """Runs 4 distinct experimental configurations to log progression telemetry into MLflow."""
    configs = [
        ("v0.1.0-raw-baseline", 5, 2, 0, 5e-3),
        ("v0.2.0-unfreeze-backbone", 10, 5, 2, 2e-3),
        ("v0.3.0-cleanroom-augmented", 15, 10, 4, 1.5e-3),
        ("v1.0.0-final-vfm", 25, 10, 7, 1e-3),
    ]
    for version, ep, k, n_aug, lr in configs:
        fine_tune_vfm(version=version, epochs=ep, k_shot=k, num_aug=n_aug, learning_rate=lr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Vision Foundation Model for Die Metrology")
    parser.add_argument("--version", type=str, default="v1.0.0", help="Model version tag")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--k-shot", type=int, default=10, help="Support samples per defect class")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num-aug", type=int, default=7, help="Cleanroom optical augmentations per sample (linear-probe mode only)")
    parser.add_argument("--unfreeze-blocks", type=int, default=0, help="Number of final DINOv2 transformer blocks to jointly fine-tune (0 = frozen linear probe)")
    parser.add_argument("--backbone-lr", type=float, default=1e-5, help="Learning rate for unfrozen backbone blocks")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--backbone", type=str, default="dinov2_vitb14", help="DINOv2 backbone version")
    parser.add_argument("--data-path", type=str, default="data/pcb_dataset", help="Path to defect dataset")
    parser.add_argument("--progression", action="store_true", help="Run 4-stage experimental progression")

    args = parser.parse_args()

    if args.progression:
        run_progression_experiment()
    else:
        fine_tune_vfm(
            version=args.version,
            epochs=args.epochs,
            k_shot=args.k_shot,
            learning_rate=args.lr,
            num_aug=args.num_aug,
            data_path=args.data_path,
            unfreeze_blocks=args.unfreeze_blocks,
            backbone_lr=args.backbone_lr,
            batch_size=args.batch_size,
            backbone_name=args.backbone
        )
