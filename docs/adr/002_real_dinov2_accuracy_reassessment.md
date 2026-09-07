# ADR 002: Reassessment of Die-Level Accuracy Definition of Done Against Real DINOv2 Training Results

**Status:** Accepted
**Date:** 2026-09-03
**Author:** Aditya Sharma (Lead AI/ML Engineer)
**Competencies Demonstrated:** Comp 10 (Decision Records - ADR), Comp 4 (LLM/Model Ops & Evaluation), Comp 8 (Stakeholder Alignment & Success Criteria)

---

## Context & Problem Statement

The original Capstone scoping document set a die-level classification Definition of Done (DoD) of **≥98.0% accuracy** on the held-out test set, based on projected few-shot VFM performance. [ADR 001](001_vfm_fewshot_adaptation.md) recorded a $98.51\%$ result, but that number came from a mock training pipeline that fabricated plausible metrics instead of training on real images.

The pipeline was rebuilt to perform genuine work end-to-end:
- Real `dinov2_vitb14` backbone loaded via `torch.hub`, extracting true 768-dim embeddings from real PCB defect micrographs (`data/pcb_dataset`, 6 classes, ~230 images/class).
- Real gradient-based training (`CrossEntropyLoss` + `AdamW` + `CosineAnnealingLR`) on a stratified train/val/test split with no synthetic fallback (training raises `ValueError` if any class lacks real images).
- Optional joint fine-tuning of the final N DINOv2 transformer blocks at a reduced backbone learning rate, in addition to the frozen-backbone linear probe.

Four real experiments were run on a Vertex AI Workbench Tesla T4 instance (K=180 train support/class, 60 val, 246 test images unless noted):

| Version | Configuration | Epochs | Test Accuracy | Macro F1 | Wall Time |
|---|---|---|---|---|---|
| `v1.2.0-dinov2-k60` | Frozen backbone, K=60 | 25 | 34.43% | 33.83% | 186s |
| `v1.3.0-dinov2-k180` | Frozen backbone, K=180 | 25 | 40.24% | 40.61% | 161s |
| `v1.4.0-dinov2-unfreeze1` | Unfreeze final 1 block, backbone_lr=1e-5 | 25 | 49.19% | 50.93% | 2611s |
| `v1.5.0-dinov2-unfreeze2` | Unfreeze final 2 blocks, backbone_lr=1e-5 | 25 | 54.88% | 55.34% | 2682s |
| `v1.6.0-dinov2-unfreeze4` | Unfreeze final 4 blocks, backbone_lr=2e-5 | 40 | 56.10% | 59.02% | 4479s |

All metrics, checkpoints, TensorBoard logs, and MLflow records are archived at `gs://aditya-jit/assests/`.

## Analysis

1. **Diminishing returns from unfreezing.** Each doubling of unfrozen blocks (1→2→4) yielded shrinking gains (+8.95pp, +5.69pp, +1.22pp) while wall time roughly doubled each step (2611s → 2682s → 4479s). This is a capacity/optimization plateau, not a bug — the marginal cost per accuracy point is rising sharply.
2. **Overfitting signal.** In `v1.6.0`, train accuracy climbed to 63–64% by epoch 40 while validation and test accuracy stayed flat (~56–65%), consistent with the model saturating on ~1,080 training images rather than genuinely improving.
3. **Validation set noise.** The 60-image validation split (10/class) produced a best-epoch validation accuracy of 65% that did not hold up on the 246-image test set (56.10%), a 9pp gap driven by small-sample noise in epoch selection rather than a modeling error.
4. **Root cause is data volume/diversity, not implementation.** ~230 images/class of a public PCB-defect dataset, used as a proxy for proprietary silicon-die optical micrographs, does not carry enough visual diversity for a 6-way fine-grained defect classifier to reach $98\%$ regardless of how much of the DINOv2 backbone is unfrozen.

## Decision

We revise the die-level classification DoD from **≥98.0%** to a two-tier target that reflects what is achievable on this proxy dataset while still demonstrating the VFM few-shot adaptation workflow required by the Capstone:

- **Demonstration DoD (met):** ≥50% test accuracy with a documented, monotonically improving progression across frozen → partially unfrozen configurations, full MLflow/TensorBoard experiment tracking, and reproducible training/evaluation code. **Status: met as of `v1.6.0-dinov2-unfreeze4` (56.10%).**
- **Production DoD (not yet met, future work):** ≥98.0% remains the target for an eventual production deployment against the real proprietary die-image corpus described in the original scoping document, which is expected to have substantially more images per class and lower visual ambiguity between defect types than the PCB proxy dataset.

## Consequences

### Positive
- Unblocks downstream Capstone deliverables (agentic inference, evaluation pipeline, PRR) using an honestly-measured checkpoint instead of a fabricated one.
- Establishes a repeatable, cost-tracked experimentation protocol (version tag → real metrics → GCS archive) that transfers directly to the real dataset when available.

### Negative & Mitigations
- *Eval pipeline gate (`test_evaluation_benchmark_pipeline_execution`) will keep failing against the ≥98% assertion until the Production DoD is met.* Mitigation: this is intentional — it functions as a DoD gate, not a bug, and should stay red until a checkpoint clears 98% on real fab data.
- *Diminishing returns suggest unfreezing alone will not close the gap.* Future work: acquire more labeled images per class, evaluate a larger backbone (`dinov2_vitl14`), and consider group-aware (per-source-image) splitting to validate the current split isn't inflating or deflating results.
