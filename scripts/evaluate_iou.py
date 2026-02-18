import rasterio
import numpy as np
import sys
from rasterio.warp import reproject, Resampling

YEAR = sys.argv[1] if len(sys.argv) > 1 else "2025"

PRED = f"outputs/unet/{YEAR}/chennai_builtup_unet_{YEAR}.tif"
LABEL = "data/raw/training/builtup_labels.tif"

print("Evaluating year:", YEAR)

# -----------------------------
# Load prediction
# -----------------------------
with rasterio.open(PRED) as src:
    pred = src.read(1)
    meta = src.meta
    H, W = pred.shape

# -----------------------------
# Align ground truth
# -----------------------------
with rasterio.open(LABEL) as gt_src:

    gt_aligned = np.zeros((H, W), dtype="uint8")

    reproject(
        source=gt_src.read(1),
        destination=gt_aligned,
        src_transform=gt_src.transform,
        src_crs=gt_src.crs,
        dst_transform=meta["transform"],
        dst_crs=meta["crs"],
        resampling=Resampling.nearest
    )

gt_bin = gt_aligned > 0

# -----------------------------
# Choose best threshold (from earlier)
# -----------------------------
threshold = 0.35
pred_bin = pred > threshold

# -----------------------------
# Metrics
# -----------------------------
TP = np.logical_and(pred_bin, gt_bin).sum()
FP = np.logical_and(pred_bin, ~gt_bin).sum()
FN = np.logical_and(~pred_bin, gt_bin).sum()

precision = TP / (TP + FP + 1e-6)
recall = TP / (TP + FN + 1e-6)
f1 = 2 * precision * recall / (precision + recall + 1e-6)
iou = TP / (TP + FP + FN + 1e-6)

print("\n===== METRICS =====")
print(f"Threshold: {threshold}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"IoU:       {iou:.4f}")