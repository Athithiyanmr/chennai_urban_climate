import argparse
import rasterio
import numpy as np
from pathlib import Path
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt


# -------------------------------------------------
# ARGUMENTS
# -------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--year", required=True)
parser.add_argument("--aoi", required=True)
parser.add_argument("--threshold", type=float, default=None,
                    help="Optional threshold. If not given → auto search")
args = parser.parse_args()

YEAR = args.year
AOI = args.aoi
THRESHOLD = args.threshold

PRED = Path(f"outputs/unet/{YEAR}/builtup_prob_{YEAR}_{AOI}.tif")
LABEL = Path(f"data/raw/training/labels_google_{YEAR}_{AOI}.tif")

if not PRED.exists():
    raise FileNotFoundError(PRED)

if not LABEL.exists():
    raise FileNotFoundError(LABEL)

print(f"\nEvaluating: {YEAR} | {AOI}")


# -------------------------------------------------
# LOAD PREDICTION
# -------------------------------------------------
with rasterio.open(PRED) as src:
    pred = src.read(1)
    meta = src.meta
    H, W = pred.shape


# -------------------------------------------------
# ALIGN GROUND TRUTH
# -------------------------------------------------
with rasterio.open(LABEL) as gt_src:

    gt_aligned = np.zeros((H, W), dtype="uint8")

    reproject(
        gt_src.read(1),
        gt_aligned,
        src_transform=gt_src.transform,
        src_crs=gt_src.crs,
        dst_transform=meta["transform"],
        dst_crs=meta["crs"],
        resampling=Resampling.nearest
    )

gt_bin = gt_aligned > 0


# -------------------------------------------------
# AUTO THRESHOLD SEARCH
# -------------------------------------------------
if THRESHOLD is None:
    print("\nSearching best threshold...")
    thresholds = np.arange(0.2, 0.7, 0.05)

    best_iou = 0
    best_t = 0.5

    for t in thresholds:
        pb = pred > t
        inter = np.logical_and(pb, gt_bin).sum()
        union = np.logical_or(pb, gt_bin).sum()
        iou = inter / (union + 1e-6)

        print(f"Threshold {t:.2f} → IoU {iou:.4f}")

        if iou > best_iou:
            best_iou = iou
            best_t = t

    THRESHOLD = best_t
    print(f"\nBest threshold selected: {THRESHOLD:.2f}")


# -------------------------------------------------
# FINAL METRICS
# -------------------------------------------------
pred_bin = pred > THRESHOLD

TP = np.logical_and(pred_bin, gt_bin).sum()
FP = np.logical_and(pred_bin, ~gt_bin).sum()
FN = np.logical_and(~pred_bin, gt_bin).sum()

precision = TP / (TP + FP + 1e-6)
recall = TP / (TP + FN + 1e-6)
f1 = 2 * precision * recall / (precision + recall + 1e-6)
iou = TP / (TP + FP + FN + 1e-6)

print("\n===== METRICS =====")
print(f"Threshold: {THRESHOLD:.2f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"IoU:       {iou:.4f}")


# -------------------------------------------------
# CONFUSION MAP (VERY IMPORTANT)
# 0 = TN
# 1 = FP
# 2 = FN
# 3 = TP
# -------------------------------------------------
conf = np.zeros_like(pred_bin, dtype="uint8")
conf[np.logical_and(pred_bin, ~gt_bin)] = 1
conf[np.logical_and(~pred_bin, gt_bin)] = 2
conf[np.logical_and(pred_bin, gt_bin)] = 3

conf_path = PRED.parent / f"confusion_{YEAR}_{AOI}.tif"

meta.update(dtype="uint8")

with rasterio.open(conf_path, "w", **meta) as dst:
    dst.write(conf, 1)

print("✅ Confusion raster saved:", conf_path)


# -------------------------------------------------
# PROBABILITY HISTOGRAM
# -------------------------------------------------
plt.hist(pred.flatten(), bins=50)
plt.title("Prediction Probability Histogram")
plt.show()