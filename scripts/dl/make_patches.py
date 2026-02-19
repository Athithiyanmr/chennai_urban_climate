# scripts/dl/make_patches.py

import argparse
import rasterio
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil
from scipy.ndimage import binary_dilation

# -------------------------------------------------
# ARGUMENTS
# -------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--year", required=True)
parser.add_argument("--aoi", required=True)
parser.add_argument("--patch", type=int, default=64)
parser.add_argument("--stride", type=int, default=32)
parser.add_argument("--clean", action="store_true")
args = parser.parse_args()

YEAR = args.year
AOI = args.aoi
PATCH = args.patch
STRIDE = args.stride

STACK = f"data/processed/{AOI}/stack_{YEAR}.tif"
LABEL = f"data/raw/training/labels_google_{YEAR}_{AOI}.tif"

OUT_IMG = Path(f"data/dl/{YEAR}_{AOI}/images")
OUT_MSK = Path(f"data/dl/{YEAR}_{AOI}/masks")

# -------------------------------------------------
# CLEAN OLD PATCHES
# -------------------------------------------------
if args.clean:
    print("🧹 Cleaning old patches...")
    shutil.rmtree(OUT_IMG, ignore_errors=True)
    shutil.rmtree(OUT_MSK, ignore_errors=True)

OUT_IMG.mkdir(parents=True, exist_ok=True)
OUT_MSK.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------
# LOAD STACK
# -------------------------------------------------
if not Path(STACK).exists():
    raise FileNotFoundError(STACK)

if not Path(LABEL).exists():
    raise FileNotFoundError(LABEL)

print("Loading stack...")
with rasterio.open(STACK) as src:
    img = src.read().astype("float32")

print("Loading labels...")
with rasterio.open(LABEL) as lab:
    mask = lab.read(1).astype("uint8")

# -------------------------------------------------
# SMALL DILATION (fix Sentinel vs label mismatch)
# -------------------------------------------------
mask = binary_dilation(mask, iterations=1).astype("uint8")

H, W = mask.shape
count = 0
empty_count = 0
building_count = 0

print(f"\nCreating patches: {YEAR} | {AOI}")
print(f"Patch={PATCH}  Stride={STRIDE}")

# -------------------------------------------------
# PATCH LOOP
# -------------------------------------------------
for i in tqdm(range(0, H - PATCH + 1, STRIDE)):
    for j in range(0, W - PATCH + 1, STRIDE):

        x = img[:, i:i+PATCH, j:j+PATCH]
        y = mask[i:i+PATCH, j:j+PATCH]

        if np.isnan(x).any():
            continue

        # -------------------------------------------------
        # BALANCED SAMPLING
        # -------------------------------------------------
        building_ratio = y.sum() / (PATCH * PATCH)

        # keep dense + boundary
        if building_ratio > 0.02:
            keep = True
            building_count += 1

        # keep some empty patches (background learning)
        else:
            keep = np.random.rand() < 0.25
            if not keep:
                empty_count += 1

        if not keep:
            continue

        np.save(OUT_IMG / f"img_{count}.npy", x.astype("float32"))
        np.save(OUT_MSK / f"mask_{count}.npy", y.astype("uint8"))

        count += 1

print("\n✅ Patch generation complete")
print("Total patches:", count)
print("Building patches:", building_count)
print("Skipped empty:", empty_count)