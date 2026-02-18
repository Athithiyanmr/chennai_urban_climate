import argparse
import rasterio
import numpy as np
from pathlib import Path
from tqdm import tqdm

# -------------------------
# ARGUMENTS
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--year", required=True, help="Year of stack (e.g., 2025)")
args = parser.parse_args()

YEAR = args.year

STACK = f"data/processed/stack_{YEAR}_auroville.tif"
LABEL = "data/raw/training/builtup_labels_google_auroville.tif"

OUT_IMG = Path(f"data/dl/{YEAR}/images")
OUT_MSK = Path(f"data/dl/{YEAR}/masks")
OUT_IMG.mkdir(parents=True, exist_ok=True)
OUT_MSK.mkdir(parents=True, exist_ok=True)

PATCH = 16
STRIDE = 8

# -------------------------
# LOAD DATA
# -------------------------
with rasterio.open(STACK) as src:
    img = src.read().astype("float32")

with rasterio.open(LABEL) as lab:
    mask = lab.read(1).astype("uint8")

count = 0
H, W = mask.shape

# -------------------------
# CREATE PATCHES
# -------------------------
for i in tqdm(range(0, H - PATCH, STRIDE)):
    for j in range(0, W - PATCH, STRIDE):
        x = img[:, i:i+PATCH, j:j+PATCH]
        y = mask[i:i+PATCH, j:j+PATCH]

        if np.isnan(x).any():
            continue

        np.save(OUT_IMG / f"img_{count}.npy", x.astype("float32"))
        np.save(OUT_MSK / f"mask_{count}.npy", y.astype("uint8"))
        count += 1

print(f"Total patches for {YEAR}:", count)