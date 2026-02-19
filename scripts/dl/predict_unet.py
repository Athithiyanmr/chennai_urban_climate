import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import torch
import rasterio
import numpy as np
from tqdm import tqdm
from pathlib import Path
from scripts.dl.unet_model import UNet


# -------------------------------------------------
# ARGUMENTS
# -------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--year", required=True, help="Year (e.g. 2025)")
parser.add_argument("--aoi", required=True, help="AOI name (e.g. auroville)")
parser.add_argument("--threshold", type=float, default=None,
                    help="Optional threshold for binary output")
parser.add_argument("--patch", type=int, default=64)
parser.add_argument("--stride", type=int, default=32)

args = parser.parse_args()

YEAR = args.year
AOI = args.aoi
PATCH = args.patch
STRIDE = args.stride
THRESHOLD = args.threshold


# -------------------------------------------------
# PATHS
# -------------------------------------------------
STACK = Path(f"data/processed/{AOI}/stack_{YEAR}.tif")
MODEL = Path(f"models/unet_{YEAR}_{AOI}.pth")
OUT_DIR = Path(f"outputs/unet/{YEAR}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PROB = OUT_DIR / f"builtup_prob_{YEAR}_{AOI}.tif"
OUT_BIN = OUT_DIR / f"builtup_binary_{YEAR}_{AOI}.tif"


# -------------------------------------------------
# VALIDATION
# -------------------------------------------------
if not STACK.exists():
    raise FileNotFoundError(f"Stack not found: {STACK}")

if not MODEL.exists():
    raise FileNotFoundError(f"Model not found: {MODEL}")


# -------------------------------------------------
# DEVICE
# -------------------------------------------------
device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)

print("Using device:", device)
print("Stack:", STACK)
print("Model:", MODEL)


# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
model = UNet(in_channels=10)
model.load_state_dict(torch.load(MODEL, map_location=device))
model.to(device)
model.eval()


# -------------------------------------------------
# LOAD IMAGE
# -------------------------------------------------
with rasterio.open(STACK) as src:
    img = src.read().astype("float32")
    meta = src.meta.copy()
    H, W = src.height, src.width


# -------------------------------------------------
# PADDING (important for full coverage)
# -------------------------------------------------
pad_h = (PATCH - H % PATCH) % PATCH
pad_w = (PATCH - W % PATCH) % PATCH

img = np.pad(img, ((0,0),(0,pad_h),(0,pad_w)), mode="reflect")
H_new, W_new = img.shape[1], img.shape[2]

pred_sum = np.zeros((H_new, W_new), dtype="float32")
pred_cnt = np.zeros((H_new, W_new), dtype="float32")


# -------------------------------------------------
# SLIDING WINDOW INFERENCE
# -------------------------------------------------
for i in tqdm(range(0, H_new - PATCH + 1, STRIDE), desc="Rows"):
    for j in range(0, W_new - PATCH + 1, STRIDE):

        patch = img[:, i:i+PATCH, j:j+PATCH]

        if np.isnan(patch).any():
            continue

        x = torch.from_numpy(patch).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(x).squeeze().cpu().numpy()

        pred_sum[i:i+PATCH, j:j+PATCH] += pred
        pred_cnt[i:i+PATCH, j:j+PATCH] += 1


# -------------------------------------------------
# AVERAGE OVERLAPS
# -------------------------------------------------
pred_final = np.divide(
    pred_sum,
    pred_cnt,
    out=np.zeros_like(pred_sum),
    where=pred_cnt != 0
)

pred_final = pred_final[:H, :W]


# -------------------------------------------------
# SAVE PROBABILITY MAP
# -------------------------------------------------
meta.update(count=1, dtype="float32", nodata=0)

with rasterio.open(OUT_PROB, "w", **meta) as dst:
    dst.write(pred_final, 1)

print("✅ Probability raster saved:", OUT_PROB)


# -------------------------------------------------
# OPTIONAL BINARY OUTPUT
# -------------------------------------------------
if THRESHOLD is not None:
    binary = (pred_final > THRESHOLD).astype("uint8")
    meta.update(dtype="uint8")

    with rasterio.open(OUT_BIN, "w", **meta) as dst:
        dst.write(binary, 1)

    print(f"✅ Binary raster saved (threshold={THRESHOLD}):", OUT_BIN)