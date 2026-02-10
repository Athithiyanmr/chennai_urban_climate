import torch
import rasterio
import numpy as np
from tqdm import tqdm
import sys
from pathlib import Path
from scripts.dl.unet_model import UNet

# -------------------------
# ARGUMENT
# -------------------------
YEAR = sys.argv[1] if len(sys.argv) > 1 else "2025"

STACK = f"data/processed/chennai_stack_{YEAR}.tif"
MODEL = f"models/unet_builtup_{YEAR}.pth"

OUT_DIR = Path(f"outputs/unet/{YEAR}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT = OUT_DIR / f"chennai_builtup_unet_{YEAR}.tif"

PATCH = 64
STRIDE = 32

# -------------------------
# DEVICE
# -------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)

# -------------------------
# LOAD MODEL
# -------------------------
model = UNet(in_channels=10)
model.load_state_dict(torch.load(MODEL, map_location=device))
model.to(device)
model.eval()

# -------------------------
# LOAD IMAGE
# -------------------------
with rasterio.open(STACK) as src:
    img = src.read().astype("float32")
    meta = src.meta.copy()
    H, W = src.height, src.width

# -------------------------
# OUTPUT ARRAYS
# -------------------------
pred_sum = np.zeros((H, W), dtype="float32")
pred_cnt = np.zeros((H, W), dtype="float32")

# -------------------------
# SLIDING WINDOW INFERENCE
# -------------------------
for i in tqdm(range(0, H - PATCH, STRIDE), desc="Rows"):
    for j in range(0, W - PATCH, STRIDE):

        patch = img[:, i:i+PATCH, j:j+PATCH]

        if np.isnan(patch).any():
            continue

        x = torch.from_numpy(patch).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(x).squeeze().cpu().numpy()

        pred_sum[i:i+PATCH, j:j+PATCH] += pred
        pred_cnt[i:i+PATCH, j:j+PATCH] += 1

# -------------------------
# AVERAGE OVERLAPS
# -------------------------
pred_final = np.divide(
    pred_sum,
    pred_cnt,
    out=np.zeros_like(pred_sum),
    where=pred_cnt != 0
)

# -------------------------
# SAVE OUTPUT
# -------------------------
meta.update(
    count=1,
    dtype="float32",
    nodata=0
)

with rasterio.open(OUT, "w", **meta) as dst:
    dst.write(pred_final, 1)

print("✅ Built-up prediction saved:", OUT)