import argparse
import rasterio
import numpy as np
from pathlib import Path
from rasterio.warp import reproject, Resampling

# ---------------------------------------
# ARGUMENTS
# ---------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--aoi", required=True)
parser.add_argument("--year", required=True)
args = parser.parse_args()

AOI = args.aoi
YEAR = args.year

RAW_DIR = Path("data/raw/sentinel2_clipped") / AOI / YEAR
OUT_DIR = Path("data/processed") / AOI
OUT_DIR.mkdir(parents=True, exist_ok=True)

BANDS = ["B02", "B03", "B04", "B08", "B11"]

print(f"\nBuilding stack for {AOI} {YEAR}")

arrays = []

# ---------------------------------------
# Reference band
# ---------------------------------------
ref_path = RAW_DIR / "B02.tif"
if not ref_path.exists():
    raise FileNotFoundError(ref_path)

with rasterio.open(ref_path) as ref:
    ref_arr = ref.read(1).astype("float32")
    ref_meta = ref.meta.copy()
    nodata = ref.nodata

arrays.append(ref_arr)

# ---------------------------------------
# Load other bands
# ---------------------------------------
for band in BANDS[1:]:

    path = RAW_DIR / f"{band}.tif"
    if not path.exists():
        raise FileNotFoundError(path)

    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")

        if arr.shape != ref_arr.shape:
            res = np.empty(ref_arr.shape, dtype="float32")
            reproject(
                arr,
                res,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_meta["transform"],
                dst_crs=ref_meta["crs"],
                resampling=Resampling.bilinear,
            )
            arrays.append(res)
        else:
            arrays.append(arr)

b2, b3, b4, b8, b11 = arrays

# ---------------------------------------
# Mask nodata
# ---------------------------------------
mask = (b2 == nodata) if nodata is not None else np.zeros_like(b2, dtype=bool)

# ---------------------------------------
# Indices
# ---------------------------------------
ndvi = (b8 - b4) / (b8 + b4 + 1e-6)
ndbi = (b11 - b8) / (b11 + b8 + 1e-6)
ndwi = (b3 - b8) / (b3 + b8 + 1e-6)
bsi  = ((b11 + b4) - (b8 + b2)) / ((b11 + b4) + (b8 + b2) + 1e-6)
ibi  = (ndbi - (ndvi + ndwi)/2) / (ndbi + (ndvi + ndwi)/2 + 1e-6)

# ---------------------------------------
# Stack
# ---------------------------------------
stack = np.stack([b2,b3,b4,b8,b11,ndvi,ndbi,ndwi,bsi,ibi]).astype("float32")

stack[:, mask] = 0

# ---------------------------------------
# Save
# ---------------------------------------
ref_meta.update(count=10, dtype="float32", nodata=0)

out_path = OUT_DIR / f"stack_{YEAR}_{AOI}.tif"

with rasterio.open(out_path, "w", **ref_meta) as dst:
    dst.write(stack)

print("✅ Stack saved:", out_path)