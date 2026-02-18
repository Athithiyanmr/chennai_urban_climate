import argparse
import rasterio
import numpy as np
from pathlib import Path
from rasterio.warp import reproject, Resampling

# ---------------------------------------
# ARGUMENTS
# ---------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--aoi", required=True, help="AOI name")
parser.add_argument("--year", required=True, help="Year")
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
# Load reference band (B02)
# ---------------------------------------
ref_path = RAW_DIR / "B02.tif"

if not ref_path.exists():
    raise FileNotFoundError(f"Missing B02 in {RAW_DIR}")

with rasterio.open(ref_path) as ref:
    ref_arr = ref.read(1).astype("float32")
    ref_meta = ref.meta.copy()

arrays.append(ref_arr)

# ---------------------------------------
# Load other bands (resample if needed)
# ---------------------------------------
for band in BANDS[1:]:

    band_path = RAW_DIR / f"{band}.tif"

    if not band_path.exists():
        raise FileNotFoundError(f"Missing {band} in {RAW_DIR}")

    with rasterio.open(band_path) as src:
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

# ---------------------------------------
# Extract bands
# ---------------------------------------
b2, b3, b4, b8, b11 = arrays

# ---------------------------------------
# Compute indices
# ---------------------------------------
ndvi = (b8 - b4) / (b8 + b4 + 1e-6)
ndbi = (b11 - b8) / (b11 + b8 + 1e-6)
ndwi = (b3 - b8) / (b3 + b8 + 1e-6)
mndwi = (b3 - b11) / (b3 + b11 + 1e-6)
bsi = ((b11 + b4) - (b8 + b2)) / ((b11 + b4) + (b8 + b2) + 1e-6)
ibi = (ndbi - (ndvi + mndwi)/2) / (ndbi + (ndvi + mndwi)/2 + 1e-6)

# ---------------------------------------
# Final 10-band stack
# ---------------------------------------
stack = np.stack([
    b2, b3, b4, b8, b11,
    ndvi, ndbi, ndwi, bsi, ibi
]).astype("float32")

ref_meta.update(count=10, dtype="float32")

out_path = OUT_DIR / f"stack_{YEAR}.tif"

with rasterio.open(out_path, "w", **ref_meta) as dst:
    dst.write(stack)

print(f"\n✅ 10-band stack saved → {out_path}")