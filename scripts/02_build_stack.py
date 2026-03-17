import argparse
import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.enums import Resampling

# -----------------------------------------
# LOGGING
# -----------------------------------------
logging.basicConfig(
    filename="stack.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# -----------------------------------------
# ARGUMENTS
# -----------------------------------------
parser = argparse.ArgumentParser(description="Build multi-band stack with spectral indices")
parser.add_argument("--aoi",  required=True, help="AOI name")
parser.add_argument("--year", required=True, help="Year to process")
args = parser.parse_args()

AOI  = args.aoi
YEAR = args.year

RAW_DIR = Path("data/processed/sentinel2_clipped") / AOI / YEAR  # ✅ aligned with preprocess output
OUT_DIR = Path("data/processed") / AOI
OUT_DIR.mkdir(parents=True, exist_ok=True)

BANDS = ["B02", "B03", "B04", "B08", "B11"]

# Band index labels for documentation
BAND_NAMES = ["B02", "B03", "B04", "B08", "B11", "NDVI", "NDBI", "NDWI", "MNDWI", "BSI", "IBI"]

print(f"\n📦 Building stack: {AOI} {YEAR}")
log.info(f"Starting stack build: AOI={AOI}, year={YEAR}")

# -----------------------------------------
# LOAD REFERENCE BAND (B02)
# -----------------------------------------
ref_path = RAW_DIR / "B02.tif"
if not ref_path.exists():
    raise FileNotFoundError(f"Reference band not found: {ref_path}")

with rasterio.open(ref_path) as ref:
    ref_arr  = ref.read(1).astype("float32")
    ref_meta = ref.meta.copy()
    nodata   = ref.nodata if ref.nodata is not None else 0

log.info(f"Reference band loaded: shape={ref_arr.shape}, CRS={ref_meta['crs']}")

# -----------------------------------------
# LOAD + ALIGN ALL BANDS
# -----------------------------------------
arrays = [ref_arr]

for band in BANDS[1:]:
    path = RAW_DIR / f"{band}.tif"
    if not path.exists():
        raise FileNotFoundError(f"Band file not found: {path}")

    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")

        if arr.shape != ref_arr.shape or src.transform != ref_meta["transform"]:
            print(f"   🔄 Resampling {band} to match reference grid")
            log.info(f"Resampling {band}: {arr.shape} → {ref_arr.shape}")
            res = np.empty(ref_arr.shape, dtype="float32")
            reproject(
                arr, res,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_meta["transform"],
                dst_crs=ref_meta["crs"],
                resampling=Resampling.bilinear,
            )
            arrays.append(res)
        else:
            arrays.append(arr)

    print(f"   ✅ Loaded: {band}")

b2, b3, b4, b8, b11 = arrays

# -----------------------------------------
# NODATA MASK
# -----------------------------------------
nodata_mask = (b2 == nodata)
valid_px    = int(np.sum(~nodata_mask))
total_px    = b2.size
print(f"\n📊 Valid pixels : {valid_px:,} / {total_px:,} ({100*valid_px/total_px:.1f}%)")
log.info(f"Valid pixels: {valid_px}/{total_px}")

# -----------------------------------------
# REFLECTANCE SCALING
# ✅ Sentinel-2 L2A stores DN as uint16 scaled by 10000
# -----------------------------------------
SCALE = 10_000.0

def scale(arr):
    """Convert DN → reflectance, clip to [0, 1], preserve nodata."""
    scaled = np.where(arr == nodata, np.nan, arr / SCALE)
    return np.clip(scaled, 0.0, 1.0)

b2s, b3s, b4s, b8s, b11s = [scale(b) for b in [b2, b3, b4, b8, b11]]

# -----------------------------------------
# SPECTRAL INDICES
# ✅ All computed on scaled reflectance
# ✅ NDWI and MNDWI both retained (your comment was correct)
# -----------------------------------------
eps = 1e-6
# we can add more
ndvi  = (b8s  - b4s)  / (b8s  + b4s  + eps)   # Vegetation
ndbi  = (b11s - b8s)  / (b11s + b8s  + eps)   # Built-up
ndwi  = (b3s  - b8s)  / (b3s  + b8s  + eps)   # Water (Gao 1996)
mndwi = (b3s  - b11s) / (b3s  + b11s + eps)   # Modified Water (Xu 2006) ✅ fixed formula
bsi   = ((b11s + b4s) - (b8s + b2s)) / ((b11s + b4s) + (b8s + b2s) + eps)   # Bare soil
ibi   = (ndbi - (ndvi + ndwi) / 2) / (ndbi + (ndvi + ndwi) / 2 + eps)       # Index-based built-up

# -----------------------------------------
# CLIP INDEX RANGES  ✅ prevent extreme outlier values
# -----------------------------------------
for name, arr in [("NDVI", ndvi), ("NDBI", ndbi), ("NDWI", ndwi),
                  ("MNDWI", mndwi), ("BSI", bsi), ("IBI", ibi)]:
    pct1, pct99 = np.nanpercentile(arr[~nodata_mask], [1, 99])
    print(f"   {name:6s} range: [{pct1:.3f}, {pct99:.3f}]")
    log.info(f"{name} P1={pct1:.3f}, P99={pct99:.3f}")

# -----------------------------------------
# STACK  (11 bands now: 5 raw + 6 indices)
# -----------------------------------------
stack = np.stack([b2s, b3s, b4s, b8s, b11s,
                  ndvi, ndbi, ndwi, mndwi, bsi, ibi]).astype("float32")

# Apply nodata mask across all bands
stack[:, nodata_mask] = np.nan   # ✅ use NaN instead of 0 to avoid false index values

# -----------------------------------------
# SAVE WITH BAND DESCRIPTIONS
# -----------------------------------------
ref_meta.update(
    count=11,
    dtype="float32",
    nodata=np.nan,
    compress="lzw",     # ✅ compress output
    tiled=True,
    blockxsize=256,
    blockysize=256,
)

out_path = OUT_DIR / f"stack_{YEAR}.tif"

with rasterio.open(out_path, "w", **ref_meta) as dst:
    dst.write(stack)
    for i, name in enumerate(BAND_NAMES, start=1):
        dst.update_tags(i, name=name)   # ✅ tag each band with its name

size_mb = out_path.stat().st_size / 1_000_000
print(f"\n✅ Stack saved → {out_path} ({size_mb:.1f} MB)")
print(f"   Bands ({len(BAND_NAMES)}): {', '.join(BAND_NAMES)}")
log.info(f"Stack saved: {out_path}, bands={BAND_NAMES}, size={size_mb:.1f}MB")
