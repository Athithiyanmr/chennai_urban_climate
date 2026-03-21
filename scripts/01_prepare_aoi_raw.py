import argparse
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject

# -----------------------------------------
# LOGGING
# -----------------------------------------
logging.basicConfig(
    filename="preprocess.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# -----------------------------------------
# ARGUMENTS
# -----------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--aoi", required=True)
parser.add_argument("--year", required=True)
parser.add_argument("--target_crs", default="EPSG:32643")
args = parser.parse_args()

AOI_PATH = f"data/raw/boundaries/{args.aoi}.shp"
RAW_DIR = Path("data/raw/sentinel2") / args.aoi / args.year
OUT_DIR = Path("data/processed/sentinel2_clipped") / args.aoi / args.year

BANDS = ["B02", "B03", "B04", "B05", "B06", "B08", "B11", "B12"]
TARGET_CRS = args.target_crs

OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"\nAOI  : {args.aoi}")
print(f"Year : {args.year}")
print(f"CRS  : {TARGET_CRS}")

# -----------------------------------------
# LOAD AOI
# -----------------------------------------
aoi = gpd.read_file(AOI_PATH)
if aoi.crs is None:
    raise ValueError("AOI has no CRS")

# -----------------------------------------
# REPROJECT FUNCTION
# -----------------------------------------
def reproject_tile(src, target_crs):

    transform, width, height = calculate_default_transform(
        src.crs, target_crs,
        src.width, src.height,
        *src.bounds
    )

    kwargs = src.meta.copy()
    kwargs.update({
        "crs": target_crs,
        "transform": transform,
        "width": width,
        "height": height
    })

    data = src.read()
    dst_array = np.zeros((data.shape[0], height, width), dtype=data.dtype)

    for i in range(data.shape[0]):
        reproject(
            source=data[i],
            destination=dst_array[i],
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=target_crs,
            resampling=Resampling.bilinear,
        )

    memfile = rasterio.io.MemoryFile()
    dataset = memfile.open(**kwargs)
    dataset.write(dst_array)

    return dataset

# -----------------------------------------
# PROCESS BANDS
# -----------------------------------------
for band in BANDS:

    band_files = sorted(RAW_DIR.glob(f"**/*{band}.tif"))

    if not band_files:
        print(f"❌ No files for {band}")
        continue

    print(f"\nProcessing {band} ({len(band_files)} tiles)")

    reprojected_srcs = []

    for f in band_files:
        src = rasterio.open(f)

        if str(src.crs) != TARGET_CRS:
            print(f"   Reprojecting {f.name}")
            reproj = reproject_tile(src, TARGET_CRS)
            reprojected_srcs.append(reproj)
        else:
            reprojected_srcs.append(src)

    # -----------------------------------------
    # MERGE
    # -----------------------------------------
    mosaic, transform = merge(reprojected_srcs, nodata=0)

    meta = reprojected_srcs[0].meta.copy()
    meta.update({
        "transform": transform,
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "nodata": 0
    })

    # -----------------------------------------
    # CLIP
    # -----------------------------------------
    aoi_proj = aoi.to_crs(meta["crs"])
    geoms = list(aoi_proj.geometry)

    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(**meta) as tmp:
            tmp.write(mosaic)

            clipped, clipped_transform = mask(
                tmp,
                geoms,
                crop=True,
                nodata=0
            )

    # -----------------------------------------
    # OPTIMIZATION (KEY PART 🚀)
    # -----------------------------------------
    clipped = clipped.astype("uint16")

    meta.update({
        "transform": clipped_transform,
        "height": clipped.shape[1],
        "width": clipped.shape[2],
        "dtype": "uint16",
        "compress": "lzw",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "predictor": 2,
        "BIGTIFF": "IF_SAFER"
    })

    # -----------------------------------------
    # SAVE
    # -----------------------------------------
    out_path = OUT_DIR / f"{band}.tif"

    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(clipped)

    size_mb = out_path.stat().st_size / 1_000_000
    print(f"✔ Saved {band} ({size_mb:.1f} MB)")

# -----------------------------------------
# DONE
# -----------------------------------------
print("\n✅ Preprocessing complete")
print("Output:", OUT_DIR)