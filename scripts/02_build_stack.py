import rasterio
import numpy as np
from pathlib import Path
from rasterio.warp import reproject, Resampling
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--aoi", required=True)
args = parser.parse_args()

TARGET_AOI = args.aoi

RAW = Path("data/raw/sentinel2")
OUT = Path("data/processed")
BANDS = ["B02", "B03", "B04", "B08", "B11"]

OUT.mkdir(exist_ok=True)

for aoi_dir in RAW.iterdir():

    # -----------------------------------
    # Must be folder like: 2025_auroville
    # -----------------------------------
    if not aoi_dir.is_dir():
        continue

    if "_" not in aoi_dir.name:
        continue

    print("Processing:", aoi_dir.name)

    arrays = []

    # -----------------------------------
    # Load reference band (B02)
    # -----------------------------------
    ref_path = aoi_dir / "B02.tif"
    if not ref_path.exists():
        print("Missing B02:", aoi_dir)
        continue

    with rasterio.open(ref_path) as ref:
        ref_arr = ref.read(1)
        ref_meta = ref.meta.copy()

    arrays.append(ref_arr)

    # -----------------------------------
    # Load other bands
    # -----------------------------------
    for band in BANDS[1:]:
        band_path = aoi_dir / f"{band}.tif"

        if not band_path.exists():
            print("Missing band:", band_path)
            continue

        with rasterio.open(band_path) as src:
            arr = src.read(1)

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

    # -----------------------------------
    # Ensure all 5 bands exist
    # -----------------------------------
    if len(arrays) != 5:
        print("Skipping incomplete stack:", aoi_dir.name)
        continue

    b2, b3, b4, b8, b11 = arrays

    # -----------------------------------
    # Compute indices
    # -----------------------------------
    ndvi = (b8 - b4) / (b8 + b4 + 1e-6)
    ndbi = (b11 - b8) / (b11 + b8 + 1e-6)
    ndwi = (b3 - b8) / (b3 + b8 + 1e-6)
    mndwi = (b3 - b11) / (b3 + b11 + 1e-6)
    bsi = ((b11 + b4) - (b8 + b2)) / ((b11 + b4) + (b8 + b2) + 1e-6)
    ibi = (ndbi - (ndvi + mndwi) / 2) / (ndbi + (ndvi + mndwi) / 2 + 1e-6)

    # -----------------------------------
    # Final 10-band stack
    # -----------------------------------
    stack = np.stack(
        [b2, b3, b4, b8, b11, ndvi, ndbi, ndwi, bsi, ibi]
    ).astype("float32")

    ref_meta.update(count=10, dtype="float32")

    # -----------------------------------
    # Extract year + AOI name
    # -----------------------------------
    year, aoi_name = aoi_dir.name.split("_", 1)

    out_path = OUT / f"stack_{year}_{aoi_name}.tif"

    with rasterio.open(out_path, "w", **ref_meta) as dst:
        dst.write(stack)

    print(f"10-band stack written: {year}_{aoi_name}")