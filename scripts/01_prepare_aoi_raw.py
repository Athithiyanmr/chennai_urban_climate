import argparse
from pathlib import Path
import rasterio
from rasterio.merge import merge
from rasterio.mask import mask
import geopandas as gpd

# ---------------------------------------
# ARGUMENTS
# ---------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--aoi", required=True, help="AOI name (without path)")
parser.add_argument("--year", required=True, help="Year to process")
args = parser.parse_args()

AOI_PATH = f"data/raw/boundaries/{args.aoi}.shp"
RAW_DIR = Path("data/raw/sentinel2") / args.aoi / args.year
OUT_DIR = Path("data/raw/sentinel2_clipped") / args.aoi / args.year

# Add SCL for future cloud masking (optional but recommended)
BANDS = ["B02", "B03", "B04", "B08", "B11"]

OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"\nPreparing AOI: {args.aoi}")
print(f"Year: {args.year}")

# ---------------------------------------
# Load AOI
# ---------------------------------------
aoi = gpd.read_file(AOI_PATH)

# ---------------------------------------
# Process each band
# ---------------------------------------
for band in BANDS:

    band_files = sorted(RAW_DIR.glob(f"**/*{band}.tif"))

    if not band_files:
        print(f"❌ No files found for {band}")
        continue

    print(f"\nProcessing band: {band}")

    srcs = [rasterio.open(f) for f in band_files]

    # ---------------------------------------
    # Merge tiles
    # ---------------------------------------
    mosaic, transform = merge(srcs, nodata=0)

    meta = srcs[0].meta.copy()
    meta.update(
        transform=transform,
        height=mosaic.shape[1],
        width=mosaic.shape[2],
        nodata=0
    )

    # Close raster handles (important)
    for s in srcs:
        s.close()

    # ---------------------------------------
    # Reproject AOI
    # ---------------------------------------
    aoi_proj = aoi.to_crs(meta["crs"])
    geoms = list(aoi_proj.geometry)

    # ---------------------------------------
    # Clip to AOI
    # ---------------------------------------
    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(**meta) as tmp:
            tmp.write(mosaic)

            clipped, clipped_transform = mask(
                tmp,
                geoms,
                crop=True,
                nodata=0
            )

    meta.update(
        transform=clipped_transform,
        height=clipped.shape[1],
        width=clipped.shape[2]
    )

    out_path = OUT_DIR / f"{band}.tif"

    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(clipped)

    print(f"✔ {band} saved")

print(f"\n✅ AOI clipping complete for {args.aoi} {args.year}")