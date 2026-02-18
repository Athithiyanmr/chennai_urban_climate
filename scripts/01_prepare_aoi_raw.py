# scripts/01_prepare_aoi_raw.py
from pathlib import Path
import rasterio
from rasterio.merge import merge
from rasterio.mask import mask
import geopandas as gpd

AOI = "data/raw/boundaries/auroville.shp"
RAW = Path("data/raw/sentinel2")
BANDS = ["B02", "B03", "B04", "B08", "B11"]

# ---------------------------------
# Extract shapefile name dynamically
# ---------------------------------
aoi_name = Path(AOI).stem   # -> "auroville"

aoi = gpd.read_file(AOI)

for year_dir in RAW.iterdir():

    if not year_dir.is_dir():
        continue

    # Skip already processed folders
    if year_dir.name.endswith(f"_{aoi_name}") or year_dir.name.startswith("._"):
        continue

    # ---------------------------------
    # Create output folder with AOI name
    # ---------------------------------
    out_dir = RAW / f"{year_dir.name}_{aoi_name}"
    out_dir.mkdir(exist_ok=True)

    for band in BANDS:

        files = list(year_dir.glob(f"*/**/*_{band}.tif"))
        srcs = [rasterio.open(f) for f in files if f.exists()]

        if not srcs:
            continue

        aoi_proj = aoi.to_crs(srcs[0].crs)
        geoms = list(aoi_proj.geometry)

        mosaic, tr = merge(srcs)
        meta = srcs[0].meta.copy()
        meta.update(
            transform=tr,
            height=mosaic.shape[1],
            width=mosaic.shape[2]
        )

        # Use in-memory instead of /tmp file (cleaner)
        with rasterio.io.MemoryFile() as memfile:
            with memfile.open(**meta) as tmp:
                tmp.write(mosaic)

                out, tr = mask(tmp, geoms, crop=True, nodata=0)

        meta.update(
            transform=tr,
            height=out.shape[1],
            width=out.shape[2]
        )

        with rasterio.open(out_dir / f"{band}.tif", "w", **meta) as dst:
            dst.write(out)

        print(f"{year_dir.name} {band} done")

print(f"\n✔ AOI prepared for: {aoi_name}")