# scripts/01_prepare_aoi_raw.py
from pathlib import Path
import rasterio
from rasterio.merge import merge
from rasterio.mask import mask
import geopandas as gpd

AOI = "data/raw/boundaries/CMDA.shp"
RAW = Path("data/raw/sentinel2")
BANDS = ["B02", "B03", "B04", "B08", "B11"]

aoi = gpd.read_file(AOI)

for year_dir in RAW.iterdir():
    if not year_dir.is_dir():
        continue
    if year_dir.name.endswith("_aoi") or year_dir.name.startswith("._"):
        continue

    out_dir = RAW / f"{year_dir.name}_aoi"
    out_dir.mkdir(exist_ok=True)

    for band in BANDS:
        files = [f for f in year_dir.glob(f"*/**/*_{band}.tif")]
        srcs = [rasterio.open(f) for f in files if f.exists()]

        if not srcs:
            continue

        aoi_proj = aoi.to_crs(srcs[0].crs)
        geoms = list(aoi_proj.geometry)

        mosaic, tr = merge(srcs)
        meta = srcs[0].meta.copy()
        meta.update(transform=tr, height=mosaic.shape[1], width=mosaic.shape[2])

        with rasterio.open("/tmp/tmp.tif", "w", **meta) as tmp:
            tmp.write(mosaic)

        with rasterio.open("/tmp/tmp.tif") as src:
            out, tr = mask(src, geoms, crop=True, nodata=0)
            meta.update(transform=tr, height=out.shape[1], width=out.shape[2])

        with rasterio.open(out_dir / f"{band}.tif", "w", **meta) as dst:
            dst.write(out)

        print(f"{year_dir.name} {band} done")