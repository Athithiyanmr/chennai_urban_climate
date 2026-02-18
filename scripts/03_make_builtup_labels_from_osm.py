import argparse
import osmnx as ox
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from pathlib import Path

# -------------------------------------------------
# ARGUMENTS
# -------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--year", required=True)
parser.add_argument("--aoi", required=True)
args = parser.parse_args()

YEAR = args.year
AOI_NAME = args.aoi

AOI = f"data/raw/boundaries/{AOI_NAME}.shp"
REF = f"data/processed/stack_{YEAR}_{AOI_NAME}.tif"
OUT = f"data/raw/training/labels_{YEAR}_{AOI_NAME}.tif"

# -------------------------------------------------
# 1. Load AOI
# -------------------------------------------------
print("Loading AOI...")
aoi = gpd.read_file(AOI)
aoi["geometry"] = aoi.geometry.buffer(0)
aoi = aoi.to_crs("EPSG:4326")

# -------------------------------------------------
# 2. Download OSM buildings
# -------------------------------------------------
print("Downloading OSM buildings...")
buildings = ox.features_from_polygon(
    aoi.geometry.iloc[0],
    tags={"building": True}
)

if buildings.empty:
    raise RuntimeError("No buildings found in OSM.")

# Keep only polygons
buildings = buildings[
    buildings.geometry.type.isin(["Polygon", "MultiPolygon"])
]

print("Total buildings:", len(buildings))

# -------------------------------------------------
# 3. Load reference raster
# -------------------------------------------------
with rasterio.open(REF) as ref:
    meta = ref.meta.copy()
    shape = (ref.height, ref.width)
    transform = ref.transform
    crs = ref.crs

# -------------------------------------------------
# 4. Reproject buildings
# -------------------------------------------------
buildings = buildings.to_crs(crs)
buildings["geometry"] = buildings.buffer(1)

# -------------------------------------------------
# 5. Rasterize
# -------------------------------------------------
labels = rasterize(
    ((g, 1) for g in buildings.geometry),
    out_shape=shape,
    transform=transform,
    fill=0,
    dtype="uint8"
)

# -------------------------------------------------
# 6. Save
# -------------------------------------------------
meta.update(count=1, dtype="uint8", nodata=0)
Path("data/raw/training").mkdir(parents=True, exist_ok=True)

with rasterio.open(OUT, "w", **meta) as dst:
    dst.write(labels, 1)

print("✅ Built-up labels saved:", OUT)