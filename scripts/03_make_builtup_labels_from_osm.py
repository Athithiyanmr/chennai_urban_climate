# scripts/03A_make_builtup_labels_from_osm.py

import osmnx as ox
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from pathlib import Path

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
AOI = "data/raw/boundaries/CMDA.shp"
REF = "data/processed/chennai_stack_2025.tif"
OUT = "data/raw/training/builtup_labels.tif"

# -------------------------------------------------
# 1. Load & fix AOI
# -------------------------------------------------
print("Loading AOI...")
aoi = gpd.read_file(AOI)
aoi["geometry"] = aoi.geometry.buffer(0)  # fix invalid geometries
aoi = aoi.to_crs("EPSG:4326")

# -------------------------------------------------
# 2. Download OSM buildings (OSMnx v2.x)
# -------------------------------------------------
print("Downloading OSM buildings via Overpass API...")
buildings = ox.features_from_polygon(
    aoi.geometry.iloc[0],
    tags={"building": True}
)

print("Total OSM features:", len(buildings))
if buildings.empty:
    raise RuntimeError("No buildings returned from OSM. Check AOI.")

# Keep only polygons
buildings = buildings[buildings.geometry.type.isin(["Polygon", "MultiPolygon"])]
print("Polygon buildings:", len(buildings))

# -------------------------------------------------
# 3. Load Sentinel reference raster
# -------------------------------------------------
with rasterio.open(REF) as ref:
    meta = ref.meta.copy()
    shape = (ref.height, ref.width)
    transform = ref.transform
    crs = ref.crs

print("Sentinel CRS:", crs)
print("Raster shape:", shape)

# -------------------------------------------------
# 4. Reproject + buffer buildings (accuracy boost)
# -------------------------------------------------
buildings = buildings.to_crs(crs)
buildings["geometry"] = buildings.buffer(1)  # meters

# -------------------------------------------------
# 5. Rasterize
# -------------------------------------------------
print("Rasterizing buildings...")
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

print("✅ Built-up labels created:", OUT)