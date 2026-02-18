import pandas as pd
import geopandas as gpd
from shapely import wkt
import rasterio
from rasterio.features import rasterize
from pathlib import Path
import numpy as np

# ---------------------------------------
# CONFIG
# ---------------------------------------
CSV = "data/raw/training/google_buildings_av.csv"
AOI = "data/raw/boundaries/auroville.shp"
STACK = "data/processed/stack_2025_auroville.tif"
OUT = "data/raw/training/builtup_labels_google_auroville.tif"

CONF_THRESHOLD = 0.7   # change if needed

# ---------------------------------------
# 1️⃣ Read CSV
# ---------------------------------------
print("Reading CSV...")
df = pd.read_csv(CSV)

if "geometry" not in df.columns:
    raise ValueError("CSV must contain a 'geometry' column (WKT format).")

df["geometry"] = df["geometry"].apply(wkt.loads)
gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

# Optional confidence filter
if "confidence" in df.columns:
    gdf = gdf[gdf["confidence"] >= CONF_THRESHOLD]

print("Buildings after confidence filter:", len(gdf))

# ---------------------------------------
# 2️⃣ Clip to AOI
# ---------------------------------------
aoi = gpd.read_file(AOI).to_crs("EPSG:4326")
gdf = gdf[gdf.intersects(aoi.unary_union)]

print("Buildings after AOI clip:", len(gdf))

# ---------------------------------------
# 3️⃣ Align to Sentinel stack
# ---------------------------------------
with rasterio.open(STACK) as src:
    meta = src.meta.copy()
    transform = src.transform
    height = src.height
    width = src.width
    crs = src.crs

gdf = gdf.to_crs(crs)

# ---------------------------------------
# 4️⃣ Rasterize
# ---------------------------------------
print("Rasterizing buildings...")

shapes = ((geom, 1) for geom in gdf.geometry)

mask = rasterize(
    shapes,
    out_shape=(height, width),
    transform=transform,
    fill=0,
    dtype="uint8"
)

meta.update(count=1, dtype="uint8", nodata=0)

Path("data/raw/training").mkdir(parents=True, exist_ok=True)

with rasterio.open(OUT, "w", **meta) as dst:
    dst.write(mask, 1)

print("✅ Training mask created:", OUT)