import argparse
import pandas as pd
import geopandas as gpd
from shapely import wkt
import rasterio
from rasterio.features import rasterize
from pathlib import Path

# ---------------------------------------
# ARGUMENTS
# ---------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--year", required=True)
parser.add_argument("--aoi", required=True)
parser.add_argument("--csv", required=True)
parser.add_argument("--conf", type=float, default=0.7)
args = parser.parse_args()

YEAR = args.year
AOI_NAME = args.aoi
CSV = args.csv
CONF_THRESHOLD = args.conf

AOI = f"data/raw/boundaries/{AOI_NAME}.shp"
STACK = f"data/processed/{AOI_NAME}/stack_{YEAR}.tif"
OUT = f"data/raw/training/labels_google_{YEAR}_{AOI_NAME}.tif"

# ---------------------------------------
# 1️⃣ Read CSV
# ---------------------------------------
print("Reading CSV...")
df = pd.read_csv(CSV)

if "geometry" not in df.columns:
    raise ValueError("CSV must contain a 'geometry' column (WKT format).")

df["geometry"] = df["geometry"].apply(wkt.loads)

gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

# Optional confidence filtering
if "confidence" in df.columns:
    gdf = gdf[gdf["confidence"] >= CONF_THRESHOLD]

print("Buildings after confidence filter:", len(gdf))

# Fix invalid geometries
gdf["geometry"] = gdf.geometry.buffer(0)

# ---------------------------------------
# 2️⃣ Clip to AOI
# ---------------------------------------
aoi = gpd.read_file(AOI).to_crs("EPSG:4326")
gdf = gdf[gdf.intersects(aoi.geometry.iloc[0])]

print("Buildings after AOI clip:", len(gdf))

if gdf.empty:
    raise RuntimeError("No buildings left after clipping.")

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

# Optional: small buffer improves rasterization
gdf["geometry"] = gdf.buffer(0.5)

# ---------------------------------------
# 4️⃣ Rasterize
# ---------------------------------------
print("Rasterizing buildings...")

mask = rasterize(
    ((geom, 1) for geom in gdf.geometry),
    out_shape=(height, width),
    transform=transform,
    fill=0,
    dtype="uint8"
)

meta.update(count=1, dtype="uint8", nodata=0)
Path("data/raw/training").mkdir(parents=True, exist_ok=True)

with rasterio.open(OUT, "w", **meta) as dst:
    dst.write(mask, 1)

print("✅ Google training mask saved:", OUT)