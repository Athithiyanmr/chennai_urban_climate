import argparse
from pathlib import Path
import geopandas as gpd
import geoai

# -----------------------------------------
# ARGUMENTS
# -----------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--aoi", required=True)
parser.add_argument("--year", type=int, required=True)
args = parser.parse_args()
AOI = f"data/raw/boundaries/{args.aoi}.shp"
YEAR = args.year
OUT = Path("data/raw/embeddings") / args.aoi / str(YEAR)
OUT.mkdir(parents=True, exist_ok=True)

# -----------------------------------------
# LOAD AOI
# -----------------------------------------
print("Loading AOI...")

aoi = gpd.read_file(AOI).to_crs("EPSG:4326")

minx, miny, maxx, maxy = aoi.total_bounds
bbox = [float(minx), float(miny), float(maxx), float(maxy)]

print("AOI bbox:", bbox)

# -----------------------------------------
# DOWNLOAD GOOGLE EMBEDDINGS
# -----------------------------------------
print("\n⬇ Downloading Google embeddings...")

geoai.download_google_satellite_embedding(
    bbox=bbox,
    years=[YEAR],
    output_dir=str(OUT)
)

print("\n✅ Embeddings downloaded")
print("Saved to:", OUT)