import argparse
from pathlib import Path
import geopandas as gpd
import pystac_client
import planetary_computer
import requests

# ----------------------------
# ARGUMENTS
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--aoi", required=True, help="AOI shapefile name (without path)")
parser.add_argument("--year", type=int, required=True)
parser.add_argument("--cloud", type=int, default=20)
args = parser.parse_args()

AOI = f"data/raw/boundaries/{args.aoi}.shp"
YEAR = args.year
CLOUD = args.cloud

OUT = Path("data/raw/sentinel2")
BANDS = ["B02", "B03", "B04", "B08", "B11"]

OUT.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Download helper
# ----------------------------
def download(url, out_path):
    with requests.get(url, stream=True, timeout=90) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)

# ----------------------------
# Load AOI
# ----------------------------
print("Loading AOI...")
aoi = gpd.read_file(AOI).to_crs("EPSG:4326")
geom = aoi.geometry.iloc[0].__geo_interface__

# ----------------------------
# Open STAC
# ----------------------------
catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1"
)

print(f"Searching Sentinel-2 for {YEAR}...")

search = catalog.search(
    collections=["sentinel-2-l2a"],
    intersects=geom,
    datetime=f"{YEAR}-01-01/{YEAR}-12-31",
    query={"eo:cloud_cover": {"lt": CLOUD}}
)

items = list(search.get_items())

if not items:
    raise RuntimeError("No Sentinel-2 scenes found.")

# ----------------------------
# Pick lowest cloud cover
# ----------------------------
best = sorted(
    items,
    key=lambda x: x.properties.get("eo:cloud_cover", 100)
)[0]

best = planetary_computer.sign(best)

print("Best scene:")
print("  ID:", best.id)
print("  Cloud cover:", best.properties["eo:cloud_cover"], "%")
print("  Date:", best.datetime)

# ----------------------------
# Create AOI → Year folder structure
# ----------------------------
out_dir = OUT / args.aoi / str(YEAR)
out_dir.mkdir(parents=True, exist_ok=True)
# ----------------------------
# Download bands
# ----------------------------
for band in BANDS:
    asset = best.assets.get(band)
    if not asset:
        continue

    out_file = out_dir / f"{band}.tif"

    if out_file.exists():
        print(f"{band} already exists. Skipping.")
        continue

    print("Downloading:", band)
    download(asset.href, out_file)

print(f"\n✅ Lowest cloud scene downloaded for {YEAR}")