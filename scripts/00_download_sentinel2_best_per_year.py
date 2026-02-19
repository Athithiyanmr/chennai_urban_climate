import argparse
from pathlib import Path
import geopandas as gpd
import pystac_client
import planetary_computer
import requests
from collections import defaultdict

# -----------------------------------------
# ARGUMENTS
# -----------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--aoi", required=True)
parser.add_argument("--year", type=int, required=True)
parser.add_argument("--cloud", type=int, default=40)
args = parser.parse_args()

AOI = f"data/raw/boundaries/{args.aoi}.shp"
YEAR = args.year
CLOUD = args.cloud

OUT = Path("data/raw/sentinel2") / args.aoi / str(YEAR)
OUT.mkdir(parents=True, exist_ok=True)

BANDS = ["B02", "B03", "B04", "B08", "B11"]

# -----------------------------------------
# SAFE DOWNLOAD (resume + corruption check)
# -----------------------------------------
def download(url, out_path):

    tmp = out_path.with_suffix(".tmp")

    try:
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1024 * 1024):
                    if chunk:
                        f.write(chunk)

        if tmp.stat().st_size < 5_000_000:
            tmp.unlink(missing_ok=True)
            raise RuntimeError("Corrupted download")

        tmp.rename(out_path)

    except Exception as e:
        print("❌ Download failed:", out_path.name)
        tmp.unlink(missing_ok=True)


# -----------------------------------------
# LOAD AOI
# -----------------------------------------
print("Loading AOI...")
aoi = gpd.read_file(AOI).to_crs("EPSG:4326")
geom = aoi.geometry.iloc[0].__geo_interface__

# -----------------------------------------
# OPEN STAC
# -----------------------------------------
catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1"
)

print(f"\n🔎 Searching Sentinel-2 {YEAR}")

search = catalog.search(
    collections=["sentinel-2-l2a"],
    intersects=geom,
    datetime=f"{YEAR}-01-01/{YEAR}-12-31",
    query={"eo:cloud_cover": {"lt": CLOUD}},
)

items = list(search.get_items())

if not items:
    raise RuntimeError("No scenes found")

# -----------------------------------------
# GROUP BY TILE
# -----------------------------------------
by_tile = defaultdict(list)

for item in items:
    tile = item.properties.get("s2:mgrs_tile")
    if tile:
        by_tile[tile].append(item)

print("Tiles intersecting AOI:", list(by_tile.keys()))

# -----------------------------------------
# DOWNLOAD BEST SCENE PER TILE
# -----------------------------------------
for tile, tile_items in by_tile.items():

    best = sorted(
        tile_items,
        key=lambda x: x.properties.get("eo:cloud_cover", 100)
    )[0]

    best = planetary_computer.sign(best)

    tile_dir = OUT / f"T{tile}"
    tile_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n⬇ Tile {tile}")
    print("   Cloud:", best.properties["eo:cloud_cover"])
    print("   Date :", best.datetime)

    for band in BANDS:
        asset = best.assets.get(band)
        if not asset:
            continue

        out_file = tile_dir / f"{band}.tif"

        if out_file.exists():
            continue

        print("   Downloading:", band)
        download(asset.href, out_file)

print("\n✅ Sentinel-2 download complete")