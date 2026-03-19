import argparse
import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import planetary_computer
import pystac_client
import requests
import rasterio
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

# -----------------------------------------
# LOGGING
# -----------------------------------------
logging.basicConfig(
    filename="download.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# -----------------------------------------
# ARGUMENTS
# -----------------------------------------
parser = argparse.ArgumentParser(description="Download Sentinel-2 (best per tile)")
parser.add_argument("--aoi", required=True, help="AOI shapefile name")
parser.add_argument("--year", type=int, required=True)
parser.add_argument("--cloud", type=int, default=40)
args = parser.parse_args()

AOI   = f"data/raw/boundaries/{args.aoi}.shp"
YEAR  = args.year
CLOUD = args.cloud

OUT = Path("data/raw/sentinel2") / args.aoi / str(YEAR)
OUT.mkdir(parents=True, exist_ok=True)

BANDS = ["B02", "B03", "B04", "B08", "B11"]

# -----------------------------------------
# DOWNLOAD FUNCTION (ROBUST)
# -----------------------------------------
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=4, max=60))
def download(url, out_path):
    tmp = out_path.with_suffix(".tmp")

    try:
        with requests.get(url, stream=True, timeout=180) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))

            with open(tmp, "wb") as f, tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                desc=out_path.name,
                leave=False,
            ) as bar:
                for chunk in r.iter_content(1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))

        # ✅ Validate GeoTIFF properly
        with rasterio.open(tmp) as src:
            _ = src.read(1)

        tmp.rename(out_path)
        log.info(f"Downloaded OK: {out_path}")

    except Exception as e:
        log.error(f"Failed: {out_path.name} — {e}")
        tmp.unlink(missing_ok=True)
        raise

# -----------------------------------------
# LOAD AOI
# -----------------------------------------
print("📍 Loading AOI...")
aoi = gpd.read_file(AOI).to_crs("EPSG:4326")

# ✅ FIX: Use bbox (handles large AOIs like Tamil Nadu)
minx, miny, maxx, maxy = aoi.total_bounds

# -----------------------------------------
# OPEN STAC
# -----------------------------------------
catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

# -----------------------------------------
# SEARCH
# -----------------------------------------
print(f"\n🔎 Searching Sentinel-2 {YEAR} | Cloud < {CLOUD}%")

search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=[minx, miny, maxx, maxy],
    datetime=f"{YEAR}-01-01/{YEAR}-12-31",
    query={"eo:cloud_cover": {"lt": CLOUD}},
)

items = search.item_collection()

if not items:
    raise RuntimeError("No scenes found")

print(f"   Found {len(items)} scenes")

# -----------------------------------------
# GROUP BY TILE
# -----------------------------------------
by_tile = defaultdict(list)

for item in items:
    tile = item.properties.get("s2:mgrs_tile")
    if tile:
        by_tile[tile].append(item)

print("📦 Tiles:", list(by_tile.keys()))

# -----------------------------------------
# DOWNLOAD BEST SCENE PER TILE
# -----------------------------------------
manifest = []

for tile, tile_items in by_tile.items():

    best = sorted(
        tile_items,
        key=lambda x: x.properties.get("eo:cloud_cover", 100)
    )[0]

    tile_dir = OUT / f"T{tile}"
    tile_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n⬇ Tile {tile}")
    print(f"   Cloud: {best.properties['eo:cloud_cover']}%")
    print(f"   Date : {best.datetime}")

    downloaded_bands = []

    for band in BANDS:
        asset = best.assets.get(band)

        if not asset:
            print(f"   ⚠️ {band} missing")
            log.warning(f"{band} missing in tile {tile}")
            continue

        out_file = tile_dir / f"{band}.tif"

        if out_file.exists():
            print(f"   ⏭ {band} exists")
            downloaded_bands.append(band)
            continue

        print(f"   ⬇ {band}")

        try:
            download(asset.href, out_file)
            downloaded_bands.append(band)

        except Exception:
            print(f"   ❌ Failed {band} (skipped)")
            log.error(f"{band} failed for tile {tile}")

    manifest.append({
        "tile": tile,
        "date": str(best.datetime),
        "cloud_cover": best.properties["eo:cloud_cover"],
        "bands": downloaded_bands,
        "downloaded_at": datetime.utcnow().isoformat(),
    })

# -----------------------------------------
# SAVE MANIFEST
# -----------------------------------------
manifest_path = OUT / "manifest.json"

with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"\n📋 Manifest saved → {manifest_path}")
print("✅ Sentinel-2 download complete")