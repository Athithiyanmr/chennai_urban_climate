from pathlib import Path
import geopandas as gpd
import pystac_client
import planetary_computer
import requests
from collections import defaultdict

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
AOI = "data/raw/boundaries/CMDA.shp"
OUT = Path("data/raw/sentinel2")

YEARS = [2023, 2024, 2025]

# Sentinel-2 bands (10 m + 20 m)
BANDS = ["B02", "B03", "B04", "B08", "B11"]

OUT.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------
# Safe download helper
# -------------------------------------------------
def download(url, out_path):
    tmp = out_path.with_suffix(".tmp")

    with requests.get(url, stream=True, timeout=90) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    # Sentinel-2 band files are large (sanity check)
    if tmp.stat().st_size < 10_000_000:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("Downloaded file too small (corrupted)")

    tmp.rename(out_path)

# -------------------------------------------------
# Load AOI (lat/lon for STAC)
# -------------------------------------------------
aoi = gpd.read_file(AOI)
aoi = aoi.to_crs("EPSG:4326")
geom = aoi.geometry.iloc[0].__geo_interface__

# -------------------------------------------------
# Open Planetary Computer STAC
# -------------------------------------------------
catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1"
)

# -------------------------------------------------
# MAIN LOOP
# -------------------------------------------------
for year in YEARS:
    print(f"\n🔍 Searching Sentinel-2 scenes for {year} ...")

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects=geom,
        datetime=f"{year}-01-01/{year}-12-31",
        query={"eo:cloud_cover": {"lt": 20}}
    )

    items = list(search.get_items())

    if not items:
        print(f"❌ No Sentinel-2 scenes found for {year}")
        continue

    # -------------------------------------------------
    # Group scenes by MGRS tile
    # -------------------------------------------------
    by_tile = defaultdict(list)
    for item in items:
        tile = item.properties.get("s2:mgrs_tile")
        if tile:
            by_tile[tile].append(item)

    print(f"📦 Found {len(by_tile)} tiles intersecting AOI")

    # -------------------------------------------------
    # Download best scene per tile
    # -------------------------------------------------
    for tile, tile_items in by_tile.items():
        best = sorted(
            tile_items,
            key=lambda x: x.properties.get("eo:cloud_cover", 100)
        )[0]

        best = planetary_computer.sign(best)

        out_dir = OUT / str(year) / f"T{tile}"
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"⬇️  {year} | Tile {tile} | Cloud {best.properties['eo:cloud_cover']}%")

        for band in BANDS:
            asset = best.assets.get(band)
            if not asset:
                continue

            out_file = out_dir / f"{best.id}_{band}.tif"
            if out_file.exists():
                continue

            download(asset.href, out_file)

    print(f"✅ Completed Sentinel-2 download for {year}")