import argparse
import logging
import sys
from pathlib import Path
import geopandas as gpd
import geoai

# -----------------------------------------
# LOGGING
# -----------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# -----------------------------------------
# ARGUMENTS
# -----------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Google satellite embeddings for a given AOI and year.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--aoi",
        required=True,
        help="Name of the AOI shapefile (without extension) inside data/raw/boundaries/",
    )
    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="Year for which to download embeddings.",
    )
    parser.add_argument(
        "--boundaries-dir",
        type=Path,
        default=Path("data/raw/boundaries"),
        help="Root directory containing boundary shapefiles.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/embeddings"),
        help="Root directory for downloaded embeddings.",
    )
    return parser.parse_args()


# -----------------------------------------
# HELPERS
# -----------------------------------------
def load_aoi(path: Path) -> list[float]:
    """Read a shapefile and return its bounding box as [minx, miny, maxx, maxy]."""
    if not path.exists():
        raise FileNotFoundError(f"AOI shapefile not found: {path}")

    log.info("Loading AOI from %s", path)
    aoi = gpd.read_file(path).to_crs("EPSG:4326")

    if aoi.empty:
        raise ValueError(f"AOI shapefile is empty: {path}")

    minx, miny, maxx, maxy = aoi.total_bounds
    bbox = [float(minx), float(miny), float(maxx), float(maxy)]
    log.info("AOI bbox: %s", bbox)
    return bbox


def download_embeddings(bbox: list[float], year: int, output_dir: Path) -> None:
    """Download Google satellite embeddings for a bounding box and year."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("Downloading Google embeddings → %s (year=%d)", output_dir, year)

    geoai.download_google_satellite_embedding(
        bbox=bbox,
        years=[year],
        output_dir=str(output_dir),
    )

    log.info("Embeddings saved to: %s", output_dir)


# -----------------------------------------
# MAIN
# -----------------------------------------
def main() -> None:
    args = parse_args()

    aoi_path = args.boundaries_dir / f"{args.aoi}.shp"
    output_dir = args.output_dir / args.aoi / str(args.year)

    try:
        bbox = load_aoi(aoi_path)
        download_embeddings(bbox, args.year, output_dir)
    except FileNotFoundError as e:
        log.error("Missing file: %s", e)
        sys.exit(1)
    except ValueError as e:
        log.error("Invalid data: %s", e)
        sys.exit(1)
    except Exception as e:
        log.exception("Unexpected error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
