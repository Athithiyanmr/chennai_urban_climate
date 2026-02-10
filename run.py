import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--year", required=True, help="Year to process")
args = parser.parse_args()

YEAR = args.year

steps = [
    # Clean before starting
    'find . -name "._*" -type f -delete',

    "python scripts/00_download_sentinel2_best_per_year.py",
    "python scripts/01_prepare_aoi_raw.py",
    "python scripts/02_build_stack.py",
    "python scripts/03_make_builtup_labels_from_osm.py",

    f"python -m scripts.dl.make_patches --year {YEAR}",

    # Clean again after patches
    'find data/dl -name "._*" -type f -delete',

    f"python -m scripts.dl.train_unet --year {YEAR}",
    f"python -m scripts.dl.predict_unet --year {YEAR}",
]
for step in steps:
    print("\nRunning:", step)
    subprocess.run(step, shell=True, check=True)