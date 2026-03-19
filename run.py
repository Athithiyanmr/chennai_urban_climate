import subprocess
import argparse
import os


# --------------------------------
# ARGUMENTS
# --------------------------------
parser = argparse.ArgumentParser()

parser.add_argument("--year", required=True)
parser.add_argument("--aoi", required=True)

# optional Google CSV
parser.add_argument("--csv", default=None)

# DL tuning
parser.add_argument("--patch", type=int, default=64)
parser.add_argument("--stride", type=int, default=32)
parser.add_argument("--threshold", type=float, default=0.35)

# optional skip flags (VERY useful later)
parser.add_argument("--skip_download", action="store_true")
parser.add_argument("--skip_train", action="store_true")

args = parser.parse_args()

YEAR = args.year
AOI = args.aoi
CSV = args.csv
PATCH = args.patch
STRIDE = args.stride
THRESHOLD = args.threshold


# --------------------------------
# Fix OpenMP crash (Mac)
# --------------------------------
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# --------------------------------
# Helper
# --------------------------------
def run(cmd):
    print("\n🚀 Running:", cmd)
    subprocess.run(cmd, shell=True, check=True)


# --------------------------------
# CLEAN hidden files
# --------------------------------
run('find . -name "._*" -type f -delete')


# --------------------------------
# PIPELINE
# --------------------------------

# 1️⃣ Download Sentinel
if not args.skip_download:
    run(f"python scripts/00_download_sentinel2_best_per_year.py --year {YEAR} --aoi {AOI}")

run('find . -name "._*" -type f -delete')
# 2️⃣ AOI clip
run(f"python scripts/01_prepare_aoi_raw.py --year {YEAR} --aoi {AOI}")

run('find . -name "._*" -type f -delete')
# 3️⃣ Build stack
run(f"python scripts/02_build_stack.py --year {YEAR} --aoi {AOI}")

run('find . -name "._*" -type f -delete')

# 4️⃣ Training labels
run(f"python scripts/03_google_csv_to_training_mask.py "
        f"--year {YEAR} --aoi {AOI} --csv {CSV}")



# 5️⃣ Create patches
run(
    f"python -m scripts.dl.make_patches "
    f"--year {YEAR} --aoi {AOI} "
    f"--patch {PATCH} --stride {STRIDE}"
)

# clean macOS hidden patch files
run('find data/dl -name "._*" -type f -delete')


# 6️⃣ Train
if not args.skip_train:
    run(f"python -m scripts.dl.train_unet --year {YEAR} --aoi {AOI}")


# 7️⃣ Predict
run(
    f"python -m scripts.dl.predict_unet "
    f"--year {YEAR} --aoi {AOI} "
    f"--patch {PATCH} --stride {STRIDE} "
    f"--threshold {THRESHOLD}"
)


# 8️⃣ Evaluate (⭐ VERY IMPORTANT)
run(f"python scripts/evaluate_iou.py --year {YEAR} --aoi {AOI}")


print("\n🎉 FULL PIPELINE COMPLETE")