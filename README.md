# 🌍 Built-up Mapping using Sentinel-2 + Deep Learning

This project performs automatic built-up extraction from Sentinel-2
imagery using spectral indices and UNet segmentation.


## Setup

conda env create -f environment.yml conda activate chennai_climate

export PYTHONPATH=\$(pwd) export KMP_DUPLICATE_LIB_OK=TRUE

## Run Pipeline

python scripts/00_download_sentinel2_best_per_year.py --aoi auroville
--year 2025 python scripts/01_prepare_aoi_raw.py --aoi auroville --year
2025 python scripts/02_build_stack.py --aoi auroville --year 2025

## Google Buildings (Colab)

Use Earth Engine to export Google Open Buildings CSV.

## Deep Learning

python -m scripts.dl.make_patches --aoi auroville --year 2025 --patch 64
--stride 32 --clean python -m scripts.dl.train_unet --aoi auroville
--year 2025 python -m scripts.dl.predict_unet --aoi auroville --year
2025 --threshold 0.35
