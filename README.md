## RUN FULL PIPELINE (Single-Shot Execution)

### 1. Create environment
conda env create -f environment.yml
conda activate chennai_climate
### 2. Set project path
export PYTHONPATH=$(pwd)
export KMP_DUPLICATE_LIB_OK=TRUE

### 3. Run full pipeline
python scripts/00_download_sentinel2_best_per_year.py
python scripts/01_prepare_aoi_raw.py
python scripts/02_build_stack.py
python scripts/03_make_builtup_labels_from_osm.py

# Deep Learning pipeline
python -m scripts.dl.make_patches
python -m scripts.dl.train_unet
python -m scripts.dl.predict_unet

# Random Forest baseline (optional)
python scripts/04_builtup_rf_yearwise.py