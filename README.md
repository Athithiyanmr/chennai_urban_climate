
# 🌍 Chennai Urban Climate
## Deep Learning-Based Built-Up Area Extraction Using Sentinel-2

Author: Athithiyan M.R  
Domain: Remote Sensing · Geospatial AI · Urban Climate Analytics  
Year: 2026  

---

# 1️⃣ Project Overview

Urban expansion significantly alters land surface properties, influencing:

- Urban Heat Island intensity  
- Flood vulnerability  
- Surface runoff dynamics  
- Carbon balance  
- Climate resilience planning  

Accurate and scalable mapping of built-up areas is therefore critical for climate-informed urban decision-making.

This project develops a reproducible deep learning pipeline to extract built-up (urban) areas from Sentinel-2 satellite imagery using a multi-spectral UNet segmentation framework.

The system integrates:

- Automated satellite acquisition  
- Multi-tile AOI handling  
- Spectral feature engineering  
- Supervised semantic segmentation  
- Quantitative performance evaluation  

---

# 2️⃣ Scientific Objective

The core objective is:

> To learn pixel-level representations of built-up surfaces using multi-spectral Sentinel-2 imagery enhanced with urban-discriminative spectral indices.

Instead of relying on traditional classification methods, this project applies deep convolutional neural networks to perform semantic segmentation at 10m resolution.

---

# 3️⃣ Data Foundation

## Sentinel-2 Level-2A

Spatial Resolution:
- 10 m (Visible + NIR)
- 20 m (SWIR, resampled)

Spectral Bands Used:

- B02 – Blue  
- B03 – Green  
- B04 – Red  
- B08 – Near Infrared  
- B11 – Shortwave Infrared  

Scenes are automatically selected using:

- Lowest cloud cover per year  
- Tile-aware search for large AOIs  
- STAC API integration via Microsoft Planetary Computer  

---

# 4️⃣ Feature Engineering

To improve urban separability, spectral indices are computed:

- NDVI — Vegetation contrast  
- NDBI — Built-up indicator  
- NDWI — Water detection  
- BSI — Bare soil detection  
- IBI — Integrated built-up index  

Final model input:

[B02, B03, B04, B08, B11, NDVI, NDBI, NDWI, BSI, IBI]

This transforms raw satellite imagery into an urban-aware feature space.

---

# 5️⃣ Ground Truth Generation

Supervised labels are derived from:

- OpenStreetMap building footprints  
- Google Open Buildings dataset (recommended)

Building polygons are:

- Reprojected to match Sentinel CRS  
- Rasterized to 10 m resolution  
- Spatially aligned with the spectral stack  

Final output:

Binary segmentation mask  
Built-up = 1  
Non built-up = 0  

---

# 6️⃣ Machine Learning Framework

## Model Architecture

A modified UNet convolutional neural network:

- Encoder-decoder structure  
- Skip connections  
- Multi-spectral 10-channel input  
- Pixel-level classification output  

UNet is chosen for its strong performance in spatial segmentation tasks.

---

## Loss Function

Combined objective:

Loss = Binary Cross Entropy (BCE) + Dice Loss

This balances:

- Pixel-wise classification accuracy  
- Region-level spatial overlap  

---

# 7️⃣ Evaluation Metric

Primary metric:

Intersection over Union (IoU)

IoU = TP / (TP + FP + FN)

Where:
- TP = True Positives  
- FP = False Positives  
- FN = False Negatives  

Performance interpretation:

- < 0.4  → Weak  
- 0.4–0.59 → Moderate  
- 0.6–0.7 → Strong  
- > 0.7 → Research-grade  

Observed validation IoU ≈ 0.60+ for Sentinel-2 10m resolution.

---

# 8️⃣ Full Pipeline Workflow

1. Download lowest-cloud Sentinel-2 scenes  
2. Mosaic and clip to AOI  
3. Build 10-band spectral stack  
4. Generate rasterized training labels  
5. Create balanced image patches  
6. Train UNet segmentation model  
7. Perform sliding-window inference  
8. Evaluate segmentation performance  

---

# 9️⃣ Applications

- Urban Heat Island modeling  
- Flood risk assessment  
- Impervious surface estimation  
- Urban growth monitoring  
- Climate resilience analysis  
- Sustainable development research  

---

# 🔟 Reproducibility

Environment setup:

conda env create -f environment.yml  
conda activate chennai_climate  
export PYTHONPATH=$(pwd)  
export KMP_DUPLICATE_LIB_OK=TRUE  

Run core pipeline:

python scripts/00_download_sentinel2_best_per_year.py  
python scripts/01_prepare_aoi_raw.py  
python scripts/02_build_stack.py  
python scripts/03_make_builtup_labels_from_osm.py  
python -m scripts.dl.make_patches  
python -m scripts.dl.train_unet  
python -m scripts.dl.predict_unet  

---

# 📜 License

MIT License

---

# 🙏 Acknowledgements

- ESA Sentinel-2 Mission  
- Microsoft Planetary Computer  
- OpenStreetMap  
- Google Open Buildings Dataset  

