
# 🏙️ Chennai Urban Climate

> **A deep learning pipeline for built-up area extraction from Sentinel-2 imagery using UNet semantic segmentation — applied to Chennai for urban climate analysis.**

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Sentinel-2](https://img.shields.io/badge/Sentinel--2-003DA5?style=flat-square)](https://sentinel.esa.int)
[![Deep Learning](https://img.shields.io/badge/UNet-Deep%20Learning-FF4500?style=flat-square)](https://arxiv.org/abs/1505.04597)
[![Planetary Computer](https://img.shields.io/badge/Microsoft%20Planetary%20Computer-0078D4?style=flat-square&logo=microsoft&logoColor=white)](https://planetarycomputer.microsoft.com)
[![MIT License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## 📌 What Is This?

Urban expansion reshapes land surfaces in ways that directly drive climate risk — intensifying heat islands, increasing flood vulnerability, and altering carbon balance. Mapping built-up areas accurately and at scale is foundational to climate-informed urban planning.

This project builds a **reproducible deep learning pipeline** that extracts built-up areas from Sentinel-2 satellite imagery using a **multi-spectral UNet segmentation model** — trained on Chennai and applicable to any Indian city.

It goes beyond traditional ML classifiers by applying convolutional neural networks to learn spatial patterns directly from satellite image patches, achieving pixel-level segmentation at 10m resolution.

---

## 🎯 Scientific Objective

To learn pixel-level representations of built-up surfaces from multi-spectral Sentinel-2 imagery, enriched with urban-discriminative spectral indices, using deep convolutional semantic segmentation.

---

## 🔄 Full Pipeline Workflow

```
1. Download lowest-cloud Sentinel-2 scenes (Planetary Computer STAC)
       ↓
2. Mosaic & clip scenes to AOI
       ↓
3. Build 10-band spectral feature stack
       ↓
4. Rasterize OSM / Google Open Buildings as binary labels
       ↓
5. Generate balanced image patches for training
       ↓
6. Train UNet segmentation model (BCE + Dice loss)
       ↓
7. Sliding-window inference over full AOI
       ↓
8. Evaluate segmentation performance (IoU)
```

---

## 🛰️ Input Data

**Sentinel-2 Level-2A bands:**

| Band | Name | Resolution |
|---|---|---|
| B02 | Blue | 10m |
| B03 | Green | 10m |
| B04 | Red | 10m |
| B08 | Near Infrared | 10m |
| B11 | Shortwave Infrared | 20m (resampled to 10m) |

**Spectral indices computed:**

| Index | Purpose |
|---|---|
| NDVI | Vegetation contrast (inverse signal for built-up) |
| NDBI | Built-up surface indicator |
| NDWI | Water body detection |
| BSI | Bare soil detection |
| IBI | Integrated Built-up Index |

**Final model input:** 10-channel feature stack `[B02, B03, B04, B08, B11, NDVI, NDBI, NDWI, BSI, IBI]`

---

## 🤖 Model Architecture

**UNet Semantic Segmentation**
- Encoder-decoder structure with skip connections
- 10-channel multi-spectral input
- Pixel-level binary classification output (built-up / non built-up)
- Chosen for strong spatial segmentation performance on remote sensing data

**Loss Function:**
```
Loss = Binary Cross-Entropy (BCE) + Dice Loss
```
BCE handles pixel-wise accuracy; Dice Loss handles region-level spatial overlap.

---

## 📈 Performance

**Primary metric: Intersection over Union (IoU)**

```
IoU = TP / (TP + FP + FN)
```

| IoU Range | Interpretation |
|---|---|
| < 0.40 | Weak |
| 0.40 – 0.59 | Moderate |
| 0.60 – 0.70 | Strong |
| > 0.70 | Research-grade |

✅ **Observed validation IoU ≈ 0.60+** at Sentinel-2 10m resolution over Chennai.

---

## 🗂️ Project Structure

```
chennai_urban_climate/
│
├── scripts/
│   ├── 00_download_sentinel2_best_per_year.py   # Satellite acquisition
│   ├── 01_prepare_aoi_raw.py                    # AOI preprocessing
│   ├── 02_build_stack.py                        # Spectral stack builder
│   ├── 03_make_builtup_labels_from_osm.py       # Label rasterization
│   └── dl/
│       ├── make_patches.py                      # Patch generation
│       ├── train_unet.py                        # UNet training
│       └── predict_unet.py                      # Inference
│
├── data/           # Satellite imagery, AOI, labels
├── models/         # Saved model checkpoints
├── outputs/        # Prediction maps and evaluation results
├── run.py          # End-to-end runner script
└── environment.yml
```

---

## ⚙️ Setup

```bash
conda env create -f environment.yml
conda activate chennai_climate
export PYTHONPATH=$(pwd)
export KMP_DUPLICATE_LIB_OK=TRUE
```

---

## ▶️ Run the Pipeline

```bash
# Step 1 — Download Sentinel-2 imagery
python scripts/00_download_sentinel2_best_per_year.py

# Step 2 — Prepare AOI
python scripts/01_prepare_aoi_raw.py

# Step 3 — Build spectral feature stack
python scripts/02_build_stack.py

# Step 4 — Generate training labels from OSM
python scripts/03_make_builtup_labels_from_osm.py

# Step 5 — Create image patches
python -m scripts.dl.make_patches

# Step 6 — Train UNet model
python -m scripts.dl.train_unet

# Step 7 — Run inference
python -m scripts.dl.predict_unet
```

---

## 🌍 Applications

- Urban Heat Island modeling
- Flood and surface runoff risk assessment
- Impervious surface area estimation
- Urban growth monitoring over time
- Climate resilience and adaptation planning
- Input layer for city-scale sustainability models

---

## 🗺️ Roadmap

- [ ] Multi-city generalization (Bengaluru, Mumbai, Hyderabad)
- [ ] Temporal change detection (built-up expansion over years)
- [ ] DeepLabV3+ comparison study
- [ ] Web map visualization of predictions

---

## 👤 Author

**Athithiyan M R** — Geospatial Data Scientist | Remote Sensing | Climate Analytics

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/athithiyan-m-r-/)
[![GitHub](https://img.shields.io/badge/GitHub-Athithiyanmr-181717?style=flat-square&logo=github)](https://github.com/Athithiyanmr)

---

## 🙏 Acknowledgements

- ESA Sentinel-2 Mission
- Microsoft Planetary Computer & STAC API
- OpenStreetMap contributors
- Google Open Buildings Dataset

---

## 📜 License

MIT License © 2026 Athithiyan M R
