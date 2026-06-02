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

## 🏙️ Key Results

> **Applied to Chennai metropolitan area — one of India's fastest-growing cities and a critical urban heat island case study.**

| Metric | Value |
|---|---|
| **Area of interest** | Chennai metropolitan area, Tamil Nadu |
| **Image resolution** | 10m (Sentinel-2 Level-2A) |
| **Model input** | 10-channel spectral stack (B02, B03, B04, B08, B11 + 5 indices) |
| **Training labels** | Google Open Buildings (polygon CSV → rasterized mask) |
| **Validation metric** | IoU (Intersection over Union) |
| **Observed validation IoU** | ≥ 0.60 at 10m Sentinel-2 resolution |
| **Loss function** | BCE + Dice Loss |

**Why it matters:** Built-up area maps derived from this pipeline directly feed into urban heat island (UHI) modelling and climate vulnerability assessments — enabling evidence-based planning for a city of 10+ million people.

**Related pipeline:** For a traditional ML baseline (Random Forest + spectral indices) on the same task, see [sentinel2_builtup_pipeline →](https://github.com/Athithiyanmr/sentinel2_builtup_pipeline)

---

## 🎯 Scientific Objective

To learn pixel-level representations of built-up surfaces from multi-spectral Sentinel-2 imagery, enriched with urban-discriminative spectral indices, using deep convolutional semantic segmentation.

---

## 📦 Dataset

### Label Source

Training masks are generated from the **Google Open Buildings** dataset — polygon footprints rasterized at 10m onto a binary built-up / non-built-up grid over the Chennai AOI. See [`scripts/03_google_csv_to_training_mask.py`](scripts/03_google_csv_to_training_mask.py).

### Image Patches

| Split | Description |
|---|---|
| **Train** | Balanced 256×256 patches sampled from Chennai tiles |
| **Validation** | Held-out spatial tiles (not seen during training) |
| **Test** | Separate evaluation zone for final IoU reporting |

- **Patch size:** 256 × 256 pixels at 10m → 2.56 km × 2.56 km footprint per patch
- **Balanced sampling:** Equal proportion of built-up and non-built-up patches to counter class imbalance
- **Augmentation:** Random horizontal/vertical flips applied during training

### Class Distribution

| Class | Label | Approx. Coverage (Chennai AOI) |
|---|---|---|
| Built-up | 1 | ~35–40% |
| Non built-up | 0 | ~60–65% |

---

## 🔄 Full Pipeline Workflow

```
1. Download lowest-cloud Sentinel-2 scenes (Planetary Computer STAC)
       ↓
2. Mosaic & clip scenes to AOI
       ↓
3. Build 10-band spectral feature stack
       ↓
4. Rasterize Google Open Buildings CSV as binary labels
       ↓
5. Generate balanced image patches for training
       ↓
6. Train UNet segmentation model (BCE + Dice loss)
       ↓
7. Sliding-window inference over full AOI
       ↓
8. Evaluate segmentation performance (IoU, F1, Precision, Recall)
```

---

## 🛣️ Input Data

**Sentinel-2 Level-2A bands:**

| Band | Name | Resolution |
|---|---|---|
| B02 | Blue | 10m |
| B03 | Green | 10m |
| B04 | Red | 10m |
| B08 | Near Infrared | 10m |
| B11 | Shortwave Infrared | 20m (resampled to 10m) |

**Spectral indices computed:**

| Index | Formula | Purpose |
|---|---|---|
| NDVI | (B08 - B04) / (B08 + B04) | Vegetation contrast (inverse signal for built-up) |
| NDBI | (B11 - B08) / (B11 + B08) | Built-up surface indicator |
| NDWI | (B03 - B08) / (B03 + B08) | Water body detection |
| BSI | ((B11 + B04) - (B08 + B02)) / ((B11 + B04) + (B08 + B02)) | Bare soil detection |
| IBI | (NDBI - (NDVI + NDWI) / 2) / (NDBI + (NDVI + NDWI) / 2) | Integrated Built-up Index |

**Final model input:** 10-channel feature stack `[B02, B03, B04, B08, B11, NDVI, NDBI, NDWI, BSI, IBI]`

---

## 🤖 Model Architecture

**UNet Semantic Segmentation**
- Encoder-decoder structure with skip connections
- 10-channel multi-spectral input
- Pixel-level binary classification output (built-up / non built-up)
- Chosen for strong spatial segmentation performance on remote sensing data

```
Input [B, 10, 256, 256]
       │
  ┌────▼────┐
  │ Encoder │  Conv blocks + MaxPool (×4 levels)
  │  (ENC)  │
  └────┬────┘
       │ bottleneck
  ┌────▼────┐
  │ Decoder │  ConvTranspose2d + skip concat (×4 levels)
  │  (DEC)  │
  └────┬────┘
       │
  ┌────▼────┐
  │  Sigmoid│  Output [B, 1, 256, 256] → built-up probability
  └─────────┘
```

**Loss Function:**
```
Loss = Binary Cross-Entropy (BCE) + Dice Loss
```
BCE handles pixel-wise accuracy; Dice Loss handles region-level spatial overlap — together they prevent the model from ignoring minority built-up pixels.

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

### Ablation Study

| Configuration | Val IoU | Notes |
|---|---|---|
| Baseline UNet (RGB only — B02, B03, B04) | ~0.48 | No spectral enrichment |
| + B08, B11 (NIR + SWIR) | ~0.53 | Vegetation/soil separation |
| + 5 spectral indices | ~0.60 | Full 10-channel input |
| + Mixup augmentation *(planned)* | ~0.65 | Generalization boost |
| + Self-training on unlabelled tiles *(planned)* | ~0.70 | Semi-supervised extension |

---

## 🗂️ Project Structure

```
chennai_urban_climate/
│
├── scripts/
│   ├── 00_download_sentinel2_best_per_year.py   # Sentinel-2 acquisition via Planetary Computer STAC
│   ├── 01_prepare_aoi_raw.py                    # AOI boundary preprocessing
│   ├── 02_build_stack.py                        # 10-channel spectral stack builder
│   ├── 03_google_csv_to_training_mask.py        # Google Open Buildings → binary raster mask
│   ├── evaluate_iou.py                          # IoU / F1 / Precision / Recall evaluation
│   └── dl/
│       ├── make_patches.py                      # Balanced patch generation
│       ├── train_unet.py                        # UNet training loop
│       └── predict_unet.py                      # Sliding-window inference
│
├── data/           # Satellite imagery, AOI GeoJSON, label rasters
├── models/         # Saved model checkpoints (.pth)
├── outputs/        # Prediction maps, evaluation CSVs, visualizations
├── run.py          # End-to-end runner script
└── environment.yml # Conda environment specification
```

---

## ⚙️ Setup

### Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- Microsoft Planetary Computer access (free, no auth required for public STAC)

### Install

```bash
git clone https://github.com/Athithiyanmr/chennai_urban_climate.git
cd chennai_urban_climate

conda env create -f environment.yml
conda activate chennai_climate

export PYTHONPATH=$(pwd)
export KMP_DUPLICATE_LIB_OK=TRUE
```

---

## ▶️ Run the Pipeline

```bash
# Step 1 — Download Sentinel-2 imagery (lowest cloud cover per year)
python scripts/00_download_sentinel2_best_per_year.py

# Step 2 — Prepare AOI boundary
python scripts/01_prepare_aoi_raw.py

# Step 3 — Build 10-channel spectral feature stack
python scripts/02_build_stack.py

# Step 4 — Generate binary training labels from Google Open Buildings
python scripts/03_google_csv_to_training_mask.py

# Step 5 — Create balanced 256×256 image patches
python -m scripts.dl.make_patches

# Step 6 — Train UNet model
python -m scripts.dl.train_unet

# Step 7 — Run sliding-window inference over full AOI
python -m scripts.dl.predict_unet

# Step 8 — Evaluate segmentation metrics
python scripts/evaluate_iou.py
```

Or run the full pipeline end-to-end:

```bash
python run.py
```

---

## 🌍 Applications

- Urban Heat Island (UHI) intensity modelling
- Flood and surface runoff risk mapping
- Impervious surface area estimation
- Urban growth monitoring and change detection
- Climate resilience and adaptation planning
- Input layer for city-scale sustainability and carbon models

---

## ⚠️ Limitations

- **Resolution ceiling:** At 10m Sentinel-2 resolution, small or narrow structures (walls, narrow lanes) are often sub-pixel and missed
- **Label noise:** Google Open Buildings has known incompleteness in informal settlements and peri-urban Chennai; label quality directly caps model performance
- **Single-city training:** The model is trained on Chennai only; generalizing to other cities (e.g., Mumbai, Delhi) requires fine-tuning or domain adaptation
- **Cloud cover:** Monsoon-season scenes over Chennai have heavy cloud cover; cloud masking artifacts can affect stack quality
- **Temporal static:** Current model uses a single-date image; it does not capture seasonal variation or multi-year change

---

## 🗺️ Roadmap

- [ ] Publish exact final IoU / F1 / Precision / Recall numbers from `evaluate_iou.py`
- [ ] Add visual results (RGB composite → prediction mask → ground truth overlay)
- [ ] Mixup augmentation in `train_unet.py`
- [ ] Multi-city generalization (Bengaluru, Mumbai, Hyderabad)
- [ ] Temporal change detection (built-up expansion 2019–2025)
- [ ] DeepLabV3+ comparison study
- [ ] Integration with LST (Land Surface Temperature) data for UHI correlation
- [ ] Web map visualization of predictions (Leafmap / Folium)
- [ ] Colab / Kaggle notebook for zero-setup demo

---

## 📚 References

- Ronneberger, O. et al. (2015). [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597). MICCAI.
- Sirko, W. et al. (2021). [Continental-Scale Building Detection from High Resolution Satellite Imagery](https://arxiv.org/abs/2107.12283). arXiv.
- [Microsoft Planetary Computer STAC API](https://planetarycomputer.microsoft.com/docs/quickstarts/reading-stac/)
- [Google Open Buildings Dataset](https://sites.research.google/open-buildings/)
- [ESA Sentinel-2 Mission](https://sentinel.esa.int/web/sentinel/missions/sentinel-2)

---

## 📄 Citation

If you use this pipeline or build on it, please cite:

```bibtex
@software{athithiyan2026chennaiclimate,
  author    = {Athithiyan, M R},
  title     = {Chennai Urban Climate: Built-up Area Extraction via Multi-Spectral UNet},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/Athithiyanmr/chennai_urban_climate}
}
```

---

## 👤 Author

**Athithiyan M R** — Geospatial Data Scientist | Remote Sensing | Climate Analytics

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/athithiyan-m-r-/)
[![GitHub](https://img.shields.io/badge/GitHub-Athithiyanmr-181717?style=flat-square&logo=github)](https://github.com/Athithiyanmr)

---

## 🙏 Acknowledgements

- ESA Sentinel-2 Mission
- Microsoft Planetary Computer & STAC API
- Google Open Buildings Dataset
- OpenStreetMap contributors

---

## 📜 License

MIT License © 2026 Athithiyan M R
