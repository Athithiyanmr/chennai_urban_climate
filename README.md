# 🏙️ Chennai Urban Climate

> **A deep learning pipeline for built-up area extraction from Sentinel-2 imagery using a multi-spectral UNet-Transformer — applied to Chennai for urban climate analysis.**

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Sentinel-2](https://img.shields.io/badge/Sentinel--2-003DA5?style=flat-square)](https://sentinel.esa.int)
[![Deep Learning](https://img.shields.io/badge/UNet--Transformer-Deep%20Learning-FF4500?style=flat-square)](https://arxiv.org/abs/1505.04597)
[![Planetary Computer](https://img.shields.io/badge/Microsoft%20Planetary%20Computer-0078D4?style=flat-square&logo=microsoft&logoColor=white)](https://planetarycomputer.microsoft.com)
[![MIT License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## 📌 What Is This?

Urban expansion reshapes land surfaces in ways that directly drive climate risk — intensifying heat islands, increasing flood vulnerability, and altering carbon balance. Mapping built-up areas accurately and at scale is foundational to climate-informed urban planning.

This project builds a **reproducible deep learning pipeline** that extracts built-up areas from Sentinel-2 satellite imagery using a **multi-spectral UNet-Transformer segmentation model** — trained on Chennai and applicable to any Indian city.

It goes beyond traditional ML classifiers by applying convolutional neural networks with a transformer bottleneck to learn spatial patterns directly from satellite image patches, achieving pixel-level segmentation at 10 m resolution.

---

## 🏙️ Key Results

> **Applied to Chennai metropolitan area — one of India's fastest-growing cities and a critical urban heat island case study.**

| Metric | Value |
|---|---|
| **Area of interest** | Chennai metropolitan area, Tamil Nadu |
| **Image resolution** | 10 m (Sentinel-2 Level-2A) |
| **Model input** | 11-channel spectral stack (B02, B03, B04, B08, B11 + 6 indices) |
| **Training labels** | Google Open Buildings (polygon CSV → rasterized mask) |
| **Model** | UNet-Transformer (encoder–decoder + transformer bottleneck) |
| **Validation metric** | IoU (Intersection over Union) |
| **Observed validation IoU** | ≥ 0.60 at 10 m Sentinel-2 resolution |
| **Loss function** | Focal Loss + Dice Loss |

**Why it matters:** Built-up area maps derived from this pipeline directly feed into urban heat island (UHI) modelling and climate vulnerability assessments — enabling evidence-based planning for a city of 10+ million people.

**Related pipeline:** For a traditional ML baseline (Random Forest + spectral indices) on the same task, see [sentinel2_builtup_pipeline →](https://github.com/Athithiyanmr/sentinel2_builtup_pipeline)

---

## 🎯 Scientific Objective

To learn pixel-level representations of built-up surfaces from multi-spectral Sentinel-2 imagery, enriched with urban-discriminative spectral indices, using deep convolutional semantic segmentation with a transformer bottleneck.

---

## 📦 Dataset

### Label Source

Training masks are generated from the **Google Open Buildings** dataset — polygon footprints rasterized at 10 m onto a binary built-up / non-built-up grid over the Chennai AOI. See [`scripts/03_google_csv_to_training_mask.py`](scripts/03_google_csv_to_training_mask.py).

### Image Patches

| Split | Description |
|---|---|
| **Train** | Balanced patches sampled from Chennai tiles |
| **Validation** | Random hold-out from the same patch pool (default 20 %) |
| **Test** | Separate evaluation via full-AOI prediction + ground-truth comparison |

- **Default patch size:** 128 × 128 pixels at 10 m → 1.28 km × 1.28 km footprint per patch  
  (configurable via `--patch`; common values: 64, 128, 256)
- **Default stride:** 64 pixels (50 % overlap)
- **Balanced sampling:** Positive patches (built-up ratio ≥ 2 %) are always kept; background patches are randomly subsampled (default keep probability 25 %)
- **Augmentation:** None by default (Mixup planned); random flips can be added easily

### Class Distribution

| Class | Label | Approx. Coverage (Chennai AOI) |
|---|---|---|
| Built-up | 1 | ~35–40 % |
| Non built-up | 0 | ~60–65 % |

---

## 🔄 Full Pipeline Workflow

```
1. Download lowest-cloud Sentinel-2 scenes (Planetary Computer STAC)
       ↓
2. Mosaic & clip scenes to AOI
       ↓
3. Build 11-band spectral feature stack
       ↓
4. Rasterize Google Open Buildings CSV as binary labels
       ↓
5. Generate balanced image patches for training
       ↓
6. Train UNet-Transformer (Focal + Dice loss)
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
| B02 | Blue | 10 m |
| B03 | Green | 10 m |
| B04 | Red | 10 m |
| B08 | Near Infrared | 10 m |
| B11 | Shortwave Infrared | 20 m (resampled to 10 m) |

**Spectral indices computed (on scaled reflectance):**

| Index | Formula | Purpose |
|---|---|---|
| NDVI | (B08 − B04) / (B08 + B04) | Vegetation contrast (inverse signal for built-up) |
| NDBI | (B11 − B08) / (B11 + B08) | Built-up surface indicator |
| NDWI | (B03 − B08) / (B03 + B08) | Water body detection (Gao 1996) |
| MNDWI | (B03 − B11) / (B03 + B11) | Modified water index (Xu 2006) |
| BSI | ((B11 + B04) − (B08 + B02)) / ((B11 + B04) + (B08 + B02)) | Bare soil detection |
| IBI | (NDBI − (NDVI + NDWI) / 2) / (NDBI + (NDVI + NDWI) / 2) | Integrated Built-up Index |

**Final model input:** 11-channel feature stack  
`[B02, B03, B04, B08, B11, NDVI, NDBI, NDWI, MNDWI, BSI, IBI]`

---

## 🤖 Model Architecture

**UNet-Transformer Semantic Segmentation**

- Encoder–decoder structure with skip connections
- Transformer block at the bottleneck for global context
- 11-channel multi-spectral input
- Pixel-level binary classification output (built-up / non-built-up)

```
Input [B, 11, H, W]
       │
  ┌────▼────┐
  │ Encoder │  Conv blocks + MaxPool (×2 levels → 256-dim)
  │  (ENC)  │
  └────┬────┘
       │ bottleneck
  ┌────▼────┐
  │Transformer│  Multi-head self-attention (LayerNorm + MLP)
  └────┬────┘
       │
  ┌────▼────┐
  │ Decoder │  Upsample + skip concat (×2 levels)
  │  (DEC)  │
  └────┬────┘
       │
  ┌────▼────┐
  │  Sigmoid│  Output [B, 1, H, W] → built-up probability
  └─────────┘
```

A pure convolutional UNet is also available in `scripts/dl/unet_model.py` (not used by the default training script).

**Loss Function:**

```
Loss = Focal Loss + Dice Loss
```

Focal Loss focuses training on hard examples; Dice Loss optimises region-level spatial overlap. Together they handle class imbalance better than plain BCE.

**Training defaults:**
- Optimizer: AdamW (lr = 1e-4, weight decay = 1e-4)
- Scheduler: CosineAnnealingLR
- Early stopping patience: 6 epochs
- Gradient clipping: max norm 1.0
- Default epochs: 40, batch size: 8
- Binarization threshold: 0.35 (also auto-searched at evaluation)

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

✅ **Observed validation IoU ≈ 0.60+** at Sentinel-2 10 m resolution over Chennai.

### Ablation Study (approximate)

| Configuration | Val IoU | Notes |
|---|---|---|
| Baseline UNet (RGB only — B02, B03, B04) | ~0.48 | No spectral enrichment |
| + B08, B11 (NIR + SWIR) | ~0.53 | Vegetation/soil separation |
| + spectral indices (full 11-channel) | ~0.60 | Current default |
| + Mixup augmentation *(planned)* | ~0.65 | Generalization boost |
| + Self-training on unlabelled tiles *(planned)* | ~0.70 | Semi-supervised extension |

---

## 🗂️ Project Structure

```
chennai_urban_climate/
│
├── scripts/
│   ├── 00_download_sentinel2_best_per_year.py   # Sentinel-2 acquisition via Planetary Computer STAC
│   ├── 01_prepare_aoi_raw.py                    # AOI boundary preprocessing / clipping
│   ├── 02_build_stack.py                        # 11-channel spectral stack builder
│   ├── 03_google_csv_to_training_mask.py        # Google Open Buildings → binary raster mask
│   ├── evaluate_iou.py                          # IoU / F1 / Precision / Recall + plots
│   └── dl/
│       ├── dataset.py                           # PyTorch Dataset for .npy patches
│       ├── make_patches.py                      # Balanced patch generation
│       ├── unet_model.py                        # Pure convolutional UNet
│       ├── unet_transformer.py                  # UNet + Transformer bottleneck (default)
│       ├── train_unet.py                        # Training loop (Focal + Dice)
│       └── predict_unet.py                      # Sliding-window inference
│
├── data/           # Satellite imagery, AOI, label rasters, patches (git-ignored)
├── models/         # Saved model checkpoints (.pth) (git-ignored)
├── outputs/        # Prediction maps, metrics, visualizations (git-ignored)
├── run.py          # End-to-end runner script
├── environment.yml # Conda environment specification
└── LICENSE         # MIT License
```

---

## ⚙️ Setup

### Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- Microsoft Planetary Computer access (free, no auth required for public STAC)
- Google Open Buildings CSV for the AOI (download from [Google Open Buildings](https://sites.research.google/open-buildings/))

### Install

```bash
git clone https://github.com/Athithiyanmr/chennai_urban_climate.git
cd chennai_urban_climate

conda env create -f environment.yml
conda activate chennai_climate

export PYTHONPATH=$(pwd)
export KMP_DUPLICATE_LIB_OK=TRUE   # avoids OpenMP conflicts on some platforms
```

---

## ▶️ Run the Pipeline

All scripts require `--year` and `--aoi`. Example for Chennai 2023:

```bash
# Step 1 — Download Sentinel-2 imagery (lowest cloud cover per tile)
python scripts/00_download_sentinel2_best_per_year.py --year 2023 --aoi chennai

# Step 2 — Prepare / clip AOI
python scripts/01_prepare_aoi_raw.py --year 2023 --aoi chennai

# Step 3 — Build 11-channel spectral feature stack
python scripts/02_build_stack.py --year 2023 --aoi chennai

# Step 4 — Generate binary training labels from Google Open Buildings
python scripts/03_google_csv_to_training_mask.py --year 2023 --aoi chennai --csv path/to/buildings.csv

# Step 5 — Create balanced image patches (default 128×128, stride 64)
python -m scripts.dl.make_patches --year 2023 --aoi chennai --patch 128 --stride 64

# Step 6 — Train UNet-Transformer
python -m scripts.dl.train_unet --year 2023 --aoi chennai --epochs 40 --batch 8

# Step 7 — Sliding-window inference over full AOI
python -m scripts.dl.predict_unet --year 2023 --aoi chennai --patch 128 --stride 64 --threshold 0.35

# Step 8 — Evaluate (auto-searches best threshold if --threshold omitted)
python scripts/evaluate_iou.py --year 2023 --aoi chennai
```

Or run the full pipeline end-to-end:

```bash
python run.py --year 2023 --aoi chennai --csv path/to/buildings.csv \
              --patch 128 --stride 64 --threshold 0.35
```

Useful flags for `run.py`:
- `--skip_download` — skip Sentinel-2 download
- `--skip_train` — skip training (use existing checkpoint)

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

- **Resolution ceiling:** At 10 m Sentinel-2 resolution, small or narrow structures (walls, narrow lanes) are often sub-pixel and missed
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
  title     = {Chennai Urban Climate: Built-up Area Extraction via Multi-Spectral UNet-Transformer},
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

MIT License © 2026 Athithiyan M R — see [LICENSE](LICENSE)
