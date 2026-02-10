# scripts/04_builtup_rf_yearwise.py
import rasterio
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

LABELS = "data/raw/training/builtup_labels.tif"

# ---------------------------
# Load labels
# ---------------------------
with rasterio.open(LABELS) as src:
    y = src.read(1).reshape(-1)

# ---------------------------
# RF model
# ---------------------------
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_leaf=2,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)

trained = False
OUT = Path("outputs/rasters")
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Loop over yearly stacks
# ---------------------------
for stack in Path("data/processed").glob("chennai_stack_*.tif"):
    year = stack.stem.split("_")[-1]

    with rasterio.open(stack) as src:
        img = src.read().astype("float32")
        bands, rows, cols = img.shape

        # reshape automatically based on band count
        X = img.reshape(bands, -1).T
        meta = src.meta

    # valid pixels
    mask = (y >= 0) & (~np.isnan(X).any(axis=1))

    # train only once
    if not trained:
        print("Training RF on", year)
        rf.fit(X[mask], y[mask])
        trained = True

    # predict
    pred = rf.predict(X).reshape(rows, cols)

    meta.update(count=1, dtype="uint8", nodata=0)

    out_path = OUT / f"chennai_builtup_rf_{year}.tif"
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(pred.astype("uint8"), 1)

    print("RF built-up map:", year)


    