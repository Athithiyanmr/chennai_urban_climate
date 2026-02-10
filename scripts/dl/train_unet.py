# scripts/dl/train_unet.py
import argparse
import torch
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
from tqdm import tqdm
from pathlib import Path

from scripts.dl.dataset import BuiltupDataset
from scripts.dl.unet_model import UNet

# --------------------
# ARGUMENTS
# --------------------
parser = argparse.ArgumentParser()
parser.add_argument("--year", required=True, help="Year to train on (e.g., 2025)")
args = parser.parse_args()

YEAR = args.year

# --------------------
# CONFIG
# --------------------
BATCH_SIZE = 8
EPOCHS = 30
VAL_SPLIT = 0.2
PATIENCE = 5
LR = 1e-4

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# --------------------
# DEVICE
# --------------------
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print("Using device:", device)

# --------------------
# DATASET
# --------------------
img_dir = f"data/dl/{YEAR}/images"
mask_dir = f"data/dl/{YEAR}/masks"

ds = BuiltupDataset(img_dir, mask_dir)

val_size = int(len(ds) * VAL_SPLIT)
train_size = len(ds) - val_size

train_ds, val_ds = random_split(ds, [train_size, val_size])

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

print(f"Year: {YEAR}")
print(f"Train patches: {len(train_ds)}")
print(f"Val patches:   {len(val_ds)}")

# --------------------
# MODEL
# --------------------
model = UNet(in_channels=10).to(device)
loss_fn = nn.BCELoss()
opt = optim.Adam(model.parameters(), lr=LR)

best_val_loss = float("inf")
patience_counter = 0

# --------------------
# TRAINING LOOP
# --------------------
for epoch in range(1, EPOCHS + 1):

    model.train()
    train_loss = 0.0

    for x, y in tqdm(train_dl, desc=f"Epoch {epoch} [Train]"):
        x, y = x.to(device), y.to(device)

        pred = model(x)
        loss = loss_fn(pred, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        train_loss += loss.item()

    train_loss /= len(train_dl)

    # ---- VALIDATION ----
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for x, y in val_dl:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            val_loss += loss.item()

    val_loss /= len(val_dl)

    print(
        f"Epoch {epoch:02d} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f}"
    )

    # ---- EARLY STOPPING ----
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0

        torch.save(
            model.state_dict(),
            MODEL_DIR / f"unet_builtup_{YEAR}.pth"
        )
        print("  ✓ Best model saved")

    else:
        patience_counter += 1
        print(f"  ⚠ No improvement ({patience_counter}/{PATIENCE})")

        if patience_counter >= PATIENCE:
            print("⏹ Early stopping triggered")
            break

print("Training complete.")
print("Best validation loss:", best_val_loss)