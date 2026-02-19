import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import torch
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
from tqdm import tqdm
from pathlib import Path

from scripts.dl.dataset import BuiltupDataset
from scripts.dl.unet_model import UNet

# -------------------------------------------------
# ARGUMENTS
# -------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--year", required=True, help="Year to train (e.g., 2025)")
parser.add_argument("--aoi", required=True, help="AOI name (e.g., auroville)")
args = parser.parse_args()

YEAR = args.year
AOI = args.aoi

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
BATCH_SIZE = 8
EPOCHS = 40
VAL_SPLIT = 0.2
PATIENCE = 6
LR = 1e-4
THRESHOLD = 0.35

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# -------------------------------------------------
# DEVICE
# -------------------------------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)

# -------------------------------------------------
# DATASET
# -------------------------------------------------
img_dir = f"data/dl/{YEAR}_{AOI}/images"
mask_dir = f"data/dl/{YEAR}_{AOI}/masks"

ds = BuiltupDataset(img_dir, mask_dir)

torch.manual_seed(42)

val_size = int(len(ds) * VAL_SPLIT)
train_size = len(ds) - val_size

train_ds, val_ds = random_split(ds, [train_size, val_size])

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

print(f"\nTraining: {YEAR} | {AOI}")
print(f"Train patches: {len(train_ds)}")
print(f"Val patches:   {len(val_ds)}")

# -------------------------------------------------
# MODEL
# -------------------------------------------------
model = UNet(in_channels=10).to(device)

# -------- Dice Loss ----------
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = preds.view(-1)
        targets = targets.view(-1)

        intersection = (preds * targets).sum()
        dice = (2 * intersection + self.smooth) / (
            preds.sum() + targets.sum() + self.smooth
        )
        return 1 - dice

bce = nn.BCELoss()
dice = DiceLoss()

def combined_loss(pred, target):
    return bce(pred, target) + dice(pred, target)

optimizer = optim.Adam(model.parameters(), lr=LR)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3
)

# -------------------------------------------------
# TRAINING LOOP
# -------------------------------------------------
best_val_loss = float("inf")
patience_counter = 0

for epoch in range(1, EPOCHS + 1):

    # ----------------- TRAIN -----------------
    model.train()
    train_loss = 0.0

    for x, y in tqdm(train_dl, desc=f"Epoch {epoch} [Train]"):
        x, y = x.to(device), y.to(device)

        pred = model(x)
        loss = combined_loss(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_dl)

    # ----------------- VALIDATION -----------------
    model.eval()
    val_loss = 0.0
    total_iou = 0.0

    with torch.no_grad():
        for x, y in val_dl:
            x, y = x.to(device), y.to(device)
            pred = model(x)

            loss = combined_loss(pred, y)
            val_loss += loss.item()

            # IoU metric
            pred_bin = (pred > THRESHOLD).float()
            intersection = (pred_bin * y).sum()
            union = pred_bin.sum() + y.sum() - intersection
            iou = intersection / (union + 1e-6)
            total_iou += iou.item()

    val_loss /= len(val_dl)
    avg_iou = total_iou / len(val_dl)

    scheduler.step(val_loss)

    print(
        f"Epoch {epoch:02d} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"IoU: {avg_iou:.4f}"
    )

    # ----------------- EARLY STOPPING -----------------
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0

        torch.save(
            model.state_dict(),
            MODEL_DIR / f"unet_{YEAR}_{AOI}.pth"
        )

        print("  ✓ Best model saved")

    else:
        patience_counter += 1
        print(f"  ⚠ No improvement ({patience_counter}/{PATIENCE})")

        if patience_counter >= PATIENCE:
            print("⏹ Early stopping triggered")
            break

print("\nTraining complete.")
print("Best validation loss:", best_val_loss)