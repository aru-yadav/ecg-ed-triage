"""Train 3 ensemble models on folds 1-8, validate on fold 9.

Config ported VERBATIM from 02_production_model.ipynb (cells 15, 16, 18):
  - on-the-fly AugmentedECGDataset (mi_prob=0.9, normal_prob=0.3), batch 64
  - FocalLoss(alpha=0.25, gamma=2.0, weight=[1.0, pos_weight])
  - Adam(lr=1e-3, weight_decay=1e-4), CosineAnnealingLR(T_max=30, eta_min=1e-6)
  - 30 epochs, early-stop patience 12, save BEST-by-val-accuracy
  - per-model seed = 42 + model_idx*100  (exactly as the notebook)

Weights saved to models_foldcv/ so existing models/ weights are untouched.
"""
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

from .data import get_splits, REPO_ROOT
from .model import ImprovedECGModel

GLOBAL_SEED = 42
# Override with env var MODELS_FOLDCV_DIR (e.g. a Google Drive path) so the 3
# best-checkpoints are written straight to durable storage during training.
OUT_DIR = Path(os.environ.get("MODELS_FOLDCV_DIR", REPO_ROOT / "models_foldcv"))


# ---- augmentation (verbatim, cells 2 & 15) ----
class ECGAugmentation:
    @staticmethod
    def add_gaussian_noise(ecg, noise_factor=0.01):
        return ecg + np.random.normal(0, noise_factor, ecg.shape)

    @staticmethod
    def add_baseline_wander(ecg, freq=0.5, amplitude=0.1, sr=100):
        t = np.arange(ecg.shape[1]) / sr
        wander = amplitude * np.sin(2 * np.pi * freq * t)
        return ecg + wander[np.newaxis, :]

    @staticmethod
    def random_amplitude_scale(ecg, low=0.9, high=1.1):
        return ecg * np.random.uniform(low, high)


class AugmentedECGDataset(Dataset):
    def __init__(self, X, y, augment_mi_prob=0.9, augment_normal_prob=0.3):
        self.X = X
        self.y = y
        self.augment_mi_prob = augment_mi_prob
        self.augment_normal_prob = augment_normal_prob

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        ecg = self.X[idx].copy()
        label = self.y[idx]
        if label == 1 and np.random.rand() < self.augment_mi_prob:
            ecg = ECGAugmentation.add_gaussian_noise(ecg, noise_factor=0.015)
            if np.random.rand() > 0.5:
                ecg = ECGAugmentation.add_baseline_wander(ecg)
            if np.random.rand() > 0.5:
                ecg = ECGAugmentation.random_amplitude_scale(ecg, low=0.90, high=1.10)
        elif label == 0 and np.random.rand() < self.augment_normal_prob:
            ecg = ECGAugmentation.add_gaussian_noise(ecg, noise_factor=0.01)
        return torch.FloatTensor(ecg), torch.LongTensor([label])[0]


# ---- focal loss (verbatim, cell 16) ----
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for ecgs, labels in loader:
            ecgs, labels = ecgs.to(device), labels.to(device)
            outputs = model(ecgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * ecgs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return running_loss / total, 100. * correct / total


def main():
    # ---- seeds: fixed AND printed ----
    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)
    torch.use_deterministic_algorithms(False)  # cudnn off (CPU); keep BN/conv defaults
    print(f"[seeds] GLOBAL random/numpy/torch = {GLOBAL_SEED}")
    print(f"[seeds] per-model (numpy+torch)   = 42 + model_idx*100  -> [42, 142, 242]")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    splits = get_splits()
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    print(f"[data] train n={len(X_train)} (MI {int(y_train.sum())}) | "
          f"val(fold9) n={len(X_val)} (MI {int(y_val.sum())})")

    # class weights from TRAIN only (folds 1-8)
    mi_count = y_train.sum()
    normal_count = (y_train == 0).sum()
    pos_weight = normal_count / mi_count
    class_weights = torch.FloatTensor([1.0, float(pos_weight)]).to(device)
    print(f"[loss] pos_weight (Normal/MI on train) = {pos_weight:.4f}")

    # val loader (fold 9), batch 32, no shuffle  (as notebook cell 7)
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)),
        batch_size=32, shuffle=False,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    num_epochs = 30
    early_stop_patience = 12
    batch_size = 64

    for model_idx in range(3):
        print(f"\n{'='*60}\nTRAINING MODEL {model_idx+1}/3\n{'='*60}")
        # per-model seed — exactly as the notebook
        torch.manual_seed(42 + model_idx * 100)
        np.random.seed(42 + model_idx * 100)

        train_loader = DataLoader(
            AugmentedECGDataset(X_train, y_train, augment_mi_prob=0.9, augment_normal_prob=0.3),
            batch_size=batch_size, shuffle=True, num_workers=0,
        )

        model = ImprovedECGModel(num_classes=2).to(device)
        criterion = FocalLoss(alpha=0.25, gamma=2.0, weight=class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)

        best_val_acc = 0.0
        patience_counter = 0
        save_path = OUT_DIR / f"ensemble_model_{model_idx+1}.pt"

        for epoch in range(1, num_epochs + 1):
            t0 = time.time()
            model.train()
            running_loss, correct, total = 0.0, 0, 0
            for ecgs, labels in train_loader:
                ecgs, labels = ecgs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(ecgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * ecgs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            train_loss = running_loss / total
            train_acc = 100. * correct / total
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            scheduler.step()
            print(f"Model {model_idx+1} | Epoch {epoch:2d}/{num_epochs} | "
                  f"Train {train_loss:.4f}/{train_acc:.1f}% | Val {val_loss:.4f}/{val_acc:.1f}% | "
                  f"{time.time()-t0:.0f}s", flush=True)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), save_path)
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"   early stop at epoch {epoch}")
                break
        print(f"Model {model_idx+1} done. best val acc = {best_val_acc:.2f}% -> {save_path}")

    print("\nALL 3 MODELS TRAINED.")


if __name__ == "__main__":
    main()
