#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train U-Net for Kaggle Brain MRI segmentation.

This script is designed for Kaggle Notebook, but it can also run locally if the
dataset directory is available.
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class BrainMRIDataset(Dataset):
    """Brain MRI image-mask dataset."""

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        image_size: int = 256,
        augment: bool = False,
        seed: int = 42,
        limit: int | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size
        self.augment = augment and split == "train"

        all_images = sorted([
            p for p in self.data_dir.rglob("*.tif")
            if "_mask" not in p.name
        ])

        pairs = []
        for img_path in all_images:
            mask_path = img_path.with_name(img_path.stem + "_mask.tif")
            if mask_path.exists():
                pairs.append((img_path, mask_path))

        random.Random(seed).shuffle(pairs)

        n = len(pairs)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)

        if split == "train":
            self.pairs = pairs[:n_train]
        elif split == "val":
            self.pairs = pairs[n_train:n_train + n_val]
        elif split == "test":
            self.pairs = pairs[n_train + n_val:]
        else:
            raise ValueError("split must be one of train / val / test")

        if limit is not None:
            self.pairs = self.pairs[:limit]

        print(f"{split} samples: {len(self.pairs)}")

    def __len__(self):
        return len(self.pairs)

    def _resize(self, img: Image.Image, is_mask: bool = False) -> Image.Image:
        resampling = Image.NEAREST if is_mask else Image.BILINEAR
        return img.resize((self.image_size, self.image_size), resampling)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self._resize(image, is_mask=False)
        mask = self._resize(mask, is_mask=True)

        image = np.array(image, dtype=np.float32) / 255.0
        mask = (np.array(mask, dtype=np.float32) > 0).astype(np.float32)

        if self.augment:
            image, mask = apply_augmentation(image, mask)

        image = np.ascontiguousarray(image.transpose(2, 0, 1))
        mask = np.ascontiguousarray(mask[None, :, :])

        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32), str(img_path)


def apply_augmentation(image: np.ndarray, mask: np.ndarray):
    """Synchronized geometric augmentation for image and mask."""
    if random.random() < 0.5:
        image = np.flip(image, axis=1)
        mask = np.flip(mask, axis=1)

    if random.random() < 0.5:
        image = np.flip(image, axis=0)
        mask = np.flip(mask, axis=0)

    if random.random() < 0.5:
        k = random.randint(1, 3)
        image = np.rot90(image, k, axes=(0, 1))
        mask = np.rot90(mask, k, axes=(0, 1))

    if random.random() < 0.5:
        factor = random.uniform(0.85, 1.15)
        image = np.clip(image * factor, 0, 1)

    return image, mask


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, base: int = 32):
        super().__init__()

        self.inc = DoubleConv(in_channels, base)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base, base * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base * 2, base * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base * 4, base * 8))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base * 8, base * 16))

        self.up1 = nn.ConvTranspose2d(base * 16, base * 8, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(base * 16, base * 8)

        self.up2 = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(base * 8, base * 4)

        self.up3 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(base * 4, base * 2)

        self.up4 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(base * 2, base)

        self.outc = nn.Conv2d(base, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5)
        x = torch.cat([x4, x], dim=1)
        x = self.conv1(x)

        x = self.up2(x)
        x = torch.cat([x3, x], dim=1)
        x = self.conv2(x)

        x = self.up3(x)
        x = torch.cat([x2, x], dim=1)
        x = self.conv3(x)

        x = self.up4(x)
        x = torch.cat([x1, x], dim=1)
        x = self.conv4(x)

        return self.outc(x)


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.reshape(probs.size(0), -1)
        targets = targets.reshape(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        dice = (2 * intersection + self.smooth) / (
            probs.sum(dim=1) + targets.sum(dim=1) + self.smooth
        )
        return 1 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        return self.bce(logits, targets) + self.dice(logits, targets)


class MetricAccumulator:
    def __init__(self, eps: float = 1e-7):
        self.eps = eps
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0

    def update(self, logits, masks, threshold: float = 0.5):
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()

        preds = preds.reshape(-1)
        masks = masks.reshape(-1)

        self.tp += (preds * masks).sum().item()
        self.fp += (preds * (1 - masks)).sum().item()
        self.fn += ((1 - preds) * masks).sum().item()

    def compute(self):
        dice = (2 * self.tp + self.eps) / (2 * self.tp + self.fp + self.fn + self.eps)
        iou = (self.tp + self.eps) / (self.tp + self.fp + self.fn + self.eps)
        precision = (self.tp + self.eps) / (self.tp + self.fp + self.eps)
        recall = (self.tp + self.eps) / (self.tp + self.fn + self.eps)
        return dice, iou, precision, recall


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for images, masks, _ in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device, desc: str = "Val"):
    model.eval()
    total_loss = 0.0
    metrics = MetricAccumulator()

    for images, masks, _ in tqdm(loader, desc=desc, leave=False):
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = criterion(logits, masks)

        total_loss += loss.item() * images.size(0)
        metrics.update(logits.cpu(), masks.cpu())

    dice, iou, precision, recall = metrics.compute()

    return {
        "loss": total_loss / len(loader.dataset),
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
    }


def plot_curves(log_df: pd.DataFrame, output_dir: Path, mode: str):
    curve_dir = output_dir / "curves"
    curve_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(log_df["epoch"], log_df["train_loss"], label="train_loss")
    plt.plot(log_df["epoch"], log_df["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{mode.capitalize()} Loss Curve")
    plt.legend()
    plt.savefig(curve_dir / f"loss_curve_{mode}.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(log_df["epoch"], log_df["val_dice"], label="val_dice")
    plt.plot(log_df["epoch"], log_df["val_iou"], label="val_iou")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title(f"{mode.capitalize()} Dice / IoU Curve")
    plt.legend()
    plt.savefig(curve_dir / f"dice_iou_curve_{mode}.png", dpi=200, bbox_inches="tight")
    plt.close()


@torch.no_grad()
def save_predictions(model, dataset, device, output_dir: Path, mode: str, num_samples: int = 8):
    model.eval()
    save_dir = output_dir / "prediction_visualizations"
    save_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for img_path, mask_path in dataset.pairs:
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image_np = np.array(image, dtype=np.float32) / 255.0
        mask_np = (np.array(mask, dtype=np.float32) > 0).astype(np.float32)

        if mask_np.sum() == 0:
            continue

        image_tensor = torch.tensor(
            image_np.transpose(2, 0, 1),
            dtype=torch.float32
        ).unsqueeze(0).to(device)

        logits = model(image_tensor)
        pred = (torch.sigmoid(logits).cpu().squeeze().numpy() > 0.5).astype(np.float32)

        plt.figure(figsize=(16, 4))

        plt.subplot(1, 4, 1)
        plt.imshow(image_np)
        plt.title("Original MRI")
        plt.axis("off")

        plt.subplot(1, 4, 2)
        plt.imshow(mask_np, cmap="gray")
        plt.title("Ground Truth")
        plt.axis("off")

        plt.subplot(1, 4, 3)
        plt.imshow(pred, cmap="gray")
        plt.title(f"{mode.capitalize()} Prediction")
        plt.axis("off")

        plt.subplot(1, 4, 4)
        plt.imshow(image_np)
        plt.imshow(mask_np, cmap="Reds", alpha=0.35)
        plt.imshow(pred, cmap="Greens", alpha=0.35)
        plt.title("GT + Prediction")
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(save_dir / f"{mode}_sample_{saved + 1}.png", dpi=200, bbox_inches="tight")
        plt.close()

        saved += 1
        if saved >= num_samples:
            break

    print(f"Saved {saved} prediction visualizations to {save_dir}")


def build_loader(dataset, batch_size: int, shuffle: bool, num_workers: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["baseline", "improved"], default="baseline")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit_train", type=int, default=None)
    parser.add_argument("--limit_val", type=int, default=None)
    parser.add_argument("--limit_test", type=int, default=None)
    args = parser.parse_args()

    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "weights").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_set = BrainMRIDataset(
        args.data_dir,
        split="train",
        image_size=args.image_size,
        augment=args.mode == "improved",
        seed=args.seed,
        limit=args.limit_train,
    )
    val_set = BrainMRIDataset(
        args.data_dir,
        split="val",
        image_size=args.image_size,
        augment=False,
        seed=args.seed,
        limit=args.limit_val,
    )
    test_set = BrainMRIDataset(
        args.data_dir,
        split="test",
        image_size=args.image_size,
        augment=False,
        seed=args.seed,
        limit=args.limit_test,
    )

    train_loader = build_loader(train_set, args.batch_size, True, args.num_workers)
    val_loader = build_loader(val_set, args.batch_size, False, args.num_workers)
    test_loader = build_loader(test_set, args.batch_size, False, args.num_workers)

    model = UNet(in_channels=3, out_channels=1, base=32).to(device)

    if args.mode == "baseline":
        criterion = nn.BCEWithLogitsLoss()
        loss_name = "BCE Loss"
        aug_name = "No"
        model_name = "U-Net Baseline"
    else:
        criterion = BCEDiceLoss()
        loss_name = "BCE + Dice Loss"
        aug_name = "Flip / Rotate / Brightness"
        model_name = "U-Net + BCE Dice + Aug"

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    logs = []
    best_dice = -1.0

    for epoch in range(1, args.epochs + 1):
        print(f"\n========== {args.mode.capitalize()} Epoch {epoch}/{args.epochs} ==========")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device, desc="Val")

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
        }
        logs.append(row)
        print(row)

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            torch.save(model.state_dict(), output_dir / "weights" / f"unet_{args.mode}_best.pth")
            print(f"Saved best model, best val Dice = {best_dice:.4f}")

    torch.save(model.state_dict(), output_dir / "weights" / f"unet_{args.mode}_final.pth")

    log_df = pd.DataFrame(logs)
    log_df.to_csv(output_dir / f"training_log_{args.mode}.csv", index=False)
    plot_curves(log_df, output_dir, args.mode)

    best_weight = output_dir / "weights" / f"unet_{args.mode}_best.pth"
    model.load_state_dict(torch.load(best_weight, map_location=device))

    test_metrics = evaluate(model, test_loader, criterion, device, desc="Test")

    summary = pd.DataFrame([{
        "model": model_name,
        "loss": loss_name,
        "augmentation": aug_name,
        "test_loss": test_metrics["loss"],
        "test_dice": test_metrics["dice"],
        "test_iou": test_metrics["iou"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
    }])

    summary.to_csv(output_dir / f"metrics_summary_{args.mode}.csv", index=False)
    print("\nFinal test metrics:")
    print(summary)

    save_predictions(model, test_set, device, output_dir, args.mode, num_samples=8)

    print("\nDone. Outputs saved to:", output_dir)


if __name__ == "__main__":
    main()
