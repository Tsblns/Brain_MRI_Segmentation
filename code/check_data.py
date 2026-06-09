#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check Brain MRI segmentation dataset path and image-mask matching."""

import argparse
from pathlib import Path
from PIL import Image
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to kaggle_3m dataset directory."
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    print("Data directory:", data_dir)
    print("Path exists:", data_dir.exists())

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    patient_dirs = sorted([
        d for d in data_dir.iterdir()
        if d.is_dir() and d.name.startswith("TCGA")
    ])

    image_files = sorted([
        p for p in data_dir.rglob("*.tif")
        if "_mask" not in p.name
    ])

    mask_files = sorted([
        p for p in data_dir.rglob("*.tif")
        if "_mask" in p.name
    ])

    missing_masks = []
    for img_path in image_files:
        mask_path = img_path.with_name(img_path.stem + "_mask.tif")
        if not mask_path.exists():
            missing_masks.append((img_path, mask_path))

    print("Patient directory count:", len(patient_dirs))
    print("Image count:", len(image_files))
    print("Mask count:", len(mask_files))
    print("Missing mask count:", len(missing_masks))

    if patient_dirs:
        print("First 5 patient dirs:", [p.name for p in patient_dirs[:5]])

    if image_files:
        sample_img = image_files[0]
        sample_mask = sample_img.with_name(sample_img.stem + "_mask.tif")

        print("Sample image:", sample_img)
        print("Sample mask :", sample_mask)

        image = Image.open(sample_img).convert("RGB")
        mask = Image.open(sample_mask).convert("L")

        print("Image size:", image.size)
        print("Mask size :", mask.size)
        print("Mask unique values:", np.unique(np.array(mask))[:10])

    if missing_masks:
        print("First missing examples:")
        for img, mask in missing_masks[:5]:
            print("image:", img)
            print("mask :", mask)
        raise RuntimeError("Some images do not have matching masks.")

    print("Dataset check passed.")


if __name__ == "__main__":
    main()
