#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Package experiment outputs into a final zip file."""

import argparse
from pathlib import Path
import shutil
import os


def copy_if_exists(src: Path, dst_dir: Path):
    if src.exists():
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst_dir / src.name)


def copy_pngs(src_dir: Path, dst_dir: Path):
    if src_dir.exists():
        dst_dir.mkdir(parents=True, exist_ok=True)
        for file in src_dir.glob("*.png"):
            shutil.copy(file, dst_dir / file.name)


def copy_pths(src_dir: Path, dst_dir: Path):
    if src_dir.exists():
        dst_dir.mkdir(parents=True, exist_ok=True)
        for file in src_dir.glob("*.pth"):
            shutil.copy(file, dst_dir / file.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_dir", type=str, required=True)
    parser.add_argument("--improved_dir", type=str, required=True)
    parser.add_argument("--final_dir", type=str, required=True)
    parser.add_argument("--package_dir", type=str, required=True)
    args = parser.parse_args()

    baseline_dir = Path(args.baseline_dir)
    improved_dir = Path(args.improved_dir)
    final_dir = Path(args.final_dir)
    package_dir = Path(args.package_dir)

    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)

    metrics_dir = package_dir / "metrics"
    logs_dir = package_dir / "training_logs"
    curves_dir = package_dir / "curves"
    vis_dir = package_dir / "prediction_visualizations"
    weights_dir = package_dir / "weights"

    for file in [
        final_dir / "metrics_comparison_table.csv",
        final_dir / "metrics_comparison_table.xlsx",
        baseline_dir / "metrics_summary_baseline.csv",
        improved_dir / "metrics_summary_improved.csv",
    ]:
        copy_if_exists(file, metrics_dir)

    for file in [
        baseline_dir / "training_log_baseline.csv",
        improved_dir / "training_log_improved.csv",
    ]:
        copy_if_exists(file, logs_dir)

    copy_pngs(baseline_dir / "curves", curves_dir)
    copy_pngs(improved_dir / "curves", curves_dir)

    copy_pngs(baseline_dir / "prediction_visualizations", vis_dir)
    copy_pngs(improved_dir / "prediction_visualizations", vis_dir)

    copy_pths(baseline_dir / "weights", weights_dir)
    copy_pths(improved_dir / "weights", weights_dir)

    readme_text = """基于 U-Net 的脑 MRI 低级别胶质瘤区域自动分割实验结果

数据集：Kaggle Brain MRI segmentation / LGG MRI Segmentation
数据规模：3929 对 image-mask 数据
划分方式：训练集 2750 张，验证集 589 张，测试集 590 张
输入尺寸：256×256
模型：U-Net
Baseline：BCE Loss，无数据增强
Improved：BCE + Dice Loss，加入翻转、旋转、亮度扰动

主要输出：
1. metrics/：测试集指标表和模型对比表
2. training_logs/：baseline 和 improved 训练日志
3. curves/：loss 曲线和 Dice/IoU 曲线
4. prediction_visualizations/：测试集分割结果可视化图
5. weights/：baseline 和 improved 的 best/final 权重文件
"""
    (package_dir / "实验结果说明.txt").write_text(readme_text, encoding="utf-8")

    zip_path = shutil.make_archive(str(package_dir), "zip", package_dir)

    print("Package finished:", zip_path)
    print("\nPackage contents:")
    for root, _, files in os.walk(package_dir):
        level = root.replace(str(package_dir), "").count(os.sep)
        indent = "  " * level
        print(indent + Path(root).name + "/")
        for f in files:
            print(indent + "  " + f)


if __name__ == "__main__":
    main()
