#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Merge baseline and improved metrics into one comparison table."""

import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_csv", type=str, required=True)
    parser.add_argument("--improved_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_df = pd.read_csv(args.baseline_csv)
    improved_df = pd.read_csv(args.improved_csv)

    metrics_table = pd.concat([baseline_df, improved_df], ignore_index=True)

    display_table = metrics_table.copy()
    for col in ["test_loss", "test_dice", "test_iou", "test_precision", "test_recall"]:
        if col in display_table.columns:
            display_table[col] = display_table[col].round(4)

    display_table.to_csv(output_dir / "metrics_comparison_table.csv", index=False)
    display_table.to_excel(output_dir / "metrics_comparison_table.xlsx", index=False)

    print("Final comparison table:")
    print(display_table)
    print("\nSaved to:", output_dir)


if __name__ == "__main__":
    main()
