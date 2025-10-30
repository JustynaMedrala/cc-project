#!/usr/bin/env python3
"""
find_plateau.py
----------------

This script identifies plateau points (i.e., near-optimal parameter sets)
from offline calorimeter sampling results. It filters results that are
within a small margin (delta) of the best score.

Usage examples:
    python3 find_plateau.py --file all
    python3 find_plateau.py --file inner --delta 0.02

Author: Justyna Mędrala-Sowa
Date: 2025-10-30
"""

import pandas as pd
import numpy as np
import logging
import argparse
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def find_plateau(
    file_path: str,
    score_col: str = "score",
    delta: float = None,
    quantile: float = 0.05,
    save_path: str = None
) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)
    if score_col not in df.columns:
        raise ValueError(f"Column '{score_col}' does not exist in {file_path}")

    best_score = df[score_col].min()
    if delta is None:
        cutoff = df[score_col].quantile(quantile)
        delta = cutoff - best_score
        logger.info(f"Automatically determined delta: {delta:.6f} (quantile={quantile})")

    plateau_df = df[df[score_col] <= best_score + delta].copy().sort_values(by=score_col)

    logger.info("🔹 Plateau Detection Summary:")
    logger.info(f"  • Total points: {len(df)}")
    logger.info(f"  • Plateau points: {len(plateau_df)}")
    logger.info(f"  • Best score: {best_score:.6f}")
    logger.info(f"  • Delta: {delta:.6f}")
    logger.info(f"  • Mean plateau score: {plateau_df[score_col].mean():.6f}")

    if save_path is None:
        save_path = file_path.replace(".csv", "_plateau.csv")
    plateau_df.to_csv(save_path, index=False)
    logger.info(f"Plateau points saved → {save_path}")

    print("\nTop 10 plateau points:")
    print(plateau_df.head(10))
    return plateau_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find plateau points in offline sampling results.")
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        choices=["all", "inner"],
        help="Choose which dataset to analyze: 'all' or 'inner'"
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=None,
        help="Optional delta for plateau detection (if not provided, auto-calculated)."
    )
    args = parser.parse_args()

    file_map = {
        "all": "offline_sampling_results_all.csv",
        "inner": "offline_sampling_results_inner.csv"
    }

    file_path = file_map[args.file]
    logger.info(f"Processing region: {args.file.upper()} ({file_path})")

    find_plateau(file_path=file_path, delta=args.delta)
