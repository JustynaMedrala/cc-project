#!/usr/bin/env python3
"""
offline_sampling.py
-------------------

Performs offline sampling and optimization of calorimeter geometry parameters
using Latin Hypercube Sampling (LHS) and parallelized evaluation.

Each parameter set defines a different calorimeter discretization:
    - rho_bins, phi_bins, z_bins → number of bins
    - dr, dz → bin sizes (mm)

For each combination, simulated calorimeter energy hits are compared with
real calorimeter data to compute a similarity score.

Example:
    python3 offline_sampling.py --file inner --n_samples 500 --max_jobs 12

Author: Justyna Mędrala-Sowa
Date: 2025-10-30
"""

import numpy as np
import pandas as pd
from skopt.sampler import Lhs
from skopt.space import Real, Integer
from joblib import Parallel, delayed
import logging
import os
import time
import argparse

from generate_data import generate_calorimeter_file_in_memory
from compare_utils import load_hits_flat, validate_dataframe, compare_energy_distributions_eventwise


LOG_FILE = "offline_sampling.log"
open(LOG_FILE, "w").close() 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="a"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

def objective(params, real_hits, input_file, histogram_flag):
    """
    Evaluate a single parameter combination by generating calorimeter data in memory
    and comparing with real calorimeter hits.

    Returns:
        score (float): distance or similarity metric (lower = better)
    """
    rho_bins, phi_bins, z_bins, dr, dz = params
    rho_bins, phi_bins, z_bins = map(lambda x: int(round(x)), (rho_bins, phi_bins, z_bins))
    try:
        sim_hits = generate_calorimeter_file_in_memory(
            input_file=input_file,
            rho_bins=rho_bins,
            phi_bins=phi_bins,
            z_bins=z_bins,
            dr=dr,
            dz=dz,
            max_events=200,
            histogram_flag=histogram_flag,
        )
        score = compare_energy_distributions_eventwise(sim_hits, real_hits)
        return score
    except Exception as e:
        logger.error(f"Error evaluating params {params}: {e}")
        return np.inf

def eval_and_save(i, params, real_hits, input_file, save_path, histogram_flag):
    """
    Evaluate parameters and append result to a CSV file.
    """
    start_time = time.time()
    score = objective(params, real_hits, input_file, histogram_flag)
    rho_bins, phi_bins, z_bins = map(lambda x: int(round(x)), params[:3])
    dr, dz = params[3], params[4]
    elapsed = time.time() - start_time

    result = {
        "index": i,
        "rho_bins": rho_bins,
        "phi_bins": phi_bins,
        "z_bins": z_bins,
        "dr": float(dr),
        "dz": float(dz),
        "score": float(score),
        "elapsed_s": round(elapsed, 2),
    }

    header = not os.path.exists(save_path)
    pd.DataFrame([result]).to_csv(save_path, mode="a", header=header, index=False)
    logger.info(f"[{i}] Done in {elapsed:.1f}s -> score={score:.4f}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Offline calorimeter sampling with resume support.")
    parser.add_argument("--file", type=str, required=True, choices=["all", "inner"],
                        help="Choose dataset: 'all' or 'inner'")
    parser.add_argument("--delta", type=float, default=None, help="Optional delta for plateau detection")
    parser.add_argument("--n_samples", type=int, default=1000, help="Number of LHS samples to generate")
    parser.add_argument("--max_jobs", type=int, default=18, help="Maximum parallel jobs")
    args = parser.parse_args()

    file_map = {
        "all": (
            "calorimeter_coordinates_no_cylindrical.root",
            "simulated_showers.root",
            "histogram_energy_xy",
            "offline_sampling_results_all.csv",
        ),
        "inner": (
            "calorimeter_coordinates_no_cylindrical_inner.root",
            "simulated_showers_inner.root",
            "histogram_energy_xy_only_inner",
            "offline_sampling_results_inner.csv",
        ),
    }

    real_hits_file, input_file, histogram_flag, save_path = file_map[args.file]
    lhs_file = f"lhs_params_{args.file}.csv"

    logger.info(f"Running offline sampling for region: {args.file.upper()}")
    logger.info(f"Real hits: {real_hits_file}")
    logger.info(f"Simulated input: {input_file}")
    logger.info(f"Results file: {save_path}")

    real_hits = load_hits_flat(real_hits_file)
    if not validate_dataframe(real_hits):
        logger.error("Invalid or empty real hits file. Exiting.")
        return

    space = [
        Integer(26, 30, name="rho_bins"),
        Integer(64, 70, name="phi_bins"),
        Integer(41, 69, name="z_bins"),
        Real(3.0, 3.2, name="dr"),
        Real(10.8, 19.2, name="dz"),
    ]

    if not os.path.exists(lhs_file):
        logger.info(f"Generating new LHS parameters -> {lhs_file}")
        lhs = Lhs(lhs_type="classic", criterion=None)
        X = lhs.generate(dimensions=space, n_samples=args.n_samples)
        lhs_df = pd.DataFrame(X, columns=[s.name for s in space])
        lhs_df.to_csv(lhs_file, index=False)
    else:
        logger.info(f"Loading existing LHS parameters from {lhs_file}")
        lhs_df = pd.read_csv(lhs_file)

    done_indices = set()
    if os.path.exists(save_path):
        try:
            done_indices = set(pd.read_csv(save_path)["index"])
        except Exception:
            logger.warning("Could not read existing results file; starting from scratch.")
            done_indices = set()

    tasks = [(i + 1, row.values) for i, row in lhs_df.iterrows() if (i + 1) not in done_indices]
    logger.info(f"Pending evaluations: {len(tasks)} / {len(lhs_df)}")

    if not tasks:
        logger.info("All parameter sets already evaluated. Exiting.")
        return

    results = Parallel(n_jobs=args.max_jobs, prefer="processes")(
        delayed(eval_and_save)(i, params, real_hits, input_file, save_path, histogram_flag)
        for i, params in tasks
    )

    df = pd.read_csv(save_path).sort_values("index").reset_index(drop=True)
    df.to_csv(save_path, index=False)
    logger.info(f"Offline sampling complete. Saved {len(df)} results to {save_path}")

    best_score = df["score"].min()
    delta = args.delta if args.delta is not None else 0.01
    plateau_df = df[df["score"] <= best_score + delta]
    plateau_save_path = save_path.replace(".csv", "_plateau.csv")
    plateau_df.to_csv(plateau_save_path, index=False)
    logger.info(f"Plateau points: {len(plateau_df)}, best score: {best_score:.4f}")
    logger.info(f"Saved to {plateau_save_path}")


if __name__ == "__main__":
    main()
