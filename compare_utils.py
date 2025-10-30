"""
============================================================
compare_utils.py
============================================================

This module provides a set of utility functions for comparing calorimeter hit 
data between simulated and real (reference) datasets. It includes tools for 
loading ROOT-based data into pandas DataFrames, validating dataset structures, 
extracting geometry parameters from filenames, and computing event-level 
energy distribution similarity metrics.

------------------------------------------------------------
Main Functionalities
------------------------------------------------------------
1. **Data Validation**
   - Ensures input DataFrames contain the expected calorimeter hit columns.

2. **ROOT Data Loading**
   - Uses `uproot` and `awkward` to efficiently load calorimeter hit data 
     from ROOT files into pandas DataFrames.

3. **Parameter Extraction**
   - Parses filenames to extract cylindrical geometry parameters such as 
     the number of bins (ρ, φ, z) and cell sizes (Δr, Δz).

4. **Energy Distribution Comparison**
   - Compares the energy deposition patterns between real and simulated 
     calorimeter hits on an *event-by-event* basis.
   - Uses the Wasserstein distance to quantify 
     the difference between distributions.

------------------------------------------------------------
Functions
------------------------------------------------------------
validate_dataframe(df: pd.DataFrame) -> bool
    Checks whether the DataFrame is valid and contains the required columns:
    {"Event_ID", "Cell_X", "Cell_Y", "Cell_Size", "Active_Energy"}.

load_hits_flat(filename: str, tree_path: str = "CalorimeterHits") -> pd.DataFrame
    Loads calorimeter hits from a ROOT file into a pandas DataFrame using uproot.
    Returns an empty DataFrame if the file or TTree is missing or unreadable.

extract_params_from_filename(filename: str) -> tuple
    Extracts the calorimeter geometry parameters (rho_bins, phi_bins, z_bins, dr, dz)
    from a standardized filename (e.g., "rho25_phi60_z52_dr3_dz12_5.root").

compare_energy_distributions_eventwise(simulated_df, real_df) -> float
    Compares simulated and real calorimeter energy distributions across events.
    - Groups hits by (Cell_X, Cell_Y, Cell_Size).
    - Aligns cells between real and simulated data.
    - Computes the Wasserstein distance for each event.
    - Returns the mean distance (lower = better agreement).

Author: Justyna Mędrala-Sowa
Created: 30.10.2025
------------------------------------------------------------
"""



import os
import re
import uproot as ur
import awkward as ak
import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance


def validate_dataframe(df: pd.DataFrame) -> bool:
    required = {"Event_ID", "Cell_X", "Cell_Y", "Cell_Size", "Active_Energy"}
    return isinstance(df, pd.DataFrame) and not df.empty and not (required - set(df.columns))


def load_hits_flat(filename: str, tree_path: str = "CalorimeterHits") -> pd.DataFrame:
    if not os.path.exists(filename):
        return pd.DataFrame()
    try:
        with ur.open(filename) as f:
            if tree_path not in f:
                return pd.DataFrame()
            arrays = f[tree_path].arrays(library="ak")
        return ak.to_dataframe(arrays)
    except Exception:
        return pd.DataFrame()


def extract_params_from_filename(filename: str):
    match = re.search(r"rho(\d+)_phi(\d+)_z(\d+)_dr(\d+)_dz(\d+_\d+)", filename)
    if match:
        rho_no, phi_no, z_no = map(int, match.group(1, 2, 3))
        dr = int(match.group(4))
        dz = float(match.group(5).replace("_", "."))
        return rho_no, phi_no, z_no, dr, dz
    return None, None, None, None, None


def compare_energy_distributions_eventwise(simulated_df, real_df) -> float:
    if not validate_dataframe(simulated_df) or not validate_dataframe(real_df):
        return float("inf")

    common_events = np.intersect1d(
        simulated_df["Event_ID"].unique(), real_df["Event_ID"].unique()
    )

    if len(common_events) == 0:
        return float("inf")

    wd_list = []
    for event_id in common_events:
        sim_evt = simulated_df[simulated_df["Event_ID"] == event_id]
        real_evt = real_df[real_df["Event_ID"] == event_id]
        if len(sim_evt) < 1 or len(real_evt) < 1:
            continue

        sim_grouped = sim_evt.groupby(["Cell_X", "Cell_Y", "Cell_Size"])[
            "Active_Energy"
        ].sum()
        real_grouped = real_evt.groupby(["Cell_X", "Cell_Y", "Cell_Size"])[
            "Active_Energy"
        ].sum()
        all_cells = sim_grouped.index.union(real_grouped.index)

        sim_vals = sim_grouped.reindex(all_cells, fill_value=0).values
        real_vals = real_grouped.reindex(all_cells, fill_value=0).values
        if np.all(sim_vals == 0) or np.all(real_vals == 0):
            continue
        wd = wasserstein_distance(sim_vals, real_vals)
        if not np.isnan(wd):
            wd_list.append(wd)

    return np.mean(wd_list) if wd_list else float("inf")

