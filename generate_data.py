"""
============================================================
generate_calorimeter_file_in_memory.py
============================================================

This module provides a utility function for generating calorimeter hit data 
directly in memory from simulated particle shower ROOT files. It performs 
cylindrical binning, coordinate transformations, and energy histogramming 
to produce a flattened DataFrame of calorimeter hits.

The generated data mimics the detector response by converting locally simulated 
energy deposition points into calorimeter cell coordinates, optionally limited 
to the inner region of the ECAL.

------------------------------------------------------------
Main Function
------------------------------------------------------------
generate_calorimeter_file_in_memory(
    input_file: str,
    rho_bins: int,
    phi_bins: int,
    z_bins: int,
    dr: float,
    dz: float,
    n_points: int = 100000,
    energy_scale_factor: float = 7.85,
    max_events: int = 200,
    histogram_flag: str = "histogram_energy_xy_only_inner"
) -> pd.DataFrame

Description:
    - Loads simulated shower data from a ROOT file using `uproot`.
    - Bins the data in cylindrical coordinates (ρ, φ, z).
    - Converts cell centers from cylindrical to local Cartesian coordinates.
    - Transforms local coordinates into the world (ECAL) frame.
    - Applies energy histogramming using one of:
        • `histogram_energy_xy`  → for the full ECAL region
        • `histogram_energy_xy_only_inner` → for the inner region only
    - Collects the resulting calorimeter hits in a pandas DataFrame.

Parameters:
    input_file (str): Path to the input ROOT file containing the TTree "SimulatedShowers".
    rho_bins (int): Number of radial bins.
    phi_bins (int): Number of azimuthal bins.
    z_bins (int): Number of longitudinal bins.
    dr (float): Radial bin (cell) size.
    dz (float): Longitudinal bin (cell) size.
    n_points (int, optional): Number of points per event for histogramming. Default is 100000.
    energy_scale_factor (float, optional): Energy scaling factor. Default is 7.85.
    max_events (int, optional): Maximum number of events to process. Default is 200.
    histogram_flag (str, optional): Determines which histogram function to use — 
        "histogram_energy_xy" for all regions or 
        "histogram_energy_xy_only_inner" for inner ECAL. Default is "histogram_energy_xy_only_inner".

Returns:
    pd.DataFrame:
        A flattened DataFrame with the following columns:
            - Event_ID
            - Cell_X
            - Cell_Y
            - Cell_Size
            - Active_Energy


Author: Justyna Mędrala-Sowa
Created: 30.10.2025
------------------------------------------------------------
"""

import os
import time
import numpy as np
import pandas as pd
import uproot as ur

from calorimeter_utils import (
    cylindrical,
    cylindrical_centers_to_local_cartesian,
    local_to_world,
    histogram_energy_xy,
    histogram_energy_xy_only_inner,
)

def generate_calorimeter_file_in_memory(
    input_file: str,
    rho_bins: int,
    phi_bins: int,
    z_bins: int,
    dr: float,
    dz: float,
    n_points: int = 100000,
    energy_scale_factor: float = 7.85,
    max_events: int = 200,
    histogram_flag: str = "histogram_energy_xy_only_inner"
) -> pd.DataFrame:
    """
    Generates calorimeter hits in memory, binning and converting to world coordinates.

    Parameters:
        input_file (str): ROOT file with simulated showers.
        rho_bins, phi_bins, z_bins (int): Cylindrical binning parameters.
        dr, dz (float): Cell sizes.
        n_points (int): Number of points for energy histogram.
        energy_scale_factor (float): Scale factor for energy.
        max_events (int): Max number of events to process.
        histogram_flag (str): Which histogram function to use ("histogram_energy_xy" or "histogram_energy_xy_only_inner").

    Returns:
        pd.DataFrame: Flattened DataFrame of hits with Event_ID, Cell_X, Cell_Y, Cell_Size, Active_Energy
    """
    start_time = time.time()
    all_data = {
        "Event_ID": [],
        "Cell_X": [],
        "Cell_Y": [],
        "Cell_Size": [],
        "Active_Energy": [],
    }

    processed_events = set()
    stop = False

    # Iterate over ROOT TTree in chunks
    for df_chunk in ur.iterate(f"{input_file}:SimulatedShowers", library="pd", step_size=10000):
        if stop:
            break

        for event_id, df_event in df_chunk.groupby("Event_ID"):
            if event_id in processed_events:
                continue

            E = df_event["Energy"].iloc[0]

            df_cyl = cylindrical(
                df_event,
                event_id,
                rho_bins,
                phi_bins,
                z_bins,
                dr,
                dz,
                E,
                n_points,
            )

            if df_cyl.empty:
                continue

            local_points = cylindrical_centers_to_local_cartesian(df_cyl)
            direction = df_cyl[["direction_x", "direction_y", "direction_z"]].iloc[0].values
            destination = df_cyl[["destination_x", "destination_y", "destination_z"]].iloc[0].values
            world_points = local_to_world(local_points, direction, destination)

            xw, yw = world_points[:, 0], world_points[:, 1]

            # Choose histogram function based on flag
            if histogram_flag == "histogram_energy_xy":
                df_xy = histogram_energy_xy(xw, yw, E, energy_scale_factor, n_points)
            else:
                df_xy = histogram_energy_xy_only_inner(xw, yw, E, energy_scale_factor, n_points)

            if df_xy.empty:
                continue

            n_cells = len(df_xy)
            all_data["Event_ID"].extend([event_id] * n_cells)
            all_data["Cell_X"].extend(df_xy["Cell_X"].values)
            all_data["Cell_Y"].extend(df_xy["Cell_Y"].values)
            all_data["Cell_Size"].extend(df_xy["Cell_Size"].values)
            all_data["Active_Energy"].extend(df_xy["Active_Energy"].values)

            processed_events.add(event_id)

            if len(processed_events) >= max_events:
                stop = True
                break

    elapsed = time.time() - start_time

    return pd.DataFrame(all_data)