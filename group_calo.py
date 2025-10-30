#!/usr/bin/env python3
"""
group_calo.py
-------------

Groups simulated ECAL shower hits into calorimeter cells
and computes total active energy deposition per event.

This script can process data for either the 'inner' ECAL region
or for 'all' regions combined, depending on the command-line argument.

Usage:
    python3 group_calo.py --region inner
    python3 group_calo.py --region all

Author: Justyna Mędrala-Sowa
Date: 2025-10-30
"""

import time
import argparse
import uproot as ur
import numpy as np
import pandas as pd
from tqdm import tqdm

from ecal import BOUNDARIES, EcalModXYSize
from calorimeter_utils import (
    cylindrical_centers_to_local_cartesian,
    local_to_world,
    histogram_energy_xy,
    histogram_energy_xy_only_inner,
    deduce_ecal_region,
)

# ======================================
# === Command-line arguments ===
# ======================================
parser = argparse.ArgumentParser(description="Group simulated ECAL showers into calorimeter cells")
parser.add_argument(
    "--region",
    type=str,
    choices=["inner", "all"],
    default="inner",
    help="ECAL region to process ('inner' or 'all')"
)
args = parser.parse_args()
region = args.region

print(f"🔹 Grouping ECAL data for region: {region}")

# ======================================
# === Configuration ===
# ======================================
n_points = 100000             
energy_scale_factor = 7.85     
max_events = 10000             

input_file = (
    f"simulated_showers_{region}.root" if region != "all"
    else "simulated_showers.root"
)
output_file_name = (
    f"calorimeter_coordinates_no_cylindrical_{region}.root" if region != "all"
    else "calorimeter_coordinates_no_cylindrical.root"
)

print(f"Reading simulated data from {input_file} ...")
print(f"\n[→] Processing {output_file_name}")

start_time = time.time()

with ur.recreate(output_file_name) as output_file:
    output_file["CalorimeterHits"] = {
        "Event_ID": np.empty(0, dtype=np.int32),
        "Cell_X": np.empty(0, dtype=np.float32),
        "Cell_Y": np.empty(0, dtype=np.float32),
        "Cell_Size": np.empty(0, dtype=np.float32),
        "Active_Energy": np.empty(0, dtype=np.float32),
    }

    total_active_energy = 0.0
    processed_events = set()

    with tqdm(total=max_events, desc="Events", unit="event") as pbar:
        for df_chunk in ur.iterate(f"{input_file}:SimulatedShowers", library="pd", step_size=10000):
            for event_id, df_event in df_chunk.groupby("Event_ID"):
                if event_id in processed_events:
                    continue
                if len(processed_events) >= max_events:
                    break

                E = df_event["Energy"].iloc[0]
                local_points = df_event[["x_local", "y_local", "z_local"]].values
                direction = df_event[["direction_x", "direction_y", "direction_z"]].iloc[0].values
                destination = df_event[["destination_x", "destination_y", "destination_z"]].iloc[0].values

                world_points = local_to_world(local_points, direction, destination)
                xw, yw = world_points[:, 0], world_points[:, 1]

                if region == "inner":
                    df_xy = histogram_energy_xy_only_inner(xw, yw, E, energy_scale_factor, n_points)
                else:
                    df_xy = histogram_energy_xy(xw, yw, E, energy_scale_factor, n_points)

                if df_xy.empty:
                    processed_events.add(event_id)
                    pbar.update(1)
                    continue

                n_cells = len(df_xy)
                event_ids = np.full(n_cells, event_id, dtype=np.int32)

                output_file["CalorimeterHits"].extend({
                    "Event_ID": event_ids,
                    "Cell_X": df_xy["Cell_X"].values.astype(np.float32),
                    "Cell_Y": df_xy["Cell_Y"].values.astype(np.float32),
                    "Cell_Size": df_xy["Cell_Size"].values.astype(np.float32),
                    "Active_Energy": df_xy["Active_Energy"].values.astype(np.float32),
                })

                total_active_energy += df_xy["Active_Energy"].sum()
                processed_events.add(event_id)
                pbar.update(1)

            if len(processed_events) >= max_events:
                break

    print(f"Total Active Energy (All Events): {total_active_energy:.4f}")

elapsed = time.time() - start_time
print(f"Saved {output_file_name} in {elapsed:.2f} seconds")
