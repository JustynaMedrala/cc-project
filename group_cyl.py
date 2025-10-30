#!/usr/bin/env python3
"""
group_calo_cylindrical.py
-------------------------

Converts simulated ECAL showers into calorimeter cell coordinates
in cylindrical geometry and computes energy deposition histograms.

Supports ECAL region selection via command line:
    python3 group_calo_cylindrical.py --region inner
    python3 group_calo_cylindrical.py --region all

Author: Justyna Mędrala-Sowa
Date: 2025-10-30
"""

import time
import argparse
import uproot as ur
import numpy as np
from tqdm import tqdm

from calorimeter_utils import (
    cylindrical,
    cylindrical_centers_to_local_cartesian,
    local_to_world,
    histogram_energy_xy,
    histogram_energy_xy_only_inner,
)

parser = argparse.ArgumentParser(description="Group simulated ECAL showers (cylindrical geometry)")
parser.add_argument(
    "--region",
    type=str,
    choices=["inner", "all"],
    default="inner",
    help="ECAL region to process ('inner' or 'all')"
)
args = parser.parse_args()
region = args.region

print(f"🔹 Grouping ECAL cylindrical data for region: {region}")

n_points = 100000
energy_scale_factor = 7.85
max_events = 10000

rho_cells_no_values = [25]
phi_cells_no_values = [60]
z_cells_no_values = [52]
rho_cells_size_values = [3.632387237]
z_cells_size_values = [12.47759681]

input_file = (
    f"simulated_showers_{region}.root" if region != "all"
    else "simulated_showers.root"
)
print(f"Reading simulated data from {input_file} ...")

for rho_cells_no in rho_cells_no_values:
    for phi_cells_no in phi_cells_no_values:
        phi_cells_size = 2 * np.pi / phi_cells_no
        for z_cells_no in z_cells_no_values:
            for rho_cells_size in rho_cells_size_values:
                for z_cells_size in z_cells_size_values:

                    output_file_name = (
                        f"calorimeter_coordinates_{region}_"
                        f"rho{rho_cells_no}_phi{phi_cells_no}_z{z_cells_no}_"
                        f"dr{int(rho_cells_size)}_dz{str(z_cells_size).replace('.', '_')}.root"
                    )

                    print(f"\n[→] Processing {output_file_name}")
                    start_time = time.time()

                    with ur.recreate(output_file_name) as output_file:
                        # Initialize ROOT tree schema
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

                                    df_cyl = cylindrical(
                                        df_event, event_id,
                                        rho_cells_no, phi_cells_no, z_cells_no,
                                        rho_cells_size, z_cells_size, E, n_points
                                    )

                                    if df_cyl.empty:
                                        processed_events.add(event_id)
                                        pbar.update(1)
                                        continue

                                    local_points = cylindrical_centers_to_local_cartesian(df_cyl)
                                    direction = df_cyl[['direction_x', 'direction_y', 'direction_z']].iloc[0].values
                                    destination = df_cyl[['destination_x', 'destination_y', 'destination_z']].iloc[0].values
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
