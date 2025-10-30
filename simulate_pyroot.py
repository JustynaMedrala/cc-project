#!/usr/bin/env python3
"""
simulate_pyroot.py
------------------

Simulates electromagnetic showers in a simplified ECAL geometry using ROOT.
Each simulated shower is generated according to an analytical longitudinal and
radial energy distribution, then filtered by ECAL subregion boundaries
("inner", "middle", "outer", or "all").

The script:
 - Reads particle kinematics from a Gauss-based ROOT file.
 - Generates random shower points in local coordinates.
 - Rotates and translates them into global coordinates.
 - Keeps only points that fall into the requested ECAL region.
 - Writes all accepted hits to a new ROOT file.

Usage:
    python3 simulate_pyroot.py --region inner
    python3 simulate_pyroot.py --region middle
    python3 simulate_pyroot.py --region outer
    python3 simulate_pyroot.py --region all

Outputs:
    simulated_showers_<region>.root   (or simulated_showers.root for "all")

Author: Justyna Mędrala-Sowa
Date: 2025-10-30
"""

import argparse
import numpy as np
from array import array
from math import sqrt, log, acos, pi
import ROOT
from tqdm import tqdm
from ecal import BOUNDARIES

# ROOT in batch mode
ROOT.gROOT.SetBatch(True)

# ======================================
# === Command-line arguments ===
# ======================================
parser = argparse.ArgumentParser(description="Simulate particle showers in ECAL")
parser.add_argument(
    "--region",
    type=str,
    choices=["inner", "middle", "outer", "all"],
    default="inner",
    help="ECAL region to simulate ('inner', 'middle', 'outer', or 'all')"
)
args = parser.parse_args()
region = args.region
print(f"Simulating ECAL region: {region}")

input_file_name = (
    "Gauss_CaloChallenge_MomentumRange_[10.0-100.0]GeV_DetailedSimulation_cell_rho-no18_phi-no50_z-no45_"
    "rho-size9.0_z-size13.824__INTERopths_5_INTRAopths_1-10000ev-20231212-TrainingData.root"
)
output_file_name = f"simulated_showers_{region}.root" if region != "all" else "simulated_showers.root"

Z = 82
X0 = 17.28  # mm
beta = 0.7
Ec = 610. / (Z + 1.24)
RM = 36
sigma = RM / 1.645
n_points = 100000
max_events = 7500  

mass_table = {
    11: 0.000511, -11: 0.000511,
    13: 0.105658, -13: 0.105658,
    22: 0.0, 211: 0.13957, -211: 0.13957,
    2212: 0.938272, -2212: 0.938272
}

input_file = ROOT.TFile.Open(input_file_name)
tree = input_file.Get("CaloCollector/Particles")
if not tree:
    raise ValueError("Could not find TTree 'CaloCollector/Particles' in input file.")
print(f"Loaded {tree.GetEntries()} entries")

output_file = ROOT.TFile(output_file_name, "RECREATE")
out_tree = ROOT.TTree("SimulatedShowers", "Simulated particle showers")

vars_to_write = {
    'Event_ID': array('i', [0]),
    'Particle_Index': array('i', [0]),
    'x_local': array('f', [0.]),
    'y_local': array('f', [0.]),
    'z_local': array('f', [0.]),
    'direction_x': array('f', [0.]),
    'direction_y': array('f', [0.]),
    'direction_z': array('f', [0.]),
    'destination_x': array('f', [0.]),
    'destination_y': array('f', [0.]),
    'destination_z': array('f', [0.]),
    'Energy': array('f', [0.])
}
for name, arr in vars_to_write.items():
    out_tree.Branch(name, arr, f"{name}/{'I' if arr.typecode == 'i' else 'F'}")

def compute_energy(px, py, pz, pid):
    p = sqrt(px**2 + py**2 + pz**2)
    m = mass_table.get(pid, 0.0)
    return sqrt(p**2 + m**2)

def rotate(local_coords, direction):
    direction /= np.linalg.norm(direction)
    z_axis = np.array([0., 0., 1.])
    axis = np.cross(z_axis, direction)
    if np.linalg.norm(axis) < 1e-6:
        return local_coords
    axis /= np.linalg.norm(axis)
    angle = acos(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle)*K + (1 - np.cos(angle))*(K @ K)
    return local_coords @ R.T

def in_region(global_coords, region):
    x, y = global_coords[:, 0], global_coords[:, 1]

    if region == "all":
        print("→ Keeping all points")
        return np.ones(len(global_coords), dtype=bool)

    reg = BOUNDARIES[region]
    mask = (x >= reg["x"]["min"]) & (x <= reg["x"]["max"]) & (y >= reg["y"]["min"]) & (y <= reg["y"]["max"])


    if region == "middle":
        for sub in ["inner"]:
            b = BOUNDARIES[sub]
            sub_mask = (x >= b["x"]["min"]) & (x <= b["x"]["max"]) & (y >= b["y"]["min"]) & (y <= b["y"]["max"])
            mask &= ~sub_mask
    elif region == "outer":
        for sub in ["middle"]:
            b = BOUNDARIES[sub]
            sub_mask = (x >= b["x"]["min"]) & (x <= b["x"]["max"]) & (y >= b["y"]["min"]) & (y <= b["y"]["max"])
            mask &= ~sub_mask
    return mask

seen_events = set()
event_iter = (entry for entry in tree if entry.Event_ID not in seen_events)
event_iter = tqdm(event_iter, total=max_events, desc="Simulating Events", unit="event")

for entry in event_iter:
    if len(seen_events) >= max_events:
        break

    px, py, pz = entry.Momentum_X, entry.Momentum_Y, entry.Momentum_Z
    pid = entry.Particle_PID
    energy = compute_energy(px, py, pz, pid)

    origin = np.array([0., 0., 0.])
    destination = np.array([entry.Entry_X, entry.Entry_Y, entry.Entry_Z])
    direction_unit = destination - origin
    direction_unit /= np.linalg.norm(direction_unit)

    Y = energy / Ec
    tmax = log(Y) - 0.5
    alpha = tmax * beta + 1
    depths = np.random.gamma(shape=alpha, scale=1/beta, size=n_points)
    z_local = depths * X0
    r = np.abs(np.random.normal(scale=sigma, size=n_points))
    theta = np.random.uniform(0, 2*pi, size=n_points)
    x_local = r * np.cos(theta)
    y_local = r * np.sin(theta)

    local_coords = np.vstack([x_local, y_local, z_local]).T
    global_coords = rotate(local_coords, direction_unit)
    global_coords += origin

    mask = in_region(global_coords, "inner")

    x_local = local_coords[mask, 0]
    y_local = local_coords[mask, 1]
    z_local = local_coords[mask, 2]

    seen_events.add(entry.Event_ID)

    for j in range(len(x_local)):
        vars_to_write['Event_ID'][0] = entry.Event_ID
        vars_to_write['Particle_Index'][0] = entry.Particle_Index
        vars_to_write['x_local'][0] = x_local[j]
        vars_to_write['y_local'][0] = y_local[j]
        vars_to_write['z_local'][0] = z_local[j]
        vars_to_write['direction_x'][0] = direction_unit[0]
        vars_to_write['direction_y'][0] = direction_unit[1]
        vars_to_write['direction_z'][0] = direction_unit[2]
        vars_to_write['destination_x'][0] = destination[0]
        vars_to_write['destination_y'][0] = destination[1]
        vars_to_write['destination_z'][0] = destination[2]
        vars_to_write['Energy'][0] = energy
        out_tree.Fill()

output_file.Write()
output_file.Close()
input_file.Close()
print(f"\nSimulated showers saved to {output_file_name}")
