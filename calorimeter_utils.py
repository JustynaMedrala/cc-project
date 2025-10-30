"""
============================================================
calorimeter_utils.py
============================================================

This module provides geometric binning, coordinate transformation, and 
energy histogram utilities for electromagnetic calorimeter (ECAL) simulation 
and analysis workflows. It supports multiple coordinate systems (cylindrical, 
spherical, conical) and enables mapping simulated shower hits from local to 
global (world) coordinates for comparison and visualization.

------------------------------------------------------------
Main Functionalities
------------------------------------------------------------
1. **Binning in Different Geometries**
   - `cylindrical()`: Discretizes hits into cylindrical (ρ, φ, z) bins.
   - `conical()`: Bins hits inside a conical volume with radius increasing linearly with z.
   - `spherical()`: Bins hits in spherical coordinates (r, θ, φ).

2. **Coordinate Conversion**
   - Converts binned cylindrical/conical/spherical centers into local Cartesian coordinates.
   - Transforms local coordinates into world coordinates using rotation matrices
     derived from shower direction and destination.

3. **ECAL Energy Histogramming**
   - `histogram_energy_xy()`: Produces energy maps across ECAL layers (inner, middle, outer),
     excluding beam regions.
   - `histogram_energy_xy_only_inner()`: Restricts histogramming to the inner ECAL region.
   - Uses realistic ECAL boundaries (`BOUNDARIES`) and cell size scaling from the `ecal` module.

4. **Region Deduction & Evaluation**
   - `deduce_ecal_region()`: Determines which ECAL region (inner/middle/outer/beam)
     a given (x, y) coordinate belongs to.
   - `evaluate_configuration()`: Computes simple metrics (energy ratio, cell count)
     to assess binning performance and reconstruction quality.

------------------------------------------------------------
Functions Overview
------------------------------------------------------------
cylindrical(points_df, event_id, rho_cells_no, phi_cells_no, z_cells_no,
            rho_cells_size, z_cells_size, E, n_points) -> pd.DataFrame
    Bins simulated shower hits in cylindrical coordinates.

conical(points_df, event_id, r_cells_no, phi_cells_no, z_cells_no,
        r_max_base, z_cells_size, E, n_points) -> pd.DataFrame
    Bins energy deposits inside a conical geometry aligned with the beam.

spherical(points_df, event_id, r_cells_no, theta_cells_no, phi_cells_no,
          r_cells_size, E, n_points) -> pd.DataFrame
    Bins energy deposits in spherical shells and angular sectors.

cylindrical_centers_to_local_cartesian(df) -> np.ndarray
conical_centers_to_local_cartesian(df) -> np.ndarray
spherical_centers_to_local_cartesian(df) -> np.ndarray
    Convert geometry bin centers back to local 3D Cartesian coordinates.

local_to_world(points_local, direction_unit, destination) -> np.ndarray
    Applies rotation and translation to map local coordinates into the 
    detector/world frame based on shower direction and destination point.

histogram_energy_xy(xw, yw, E, energy_scale_factor, n_points) -> pd.DataFrame
    Builds ECAL energy histograms for inner, middle, and outer layers.

histogram_energy_xy_only_inner(xw, yw, E, energy_scale_factor, n_points) -> pd.DataFrame
    Builds ECAL energy histograms restricted to the inner region only.

deduce_ecal_region(coords) -> str
    Returns ECAL region name ("inner", "middle", "outer", or "beam") 
    for a given (x, y) coordinate.

evaluate_configuration(df_cyl, df_xy, E_true, energy_scale_factor) -> dict
    Evaluates a binning configuration based on energy conservation and 
    the number of active cells.

Author: Justyna Mędrala-Sowa
Created: 30.10.2025
------------------------------------------------------------
"""



import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from ecal import BOUNDARIES, EcalModXYSize

def cylindrical(points_df, event_id, rho_cells_no, phi_cells_no, z_cells_no,
                rho_cells_size, z_cells_size, E, n_points):

    event_df = points_df[points_df["Event_ID"] == event_id]
    x_local = event_df['x_local'].values
    y_local = event_df['y_local'].values
    z_local = event_df['z_local'].values

    rho = np.sqrt(x_local**2 + y_local**2)
    phi = np.arctan2(y_local, x_local)
    z = z_local

    rho_max = rho_cells_no * rho_cells_size
    rho_bins = np.linspace(0, rho_max, rho_cells_no + 1)
    phi_bins = np.linspace(-np.pi*1.05, np.pi, phi_cells_no + 1)
    z_bins = np.linspace(0, z_cells_no*z_cells_size, z_cells_no + 1)

    hist, edges =  np.histogramdd((rho, phi, z), bins=(rho_bins, phi_bins, z_bins))

    rho_centers = 0.5 * (rho_bins[:-1] + rho_bins[1:])
    phi_centers = 0.5 * (phi_bins[:-1] + phi_bins[1:])
    z_centers = 0.5 * (z_bins[:-1] + z_bins[1:])
    rho_grid, phi_grid, z_grid = np.meshgrid(rho_centers, phi_centers, z_centers, indexing='ij')

    df = pd.DataFrame({
        "rho_center": rho_grid.flatten(),
        "phi_center": phi_grid.flatten(),
        "z_center": z_grid.flatten(),
        "energy": hist.flatten(),
    })

    #print(f"[Cylindrical] Total Energy in Bins: {df['energy'].sum()}")

    df["Event_ID"] = event_id
    df["E_total"] = E
    for coord in ['x', 'y', 'z']:
        df[f"direction_{coord}"] = event_df[f"direction_{coord}"].values[0]
        df[f"destination_{coord}"] = event_df[f"destination_{coord}"].values[0]

    return df.dropna(subset=["energy"]).query("energy > 0")

def conical(points_df, event_id, r_cells_no, phi_cells_no, z_cells_no,
            r_max_base, z_cells_size, E, n_points):
    """
    Binning energy deposits inside a conical volume.
    
    Parameters:
        points_df: DataFrame with hit positions
        event_id: Current event ID
        r_cells_no: Number of radial bins (radius grows linearly with height)
        phi_cells_no: Number of azimuthal bins
        z_cells_no: Number of height bins
        r_max_base: Max radius at base of cone (top z)
        z_cells_size: Cell size along z
        E: Total event energy
        n_points: Number of points
    
    Returns:
        DataFrame with binned energy deposits
    """
    event_df = points_df[points_df["Event_ID"] == event_id]
    x_local = event_df['x_local'].values
    y_local = event_df['y_local'].values
    z_local = event_df['z_local'].values

    # Convert to cylindrical coords for angle calculation
    rho = np.sqrt(x_local**2 + y_local**2)
    phi = np.arctan2(y_local, x_local)
    z = z_local
    
    # Max cone height
    z_max = z_cells_no * z_cells_size
    
    # Calculate max radius at each height (linear growth from 0 to r_max_base)
    r_max_z = (z / z_max) * r_max_base
    
    # Normalize radius within cone at each z
    r_normalized = rho / (r_max_z + 1e-9)
    
    # Filter points outside cone (r > r_max at z)
    inside_cone = r_normalized <= 1
    rho = rho[inside_cone]
    phi = phi[inside_cone]
    z = z[inside_cone]

    # Radial bins now normalized 0-1 at each z slice
    r_bins = np.linspace(0, 1, r_cells_no + 1)
    phi_bins = np.linspace(-np.pi, np.pi, phi_cells_no + 1)
    z_bins = np.linspace(0, z_max, z_cells_no + 1)

    hist, edges = np.histogramdd((r_normalized[inside_cone], phi, z),
                                 bins=(r_bins, phi_bins, z_bins))

    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
    phi_centers = 0.5 * (phi_bins[:-1] + phi_bins[1:])
    z_centers = 0.5 * (z_bins[:-1] + z_bins[1:])
    
    r_grid, phi_grid, z_grid = np.meshgrid(r_centers, phi_centers, z_centers, indexing='ij')

    # Convert normalized radial bins back to actual radius at center of each z bin
    r_actual = r_grid * (z_grid / z_max) * r_max_base

    df = pd.DataFrame({
        "r_center": r_actual.flatten(),
        "phi_center": phi_grid.flatten(),
        "z_center": z_grid.flatten(),
        "energy": hist.flatten(),
    })

    df["Event_ID"] = event_id
    df["E_total"] = E
    for coord in ['x', 'y', 'z']:
        df[f"direction_{coord}"] = event_df[f"direction_{coord}"].values[0]
        df[f"destination_{coord}"] = event_df[f"destination_{coord}"].values[0]

    return df.query("energy > 0")



def spherical(points_df, event_id, r_cells_no, theta_cells_no, phi_cells_no,
              r_cells_size, E, n_points):
    event_df = points_df[points_df["Event_ID"] == event_id]
    x_local = event_df['x_local'].values
    y_local = event_df['y_local'].values
    z_local = event_df['z_local'].values

    # Convert to spherical coordinates
    r = np.sqrt(x_local**2 + y_local**2 + z_local**2)
    theta = np.arccos(np.clip(z_local / (r + 1e-9), -1, 1))  # polar angle [0, pi]
    phi = np.arctan2(y_local, x_local)  # azimuth [-pi, pi]

    r_max = r_cells_no * r_cells_size
    r_bins = np.linspace(0, r_max, r_cells_no + 1)
    theta_bins = np.linspace(0, np.pi, theta_cells_no + 1)
    phi_bins = np.linspace(-np.pi, np.pi, phi_cells_no + 1)

    hist, edges = np.histogramdd((r, theta, phi), bins=(r_bins, theta_bins, phi_bins))

    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
    theta_centers = 0.5 * (theta_bins[:-1] + theta_bins[1:])
    phi_centers = 0.5 * (phi_bins[:-1] + phi_bins[1:])

    r_grid, theta_grid, phi_grid = np.meshgrid(r_centers, theta_centers, phi_centers, indexing='ij')

    df = pd.DataFrame({
        "r_center": r_grid.flatten(),
        "theta_center": theta_grid.flatten(),
        "phi_center": phi_grid.flatten(),
        "energy": hist.flatten(),
    })

    df["Event_ID"] = event_id
    df["E_total"] = E
    for coord in ['x', 'y', 'z']:
        df[f"direction_{coord}"] = event_df[f"direction_{coord}"].values[0]
        df[f"destination_{coord}"] = event_df[f"destination_{coord}"].values[0]

    return df.query("energy > 0")


def cylindrical_centers_to_local_cartesian(df):
    rho = df["rho_center"].values
    phi = df["phi_center"].values
    z = df["z_center"].values
    
    x_local = rho * np.cos(phi)
    y_local = rho * np.sin(phi)
    z_local = z
    
    points = np.vstack([x_local, y_local, z_local]).T
    
    energies = df["energy"].values.astype(int)

    repeated_points = np.repeat(points, energies, axis=0)
    
    return repeated_points

def conical_centers_to_local_cartesian(df):
    r = df["r_center"].values
    phi = df["phi_center"].values
    z = df["z_center"].values
    
    # Convert from polar (r, phi) to Cartesian (x, y)
    x_local = r * np.cos(phi)
    y_local = r * np.sin(phi)
    z_local = z
    
    points = np.vstack([x_local, y_local, z_local]).T
    
    energies = df["energy"].values.astype(int)
    
    # Repeat points according to energy values (number of hits)
    repeated_points = np.repeat(points, energies, axis=0)
    
    return repeated_points


def spherical_centers_to_local_cartesian(df):
    r = df["r_center"].values
    theta = df["theta_center"].values  # polar angle: 0 = z+, π = z-
    phi = df["phi_center"].values      # azimuthal angle: 0 = x+, π/2 = y+

    # Convert spherical to Cartesian coordinates
    x_local = r * np.sin(theta) * np.cos(phi)
    y_local = r * np.sin(theta) * np.sin(phi)
    z_local = r * np.cos(theta)

    points = np.vstack([x_local, y_local, z_local]).T

    energies = df["energy"].values.astype(int)
    repeated_points = np.repeat(points, energies, axis=0)

    return repeated_points




def local_to_world(points_local, direction_unit, destination):
    z_axis = np.array([0, 0, 1])
    rotation_vector = np.cross(z_axis, direction_unit)
    rotation_angle = np.arccos(np.clip(np.dot(z_axis, direction_unit), -1.0, 1.0))

    if np.linalg.norm(rotation_vector) < 1e-6:
        rot_matrix = np.eye(3)
    else:
        rot = R.from_rotvec(rotation_angle * rotation_vector / np.linalg.norm(rotation_vector))
        rot_matrix = rot.as_matrix()

    return points_local @ rot_matrix.T + destination

def histogram_energy_xy(xw, yw, E, energy_scale_factor, n_points):
    regions = ["inner", "middle", "outer"]

    def get_cell_size(region):
        if region == "inner":
            return EcalModXYSize / 3
        elif region == "middle":
            return EcalModXYSize / 2
        else:
            return EcalModXYSize

    results = []
    E_per_hit = E / n_points

    # Get boundaries
    inner_bounds = BOUNDARIES["inner"]
    middle_bounds = BOUNDARIES["middle"]
    outer_bounds = BOUNDARIES["outer"]
    beam_bounds = BOUNDARIES["beam"]  # Assuming beam is defined here

    # Beam mask
    mask_beam = (
        (xw >= beam_bounds["x"]["min"]) & (xw <= beam_bounds["x"]["max"]) &
        (yw >= beam_bounds["y"]["min"]) & (yw <= beam_bounds["y"]["max"])
    )

    # Inner region excluding beam
    mask_inner = (
        (xw >= inner_bounds["x"]["min"]) & (xw <= inner_bounds["x"]["max"]) &
        (yw >= inner_bounds["y"]["min"]) & (yw <= inner_bounds["y"]["max"]) &
        (~mask_beam)
    )

    # Middle region excluding inner and beam
    mask_middle = (
        (xw >= middle_bounds["x"]["min"]) & (xw <= middle_bounds["x"]["max"]) &
        (yw >= middle_bounds["y"]["min"]) & (yw <= middle_bounds["y"]["max"]) &
        (~mask_inner) & (~mask_beam)
    )

    # Outer region excluding middle, inner, and beam
    mask_outer = (
        (xw >= outer_bounds["x"]["min"]) & (xw <= outer_bounds["x"]["max"]) &
        (yw >= outer_bounds["y"]["min"]) & (yw <= outer_bounds["y"]["max"]) &
        (~mask_inner) & (~mask_middle) & (~mask_beam)
    )

    region_masks = {
        "inner": mask_inner,
        "middle": mask_middle,
        "outer": mask_outer
    }

    for region in regions:
        cell_size = get_cell_size(region)
        bounds = BOUNDARIES[region]

        mask = region_masks[region]
        xw_region = xw[mask]
        yw_region = yw[mask]

        x_min = bounds["x"]["min"]
        x_max = bounds["x"]["max"]
        y_min = bounds["y"]["min"]
        y_max = bounds["y"]["max"]

        x_bins = np.arange(x_min, x_max + cell_size, cell_size)
        y_bins = np.arange(y_min, y_max + cell_size, cell_size)

        hist, x_edges, y_edges = np.histogram2d(xw_region, yw_region, bins=[x_bins, y_bins])

        hist = hist * E_per_hit / energy_scale_factor

        nonzero_indices = np.array(np.nonzero(hist))
        cell_x = (x_edges[:-1][nonzero_indices[0]] + x_edges[1:][nonzero_indices[0]]) / 2
        cell_y = (y_edges[:-1][nonzero_indices[1]] + y_edges[1:][nonzero_indices[1]]) / 2
        active_energy = hist[nonzero_indices[0], nonzero_indices[1]]

        df_region = pd.DataFrame({
            "Cell_X": cell_x.astype(np.float32),
            "Cell_Y": cell_y.astype(np.float32),
            "Cell_Size": np.full_like(cell_x, cell_size, dtype=np.float32),
            "Active_Energy": active_energy.astype(np.float32),
            "Region": region
        })

        results.append(df_region)

    return pd.concat(results, ignore_index=True)

def histogram_energy_xy_only_inner(xw, yw, E, energy_scale_factor, n_points):
    regions = ["inner"]

    def get_cell_size(region):
        if region == "inner":
            return EcalModXYSize / 3
        elif region == "middle":
            return EcalModXYSize / 2
        else:
            return EcalModXYSize

    results = []
    E_per_hit = E / n_points

    # Get boundaries
    inner_bounds = BOUNDARIES["inner"]
    beam_bounds = BOUNDARIES["beam"]  # Assuming beam is defined here

    # Beam mask
    mask_beam = (
        (xw >= beam_bounds["x"]["min"]) & (xw <= beam_bounds["x"]["max"]) &
        (yw >= beam_bounds["y"]["min"]) & (yw <= beam_bounds["y"]["max"])
    )

    # Inner region excluding beam
    mask_inner = (
        (xw >= inner_bounds["x"]["min"]) & (xw <= inner_bounds["x"]["max"]) &
        (yw >= inner_bounds["y"]["min"]) & (yw <= inner_bounds["y"]["max"]) &
        (~mask_beam)
    )


    region_masks = {
        "inner": mask_inner}

    for region in regions:
        cell_size = get_cell_size(region)
        bounds = BOUNDARIES[region]

        mask = region_masks[region]
        xw_region = xw[mask]
        yw_region = yw[mask]

        x_min = bounds["x"]["min"]
        x_max = bounds["x"]["max"]
        y_min = bounds["y"]["min"]
        y_max = bounds["y"]["max"]

        x_bins = np.arange(x_min, x_max + cell_size, cell_size)
        y_bins = np.arange(y_min, y_max + cell_size, cell_size)

        hist, x_edges, y_edges = np.histogram2d(xw_region, yw_region, bins=[x_bins, y_bins])

        hist = hist * E_per_hit / energy_scale_factor

        nonzero_indices = np.array(np.nonzero(hist))
        cell_x = (x_edges[:-1][nonzero_indices[0]] + x_edges[1:][nonzero_indices[0]]) / 2
        cell_y = (y_edges[:-1][nonzero_indices[1]] + y_edges[1:][nonzero_indices[1]]) / 2
        active_energy = hist[nonzero_indices[0], nonzero_indices[1]]

        df_region = pd.DataFrame({
            "Cell_X": cell_x.astype(np.float32),
            "Cell_Y": cell_y.astype(np.float32),
            "Cell_Size": np.full_like(cell_x, cell_size, dtype=np.float32),
            "Active_Energy": active_energy.astype(np.float32),
            "Region": region
        })

        results.append(df_region)

    return pd.concat(results, ignore_index=True)




def deduce_ecal_region(coords):
    x, y, _ = coords
    for region in ["inner", "middle", "outer"]:
        b = BOUNDARIES[region]
        if b["x"]["min"] <= x <= b["x"]["max"] and b["y"]["min"] <= y <= b["y"]["max"]:
            return region
    b = BOUNDARIES["beam"]
    if b["x"]["min"] <= x <= b["x"]["max"] and b["y"]["min"] <= y <= b["y"]["max"]:
        return "beam"
    return None


def evaluate_configuration(df_cyl, df_xy, E_true, energy_scale_factor):
    """Evaluate the quality of a binning configuration"""
    metrics = {}
    
    metrics['energy_ratio'] = df_xy["Active_Energy"].sum() / (E_true/energy_scale_factor)
    
    metrics['n_cells'] = len(df_xy)
    
    return metrics