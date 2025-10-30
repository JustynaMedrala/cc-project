# Calorimeter Geometry and Sampling Toolkit

description: >
  Toolkit for modeling, binning, and evaluating calorimeter energy deposits in
  simulated detector geometries. Supports cylindrical, conical, and spherical
  binning, coordinate transformations, ECAL histogramming, and event-wise
  comparison with real data.

pipeline:
  name: Calorimeter Simulation & Optimization
  description: >
    End-to-end pipeline for simulating calorimeter energy deposits, grouping
    calorimeter cell data, optimizing sampling configurations via Bayesian search,
    and detecting plateau regions in performance metrics.
  stages:
    - name: Simulation
      script: simulate_pyroot.py
      description: >
        Generates ROOT-format shower data (SimulatedShowers tree) from MC simulations.
      input: null
      output: simulated_showers.root
      command: |
        python simulate_pyroot.py --events 50000 --output simulated_showers.root
      parameters:
        events: 50000
        output: simulated_showers.root

    - name: Grouping
      script: group_calo.py
      description: >
        Converts simulation data to binned calorimeter hits.
        Applies cylindrical or conical energy binning and stores hits in ROOT format.
      input: simulated_showers.root
      output: calorimeter_coordinates_rho*_phi*_z*_dr*_dz*.root
      command: |
        python group_calo.py --geometry cylindrical --max-events 10000 --n-points 100000
      parameters:
        geometry: cylindrical
        max_events: 10000
        n_points: 100000

    - name: Optimization
      script: bayesian.py
      description: >
        Performs Bayesian optimization of geometry parameters using
        Wasserstein distance as a metric.
      input: calorimeter_coordinates_*.root
      output: bayesian_results.csv
      command: |
        python bayesian.py --iterations 100 --max-events 500
      parameters:
        iterations: 100
        max_events: 500

    - name: Plateau Detection
      script: plateau.py
      description: >
        Identifies stable parameter regions (plateaus) in the optimization results.
      input: bayesian_results.csv
      output: bayesian_results_plateau.csv
      command: |
        python plateau.py --delta 0.02
      parameters:
        delta: 0.02

workflow:
  steps:
    - simulate_pyroot.py
    - group_calo.py
    - bayesian.py
    - plateau.py

files:
  - simulate_pyroot.py
  - group_calo.py
  - bayesian.py
  - plateau.py
  - calorimeter_utils.py
  - generate_data.py
  - compare_utils.py

dependencies:
  python_packages:
    - numpy
    - pandas
    - uproot
    - awkward
    - scipy
    - tqdm
    - scikit-optimize
    - matplotlib
  python_version: ">=3.9"

notes:
  - All energy is scaled by energy_scale_factor = 7.85
  - Inner ECAL region excludes the beam area
  - Wasserstein distance is computed event-wise
  - Supports both full ECAL and inner-only analysis modes
