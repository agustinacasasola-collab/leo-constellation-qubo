# Experiment 2 — Constellation Bi-Objective (Pc + Land Cover)

Original contribution: bi-objective slot selection combining
collision avoidance and land coverage, validated on D-Wave QPU.

## Setup
Shell: 550 km, 53 deg (Starlink-like)
N = 130 candidates  (No=130, Nso=1)
k = 20 selected  (~15% of pool)
sigma = 1.0 km  (SGP4 propagation uncertainty, Vallado 2006)
Objective: minimize Pc AND maximize coverage of 20N-50N

## Edge weight (both terms quadratic and pairwise)
  w(i,j) = lambda * safety_i * safety_j
          + (1-lambda) * coverage_i * coverage_j
  safety_i = -log(Pc_i + 1e-15) / max(-log(Pc))
  coverage_i = fraction of timesteps in [20N, 50N] / max

## Pipeline
Step 1  generate_shell3r.py          -> data/shell3r_candidates.tle
Step 2  propagate_shell3r.py         -> data/propagated_shell3r.csv
Step 3  compute_pc_shell3r.py        -> data/shell3r_pc.csv
Step 4  compute_coverage_shell3r.py  -> data/shell3r_coverage.csv
Step 5  optimize_shell3r.py          -> results/shell3r_comparison.csv
Step 6  submit_qpu_shell3r.py        -> results/shell3r_QPU.csv

## Gates
Step 3: Pc > 0 for >= 10/130 candidates (sigma=1 km)
Step 4: coverage max/min ratio >= 1.05
Step 5: SA OOM > 0.5 vs random
Step 6: mean chain break fraction < 0.10

## Solvers
Step 5: SA vs SQA (local validation)
Step 6: SA vs SQA vs QPU real (D-Wave Advantage via Leap)

## Lambda sweep (Step 5)
lambda in [0.3, 0.5, 0.7, 0.9, 1.0]
Characterizes Pareto frontier between safety and coverage.

## Status
All steps pending.
