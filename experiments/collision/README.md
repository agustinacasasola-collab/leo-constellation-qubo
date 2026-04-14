# Experiment 1 — Collision Avoidance (Pure Pc)

Replication and extension of Owens-Fahrner et al. (2025).

## Setup
Shell: 550 km, 30 deg (Arnas Shell 3)
N = 1,656 candidates  (No=184, Nso=9, Nc=132)
k = 100 selected
sigma = 0.1 km  (Owens-Fahrner standard)
Objective: minimize aggregate Pc only

## Pipeline
Step 1  generate_candidates.py    -> data/shell3_synthetic.tle
Step 2  fetch_catalog.py          -> data/leo_catalog.tle
Step 3  propagate_catalog.py      -> data/propagated_catalog.csv
Step 4  propagate_orbits.py       -> data/propagated_candidates.csv
Step 5  compute_pc.py             -> data/candidates_pc.csv
Step 6  build_arnas_dataset.py    -> data/arnas_candidates.csv
Step 7  analyze_results.py        -> results/method_comparison.csv
Step 8  run_hybrid.py             -> results/hybrid_comparison.csv

## Results
Method   Best Pc     OOM vs random
------   ---------   -------------
Random   3.69e-03    --
SA       8.79e-05    +1.62
SQA      2.23e-05    +2.22
Hybrid   pending     pending

## Status
Steps 1-7 complete. Step 8 pending (requires Leap account).
