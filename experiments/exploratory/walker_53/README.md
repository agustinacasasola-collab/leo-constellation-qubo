# Walker-53 Experiment

Walker Delta constellation T=648 / P=72 / F=1 at 53 deg inclination, 550 km altitude.
Selection of k=100 satellites that minimises per-satellite collision risk and guarantees
ground-station coverage over all 36 two-hour windows.

## Constellation

| Parameter        | Value                        |
|-----------------|------------------------------|
| Total sats (T)  | 648                          |
| Planes (P)      | 72                           |
| Sats/plane (S)  | 9                            |
| Phasing (F)     | 1                            |
| Inclination     | 53 deg                       |
| Altitude        | 550 km (circular)            |
| NORAD range     | 95001 -- 95648               |
| Epoch           | Inherited from shell3_synthetic.tle |

Walker Delta formulas:
    RAAN_i   = i * (360 / P)                          (deg)
    M_{i,j}  = j * (360 / S) + i * F * (360 / T)     (deg)

## Ground Stations (11)

| Station    | Lat (deg) | Lon (deg) |
|-----------|-----------|-----------|
| Nairobi   |  -1.29    |  36.82    |
| Lagos     |   6.45    |   3.39    |
| Singapore |   1.35    | 103.82    |
| Mumbai    |  19.08    |  72.88    |
| Lima      | -12.05    | -77.04    |
| Bogota    |   4.71    | -74.07    |
| Darwin    | -12.46    | 130.84    |
| Madrid    |  40.42    |  -3.70    |
| Beijing   |  39.91    | 116.39    |
| Ottawa    |  45.42    | -75.69    |
| London    |  51.51    |  -0.13    |

Minimum elevation: 5 deg.  Coverage window: 2 h.  Simulation: 3 days (36 windows).

## Pipeline

```
Step 1   generate_walker53.py         Generate TLEs from Walker Delta formula
Step 2   propagate_walker53.py        SGP4 propagation (648 x 4321 = 2.8 M rows)
Step 3   compute_pc_walker53.py       Chan 2D Pc vs LEO catalog
Step 4   compute_visibility_walker53.py  11 stations x 36 windows
Step 5   optimize_walker53.py         QUBO (safety_norm + cardinality + GS)
```

Run from the project root:

```bash
python experiments/walker_53/src/generate_walker53.py
python experiments/walker_53/src/propagate_walker53.py
python experiments/walker_53/src/compute_pc_walker53.py
python experiments/walker_53/src/compute_visibility_walker53.py
python experiments/walker_53/src/optimize_walker53.py
```

## QUBO Formulation

Safety signal (log-scale for dynamic range):
    safety_raw_i = -log(Pc_i + 1e-15)
    safety_norm_i = safety_raw_i / max(safety_raw)

Objective:    H_obj  = -sum_i safety_norm_i * x_i           (linear, diagonal only)
Cardinality:  H_card = P_card * (sum_i x_i - k)^2
GS (linear):  H_gs   = 2 * P_gs * sum_{s,w} (1 - sum_{i in C_sw} x_i)

Note: GS penalty is LINEAR (no off-diagonal cross terms), avoiding the cardinality
drift that arises from the quadratic formulation used in the collision experiment.

P_card = 20.0 * w_max * k * N
P_gs   =  2.0 * w_max * k * N   (same order as P_card, intentionally binding)

Infeasible runs (k_raw != k or GS violations) are DISCARDED -- no fixup applied.

## Outputs

| File                             | Description                        |
|---------------------------------|------------------------------------|
| data/walker53.tle               | Generated TLE set (648 sats)       |
| data/propagated_walker53.csv    | 2.8 M-row trajectory               |
| data/walker53_pc.csv            | Per-satellite Pc + dynamic range   |
| data/walker53_visibility.csv    | Sparse visibility matrix           |
| data/walker53_coverage_count.csv| 396-row coverage count (11 x 36)   |
| results/walker53_comparison.csv | Solver comparison table            |
| results/walker53_SA_best.csv    | Best SA selection (100 sats)       |
| results/walker53_SQA_best.csv   | Best SQA selection (100 sats)      |
