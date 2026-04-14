# Simulation Results Summary

Generated: 2026-04-02  
Source files: `experiments/collision/results/`, `results/shell3r_*.csv`, `data/arnas_candidates.csv`, `data/shell3r_pc.csv`, `data/shell3r_coverage.csv`

---

## Experiment 1 — Pure Collision Avoidance

### Setup

| Parameter | Value |
|-----------|-------|
| Shell | 550 km, 30 deg (Arnas Shell 3) |
| N candidates | 1,656 |
| k selected | 100 |
| sigma | 0.1 km (Owens-Fahrner standard) |
| Objective | Minimize aggregate Pc only |
| Reference | Owens-Fahrner et al. (2025) |

---

### Candidate Pool Statistics

Source: `data/arnas_candidates.csv`  
Columns: `satellite_id`, `pc`, `coverage`, `raan_deg`, `mean_anomaly_deg`

| Statistic | Value |
|-----------|-------|
| Total candidates (N) | 1,656 |
| Candidates with Pc = 0 | 1,419 (85.7%) |
| Candidates with Pc > 0 | 237 (14.3%) |
| Min Pc (non-zero) | 1.11e-16 |
| Max Pc | 4.68e-03 |
| Mean Pc (non-zero only) | 2.51e-04 |
| Mean Pc (all candidates) | 3.59e-05 |
| Dynamic range (log₁₀ max/min non-zero) | 13.62 |

**Note:** The overwhelming majority (85.7%) of candidates carry zero collision probability under sigma = 0.1 km. The non-zero Pc values span more than 13 orders of magnitude, producing a highly sparse and heterogeneous optimization landscape.

---

### Solver Comparison (Random / SA / SQA / Tabu)

Source: `experiments/collision/results/method_comparison.csv`  
Columns: `method`, `aggregate_pc`, `oom_vs_random`, `num_reads`, `feasibility_rate`, `sa_sqa_overlap`

> **Note:** The CSV contains one `aggregate_pc` per method (best achieved across all runs). Separate `mean_pc` and `std_pc` per method are NOT AVAILABLE in this file.

| Method | Best Aggregate Pc | OOM vs Random | Runs | Feasible % | SA/SQA Overlap |
|--------|------------------|---------------|------|------------|----------------|
| Random | 3.69e-03 | — | 30 | 100.0% | — |
| SA (SimulatedAnnealingSampler) | 8.79e-05 | +1.62 | 50 | 100.0% | 4 |
| Tabu (TabuSampler) | 4.01e-04 | +0.96 | 50 | 100.0% | — |
| SQA (PathIntegralAnnealingSampler) | 2.23e-05 | +2.22 | 50 | 100.0% | 4 |
| Paper SA/QA (Owens-Fahrner 2025) | 4.84e-06 | +1.22† | — | — | — |

†OOM as recorded in CSV; computed against the paper's own random baseline (not this experiment's).

**Key observations:**
- SQA achieves the best result of all local solvers: 2.23e-05, more than 2 orders of magnitude below random.
- SA achieves +1.62 OOM, placing it solidly between Tabu and SQA.
- All three solvers maintain 100% feasibility (exactly k=100 satellites selected in every run).
- The Owens-Fahrner paper reference result (4.84e-06) is approximately 4.7× better than our SQA, suggesting room for further QPU or hybrid improvement.

---

### Solution Analysis (SA best vs SQA best)

Source: `experiments/collision/results/selected_SA_best.csv` and `selected_SQA_best.csv`

#### SA Best Solution

| Statistic | Value |
|-----------|-------|
| Satellites selected | 100 |
| Satellites with Pc_n = 0 | 90 (90.0%) |
| Satellites with Pc_n > 0 | 10 (10.0%) |
| Min Pc_n (non-zero) | 6.77e-15 |
| Max Pc_n | 8.77e-05 |
| Mean Pc_n (all selected) | 8.79e-06 |
| RAAN min | 0.00° |
| RAAN max | 356.09° |
| RAAN std | 106.73° |

#### SQA Best Solution

| Statistic | Value |
|-----------|-------|
| Satellites selected | 100 |
| Satellites with Pc_n = 0 | 83 (83.0%) |
| Satellites with Pc_n > 0 | 17 (17.0%) |
| Min Pc_n (non-zero) | 1.11e-16 |
| Max Pc_n | 1.31e-05 |
| Mean Pc_n (all selected) | 1.31e-06 |
| RAAN min | 0.00° |
| RAAN max | 356.09° |
| RAAN std | 101.43° |

#### Overlap Between SA and SQA Best Solutions

| Metric | Value |
|--------|-------|
| Satellites in common | 4 / 100 |
| Overlap fraction | 4.0% |

**Note:** Both best solutions span the full RAAN range (0°–356°), indicating that both solvers independently discovered the value of distributing satellites uniformly in longitude of ascending node. Despite reaching very different aggregate Pc values, they share only 4 of 100 satellites, confirming that multiple near-optimal but structurally distinct configurations exist in this landscape.

---

### Hybrid Solver (pending)

| Metric | Value |
|--------|-------|
| Solver | D-Wave LeapHybridSampler |
| Script | `experiments/collision/src/run_hybrid.py` |
| Status | **PENDING** — requires active Leap account |
| Expected output | `results/hybrid_comparison.csv` |

Results for this step will be added once Leap QPU access is available.

---

## Experiment 2 — Bi-Objective (Pc + Land Coverage)

### Setup

| Parameter | Value |
|-----------|-------|
| Shell | 550 km, 53 deg (Starlink-like) |
| N candidates | 130 |
| k selected | 20 |
| sigma | 1.0 km (SGP4 propagation uncertainty, Vallado 2006) |
| Objective | Minimize Pc AND maximize coverage of 20°N–50°N |
| Lambda sweep | [0.3, 0.5, 0.7, 0.9, 1.0] |

Edge weight formula:
```
w(i,j) = lambda * safety_i * safety_j + (1-lambda) * coverage_i * coverage_j
safety_i  = -log(Pc_i + 1e-15) / max(-log(Pc))
coverage_i = fraction of timesteps in [20N, 50N] / max
```

---

### Candidate Pool Statistics

Source: `data/shell3r_pc.csv`  
Columns: `norad_id`, `raan_deg`, `Pc_n`, `n_after_ap_filter`, `n_after_screening`, `min_tca_km`

| Statistic | Value |
|-----------|-------|
| Total candidates (N) | 130 |
| Candidates with Pc = 0 | 1 (0.8%) |
| Candidates with Pc > 0 | 129 (99.2%) |
| Min Pc_n (non-zero) | 1.22e-15 |
| Max Pc_n | 4.89e-05 |
| Mean Pc_n (all candidates) | 2.16e-06 |
| Dynamic range (log₁₀ max/min non-zero) | 10.60 |
| Min TCA distance (min_tca_km) | 1.99 km |
| Mean TCA distance | 4.80 km |
| Max TCA distance | 9.24 km |
| Mean conjunctions after screening (n_after_screening) | 4.45 |
| Min conjunctions after screening | 1 |
| Max conjunctions after screening | 10 |

**Note:** With sigma = 1.0 km, almost all candidates (99.2%) have nonzero Pc, in sharp contrast to Experiment 1 (85.7% with Pc = 0). The landscape is much denser and more uniform, which fundamentally changes the optimization difficulty.

**Gate check (Step 3):** Pc > 0 required for ≥ 10/130 candidates. **PASSED** — 129/130 candidates have Pc > 0.

---

### Coverage Statistics

Source: `data/shell3r_coverage.csv`  
Columns: `norad_id`, `raan_deg`, `coverage_raw`, `coverage_norm`

| Statistic | Value |
|-----------|-------|
| Total candidates (N) | 130 |
| coverage_raw (all candidates) | 0.268688 (uniform) |
| coverage_raw min | 0.268688 |
| coverage_raw max | 0.268688 |
| coverage_norm min | 1.000000 |
| coverage_norm max | 1.000000 |
| coverage_norm max / min ratio | 1.0000 |

**Gate check (Step 4):** Coverage max/min ratio required ≥ 1.05. **FAILED** — ratio = 1.0000.

**Critical finding:** All 130 candidates exhibit identical coverage of the 20°N–50°N band (coverage_raw = 0.268688 for every candidate). This means the coverage objective provides zero differentiation between satellites: the bi-objective QUBO reduces to a pure Pc minimization regardless of lambda. The `best_cov = 1.0` value seen in all solver results is consistent with this — every feasible selection trivially achieves maximum normalized coverage. The coverage gate failure indicates this experiment requires re-parameterization (e.g., a narrower band, finer time resolution, or an orbit family with greater inclination spread) before the bi-objective formulation becomes meaningful.

---

### Solver Comparison (Random / SA / SQA)

Source: `results/shell3r_comparison.csv`  
Columns: `solver`, `lambda`, `best_pc`, `mean_pc`, `std_pc`, `best_cov`  

> **Note:** This file contains results for lambda = 0.3 only. Full lambda sweep in next section.

| Method | Lambda | Best Pc | Mean Pc | Std Pc | OOM vs Random | Best Coverage |
|--------|--------|---------|---------|--------|---------------|---------------|
| Random | 0.3 | 2.08e-05 | 4.63e-05 | 1.77e-05 | — | 1.000 |
| SA | 0.3 | 7.37e-06 | 4.20e-05 | 2.55e-05 | +0.45 | 1.000 |
| SQA | 0.3 | 2.29e-05 | 6.84e-05 | 2.61e-05 | −0.04 | 1.000 |

**Key observations:**
- SA achieves +0.45 OOM vs random, below the Step 5 gate threshold of > 0.5.
- SQA performs slightly *worse* than random (OOM = −0.04) at lambda = 0.3.
- Best coverage is identically 1.000 for all methods, confirming that coverage is not a discriminating objective.
- The dramatically reduced solver advantage compared to Experiment 1 (+0.45 vs +1.62 for SA, −0.04 vs +2.22 for SQA) is consistent with the denser, more uniform Pc landscape.

---

### Lambda Sweep

Source: `results/shell3r_lambda_sweep.csv`  
Columns: `lambda`, `sa_best_pc`, `sa_mean_pc`, `sa_best_cov`, `sqa_best_pc`, `sqa_mean_pc`, `sqa_best_cov`

| Lambda | SA Best Pc | SA Mean Pc | SA Best Cov | SQA Best Pc | SQA Mean Pc | SQA Best Cov |
|--------|-----------|------------|-------------|------------|-------------|--------------|
| 0.3 | 7.37e-06 | 4.46e-05 | 1.000 | 2.39e-05 | 7.74e-05 | 1.000 |
| 0.5 | 2.52e-05 | 4.81e-05 | 1.000 | 3.31e-05 | 7.44e-05 | 1.000 |
| 0.7 | 1.89e-05 | 4.92e-05 | 1.000 | 2.91e-05 | 7.27e-05 | 1.000 |
| 0.9 | 1.34e-05 | 4.29e-05 | 1.000 | 2.66e-05 | 7.51e-05 | 1.000 |
| 1.0 | 1.39e-05 | 4.34e-05 | 1.000 | 1.96e-05 | 6.35e-05 | 1.000 |

**Best lambda for SA (lowest best_pc):** 0.3 → 7.37e-06  
**Best lambda for SQA (lowest best_pc):** 1.0 → 1.96e-05

**Note:** Since all `best_cov` values are 1.000 across all lambdas and solvers (coverage is uniform), the lambda parameter has no practical effect on the coverage trade-off. The variation in best_pc across lambdas reflects only stochastic noise and the indirect effect of lambda on the QUBO energy landscape weights, not a genuine Pareto trade-off.

---

### Best SA Solution

Source: `results/shell3r_SA_best.csv`  
Columns: `norad_id`, `raan_deg`, `Pc_n`, `coverage_norm`

| Statistic | Value |
|-----------|-------|
| Satellites selected | 20 |
| Satellites with Pc_n = 0 | 0 (0.0%) |
| Min Pc_n | 9.54e-13 |
| Max Pc_n | 1.73e-06 |
| Mean Pc_n | 3.68e-07 |
| Mean coverage_norm (selected) | 1.0000 |
| Min coverage_norm (selected) | 1.0000 |
| Max coverage_norm (selected) | 1.0000 |
| RAAN min | 11.08° |
| RAAN max | 312.92° |
| RAAN std | 92.52° |

#### Best SQA Solution

Source: `results/shell3r_SQA_best.csv`

| Statistic | Value |
|-----------|-------|
| Satellites selected | 20 |
| Satellites with Pc_n = 0 | 0 (0.0%) |
| Min Pc_n | 2.52e-14 |
| Max Pc_n | 6.21e-06 |
| Mean Pc_n | 1.14e-06 |
| Mean coverage_norm (selected) | 1.0000 |
| Min coverage_norm (selected) | 1.0000 |
| Max coverage_norm (selected) | 1.0000 |
| RAAN min | 204.92° |
| RAAN max | 354.46° |
| RAAN std | 47.13° |

#### Overlap Between SA and SQA Best Solutions

| Metric | Value |
|--------|-------|
| Satellites in common | 1 / 20 |
| Overlap fraction | 5.0% |

**Note:** The SQA best solution is concentrated in a narrower RAAN band (204°–354°, std = 47.13°) compared to the SA solution (11°–313°, std = 92.52°). Despite SQA achieving a lower best Pc (1.96e-05 at lambda=1.0 vs 7.37e-06 for SA at lambda=0.3), the SQA mean Pc is consistently higher than SA across all lambdas, suggesting SA has better average solution quality while SQA exhibits higher variance.

---

### QPU Submission (pending)

| Metric | Value |
|--------|-------|
| Solver | D-Wave Advantage (via Leap) |
| Script | `experiments/constellation_bi/src/submit_qpu_shell3r.py` |
| Status | **PENDING** — requires active Leap account |
| Expected output | `results/shell3r_QPU.csv` |
| Gate | Mean chain break fraction < 0.10 |

Results for this step will be added once QPU access is available. The uniform coverage issue should be addressed before QPU submission to ensure the bi-objective formulation is non-degenerate.

---

## Cross-Experiment Comparison

### 1. Problem Scale

| Dimension | Experiment 1 (Collision) | Experiment 2 (Constellation Bi) |
|-----------|--------------------------|----------------------------------|
| N candidates | 1,656 | 130 |
| k selected | 100 | 20 |
| k / N ratio | 6.0% | 15.4% |
| sigma | 0.1 km | 1.0 km |
| Objective terms | Pc only | Pc + land coverage |
| Solvers tested | Random, SA, Tabu, SQA | Random, SA, SQA |

### 2. SA Performance

| Metric | Experiment 1 | Experiment 2 |
|--------|-------------|-------------|
| SA best Pc | 8.79e-05 | 7.37e-06 |
| Random best Pc | 3.69e-03 | 2.08e-05 |
| SA OOM vs random | +1.62 | +0.45 |
| Gate threshold (OOM > 0.5) | PASSED | FAILED |

### 3. SQA Performance

| Metric | Experiment 1 | Experiment 2 |
|--------|-------------|-------------|
| SQA best Pc | 2.23e-05 | 2.39e-05 (λ=0.3) / 1.96e-05 (λ=1.0) |
| Random best Pc | 3.69e-03 | 2.08e-05 |
| SQA OOM vs random | +2.22 | −0.04 (λ=0.3) |
| Gate threshold (OOM > 0.5) | PASSED | FAILED |

### 4. Landscape Richness

| Metric | Experiment 1 | Experiment 2 |
|--------|-------------|-------------|
| Fraction with Pc = 0 | 85.7% (1,419 / 1,656) | 0.8% (1 / 130) |
| Fraction with Pc > 0 | 14.3% (237 / 1,656) | 99.2% (129 / 130) |
| Min Pc (non-zero) | 1.11e-16 | 1.22e-15 |
| Max Pc | 4.68e-03 | 4.89e-05 |
| Pc dynamic range (log₁₀) | 13.62 | 10.60 |
| Coverage differentiation | NOT APPLICABLE | NONE (all identical) |

### 5. Key Finding

The two experiments together reveal a critical relationship between problem density, landscape richness, and solver advantage. In Experiment 1 (sigma = 0.1 km), the extremely sparse Pc landscape — where 85.7% of candidates carry zero collision probability and non-zero values span 13 orders of magnitude — creates a high-contrast energy landscape that quantum-inspired annealing (SQA) exploits effectively, achieving +2.22 OOM versus random and outperforming SA by a factor of approximately 4. In Experiment 2 (sigma = 1.0 km), the ten-fold increase in propagation uncertainty removes nearly all sparsity: 99.2% of candidates have nonzero Pc, the dynamic range compresses to 10.6 orders of magnitude, and the Pc values cluster much closer together. In this denser, more uniform landscape, both SA and SQA lose most of their advantage — SA barely clears random (+0.45 OOM, below the 0.5 threshold) and SQA performs at or below random. Compounding this, the coverage objective provides zero differentiation (all candidates are identical in the 20°N–50°N band), collapsing the intended bi-objective problem to a pure and unstructured Pc minimization. Together, these results demonstrate that quantum-inspired annealing advantage in satellite slot selection is not intrinsic to the problem class but is instead contingent on the sharpness of the energy landscape — itself a function of the positional uncertainty model (sigma), conjunction geometry, and the discriminability of the secondary objective.
