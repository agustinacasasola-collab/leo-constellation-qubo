# LEO Constellation QUBO Optimization

Quantum-assisted graph optimization for Low Earth Orbit (LEO) satellite constellation design.

Implements the methodology of **Owens-Fahrner et al. (2025)** to solve the constellation selection problem as a Densest k-Subgraph (DkS) QUBO, solved by both classical simulated annealing and D-Wave quantum annealing.

---

## Problem Context

The rapid deployment of mega-constellations (Starlink, OneWeb, Kuiper) has dramatically increased LEO congestion. As of July 2025, LEO contains over **22,000 tracked objects** — including nearly **9,000 debris fragments** — making constellation design a safety-critical optimization problem.

Given N candidate satellite orbits, the task is to select a constellation of exactly **k satellites** that jointly:
- **Minimizes collision risk** — avoids orbital shells with high debris density
- **Maximizes observational coverage** — ensures the constellation can track LEO objects effectively

This is the **Densest k-Subgraph problem** (NP-hard), which becomes intractable for large N by exhaustive search. QUBO formulation enables heuristic solvers (classical and quantum) to find high-quality solutions efficiently.

---

## Graph-Theoretic Formulation

Each candidate satellite is a **node** in a complete weighted graph K_N.

The **edge weight** between satellites v_n and v_m encodes joint safety and coverage:

```
w(vn, vm) = x·(1 - Pcn)·(1 - Pcm) + y·(an + am)/2
```

where:
- `Pcn` is the aggregate collision probability of satellite n, computed using **Chan's 2D analytic formula** summed over all geometrically-filtered LEO catalog objects
- `an` is the coverage fraction of satellite n (fraction of time steps where the satellite has valid access to catalog objects)
- `x`, `y` are scaling factors (default 1.0 each)

**Chan's 2D formula** (valid for high relative velocity, typical in LEO):

```
u = ρ² / (σx·σz)
v = x²/σx² + z²/σz²
Pc = exp(-v/2) · (1 - exp(-u/2))
```

Aggregate Pc over multiple catalog objects (assuming independence):

```
Pcn = 1 - ∏(1 - Pc_l)   for all l in Ln
```

The **apogee/perigee filter** removes catalog objects whose orbits cannot geometrically intersect the candidate's orbit, reducing computational cost.

---

## QUBO Formulation

The Densest k-Subgraph problem is NP-hard. The QUBO (Quadratic Unconstrained Binary Optimization) formulation converts it to an energy minimization problem:

```
min  -∑_{n<m} xn·xm·w(vn,vm)  +  P·(∑n xn - k)²
```

where binary variable `xn = 1` if satellite n is selected.

Expanding with the binary identity x² = x:

| Entry | Formula | Meaning |
|---|---|---|
| Diagonal `Qnn` | `P·(1 - 2k)` | Linear penalty for each selected satellite |
| Off-diagonal `Qnm` | `-w(vn,vm) + 2P` | Objective (negative = maximize) + cross-penalty |

**Penalty constant**: `P = multiplier · max_edge_weight · k²`

If P is too small, the solver may violate the cardinality constraint. If P is too large, it flattens the energy landscape. Default multiplier = 2.0 is conservative.

---

## Solvers

### 1. Simulated Annealing (CPU)
Uses `dwave-samplers` `SimulatedAnnealingSampler`. No credentials required.

At each step, proposes flipping one variable and accepts the move with probability `exp(-ΔE/T)`, with temperature T decreasing geometrically. Runs `num_reads` independent chains.

### 2. Tabu Search (CPU)
Uses `dwave-samplers` `TabuSampler`. Maintains a tabu list of recently visited solutions to escape local optima deterministically.

### 3. Quantum Annealing (D-Wave QPU)
Uses `EmbeddingComposite(DWaveSampler())` via D-Wave Leap cloud. Requires account setup (see below).

The QPU performs the annealing schedule:
- **s=0**: Quantum tunneling Hamiltonian H_A dominates — all qubits in superposition
- **0<s<1**: H_A decreases, problem Hamiltonian H_B increases
- **s=1**: H_B dominates — each qubit collapses to 0 or 1, encoding a low-energy solution

`EmbeddingComposite` automatically performs **minor-embedding**: maps the logical QUBO graph (K_N) to chains of physical qubits on the hardware Pegasus/Zephyr topology.

---

## Installation

```bash
git clone https://github.com/agustinacasasola-collab/leo-constellation-qubo.git
cd leo-constellation-qubo
pip install -r requirements.txt
```

---

## How to Run

**Basic run (simulated annealing only):**
```bash
python main.py
```

**Custom constellation size and reads:**
```bash
python main.py --k 6 --num-reads 2000
```

**With quantum annealing (requires D-Wave Leap):**
```bash
python main.py --quantum
```

**Available arguments:**
| Argument | Default | Description |
|---|---|---|
| `--k` | 5 | Constellation size (satellites to select) |
| `--num-reads` | 1000 | Number of annealing reads |
| `--quantum` | False | Attempt QPU solving via D-Wave Leap |

---

## D-Wave Leap Setup

1. Create a free account at https://cloud.dwavesys.com/leap/
2. Retrieve your API token from the Leap dashboard
3. Configure credentials:
   ```bash
   dwave config create
   ```
   Enter your token when prompted. This creates `~/.config/dwave/dwave.conf`.
4. Verify:
   ```bash
   dwave ping
   ```
5. Run with `--quantum` flag.

---

## Project Structure

```
leo-constellation-qubo/
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── .gitignore
├── config/
│   └── settings.py             # All tunable parameters
├── data/
│   └── sample_satellites.csv   # 20 synthetic candidate satellites (5 shells)
├── src/
│   ├── __init__.py
│   ├── collision_risk.py       # Chan's Pc formula, aggregate Pc, apogee filter
│   ├── coverage.py             # Solar exclusion, Earth limb, coverage fraction
│   ├── graph_builder.py        # Complete graph K_N construction
│   ├── qubo_formulator.py      # Q matrix assembly, BQM conversion, evaluation
│   ├── classical_annealing.py  # SA and Tabu solvers
│   └── quantum_annealing.py    # D-Wave QPU interface (graceful fallback)
├── notebooks/
│   └── exploration.ipynb       # Interactive walkthrough of the full pipeline
├── results/
│   └── .gitkeep                # Output directory (CSV + PNG saved here)
└── main.py                     # CLI entry point
```

---

## Reference

Owens-Fahrner, N., Wysack, J., Kim, J. (2025). **Graph-Based Optimization for High-Density LEO Constellation Design**. *AMOS Conference*.

---

## License

MIT
