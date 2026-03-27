"""
classical_annealing.py
----------------------
Classical heuristic solvers for the LEO constellation QUBO.

Provides two CPU-based solvers from the dwave-samplers package:

1. SimulatedAnnealingSampler — stochastic temperature-based search.
2. TabuSampler — tabu-list-based local search.

Both return results in a unified format compatible with the quantum
annealing interface in quantum_annealing.py for easy comparison.

Reference:
    Owens-Fahrner, N., Wysack, J., Kim, J. (2025). Graph-Based Optimization
    for High-Density LEO Constellation Design. AMOS Conference.
"""

import math
from typing import Dict, List, Tuple

import dimod
import numpy as np

try:
    from dwave.samplers import SimulatedAnnealingSampler
except ImportError:
    from neal import SimulatedAnnealingSampler  # fallback for older installs

try:
    from dwave.samplers import TabuSampler
except ImportError:
    from tabu import TabuSampler  # fallback

try:
    from dwave.samplers import PathIntegralAnnealingSampler
except ImportError:
    PathIntegralAnnealingSampler = None

from config.settings import SA_NUM_READS, SA_NUM_SWEEPS


def filter_feasible(
    sampleset: dimod.SampleSet,
    node_idx: Dict[str, int],
    k: int,
) -> List[Tuple[Dict, float]]:
    """
    Filter a sampleset to keep only feasible solutions.

    A feasible solution is one where exactly k binary variables are set
    to 1 (i.e., exactly k satellites are selected).

    Parameters
    ----------
    sampleset : dimod.SampleSet
        Raw output from any dimod-compatible sampler.
    node_idx : dict
        Satellite-ID to index mapping from ``build_qubo``.
    k : int
        Target constellation size.

    Returns
    -------
    list of (sample_dict, energy) tuples
        Sorted by energy ascending (lowest energy / best solution first).
        Empty list if no feasible solutions were found.
    """
    feasible = []
    for sample, energy in sampleset.data(['sample', 'energy']):
        # Cast to Python int to avoid numpy int8 overflow at N > 127
        num_selected = sum(int(v) for v in sample.values())
        if num_selected == k:
            feasible.append((dict(sample), energy))

    # Sort by energy: lower is better.
    feasible.sort(key=lambda t: t[1])
    return feasible


def decode_solution(
    best_sample: Dict[int, int],
    node_idx: Dict[str, int],
) -> List[str]:
    """
    Convert a binary sample back to a list of selected satellite IDs.

    Parameters
    ----------
    best_sample : dict
        Binary assignment {variable_index: 0_or_1} from the sampler.
    node_idx : dict
        Satellite-ID to index mapping from ``build_qubo``.

    Returns
    -------
    list of str
        Satellite IDs of all selected satellites (x_i = 1).
    """
    # Reverse the node_idx mapping: index -> satellite_id.
    idx_to_node = {v: k for k, v in node_idx.items()}
    return [idx_to_node[idx] for idx, val in best_sample.items() if val == 1]


def solve_simulated_annealing(
    bqm: dimod.BinaryQuadraticModel,
    node_idx: Dict[str, int],
    k: int,
    num_reads: int = SA_NUM_READS,
    num_sweeps: int = SA_NUM_SWEEPS,
) -> Dict:
    """
    Solve the QUBO using SimulatedAnnealingSampler (CPU).

    Simulated annealing process for each of the ``num_reads`` independent runs:

    1. Start from a random binary assignment of all N variables.
    2. At each sweep, propose flipping one randomly chosen variable.
    3. If the flip reduces energy: accept unconditionally.
    4. If the flip increases energy by delta_E: accept with probability
       exp(-delta_E / T), where T is the current temperature.
    5. Reduce temperature T according to a geometric cooling schedule.
    6. After ``num_sweeps`` sweeps, record the final state and energy.

    The best feasible solution (exactly k selected, lowest energy) is returned.

    Parameters
    ----------
    bqm : dimod.BinaryQuadraticModel
        The problem BQM from ``qubo_to_bqm``.
    node_idx : dict
        Satellite-ID to index mapping from ``build_qubo``.
    k : int
        Target constellation size.
    num_reads : int, optional
        Number of independent SA runs. Default SA_NUM_READS.
    num_sweeps : int, optional
        Number of sweeps per run. Default SA_NUM_SWEEPS.

    Returns
    -------
    dict
        Keys:
            ``selected_satellites`` (list of str): selected satellite IDs.
            ``best_energy`` (float): QUBO energy of the best feasible solution.
            ``num_feasible`` (int): feasible solutions found out of num_reads.
            ``feasibility_rate`` (float): num_feasible / num_reads.
            ``sampleset`` (dimod.SampleSet): full raw results.
            ``solver`` (str): ``'simulated_annealing'``.

    Raises
    ------
    RuntimeError
        If no feasible solution (sum x = k) was found in any of the runs.
    """
    sampler = SimulatedAnnealingSampler()
    sampleset = sampler.sample(bqm, num_reads=num_reads, num_sweeps=num_sweeps)

    feasible = filter_feasible(sampleset, node_idx, k)
    num_feasible = len(feasible)
    feasibility_rate = num_feasible / num_reads

    if not feasible:
        raise RuntimeError(
            f"Simulated annealing found no feasible solution (k={k}) in "
            f"{num_reads} reads. Try increasing num_reads, num_sweeps, or "
            f"reducing the penalty multiplier."
        )

    best_sample, best_energy = feasible[0]
    selected_satellites = decode_solution(best_sample, node_idx)

    return {
        'selected_satellites': selected_satellites,
        'best_energy': best_energy,
        'num_feasible': num_feasible,
        'feasibility_rate': feasibility_rate,
        'sampleset': sampleset,
        'solver': 'simulated_annealing',
    }


def solve_tabu(
    bqm: dimod.BinaryQuadraticModel,
    node_idx: Dict[str, int],
    k: int,
    num_reads: int = SA_NUM_READS,
) -> Dict:
    """
    Solve the QUBO using TabuSampler (CPU).

    Tabu search is a deterministic local-search heuristic that avoids
    cycling by maintaining a **tabu list** of recently visited solutions
    (or recently flipped variables). At each step it moves to the best
    neighbour not on the tabu list, even if the move increases energy —
    enabling escape from local optima without the stochastic temperature
    mechanism of simulated annealing.

    Tabu search typically converges faster per read than SA but may be
    less thorough in exploring the energy landscape for highly multimodal
    problems.

    Parameters
    ----------
    bqm : dimod.BinaryQuadraticModel
        The problem BQM from ``qubo_to_bqm``.
    node_idx : dict
        Satellite-ID to index mapping from ``build_qubo``.
    k : int
        Target constellation size.
    num_reads : int, optional
        Number of independent tabu runs. Default SA_NUM_READS.

    Returns
    -------
    dict
        Same structure as ``solve_simulated_annealing``, with
        ``solver`` = ``'tabu_search'``.

    Raises
    ------
    RuntimeError
        If no feasible solution was found.
    """
    sampler = TabuSampler()
    sampleset = sampler.sample(bqm, num_reads=num_reads)

    feasible = filter_feasible(sampleset, node_idx, k)
    num_feasible = len(feasible)
    feasibility_rate = num_feasible / num_reads

    if not feasible:
        raise RuntimeError(
            f"Tabu search found no feasible solution (k={k}) in "
            f"{num_reads} reads. Try increasing num_reads."
        )

    best_sample, best_energy = feasible[0]
    selected_satellites = decode_solution(best_sample, node_idx)

    return {
        'selected_satellites': selected_satellites,
        'best_energy': best_energy,
        'num_feasible': num_feasible,
        'feasibility_rate': feasibility_rate,
        'sampleset': sampleset,
        'solver': 'tabu_search',
    }


def solve_sqa(
    bqm: dimod.BinaryQuadraticModel,
    node_idx: Dict[str, int],
    k: int,
    num_reads: int = 100,
    num_sweeps: int = SA_NUM_SWEEPS,
) -> Dict:
    """
    Solve the QUBO using PathIntegralAnnealingSampler (simulated QA).

    Simulates quantum annealing via path-integral Monte Carlo, modelling the
    transverse-field Ising Hamiltonian. The Gamma (transverse field) is
    annealed from a large value (strong quantum fluctuations) to zero
    (classical ground state), analogous to hardware quantum annealing.

    Returns the same dict format as solve_simulated_annealing, with
    ``solver`` = ``'sqa_path_integral'``.

    Raises
    ------
    RuntimeError
        If PathIntegralAnnealingSampler is not available or no feasible
        solution was found.
    """
    if PathIntegralAnnealingSampler is None:
        raise RuntimeError(
            "PathIntegralAnnealingSampler not available. "
            "Install dwave-samplers >= 1.0: pip install dwave-samplers"
        )

    sampler = PathIntegralAnnealingSampler()
    sampleset = sampler.sample(bqm, num_reads=num_reads, num_sweeps=num_sweeps)

    feasible = filter_feasible(sampleset, node_idx, k)
    num_feasible = len(feasible)
    feasibility_rate = num_feasible / num_reads

    if not feasible:
        raise RuntimeError(
            f"SQA (path integral) found no feasible solution (k={k}) in "
            f"{num_reads} reads. Try increasing num_reads or num_sweeps."
        )

    best_sample, best_energy = feasible[0]
    selected_satellites = decode_solution(best_sample, node_idx)

    return {
        'selected_satellites': selected_satellites,
        'best_energy': best_energy,
        'num_feasible': num_feasible,
        'feasibility_rate': feasibility_rate,
        'sampleset': sampleset,
        'solver': 'sqa_path_integral',
    }


def print_results(results: Dict, satellites_df) -> None:
    """
    Print a formatted results table for a solved constellation.

    Displays per-satellite Pc and coverage values, aggregate constellation
    statistics, and solver performance metrics.

    Parameters
    ----------
    results : dict
        Output from ``solve_simulated_annealing`` or ``solve_tabu``.
    satellites_df : pd.DataFrame
        Original satellite data with columns satellite_id, pc, coverage.
    """
    solver_name = results['solver'].replace('_', ' ').title()
    selected = results['selected_satellites']

    # Build a lookup dict for fast access.
    sat_lookup = satellites_df.set_index('satellite_id').to_dict('index')

    print("=" * 60)
    print(f"RESULTS — {solver_name}")
    print("=" * 60)

    pc_vals = []
    cov_vals = []
    for sat_id in sorted(selected):
        pc = sat_lookup[sat_id]['pc']
        cov = sat_lookup[sat_id]['coverage']
        pc_vals.append(pc)
        cov_vals.append(cov)

    # Print per-satellite table only for small constellations (k ≤ 20).
    if len(selected) <= 20:
        print(f"  {'Satellite':<14} {'Pc':>8} {'Coverage':>10}")
        print(f"  {'-'*14} {'-'*8} {'-'*10}")
        for sat_id in sorted(selected):
            pc = sat_lookup[sat_id]['pc']
            cov = sat_lookup[sat_id]['coverage']
            print(f"  {sat_id:<14} {pc:>8.4f} {cov:>10.4f}")
    else:
        nz = sum(1 for p in pc_vals if p > 0)
        top5 = sorted(pc_vals, reverse=True)[:5]
        print(f"  Selected {len(selected)} satellites  "
              f"(Pc>0: {nz}, Pc=0: {len(selected)-nz})")
        print(f"  Top-5 individual Pc : {[f'{p:.3e}' for p in top5]}")

    print()
    print(f"  Best energy        : {results['best_energy']:.6f}")
    print(f"  Feasible solutions : {results['num_feasible']} / "
          f"{results.get('sampleset') and len(results['sampleset']) or 'N/A'}")
    print(f"  Feasibility rate   : {results['feasibility_rate']:.2%}")
    print()

    # Aggregate Pc using independence formula: 1 - prod(1 - Pc_n).
    aggregate_pc = 1.0 - math.prod(1.0 - p for p in pc_vals)
    avg_coverage = np.mean(cov_vals)
    print(f"  Aggregate constellation Pc  : {aggregate_pc:.6f}")
    print(f"  Average constellation cov.  : {avg_coverage:.4f}")
    print("=" * 60)
