"""
quantum_annealing.py
--------------------
D-Wave QPU interface for quantum annealing of the LEO constellation QUBO.

All functions fail gracefully when D-Wave Leap credentials are not configured,
printing clear setup instructions instead of crashing.

Quantum annealing process (EmbeddingComposite + DWaveSampler):

1. **Initialisation** (s=0): A(s) >> B(s). All qubits are initialised in
   superposition — each qubit simultaneously represents 0 and 1. The system
   is governed entirely by the quantum tunnelling Hamiltonian H_A.

2. **Annealing** (0 < s < 1): A(s) decreases, B(s) increases. The problem
   Hamiltonian H_B (encoding the QUBO) gradually takes over. Quantum
   tunnelling allows the system to traverse energy barriers that would trap
   a classical annealer.

3. **Readout** (s=1): A=0, B dominates. Each qubit collapses to a definite
   0 or 1. The final spin configuration encodes a low-energy solution to
   the QUBO. This is repeated num_reads times.

EmbeddingComposite performs **minor-embedding**: maps the logical QUBO graph
(complete graph K_N) onto the sparse physical qubit connectivity graph of the
D-Wave processor (Pegasus or Zephyr topology). Chains of physical qubits
represent single logical variables.

Reference:
    Owens-Fahrner, N., Wysack, J., Kim, J. (2025). Graph-Based Optimization
    for High-Density LEO Constellation Design. AMOS Conference.
"""

import math
from typing import Dict, List, Optional

import dimod
import numpy as np

from config.settings import QA_NUM_READS, QA_ANNEALING_TIME
from src.classical_annealing import filter_feasible, decode_solution


def check_leap_access() -> bool:
    """
    Check whether D-Wave Leap credentials are configured.

    Attempts to import and instantiate a DWaveSampler, which requires
    a valid ``dwave.conf`` configuration file created by ``dwave config create``.

    Returns
    -------
    bool
        True if Leap is configured and accessible, False otherwise.
    """
    try:
        from dwave.system import DWaveSampler
        sampler = DWaveSampler()
        return True
    except Exception as e:
        print("\nD-Wave Leap is not configured or not accessible.")
        print(f"  Reason: {e}")
        print()
        print("  To enable quantum annealing, follow these steps:")
        print("  1. Create a free account at https://cloud.dwavesys.com/leap/")
        print("  2. Install D-Wave Ocean SDK: pip install dwave-ocean-sdk")
        print("  3. Configure credentials: dwave config create")
        print("     (You will need your API token from the Leap dashboard.)")
        print()
        return False


def get_solver_info() -> Optional[Dict]:
    """
    Return information about the available D-Wave QPU solver.

    The ``graph_id`` identifies the specific working graph (qubit connectivity
    map) of the QPU at sampling time. Since QPUs are periodically recalibrated
    and their working graphs may change, recording graph_id is essential for
    reproducibility of quantum results.

    Returns
    -------
    dict or None
        Keys: ``solver_name``, ``topology_type``, ``num_qubits``, ``graph_id``.
        Returns None if Leap is not configured or accessible.
    """
    try:
        from dwave.system import DWaveSampler
        sampler = DWaveSampler()
        props = sampler.properties
        return {
            'solver_name': sampler.solver.name,
            'topology_type': props.get('topology', {}).get('type', 'unknown'),
            'num_qubits': props.get('num_qubits', 'unknown'),
            'graph_id': props.get('topology', {}).get('graph_id', 'unknown'),
        }
    except Exception:
        return None


def solve_quantum(
    bqm: dimod.BinaryQuadraticModel,
    node_idx: Dict[str, int],
    k: int,
    num_reads: int = QA_NUM_READS,
    annealing_time: int = QA_ANNEALING_TIME,
) -> Dict:
    """
    Solve the QUBO using the D-Wave QPU via EmbeddingComposite.

    Quantum annealing leverages quantum superposition and tunnelling to
    explore the energy landscape simultaneously across many candidate
    solutions. Unlike simulated annealing — which traverses barriers with
    probability exp(-dE/T) — quantum annealing can tunnel through barriers,
    potentially finding lower-energy solutions faster for certain problem
    structures.

    The EmbeddingComposite automatically computes a minor-embedding that maps
    the logical QUBO variables (complete graph K_N edges) to chains of
    physical qubits on the hardware graph. For N=20 satellites, K_20 has
    190 edges; the embedding typically uses O(N) physical qubits per logical
    variable.

    Parameters
    ----------
    bqm : dimod.BinaryQuadraticModel
        The problem BQM from ``qubo_to_bqm``.
    node_idx : dict
        Satellite-ID to index mapping from ``build_qubo``.
    k : int
        Target constellation size.
    num_reads : int, optional
        Number of QPU annealing runs. Default QA_NUM_READS.
    annealing_time : int, optional
        Annealing duration in microseconds. Default QA_ANNEALING_TIME (20 µs).
        Longer times improve solution quality but consume more QPU access time.

    Returns
    -------
    dict
        Keys:
            ``selected_satellites`` (list of str): selected satellite IDs.
            ``best_energy`` (float): best feasible QUBO energy.
            ``num_feasible`` (int): feasible solutions found.
            ``feasibility_rate`` (float): num_feasible / num_reads.
            ``sampleset`` (dimod.SampleSet): full QPU results.
            ``solver`` (str): ``'quantum_annealing'``.
            ``timing`` (dict): QPU timing breakdown (qpu_sampling_time, etc.).
            ``solver_info`` (dict): QPU identity for reproducibility.

    Raises
    ------
    RuntimeError
        If Leap is not configured or no feasible solution was found.
    """
    try:
        from dwave.system import DWaveSampler, EmbeddingComposite
    except ImportError:
        raise RuntimeError(
            "dwave-system is not installed. Run: pip install dwave-ocean-sdk"
        )

    if not check_leap_access():
        raise RuntimeError(
            "D-Wave Leap is not configured. Run 'dwave config create' first."
        )

    solver_info = get_solver_info()

    # EmbeddingComposite transparently handles minor-embedding:
    # maps logical QUBO variables to physical qubit chains on the QPU.
    sampler = EmbeddingComposite(DWaveSampler())
    sampleset = sampler.sample(
        bqm,
        num_reads=num_reads,
        annealing_time=annealing_time,
    )

    feasible = filter_feasible(sampleset, node_idx, k)
    num_feasible = len(feasible)
    feasibility_rate = num_feasible / num_reads

    if not feasible:
        raise RuntimeError(
            f"Quantum annealing found no feasible solution (k={k}) in "
            f"{num_reads} reads. Try increasing num_reads or annealing_time."
        )

    best_sample, best_energy = feasible[0]
    selected_satellites = decode_solution(best_sample, node_idx)

    # Extract QPU timing information for benchmarking.
    timing = {}
    if hasattr(sampleset, 'info') and 'timing' in sampleset.info:
        timing = sampleset.info['timing']

    return {
        'selected_satellites': selected_satellites,
        'best_energy': best_energy,
        'num_feasible': num_feasible,
        'feasibility_rate': feasibility_rate,
        'sampleset': sampleset,
        'solver': 'quantum_annealing',
        'timing': timing,
        'solver_info': solver_info,
    }


def compare_solvers(
    results_sa: Dict,
    results_qa: Dict,
    satellites_df,
) -> None:
    """
    Print a side-by-side comparison of simulated annealing vs quantum annealing.

    Parameters
    ----------
    results_sa : dict
        Output from a classical solver (SA or Tabu).
    results_qa : dict
        Output from ``solve_quantum``.
    satellites_df : pd.DataFrame
        Original satellite data with columns satellite_id, pc, coverage.
    """
    sat_lookup = satellites_df.set_index('satellite_id').to_dict('index')

    sa_set = set(results_sa['selected_satellites'])
    qa_set = set(results_qa['selected_satellites'])
    overlap = sa_set & qa_set

    def agg_stats(selected):
        pcs = [sat_lookup[s]['pc'] for s in selected]
        covs = [sat_lookup[s]['coverage'] for s in selected]
        agg_pc = 1.0 - math.prod(1.0 - p for p in pcs)
        avg_cov = np.mean(covs)
        return agg_pc, avg_cov

    sa_pc, sa_cov = agg_stats(results_sa['selected_satellites'])
    qa_pc, qa_cov = agg_stats(results_qa['selected_satellites'])

    print()
    print("=" * 60)
    print("SOLVER COMPARISON: Simulated Annealing vs Quantum Annealing")
    print("=" * 60)
    print(f"  {'Metric':<30} {'SA':>12} {'QA':>12}")
    print(f"  {'-'*30} {'-'*12} {'-'*12}")
    print(f"  {'Best energy':<30} {results_sa['best_energy']:>12.6f} "
          f"{results_qa['best_energy']:>12.6f}")
    print(f"  {'Feasibility rate':<30} {results_sa['feasibility_rate']:>12.2%} "
          f"{results_qa['feasibility_rate']:>12.2%}")
    print(f"  {'Aggregate Pc':<30} {sa_pc:>12.6f} {qa_pc:>12.6f}")
    print(f"  {'Average coverage':<30} {sa_cov:>12.4f} {qa_cov:>12.4f}")
    print()
    print(f"  SA selected  : {sorted(results_sa['selected_satellites'])}")
    print(f"  QA selected  : {sorted(results_qa['selected_satellites'])}")
    print(f"  Overlap ({len(overlap)}) : {sorted(overlap)}")
    print("=" * 60)
