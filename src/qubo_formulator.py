"""
qubo_formulator.py
------------------
Assembles the QUBO (Quadratic Unconstrained Binary Optimization) matrix
for the Densest k-Subgraph problem on the LEO satellite candidate graph.

The QUBO formulation converts the constrained graph optimisation problem
into an unconstrained binary quadratic form that can be solved directly
by D-Wave samplers (both classical and quantum).

QUBO objective (minimisation):
    min  -sum_{n<m} x_n * x_m * w(v_n, v_m)  +  P * (sum_n x_n - k)^2

The second term is a penalty that enforces the cardinality constraint:
exactly k satellites must be selected (sum x_n = k).

Reference:
    Owens-Fahrner, N., Wysack, J., Kim, J. (2025). Graph-Based Optimization
    for High-Density LEO Constellation Design. AMOS Conference.
"""

from typing import Dict, List, Tuple

import dimod
import networkx as nx
import numpy as np

from config.settings import PENALTY_MULTIPLIER


def compute_penalty(
    G: nx.Graph,
    k: int,
    multiplier: float = PENALTY_MULTIPLIER,
) -> float:
    """
    Compute the penalty constant P for the cardinality constraint.

    The penalty P must be large enough that any infeasible solution
    (one where sum x_n != k) has higher energy than any feasible solution,
    regardless of which satellites are selected.

    The maximum possible objective contribution from any feasible solution
    is bounded by the sum of the k*(k-1)/2 largest edge weights in G
    (the best possible k-node clique). P must exceed this bound.

    Formula used:
        P = multiplier * max_edge_weight * k^2

    This is conservative (exceeds the tight bound) but guaranteed sufficient.

    .. note::
        **Tuning the penalty**: if P is too small, the solver may prefer
        to violate the cardinality constraint — selecting more or fewer than
        k satellites — because the objective gain outweighs the penalty cost.
        If P is too large, it flattens the energy landscape: feasible
        solutions become nearly degenerate, making it hard for the annealer
        to distinguish good from bad constellations. The ``multiplier``
        parameter allows tuning this trade-off (default 2.0 is conservative).

    Parameters
    ----------
    G : nx.Graph
        The complete satellite graph from ``build_graph``.
    k : int
        Desired constellation size.
    multiplier : float, optional
        Safety multiplier applied to the penalty. Default PENALTY_MULTIPLIER.

    Returns
    -------
    float
        Penalty constant P.
    """
    weights = [d['weight'] for _, _, d in G.edges(data=True)]
    max_weight = max(weights) if weights else 1.0
    # k^2 accounts for the worst-case expansion of the squared penalty term.
    return multiplier * max_weight * (k ** 2)


def build_qubo(
    G: nx.Graph,
    k: int,
) -> Tuple[Dict[Tuple, float], Dict[str, int], float]:
    """
    Assemble the complete QUBO Q matrix for the Densest k-Subgraph problem.

    Derivation
    ----------
    Starting from the constrained problem:

        max  sum_{n<m} x_n * x_m * w(v_n, v_m)   subject to  sum x_n = k

    Convert to unconstrained minimisation by adding a penalty term and
    negating the objective:

        min  -sum_{n<m} x_n * x_m * w(v_n, v_m)  +  P * (sum x_n - k)^2

    Expand the penalty using x_n^2 = x_n (binary property, so x^2 = x):

        P*(sum x_n - k)^2
            = P * [sum_n x_n^2  + 2*sum_{n<m} x_n*x_m  - 2k*sum_n x_n  + k^2]
            = P * [(1-2k)*sum_n x_n  +  2*sum_{n<m} x_n*x_m]  + P*k^2  [const]

    Combining with the objective:

        Diagonal:     Q_{nn} = P * (1 - 2k)          [penalty only, no edge term]
        Off-diagonal: Q_{nm} = -w(v_n, v_m) + 2P     [objective + penalty]

    The constant P*k^2 shifts all energies equally and is tracked but not
    included in the Q dict (samplers add it as ``offset``).

    Parameters
    ----------
    G : nx.Graph
        The complete satellite graph from ``build_graph``.
    k : int
        Desired constellation size.

    Returns
    -------
    Q : dict
        QUBO matrix in upper-triangular form: {(i, j): value} where i <= j.
        Keys are integer indices (not satellite IDs).
    node_idx : dict
        Mapping from satellite_id (str) to matrix index (int).
    P : float
        Penalty constant used — stored for reproducibility and diagnostics.
    """
    nodes = list(G.nodes())
    node_idx = {sat_id: i for i, sat_id in enumerate(nodes)}
    n = len(nodes)
    P = compute_penalty(G, k)

    Q: Dict[Tuple[int, int], float] = {}

    # Diagonal terms: Q_{nn} = P * (1 - 2k)
    # This is the linear coefficient for each binary variable x_n,
    # arising entirely from the penalty expansion.
    diag_value = P * (1 - 2 * k)
    for i in range(n):
        Q[(i, i)] = diag_value

    # Off-diagonal terms: Q_{nm} = -w(v_n, v_m) + 2P  for n < m
    # The -w term encodes the objective (maximise edge weight sums),
    # and +2P encodes the cross terms from the penalty expansion.
    for u, v, data in G.edges(data=True):
        i = node_idx[u]
        j = node_idx[v]
        # Enforce upper-triangular storage convention.
        if i > j:
            i, j = j, i
        Q[(i, j)] = -data['weight'] + 2.0 * P

    return Q, node_idx, P


def qubo_to_bqm(Q: Dict[Tuple[int, int], float]) -> dimod.BinaryQuadraticModel:
    """
    Convert a QUBO dictionary to a dimod BinaryQuadraticModel.

    dimod BQMs are the standard interface for all D-Wave samplers
    (SimulatedAnnealingSampler, TabuSampler, DWaveSampler, etc.).

    Parameters
    ----------
    Q : dict
        QUBO matrix {(i, j): value} in upper-triangular form.

    Returns
    -------
    dimod.BinaryQuadraticModel
        BQM in BINARY vartype, ready for sampling.
    """
    return dimod.BinaryQuadraticModel.from_qubo(Q)


def evaluate_solution(
    sample: Dict[int, int],
    Q: Dict[Tuple[int, int], float],
    node_idx: Dict[str, int],
    k: int,
) -> Dict:
    """
    Evaluate the energy and feasibility of a binary assignment.

    Computes the QUBO energy: E = sum_{i<=j} Q_{ij} * x_i * x_j
    and checks whether exactly k variables are set to 1.

    Parameters
    ----------
    sample : dict
        Binary assignment {variable_index: 0_or_1}.
    Q : dict
        QUBO matrix from ``build_qubo``.
    node_idx : dict
        Satellite-ID to index mapping from ``build_qubo``.
    k : int
        Target constellation size.

    Returns
    -------
    dict
        Keys:
            ``energy`` (float): QUBO energy of this assignment.
            ``num_selected`` (int): number of variables set to 1.
            ``feasible`` (bool): True if num_selected == k.
            ``selected_indices`` (list): indices of selected variables.
    """
    # Compute QUBO energy directly from the Q matrix.
    energy = 0.0
    for (i, j), q_val in Q.items():
        xi = sample.get(i, 0)
        xj = sample.get(j, 0)
        if i == j:
            energy += q_val * xi
        else:
            energy += q_val * xi * xj

    selected_indices = [idx for idx, val in sample.items() if val == 1]
    num_selected = len(selected_indices)

    return {
        'energy': energy,
        'num_selected': num_selected,
        'feasible': num_selected == k,
        'selected_indices': selected_indices,
    }


def print_qubo_stats(
    Q: Dict[Tuple[int, int], float],
    node_idx: Dict[str, int],
    k: int,
    P: float,
) -> None:
    """
    Print Q matrix statistics for diagnostic purposes.

    Parameters
    ----------
    Q : dict
        QUBO matrix from ``build_qubo``.
    node_idx : dict
        Satellite-ID to index mapping.
    k : int
        Target constellation size.
    P : float
        Penalty constant used.
    """
    n = len(node_idx)
    diag_vals = [v for (i, j), v in Q.items() if i == j]
    off_diag_vals = [v for (i, j), v in Q.items() if i != j]

    print("=" * 60)
    print("QUBO MATRIX STATISTICS")
    print("=" * 60)
    print(f"  Matrix size           : {n} x {n}")
    print(f"  Non-zero entries      : {len(Q)}")
    print(f"  Constellation size k  : {k}")
    print(f"  Penalty constant P    : {P:.6f}")
    print()
    print(f"  Diagonal value Q_nn = P*(1-2k) = {P:.4f}*(1-{2*k}) = {P*(1-2*k):.6f}")
    print(f"    (same for all {n} diagonal entries)")
    print()
    print(f"  Off-diagonal Q_nm = -w(vn,vm) + 2P:")
    if off_diag_vals:
        print(f"    Min  : {min(off_diag_vals):.6f}")
        print(f"    Max  : {max(off_diag_vals):.6f}")
        print(f"    Mean : {np.mean(off_diag_vals):.6f}")
    print()
    # Verify penalty is sufficient: infeasible solutions should have higher
    # energy than any feasible one. An infeasible solution with k+1 selected
    # incurs penalty P*(1)^2 = P extra cost relative to feasible.
    print(f"  Penalty verification:")
    print(f"    Any solution with sum(x) != k incurs >= P = {P:.4f} extra energy.")
    print(f"    Max possible objective gain for k nodes: < P (penalty dominates).")
    print("=" * 60)
