"""
graph_builder.py
----------------
Constructs the complete weighted graph over candidate satellites.

Each node represents a candidate satellite. Each edge weight encodes the
joint desirability of including both endpoints in the final constellation,
balancing collision safety against coverage contribution.

The Densest k-Subgraph problem then selects k nodes whose induced subgraph
has maximum total internal edge weight — equivalently, the k-satellite
constellation that best trades off collision risk and observing coverage.

Reference:
    Owens-Fahrner, N., Wysack, J., Kim, J. (2025). Graph-Based Optimization
    for High-Density LEO Constellation Design. AMOS Conference.
"""

import math
from itertools import combinations
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd

from config.settings import X_SCALE, Y_SCALE


def compute_edge_weight(
    pc_n: float,
    pc_m: float,
    a_n: float,
    a_m: float,
    x: float = X_SCALE,
    y: float = Y_SCALE,
) -> float:
    """
    Compute the edge weight between two candidate satellite nodes.

    The edge weight captures two complementary objectives:

    **Safety term** ``(1 - Pc_n) * (1 - Pc_m)``:
        Joint probability that *neither* satellite collides with any LEO
        object, assuming statistical independence of their conjunction events.
        Equals 1.0 when both satellites are perfectly safe; approaches 0 when
        either has high collision risk. A safe satellite paired with a
        dangerous one is penalised.

    **Coverage term** ``(a_n + a_m) / 2``:
        Average individual coverage fraction. The average (not product) is
        used because coverage is additive — two satellites observing different
        sky regions contribute more than one satellite alone. The product would
        instead model joint probability of simultaneous coverage, which is not
        the physical quantity of interest here.

    .. note::
        A high edge weight requires **both** good safety **and** good coverage.
        A safe satellite that observes nothing (a ≈ 0) produces low-weight
        edges. An excellent-coverage satellite with high Pc also produces
        low-weight edges. The optimiser therefore naturally selects satellites
        that are simultaneously safe and useful.

    Formula
    -------
    .. math::

        w(v_n, v_m) = x \\cdot (1 - P_{c,n})(1 - P_{c,m})
                    + y \\cdot \\frac{a_n + a_m}{2}

    Parameters
    ----------
    pc_n : float
        Aggregate collision probability of satellite n. In [0, 1].
    pc_m : float
        Aggregate collision probability of satellite m. In [0, 1].
    a_n : float
        Coverage fraction of satellite n. In [0, 1].
    a_m : float
        Coverage fraction of satellite m. In [0, 1].
    x : float, optional
        Scaling factor for the collision-risk safety term. Default X_SCALE.
    y : float, optional
        Scaling factor for the coverage term. Default Y_SCALE.

    Returns
    -------
    float
        Edge weight. Range is [0, x + y] for valid inputs.
    """
    # Safety term: joint survival probability (independent events).
    safety = (1.0 - pc_n) * (1.0 - pc_m)

    # Coverage term: average individual coverage fraction.
    coverage_avg = (a_n + a_m) / 2.0

    return x * safety + y * coverage_avg


def build_graph(satellites_df: pd.DataFrame) -> nx.Graph:
    """
    Build a complete weighted graph (K_N) from candidate satellite data.

    This is a complete graph because every satellite in the constellation
    interacts with every other satellite in terms of joint safety and
    coverage. For N candidates there are N*(N-1)/2 edges. The Densest
    k-Subgraph problem then selects k nodes whose induced subgraph —
    the subgraph containing only those k nodes and the edges between them —
    has maximum total internal edge weight.

    Nodes carry attributes:
        - ``pc``: aggregate collision probability
        - ``coverage``: coverage fraction

    Edges carry attribute:
        - ``weight``: value from ``compute_edge_weight``

    Parameters
    ----------
    satellites_df : pd.DataFrame
        DataFrame with columns: ``satellite_id``, ``pc``, ``coverage``.
        Additional columns (altitude_km, inclination_deg, shell) are
        preserved as node attributes if present.

    Returns
    -------
    nx.Graph
        Complete weighted graph with N nodes and N*(N-1)/2 edges.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.read_csv('data/sample_satellites.csv')
    >>> G = build_graph(df)
    >>> len(G.nodes)
    20
    >>> len(G.edges)
    190
    """
    G = nx.Graph()

    # Add one node per candidate satellite, storing all data as attributes.
    for _, row in satellites_df.iterrows():
        node_attrs = row.to_dict()
        sat_id = node_attrs.pop('satellite_id')
        G.add_node(sat_id, **node_attrs)

    # Add one edge per pair, computing the joint edge weight.
    # This creates the complete graph K_N.
    node_ids = list(G.nodes())
    for n, m in combinations(node_ids, 2):
        pc_n = G.nodes[n]['pc']
        pc_m = G.nodes[m]['pc']
        a_n = G.nodes[n]['coverage']
        a_m = G.nodes[m]['coverage']
        w = compute_edge_weight(pc_n, pc_m, a_n, a_m)
        G.add_edge(n, m, weight=w)

    return G


def graph_summary(G: nx.Graph, k: int) -> None:
    """
    Print a formatted summary of graph properties.

    Includes node and edge counts, edge weight statistics, the top 5
    highest-weight edges (best candidate pairs), and the combinatorial
    search space size C(N, k).

    Parameters
    ----------
    G : nx.Graph
        The complete satellite graph from ``build_graph``.
    k : int
        Target constellation size (number of satellites to select).
    """
    n = len(G.nodes())
    e = len(G.edges())
    weights = [d['weight'] for _, _, d in G.edges(data=True)]

    print("=" * 60)
    print("GRAPH SUMMARY")
    print("=" * 60)
    print(f"  Nodes (candidate satellites) : {n}")
    print(f"  Edges (satellite pairs)      : {e}  [K_{n} complete graph]")
    print()
    print("  Edge weight statistics:")
    print(f"    Min  : {min(weights):.6f}")
    print(f"    Max  : {max(weights):.6f}")
    print(f"    Mean : {np.mean(weights):.6f}")
    print(f"    Std  : {np.std(weights):.6f}")
    print()
    print("  Top 5 highest-weight edges (best candidate pairs):")
    sorted_edges = sorted(
        G.edges(data=True), key=lambda t: t[2]['weight'], reverse=True
    )
    for i, (u, v, d) in enumerate(sorted_edges[:5]):
        print(f"    {i+1}. ({u}, {v})  w = {d['weight']:.6f}")

    # Combinatorial search space: C(N, k) — number of possible constellations.
    num_subgraphs = math.comb(n, k)
    print()
    print(f"  Possible k={k} subgraphs C({n},{k}) : {num_subgraphs:,}")
    if num_subgraphs > 1_000_000:
        print(
            f"  NOTE: {num_subgraphs:,} subgraphs is computationally intractable "
            "for exhaustive search — QUBO/annealing required."
        )
    print("=" * 60)
