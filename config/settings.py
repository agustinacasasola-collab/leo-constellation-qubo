# config/settings.py
# Configuration parameters for LEO constellation QUBO optimization

# ---------------------------------------------------------------------------
# Edge weight scaling factors
# w(vn, vm) = X_SCALE * (1 - Pcn) * (1 - Pcm) + Y_SCALE * avg(coverage)
# ---------------------------------------------------------------------------
X_SCALE = 1.0   # weight for the collision-risk safety term
Y_SCALE = 1.0   # weight for the coverage term

# ---------------------------------------------------------------------------
# Penalty multiplier for the cardinality constraint
# P = PENALTY_MULTIPLIER * max_edge_weight * k^2
# ---------------------------------------------------------------------------
PENALTY_MULTIPLIER = 2.0

# ---------------------------------------------------------------------------
# Simulated annealing parameters
# ---------------------------------------------------------------------------
SA_NUM_READS = 1000    # number of independent SA runs
SA_NUM_SWEEPS = 1000   # number of sweeps per run

# ---------------------------------------------------------------------------
# Quantum annealing parameters (D-Wave Leap)
# ---------------------------------------------------------------------------
QA_NUM_READS = 1000
QA_ANNEALING_TIME = 20   # annealing time in microseconds

# ---------------------------------------------------------------------------
# Default constellation size
# ---------------------------------------------------------------------------
DEFAULT_K = 5

# ---------------------------------------------------------------------------
# Conjunction screening parameters
# ---------------------------------------------------------------------------
APOGEE_PERIGEE_MARGIN_KM = 10.0   # minimum orbital separation to filter (km)
SCREENING_VOLUME_KM = 10.0        # radius of screening sphere (km)
