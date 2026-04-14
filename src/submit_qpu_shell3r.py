"""
submit_qpu_shell3r.py  --  STEP 6
-----------------------------------
Submit shell3r QUBO to D-Wave Advantage QPU via Leap.

Prerequisites:
    - STEP 5 (optimize_shell3r.py) must have passed the OOM gate.
    - D-Wave Leap API token must be configured:
        dwave config create   (or set DWAVE_API_TOKEN env var)

Parameters:
    K          = 20
    LAMBDA_VAL = 0.5
    (Rebuilds Q from shell3r_pc.csv + shell3r_coverage.csv — same
     as STEP 5 so formulation is guaranteed consistent.)

QPU protocol:
    1. Embedding check (minorminer)
    2. Chain strength sweep  (~60 QPU reads total)
    3. Annealing time sweep  (~60 QPU reads total)
    4. Final QPU run         (~200 QPU reads)

Post-processing:
    - Only solutions with chain_break_fraction < 0.05 are kept.
    - Only solutions with exactly K=20 selected satellites are scored.

Output:
    results/shell3r_QPU.csv
      Columns: read_idx, n_selected, agg_pc, mean_cov,
               chain_break_frac, energy, valid

Usage:
    python src/submit_qpu_shell3r.py

WARNING: This script submits real QPU jobs and consumes D-Wave
Leap credits. Run only after STEP 5 gate has passed.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR    = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
PC_CSV      = DATA_DIR    / "shell3r_pc.csv"
COV_CSV     = DATA_DIR    / "shell3r_coverage.csv"
OUTPUT_CSV  = RESULTS_DIR / "shell3r_QPU.csv"

# Previous SA/SQA results for comparison (from Step 5)
COMPARISON_CSV = RESULTS_DIR / "shell3r_comparison.csv"

# ---------------------------------------------------------------------------
# Parameters — must match optimize_shell3r.py exactly
# ---------------------------------------------------------------------------
K          = 20
LAMBDA_VAL = 0.5
CBF_LIMIT  = 0.05     # chain break fraction threshold for valid reads

# QPU budget
READS_CS_SWEEP = 15   # reads per chain_strength value
READS_AT_SWEEP = 20   # reads per annealing_time value
READS_FINAL    = 200  # final production run


# ---------------------------------------------------------------------------
# Rebuild QUBO (mirrors optimize_shell3r.py exactly)
# ---------------------------------------------------------------------------

def build_qubo(
    safety_norm: np.ndarray,
    cov_norm: np.ndarray,
    k: int,
    P: float,
    lam: float,
) -> np.ndarray:
    N = len(safety_norm)
    Q = np.zeros((N, N), dtype=np.float64)
    np.fill_diagonal(Q, P * (1.0 - 2.0 * k))
    for i in range(N):
        for j in range(i + 1, N):
            w       = lam * safety_norm[i] * safety_norm[j] \
                    + (1.0 - lam) * cov_norm[i] * cov_norm[j]
            Q[i, j] = -w + 2.0 * P
    return Q


def aggregate_pc(selected: list[int], pc: np.ndarray) -> float:
    s = 1.0
    for i in selected:
        s *= 1.0 - pc[i]
    return 1.0 - s


def mean_coverage(selected: list[int], cov: np.ndarray) -> float:
    return float(np.mean(cov[selected])) if selected else 0.0


def _f(v: float, fmt: str = '.4e') -> str:
    return f"{v:{fmt}}" if not math.isnan(v) else "N/A"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 72)
    print("Shell3r QPU Submission  --  STEP 6")
    print(f"  K={K}  LAMBDA={LAMBDA_VAL}")
    print("=" * 72)

    # --- Prerequisites -------------------------------------------------------
    for p in [PC_CSV, COV_CSV]:
        if not p.exists():
            print(f"\n  ERROR: {p.name} not found. Run STEPs 3-5 first.")
            sys.exit(1)

    # Check Step 5 gate result
    if COMPARISON_CSV.exists():
        df_cmp = pd.read_csv(COMPARISON_CSV)
        sa_row = df_cmp[df_cmp['solver'] == 'SA']
        rnd_row = df_cmp[df_cmp['solver'] == 'Random']
        if len(sa_row) > 0 and len(rnd_row) > 0:
            pc_sa  = float(sa_row['best_pc'].iloc[0])
            pc_rnd = float(rnd_row['best_pc'].iloc[0])
            oom    = math.log10(pc_rnd / pc_sa) if pc_sa > 0 and pc_rnd > 0 else float('nan')
            print(f"\n  Step 5 gate: SA OOM = {oom:.2f} vs random")
            if not math.isnan(oom) and oom < 0.5:
                print("  ERROR: Step 5 gate not passed (OOM < 0.5). Aborting QPU run.")
                sys.exit(1)
    else:
        print("\n  WARNING: shell3r_comparison.csv not found. Cannot verify Step 5 gate.")
        print("  Proceeding — ensure Step 5 passed before running this step.")

    # --- Import D-Wave libraries ---------------------------------------------
    try:
        from dwave.system import DWaveSampler, EmbeddingComposite
        import minorminer
    except ImportError as e:
        print(f"\n  ERROR: D-Wave Ocean SDK not installed: {e}")
        print("  Install with: pip install dwave-ocean-sdk")
        sys.exit(1)

    # --- Load and join on norad_id -------------------------------------------
    df_pc  = pd.read_csv(PC_CSV)
    df_cov = pd.read_csv(COV_CSV)
    df     = df_pc[['norad_id', 'raan_deg', 'Pc_n']].merge(
        df_cov[['norad_id', 'coverage_norm']], on='norad_id', how='inner'
    )
    N = len(df)
    print(f"\n  Candidates joined by norad_id: {N}")

    pc_values  = df['Pc_n'].values.astype(np.float64)
    cov_norm   = df['coverage_norm'].values.astype(np.float64)
    norad_ids  = df['norad_id'].tolist()
    raan_vals  = df['raan_deg'].values

    # --- Safety signal and QUBO (identical to Step 5) ------------------------
    safety_raw  = -np.log(pc_values + 1e-15)
    safety_norm = safety_raw / safety_raw.max()

    w_vals = [LAMBDA_VAL * safety_norm[i] * safety_norm[j]
              + (1.0 - LAMBDA_VAL) * cov_norm[i] * cov_norm[j]
              for i in range(N) for j in range(i + 1, N)]
    w_max = float(max(w_vals))
    P     = 2.0 * w_max * K * N

    Q_np = build_qubo(safety_norm, cov_norm, K, P, LAMBDA_VAL)

    # Convert to dict format required by D-Wave sampler
    Q_dict = {}
    for i in range(N):
        Q_dict[(i, i)] = Q_np[i, i]
    for i in range(N):
        for j in range(i + 1, N):
            if Q_np[i, j] != 0.0:
                Q_dict[(i, j)] = Q_np[i, j]

    print(f"  Q entries: {len(Q_dict):,}  (diagonal + upper triangular)")

    # =========================================================================
    # STEP 6a — Embedding check
    # =========================================================================
    print(f"\n{'='*72}")
    print("STEP 6a — Embedding check")
    print(f"{'='*72}")

    try:
        physical_sampler = DWaveSampler()
        print(f"  QPU topology : {physical_sampler.properties.get('topology', {}).get('type', 'unknown')}")
        print(f"  Qubits       : {len(physical_sampler.nodelist)}")
        print(f"  Couplers     : {len(physical_sampler.edgelist)}")
    except Exception as e:
        print(f"\n  ERROR: could not connect to D-Wave QPU: {e}")
        print("  Check Leap API token and network connectivity.")
        sys.exit(1)

    print(f"\n  Finding embedding for K_{N} ({N*(N-1)//2:,} edges) ...")
    edges = [(i, j) for (i, j) in Q_dict.keys() if i != j]

    try:
        embedding = minorminer.find_embedding(edges, physical_sampler.edgelist)
    except Exception as e:
        print(f"  ERROR during embedding: {e}")
        sys.exit(1)

    if not embedding:
        print("  ERROR: problem does not embed in available QPU.")
        print("  Options:")
        print("    (a) Reduce N (currently {N}) — try N=80 or N=50")
        print("    (b) Use Leap Hybrid Solver (no embedding limit)")
        sys.exit(1)

    max_chain_len = max(len(v) for v in embedding.values())
    mean_chain_len = float(np.mean([len(v) for v in embedding.values()]))
    total_phys_qubits = sum(len(v) for v in embedding.values())
    print(f"  Embedding OK")
    print(f"  Max chain length  : {max_chain_len}")
    print(f"  Mean chain length : {mean_chain_len:.1f}")
    print(f"  Physical qubits   : {total_phys_qubits} / {len(physical_sampler.nodelist)}")
    if max_chain_len > 10:
        print(f"  WARNING: max chain length {max_chain_len} > 10. "
              f"Long chains may degrade solution quality.")

    # =========================================================================
    # STEP 6b — Chain strength sweep
    # =========================================================================
    print(f"\n{'='*72}")
    print(f"STEP 6b — Chain strength sweep  ({READS_CS_SWEEP} reads each)")
    print(f"{'='*72}")

    sampler = EmbeddingComposite(physical_sampler)
    CS_VALUES = [0.5, 1.0, 2.0, 4.0]
    best_cs   = None

    for cs in CS_VALUES:
        try:
            res = sampler.sample_qubo(Q_dict,
                chain_strength=cs,
                num_reads=READS_CS_SWEEP,
                annealing_time=20)
            cbf_mean = float(np.mean(res.record.chain_break_fraction))
            print(f"  chain_strength={cs:.1f}: mean CBF={cbf_mean:.3f}")
            if cbf_mean < CBF_LIMIT and best_cs is None:
                best_cs = cs
        except Exception as e:
            print(f"  chain_strength={cs:.1f}: ERROR ({e})")

    if best_cs is None:
        best_cs = max(CS_VALUES)
        print(f"  WARNING: CBF > {CBF_LIMIT} for all tested values. Using {best_cs}.")
    else:
        print(f"  Selected chain_strength = {best_cs}")

    # =========================================================================
    # STEP 6c — Annealing time sweep
    # =========================================================================
    print(f"\n{'='*72}")
    print(f"STEP 6c — Annealing time sweep  ({READS_AT_SWEEP} reads each)")
    print(f"{'='*72}")

    AT_VALUES   = [20, 100, 500]
    best_at     = 20
    best_energy = float('inf')

    for at in AT_VALUES:
        try:
            res = sampler.sample_qubo(Q_dict,
                chain_strength=best_cs,
                num_reads=READS_AT_SWEEP,
                annealing_time=at)
            e = res.first.energy
            cbf = float(np.mean(res.record.chain_break_fraction))
            print(f"  annealing_time={at:4d} us: best_energy={e:.6f}  "
                  f"mean_CBF={cbf:.3f}")
            if e < best_energy:
                best_energy = e
                best_at     = at
        except Exception as e_exc:
            print(f"  annealing_time={at:4d} us: ERROR ({e_exc})")

    print(f"\n  Selected annealing_time = {best_at} us")

    # =========================================================================
    # STEP 6d — Final QPU run
    # =========================================================================
    print(f"\n{'='*72}")
    print(f"STEP 6d — Final QPU run  ({READS_FINAL} reads, "
          f"chain_strength={best_cs}, annealing_time={best_at} us)")
    print(f"{'='*72}")

    try:
        res_qpu = sampler.sample_qubo(Q_dict,
            chain_strength=best_cs,
            annealing_time=best_at,
            num_reads=READS_FINAL)
    except Exception as e:
        print(f"\n  ERROR during final QPU run: {e}")
        sys.exit(1)

    # =========================================================================
    # Post-processing
    # =========================================================================
    print(f"\n{'='*72}")
    print("Post-processing")
    print(f"{'='*72}")

    all_samples = list(res_qpu.samples())
    cbf_arr     = res_qpu.record.chain_break_fraction
    energy_arr  = res_qpu.record.energy

    qpu_rows    = []
    best_qpu_pc = float('inf')
    best_qpu_cov = 0.0
    n_valid     = 0
    n_exact_k   = 0

    for idx, (sample, cbf, energy) in enumerate(zip(all_samples, cbf_arr, energy_arr)):
        selected    = [i for i, v in sample.items() if v == 1]
        n_sel       = len(selected)
        is_valid_cbf = cbf < CBF_LIMIT
        is_exact_k   = n_sel == K

        agg_pc   = aggregate_pc(selected, pc_values) if is_exact_k else float('nan')
        mean_cov = mean_coverage(selected, cov_norm)   if is_exact_k else float('nan')

        if is_valid_cbf:
            n_valid += 1
        if is_valid_cbf and is_exact_k:
            n_exact_k += 1
            if agg_pc < best_qpu_pc:
                best_qpu_pc  = agg_pc
                best_qpu_cov = mean_cov

        qpu_rows.append({
            'read_idx':         idx,
            'n_selected':       n_sel,
            'agg_pc':           agg_pc,
            'mean_cov':         mean_cov,
            'chain_break_frac': float(cbf),
            'energy':           float(energy),
            'valid':            int(is_valid_cbf and is_exact_k),
        })

    print(f"\n  Valid reads (CBF<{CBF_LIMIT}): {n_valid}/{READS_FINAL} "
          f"({100*n_valid/READS_FINAL:.0f}%)")
    print(f"  Valid + exactly K={K}: {n_exact_k}/{READS_FINAL} "
          f"({100*n_exact_k/READS_FINAL:.0f}%)")

    if n_exact_k > 0:
        print(f"\n  QPU best aggregate Pc  : {best_qpu_pc:.4e}")
        print(f"  QPU best mean coverage : {best_qpu_cov:.4f}")
    else:
        print("\n  WARNING: no valid feasible (K=20) solutions found.")
        print("  Try increasing chain_strength or num_reads.")

    # =========================================================================
    # Final comparison table
    # =========================================================================
    if COMPARISON_CSV.exists():
        df_cmp = pd.read_csv(COMPARISON_CSV)
        rnd_pc  = float(df_cmp.loc[df_cmp['solver']=='Random', 'best_pc'].iloc[0]) \
                  if 'Random' in df_cmp['solver'].values else float('nan')
        sa_pc   = float(df_cmp.loc[df_cmp['solver']=='SA',     'best_pc'].iloc[0]) \
                  if 'SA'  in df_cmp['solver'].values else float('nan')
        sqa_pc  = float(df_cmp.loc[df_cmp['solver']=='SQA',    'best_pc'].iloc[0]) \
                  if 'SQA' in df_cmp['solver'].values else float('nan')
        rnd_cov = float(df_cmp.loc[df_cmp['solver']=='Random', 'best_cov'].iloc[0]) \
                  if 'Random' in df_cmp['solver'].values else float('nan')
        sa_cov  = float(df_cmp.loc[df_cmp['solver']=='SA',     'best_cov'].iloc[0]) \
                  if 'SA'  in df_cmp['solver'].values else float('nan')
        sqa_cov = float(df_cmp.loc[df_cmp['solver']=='SQA',    'best_cov'].iloc[0]) \
                  if 'SQA' in df_cmp['solver'].values else float('nan')
        qpu_cov = best_qpu_cov if n_exact_k > 0 else float('nan')
        qpu_pc  = best_qpu_pc  if n_exact_k > 0 else float('nan')

        oom_sa  = math.log10(rnd_pc/sa_pc)   if sa_pc>0  and rnd_pc>0 else float('nan')
        oom_sqa = math.log10(rnd_pc/sqa_pc)  if sqa_pc>0 and rnd_pc>0 else float('nan')
        oom_qpu = math.log10(rnd_pc/qpu_pc)  if qpu_pc>0 and rnd_pc>0 else float('nan')
        cbf_mean_final = float(np.mean(cbf_arr))

        W = 10
        print()
        print(f"  +{'-'*26}+{'-'*W}+{'-'*W}+{'-'*W}+{'-'*W}+")
        print(f"  | {'Metric':<24} | {'Random':^{W-2}} | {'SA':^{W-2}} "
              f"| {'SQA':^{W-2}} | {'QPU':^{W-2}} |")
        print(f"  +{'-'*26}+{'-'*W}+{'-'*W}+{'-'*W}+{'-'*W}+")
        print(f"  | {'Best aggregate Pc':<24} | {_f(rnd_pc):^{W-2}} "
              f"| {_f(sa_pc):^{W-2}} | {_f(sqa_pc):^{W-2}} | {_f(qpu_pc):^{W-2}} |")
        print(f"  | {'Best mean coverage':<24} | {_f(rnd_cov,'.4f'):^{W-2}} "
              f"| {_f(sa_cov,'.4f'):^{W-2}} | {_f(sqa_cov,'.4f'):^{W-2}} "
              f"| {_f(qpu_cov,'.4f'):^{W-2}} |")
        print(f"  | {'OOM vs random (Pc)':<24} | {'---':^{W-2}} "
              f"| {_f(oom_sa,'.2f'):^{W-2}} | {_f(oom_sqa,'.2f'):^{W-2}} "
              f"| {_f(oom_qpu,'.2f'):^{W-2}} |")
        print(f"  | {'Mean chain break frac':<24} | {'---':^{W-2}} "
              f"| {'---':^{W-2}} | {'---':^{W-2}} | {cbf_mean_final:.3f}   |")
        print(f"  | {'Valid reads (%)':<24} | {'---':^{W-2}} "
              f"| {'---':^{W-2}} | {'---':^{W-2}} | {100*n_exact_k/READS_FINAL:.0f}%     |")
        print(f"  +{'-'*26}+{'-'*W}+{'-'*W}+{'-'*W}+{'-'*W}+")

    # --- Save QPU results ----------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df_qpu = pd.DataFrame(qpu_rows)
    df_qpu.to_csv(OUTPUT_CSV, index=False, float_format='%.6e')
    print(f"\n  Saved {len(df_qpu)} rows to: {OUTPUT_CSV.name}")
    print("Done.")


if __name__ == "__main__":
    main()
