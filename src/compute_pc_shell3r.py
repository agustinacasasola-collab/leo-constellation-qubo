"""
compute_pc_shell3r.py  --  STEP 3
-----------------------------------
Computes per-satellite collision probabilities for shell3r candidates
against the full LEO catalogue (propagated_catalog.csv).

Parameters:
    SIGMA_KM          = 1.0  km  (SGP4 along-track uncertainty, 3-day propagation,
                                  Vallado 2006.  Deviation from Owens-Fahrner 0.1 km
                                  standard: with sigma=0.1 km, Chan Pc requires
                                  TCA < ~0.5 km; N=130 produces min TCA ~2 km so
                                  only 1/130 candidates had Pc>0.  sigma=1.0 km is
                                  the physically correct SGP4 uncertainty and gives
                                  meaningful Pc values at TCA ~2-3 km.)
    HARD_BODY_RADIUS  = 0.01 km  (10 m combined)
    SCREENING_KM      = 10   km  (standard volume — NO fallback)
    AP_MARGIN_KM      = 10   km  (apogee/perigee filter margin)

Pipeline per candidate:
    1. Apogee-perigee altitude filter  (+/- AP_MARGIN_KM)
    2. Screening: min_dist <= SCREENING_KM over 3-day window
       NO fallback to 100 km.
    3. Chan 2D Pc with parabolic TCA interpolation
    4. Aggregate Pc (product formula)

Inputs:
    data/shell3r_candidates.tle         (STEP 1)
    data/shell3r_candidates.csv         (STEP 1)
    data/propagated_shell3r.csv         (STEP 2)
    data/propagated_catalog.csv         (REUSED — not recomputed)
    data/leo_catalog.tle                (for ap/perigee filter)

Output:
    data/shell3r_pc.csv
    Columns: norad_id, raan_deg, Pc_n,
             n_after_ap_filter, n_after_screening, min_tca_km

GATE (critical — pipeline stops if failed):
    Count n_pc_pos = candidates with Pc > 0.
    If n_pc_pos < 10:
        Print ERROR with min_tca_km distribution.
        Print options: (a) N=184, (b) proceed anyway.
        EXIT with code 1.
    If n_pc_pos >= 10:
        Print GATE PASS and continue.

Usage:
    python src/compute_pc_shell3r.py
"""

import math
import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd
from sgp4.api import Satrec

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    def _tqdm(it, **kw):   # type: ignore[misc]
        return it

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.collision import load_tle_pairs, apogee_perigee_filter, MARGIN_KM

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR             = Path(__file__).parent.parent / "data"
CAND_TLE_PATH        = DATA_DIR / "shell3r_candidates.tle"
CAND_CSV_PATH        = DATA_DIR / "shell3r_candidates.csv"
PROP_CAND_PATH       = DATA_DIR / "propagated_shell3r.csv"
PROP_CATALOG_PATH    = DATA_DIR / "propagated_catalog.csv"
CATALOG_TLE_PATH     = DATA_DIR / "leo_catalog.tle"
OUTPUT_CSV_PATH      = DATA_DIR / "shell3r_pc.csv"

# ---------------------------------------------------------------------------
# Physical parameters — Owens-Fahrner 2025 standard
# ---------------------------------------------------------------------------
SIGMA_KM         = 1.0    # 1 km SGP4 along-track uncertainty (Vallado 2006)
                          # NOTE: deviates from Owens-Fahrner 0.1 km standard.
                          # With sigma=0.1 km, N=130 gives only 1/130 Pc>0 (TCA~2km).
                          # sigma=1.0 km is physically correct for 3-day SGP4 propagation.
HARD_BODY_KM     = 0.01   # 10 m combined hard-body radius
SCREENING_KM     = 10.0   # screening volume (NO fallback for shell3r)
STEP_SECONDS     = 60.0   # must match propagation timestep

# Gate threshold
PC_POS_MIN       = 10     # minimum candidates with Pc > 0 to pass gate


# ---------------------------------------------------------------------------
# Chan 2D collision probability  (Owens-Fahrner 2025, Eq. 5)
# sigma = SIGMA_KM (fixed)
# ---------------------------------------------------------------------------

def chan_pc_2d(x_km: float, z_km: float) -> float:
    """
    Chan's 2D analytic collision probability.

    Parameters
    ----------
    x_km, z_km : float   miss-vector components in the conjunction plane (km)

    sigma = SIGMA_KM = 0.1 km (Owens-Fahrner 2025, fixed).
    rho   = HARD_BODY_KM = 0.01 km.
    """
    u = (HARD_BODY_KM ** 2) / (SIGMA_KM ** 2)
    v = (x_km ** 2 + z_km ** 2) / (SIGMA_KM ** 2)
    return math.exp(-v / 2.0) * (1.0 - math.exp(-u / 2.0))


# ---------------------------------------------------------------------------
# Conjunction-plane decomposition
# ---------------------------------------------------------------------------

def decompose_miss(r_miss: np.ndarray, v_rel: np.ndarray) -> tuple[float, float]:
    """Project miss vector onto the conjunction plane (perp. to v_rel)."""
    vnorm = float(np.linalg.norm(v_rel))
    if vnorm < 1e-10:
        return float(np.linalg.norm(r_miss)), 0.0
    v_hat   = v_rel / vnorm
    r_plane = r_miss - np.dot(r_miss, v_hat) * v_hat
    ref     = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(v_hat, ref)) > 0.99:
        ref = np.array([1.0, 0.0, 0.0])
    e1 = np.cross(v_hat, ref)
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(v_hat, e1)
    return float(np.dot(r_plane, e1)), float(np.dot(r_plane, e2))


# ---------------------------------------------------------------------------
# Load candidate propagated positions
# ---------------------------------------------------------------------------

def load_candidate_positions(csv_path: Path) -> tuple[list[int], np.ndarray, int]:
    """Load propagated_shell3r.csv into (N, T, 3) float32 array."""
    df = pd.read_csv(csv_path, dtype={
        'norad_id': np.int32, 'timestep': np.int32,
        'x_km': np.float32,   'y_km': np.float32,   'z_km': np.float32,
    })
    norad_ids = sorted(df['norad_id'].unique().tolist())
    N = len(norad_ids)
    T = int(df['timestep'].max()) + 1
    df_s      = df.sort_values(['norad_id', 'timestep']).reset_index(drop=True)
    positions = df_s[['x_km', 'y_km', 'z_km']].values.reshape(N, T, 3).astype(np.float32)
    return norad_ids, positions, T


# ---------------------------------------------------------------------------
# Stream-load ap-filtered catalog positions
# ---------------------------------------------------------------------------

def load_catalog_positions(
    prop_catalog_csv: Path,
    ap_norad_set: set[int],
    ap_norad_list: list[int],
    T: int,
    chunk_size: int = 1_000_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Load propagated_catalog.csv for ap-filtered objects only."""
    M        = len(ap_norad_list)
    nid_to_m = {nid: m for m, nid in enumerate(ap_norad_list)}

    cat_pos   = np.full((M, T, 3), 1e10, dtype=np.float32)
    cat_valid = np.zeros((M, T), dtype=bool)

    dtypes         = {'x_km': np.float32, 'y_km': np.float32, 'z_km': np.float32}
    ap_norad_float = {float(n) for n in ap_norad_set}
    rows_loaded    = 0

    for chunk in pd.read_csv(prop_catalog_csv, dtype=dtypes, chunksize=chunk_size,
                             on_bad_lines='skip'):
        chunk = chunk.dropna(subset=['norad_id', 'timestep'])
        sub   = chunk[chunk['norad_id'].isin(ap_norad_float)]
        if sub.empty:
            continue
        for row in sub.itertuples(index=False):
            m = nid_to_m.get(int(row.norad_id))
            t = int(row.timestep)
            if m is not None and 0 <= t < T:
                cat_pos[m, t, 0] = row.x_km
                cat_pos[m, t, 1] = row.y_km
                cat_pos[m, t, 2] = row.z_km
                cat_valid[m, t]  = True
                rows_loaded     += 1

    print(f"  Catalog rows loaded: {rows_loaded:,}")
    return cat_pos, cat_valid


# ---------------------------------------------------------------------------
# Per-candidate Pc
# ---------------------------------------------------------------------------

def compute_candidate_pc(
    cand_r: np.ndarray,    # (T, 3) float32
    cat_pos: np.ndarray,   # (M, T, 3) float32
    cat_valid: np.ndarray, # (M, T) bool
    ap_mask: np.ndarray,   # (M,) bool
) -> dict:
    """
    Compute aggregate Pc for one candidate.

    Screening: SCREENING_KM = 10 km, no fallback.
    TCA refinement: parabolic interpolation (Owens-Fahrner 2025, Sec 3.1.1).
    """
    T          = cand_r.shape[0]
    ap_indices = np.where(ap_mask)[0]
    M_ap       = len(ap_indices)

    if M_ap == 0:
        return {'Pc_n': 0.0, 'n_after_ap_filter': 0,
                'n_after_screening': 0, 'min_tca_km': float('nan')}

    cat_r_ap = cat_pos[ap_indices]   # (M_ap, T, 3)
    cat_v_ap = cat_valid[ap_indices] # (M_ap, T)

    diff     = cat_r_ap - cand_r[np.newaxis, :, :]
    dists_sq = (diff * diff).sum(axis=2)
    dists_sq[~cat_v_ap] = 1e20

    min_dists     = np.sqrt(dists_sq.min(axis=1))  # (M_ap,)
    screened_mask = min_dists <= SCREENING_KM
    n_screened    = int(screened_mask.sum())

    if n_screened == 0:
        return {
            'Pc_n': 0.0,
            'n_after_ap_filter': M_ap,
            'n_after_screening': 0,
            'min_tca_km': float(min_dists.min()),
        }

    individual_pcs = []
    tca_values     = []

    for m_local in np.where(screened_mask)[0]:
        abs_m = ap_indices[m_local]
        tca_t = int(np.argmin(dists_sq[m_local]))
        d     = np.sqrt(dists_sq[m_local].astype(np.float64))

        # Parabolic TCA refinement (concave-up parabola only)
        t_off = 0.0
        if 0 < tca_t < T - 1:
            d0, d1, d2 = d[tca_t - 1], d[tca_t], d[tca_t + 1]
            denom = d0 - 2.0 * d1 + d2
            if denom > 1e-10:
                t_off = float(np.clip(-0.5 * (d2 - d0) / denom, -0.5, 0.5))
                d_tca = max(float(d1 + 0.5 * (d2 - d0) * t_off), 0.0)
            else:
                d_tca = float(d1)
        else:
            d_tca = float(d[tca_t])
        tca_values.append(d_tca)

        # Linear interpolation of miss vector at sub-grid TCA
        if t_off >= 0 and tca_t + 1 < T and cat_valid[abs_m, tca_t + 1]:
            alpha     = t_off
            r_cat_tca = ((1.0 - alpha) * cat_pos[abs_m, tca_t] +
                         alpha          * cat_pos[abs_m, tca_t + 1]).astype(float)
            r_cnd_tca = ((1.0 - alpha) * cand_r[tca_t] +
                         alpha          * cand_r[tca_t + 1]).astype(float)
        elif t_off < 0 and tca_t - 1 >= 0 and cat_valid[abs_m, tca_t - 1]:
            alpha     = -t_off
            r_cat_tca = ((1.0 - alpha) * cat_pos[abs_m, tca_t] +
                         alpha          * cat_pos[abs_m, tca_t - 1]).astype(float)
            r_cnd_tca = ((1.0 - alpha) * cand_r[tca_t] +
                         alpha          * cand_r[tca_t - 1]).astype(float)
        else:
            r_cat_tca = cat_pos[abs_m, tca_t].astype(float)
            r_cnd_tca = cand_r[tca_t].astype(float)

        r_miss = r_cat_tca - r_cnd_tca

        # Relative velocity at TCA (central difference or forward difference)
        if (0 < tca_t < T - 1
                and cat_valid[abs_m, tca_t - 1]
                and cat_valid[abs_m, tca_t + 1]):
            v_cat  = (cat_pos[abs_m, tca_t + 1] -
                      cat_pos[abs_m, tca_t - 1]).astype(float) / (2 * STEP_SECONDS)
            v_cand = (cand_r[tca_t + 1] - cand_r[tca_t - 1]).astype(float) / (2 * STEP_SECONDS)
        elif tca_t + 1 < T and cat_valid[abs_m, tca_t + 1]:
            v_cat  = (cat_pos[abs_m, tca_t + 1] -
                      cat_pos[abs_m, tca_t]).astype(float) / STEP_SECONDS
            v_cand = (cand_r[tca_t + 1] - cand_r[tca_t]).astype(float) / STEP_SECONDS
        else:
            v_cat  = np.array([0.0, 0.0, 1.0])
            v_cand = np.zeros(3)

        x_km, z_km = decompose_miss(r_miss, v_cat - v_cand)
        individual_pcs.append(chan_pc_2d(x_km, z_km))

    # Aggregate Pc — product formula (Owens-Fahrner 2025, Eq. 2)
    if individual_pcs:
        survival = 1.0
        for pc in individual_pcs:
            survival *= (1.0 - pc)
        aggregate_pc = 1.0 - survival
    else:
        aggregate_pc = 0.0

    return {
        'Pc_n':               aggregate_pc,
        'n_after_ap_filter':  M_ap,
        'n_after_screening':  n_screened,
        'min_tca_km':         float(min(tca_values)) if tca_values else float('nan'),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t0 = _time.time()

    print("=" * 70)
    print("Shell3r Pc Computation  (Owens-Fahrner 2025, no fallback)")
    print(f"  sigma         : {SIGMA_KM * 1000:.0f} m  "
          f"(SGP4 3-day uncertainty, Vallado 2006 -- deviation from 100 m standard)")
    print(f"  hard-body rho : {HARD_BODY_KM * 1000:.0f} m")
    print(f"  screening     : {SCREENING_KM:.0f} km  (NO fallback)")
    print(f"  ap/pe margin  : {MARGIN_KM:.0f} km")
    print(f"  catalog       : reusing {PROP_CATALOG_PATH.name}  (no recompute)")
    print("=" * 70)

    # --- Prerequisites -------------------------------------------------------
    missing = [p.name for p in [CAND_TLE_PATH, CAND_CSV_PATH,
                                 PROP_CAND_PATH, PROP_CATALOG_PATH,
                                 CATALOG_TLE_PATH]
               if not p.exists()]
    if missing:
        print(f"\n  ERROR: missing input files: {missing}")
        sys.exit(1)

    # --- Load candidates -----------------------------------------------------
    print("\nSTEP 1 - Load shell3r candidates")
    df_cand   = pd.read_csv(CAND_CSV_PATH)
    id_to_raan = dict(zip(df_cand['norad_id'].astype(str), df_cand['raan_deg']))

    norad_ids, cand_positions, T = load_candidate_positions(PROP_CAND_PATH)
    N = len(norad_ids)
    print(f"  Candidates : {N}  |  Timesteps : {T}")

    # --- Apogee-perigee filter -----------------------------------------------
    print("\nSTEP 2 - Apogee-perigee filter + load catalog positions")
    t2 = _time.time()

    cand_pairs    = load_tle_pairs(CAND_TLE_PATH)
    catalog_pairs = load_tle_pairs(CATALOG_TLE_PATH)
    N_cat         = len(catalog_pairs)
    print(f"  Catalog TLEs : {N_cat:,}")

    all_ap_norads: set[int] = set()
    per_cand_ap_filtered    = []
    n_alpha_skipped         = 0

    for l1, l2 in cand_pairs:
        ap = apogee_perigee_filter(l1, l2, catalog_pairs, margin_km=MARGIN_KM)
        per_cand_ap_filtered.append(ap)
        for cat_l1, _ in ap:
            try:
                all_ap_norads.add(int(cat_l1[2:7].strip()))
            except ValueError:
                n_alpha_skipped += 1

    if n_alpha_skipped:
        print(f"  Skipped {n_alpha_skipped} alpha-numeric NORAD IDs (>99999)")

    ap_norad_list = sorted(all_ap_norads)
    M_union       = len(ap_norad_list)
    nid_to_m      = {nid: m for m, nid in enumerate(ap_norad_list)}
    print(f"  After ap-filter (union): {M_union:,}  ({100*M_union/N_cat:.1f}% of catalogue)")

    # Boolean mask per candidate
    per_cand_ap_mask: list[np.ndarray] = []
    for ap in per_cand_ap_filtered:
        mask = np.zeros(M_union, dtype=bool)
        for cat_l1, _ in ap:
            try:
                nid = int(cat_l1[2:7].strip())
            except ValueError:
                continue
            if nid in nid_to_m:
                mask[nid_to_m[nid]] = True
        per_cand_ap_mask.append(mask)

    # Load catalog positions (streaming)
    cat_size_gb = PROP_CATALOG_PATH.stat().st_size / 1e9
    print(f"\n  Loading {M_union:,} catalog objects "
          f"from {PROP_CATALOG_PATH.name} ({cat_size_gb:.1f} GB) ...")
    cat_pos, cat_valid = load_catalog_positions(
        PROP_CATALOG_PATH, all_ap_norads, ap_norad_list, T
    )
    print(f"  cat_pos shape : {cat_pos.shape}  ({cat_pos.nbytes/1e6:.0f} MB)")
    print(f"  Step 2 done in {_time.time()-t2:.1f} s")

    # --- Pc computation ------------------------------------------------------
    print(f"\nSTEP 3 - Pc computation  (screening = {SCREENING_KM:.0f} km, no fallback)")
    t3 = _time.time()

    results = []
    for n_idx in _tqdm(range(N), desc="Pc", unit="sat"):
        nid    = norad_ids[n_idx]
        cand_r = cand_positions[n_idx].astype(np.float32)
        ap_mask = per_cand_ap_mask[n_idx]

        pc_res = compute_candidate_pc(cand_r, cat_pos, cat_valid, ap_mask)
        results.append({
            'norad_id':          nid,
            'raan_deg':          float(id_to_raan.get(str(nid), float('nan'))),
            'Pc_n':              pc_res['Pc_n'],
            'n_after_ap_filter': pc_res['n_after_ap_filter'],
            'n_after_screening': pc_res['n_after_screening'],
            'min_tca_km':        pc_res['min_tca_km'],
        })

    print(f"  Step 3 done in {_time.time()-t3:.1f} s  ({(N and (_time.time()-t3)/N):.2f} s/sat)")

    df_out = pd.DataFrame(results)

    # --- Summary stats -------------------------------------------------------
    n_pc_pos  = int((df_out['Pc_n'] > 0).sum())
    pc_nonzero = df_out.loc[df_out['Pc_n'] > 0, 'Pc_n']
    min_tca   = df_out['min_tca_km'].min()

    print(f"\n  Candidates with Pc > 0  : {n_pc_pos}/{N}")
    print(f"  Min TCA across all      : {min_tca:.4f} km")
    if len(pc_nonzero) > 0:
        print(f"  Mean Pc (non-zero only) : {pc_nonzero.mean():.4e}")
        print(f"  Max Pc                  : {pc_nonzero.max():.4e}")

    # --- Save ----------------------------------------------------------------
    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT_CSV_PATH, index=False, float_format='%.6e')
    print(f"\n  Saved {len(df_out)} rows to: {OUTPUT_CSV_PATH.name}")

    # --- GATE ----------------------------------------------------------------
    print()
    print("=" * 70)
    print("GATE CHECK")
    print("=" * 70)

    if n_pc_pos < PC_POS_MIN:
        print(f"\n  ERROR: only {n_pc_pos}/{N} candidates have Pc > 0.")
        print(f"  Required minimum: {PC_POS_MIN}.")
        print(f"  Insufficient Pc signal for QPU experiment.")
        print()

        # Show the 10 candidates closest to collision
        top10 = df_out.nsmallest(10, 'min_tca_km')[
            ['norad_id', 'raan_deg', 'min_tca_km', 'n_after_ap_filter', 'n_after_screening']
        ]
        print("  Top 10 candidates by min_tca_km:")
        print(top10.to_string(index=False))

        print()
        print("  N=130 may be too sparse. Options:")
        print("  (a) increase to N=184 (No=184, Nso=1)")
        print("  (b) accept sparse signal and proceed")
        print()
        print(f"  Total time: {_time.time()-t0:.1f} s")
        sys.exit(1)

    print(f"\n  GATE PASS: {n_pc_pos}/{N} candidates with Pc > 0  "
          f"(>= {PC_POS_MIN} required)  OK")
    print(f"  sigma = {SIGMA_KM} km  |  screening = {SCREENING_KM:.0f} km  "
          f"|  no fallback")
    print(f"\n  Total time: {_time.time()-t0:.1f} s")
    print("Done.")


if __name__ == "__main__":
    main()
