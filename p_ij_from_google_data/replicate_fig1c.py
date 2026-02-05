"""
Replicate Figure 1c from arXiv:2408.13687v1

This script:
1. Builds a detector error model (.dem) with values computed from syndrome correlations
   using formulas from Appendix E of arXiv:2502.17722v1
2. Saves the DEM to each data folder
3. Decodes REAL detection events using pymatching with the p_ij DEM
4. Optionally samples syndromes from the DEM for comparison
5. Plots logical error rate vs rounds, comparing real and simulated decoding
"""

import stim
import numpy as np
import pymatching
import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[
    "#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3",
    "#a6d854", "#ffd92f", "#e5c494", "#b3b3b3"])
import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, Tuple, List, Optional
import argparse
import csv


# =============================================================================
# Appendix E formulas for computing p_ij from syndrome correlations
# =============================================================================

def correlator(data: np.ndarray, idxs: List[int], cache: Dict) -> float:
    """
    Compute <σ̃_i σ̃_j ...> where σ̃ = 1 - 2σ maps {0,1} → {+1,-1}
    """
    key = tuple(sorted(idxs))
    if key not in cache:
        vals = 1 - 2 * data[:, idxs]  # map {0,1} → {+1,-1}
        cache[key] = np.mean(np.prod(vals, axis=1))
    return cache[key]


def compute_pijkl(i: int, j: int, k: int, l: int,
                  data: np.ndarray, cache: Dict) -> float:
    """(E4) Four-body error probability"""
    num = (correlator(data, [i], cache) *
           correlator(data, [j], cache) *
           correlator(data, [k], cache) *
           correlator(data, [l], cache) *
           correlator(data, [i, j, k], cache) *
           correlator(data, [i, j, l], cache) *
           correlator(data, [i, k, l], cache) *
           correlator(data, [j, k, l], cache))

    den = (correlator(data, [i, j], cache) *
           correlator(data, [i, k], cache) *
           correlator(data, [i, l], cache) *
           correlator(data, [j, k], cache) *
           correlator(data, [j, l], cache) *
           correlator(data, [k, l], cache) *
           correlator(data, [i, j, k, l], cache))

    if den == 0 or num / den < 0:
        return 0.0
    return 0.5 - 0.5 * (num / den) ** (1 / 8)


def compute_pijk(i: int, j: int, k: int, data: np.ndarray,
                 cache: Dict, p_ijkl_keys: List[Tuple]) -> float:
    """(E3) Three-body error probability"""
    num = (correlator(data, [i], cache) *
           correlator(data, [j], cache) *
           correlator(data, [k], cache) *
           correlator(data, [i, j, k], cache))

    den = (correlator(data, [i, j], cache) *
           correlator(data, [i, k], cache) *
           correlator(data, [j, k], cache))

    prod = 1.0
    for (a, b, c, d) in p_ijkl_keys:
        if set((i, j, k)) <= set((a, b, c, d)):
            ll = list(set((a, b, c, d)) - {i, j, k})[0]
            val = compute_pijkl(i, j, k, ll, data, cache)
            if val < 0.5:
                prod *= 1.0 / (1.0 - 2.0 * val)

    if den == 0 or num / den < 0:
        return 0.0
    return 0.5 - 0.5 * (num / den) ** (1 / 4) * prod


def compute_pij(i: int, j: int, data: np.ndarray, cache: Dict,
                p_ijk_keys: List[Tuple], p_ijkl_keys: List[Tuple]) -> float:
    """(E2) Two-body error probability"""
    si = correlator(data, [i], cache)
    sj = correlator(data, [j], cache)
    sij = correlator(data, [i, j], cache)

    if sij == 0 or si * sj / sij < 0:
        return 0.0

    sqrt_term = np.sqrt(si * sj / sij)
    prod = 1.0

    for (a, b, c) in p_ijk_keys:
        if {i, j} <= {a, b, c}:
            k = list({a, b, c} - {i, j})[0]
            val = compute_pijk(i, j, k, data, cache, p_ijkl_keys)
            if val < 0.5:
                prod *= 1.0 / (1.0 - 2.0 * val)

    for (a, b, c, d) in p_ijkl_keys:
        if {i, j} <= {a, b, c, d}:
            others = list({a, b, c, d} - {i, j})
            k, ll = others
            val = compute_pijkl(i, j, k, ll, data, cache)
            if val < 0.5:
                prod *= 1.0 / (1.0 - 2.0 * val)

    return 0.5 - 0.5 * sqrt_term * prod


def compute_pi(i: int, data: np.ndarray, cache: Dict,
               p_ij_keys: List[Tuple], p_ijk_keys: List[Tuple],
               p_ijkl_keys: List[Tuple]) -> float:
    """(E1) Single-body error probability"""
    si = correlator(data, [i], cache)
    prod = 1.0

    for (a, b) in p_ij_keys:
        if i in (a, b):
            j = b if a == i else a
            val = compute_pij(i, j, data, cache, p_ijk_keys, p_ijkl_keys)
            if val < 0.5:
                prod *= 1.0 / (1.0 - 2.0 * val)

    for (a, b, c) in p_ijk_keys:
        if i in (a, b, c):
            j, k = [x for x in (a, b, c) if x != i]
            val = compute_pijk(i, j, k, data, cache, p_ijkl_keys)
            if val < 0.5:
                prod *= 1.0 / (1.0 - 2.0 * val)

    for (a, b, c, d) in p_ijkl_keys:
        if i in (a, b, c, d):
            j, k, ll = [x for x in (a, b, c, d) if x != i]
            val = compute_pijkl(i, j, k, ll, data, cache)
            if val < 0.5:
                prod *= 1.0 / (1.0 - 2.0 * val)

    return 0.5 - 0.5 * si * prod


# =============================================================================
# DEM construction
# =============================================================================

def extract_dem_structure(dem: stim.DetectorErrorModel) -> Tuple[Dict, Dict, Dict, Dict, List]:
    """Extract the structure of error terms from a detector error model."""
    p_i, p_ij, p_ijk, p_ijkl = {}, {}, {}, {}
    all_errors = []

    for error in dem:
        if error.type == 'error':
            p = error.args_copy()[0]
            targets = error.targets_copy()
            det_targets = [t.val for t in targets if t.is_relative_detector_id()]
            log_targets = [t.val for t in targets if t.is_logical_observable_id()]
            key = tuple(sorted(det_targets))

            if len(key) == 1:
                p_i[key] = p
            elif len(key) == 2:
                p_ij[key] = p
            elif len(key) == 3:
                p_ijk[key] = p
            elif len(key) == 4:
                p_ijkl[key] = p

            all_errors.append((key, log_targets))

    return p_i, p_ij, p_ijk, p_ijkl, all_errors


def compute_p_ij_from_data(data: np.ndarray,
                           p_i_keys: List[Tuple],
                           p_ij_keys: List[Tuple],
                           p_ijk_keys: List[Tuple],
                           p_ijkl_keys: List[Tuple]) -> Tuple[Dict, Dict, Dict, Dict]:
    """Compute all error probabilities from detection event data."""
    cache = {}

    p_ijkl_computed = {}
    for key in p_ijkl_keys:
        i, j, k, l = key
        p_ijkl_computed[key] = compute_pijkl(i, j, k, l, data, cache)

    p_ijk_computed = {}
    for key in p_ijk_keys:
        i, j, k = key
        p_ijk_computed[key] = compute_pijk(i, j, k, data, cache, list(p_ijkl_keys))

    p_ij_computed = {}
    for key in p_ij_keys:
        i, j = key
        p_ij_computed[key] = compute_pij(i, j, data, cache, list(p_ijk_keys), list(p_ijkl_keys))

    p_i_computed = {}
    for key in p_i_keys:
        i = key[0]
        p_i_computed[key] = compute_pi(i, data, cache, list(p_ij_keys), list(p_ijk_keys), list(p_ijkl_keys))

    return p_i_computed, p_ij_computed, p_ijk_computed, p_ijkl_computed


def build_dem_from_computed_probs(all_errors: List,
                                  p_i: Dict, p_ij: Dict,
                                  p_ijk: Dict, p_ijkl: Dict) -> stim.DetectorErrorModel:
    """Build a new DEM with computed probabilities."""
    dem = stim.DetectorErrorModel()

    for det_key, log_mask in all_errors:
        if len(det_key) == 1:
            prob = p_i.get(det_key, 0.0)
        elif len(det_key) == 2:
            prob = p_ij.get(det_key, 0.0)
        elif len(det_key) == 3:
            prob = p_ijk.get(det_key, 0.0)
        elif len(det_key) == 4:
            prob = p_ijkl.get(det_key, 0.0)
        else:
            prob = 0.0

        prob = max(0.0, min(prob, 0.5 - 1e-10))

        if prob > 1e-15:
            targets = [stim.target_relative_detector_id(d) for d in det_key]
            targets.extend([stim.target_logical_observable_id(l) for l in log_mask])
            dem.append("error", prob, targets)

    return dem


def save_dem(dem: stim.DetectorErrorModel, path: Path):
    """Save DEM to file."""
    with open(path, 'w') as f:
        f.write(str(dem))


# =============================================================================
# Decoding functions
# =============================================================================

def decode_real_data(dem: stim.DetectorErrorModel,
                     det_data: np.ndarray,
                     obs_actual: np.ndarray) -> Tuple[float, int]:
    """
    Decode real detection events using pymatching.
    Returns (logical_error_rate, num_shots)
    """
    matcher = pymatching.Matching.from_detector_error_model(dem)
    predicted = matcher.decode_batch(det_data)
    errors = np.any(obs_actual != predicted, axis=1)
    return np.mean(errors), len(det_data)


def sample_and_decode(dem: stim.DetectorErrorModel,
                      num_shots: int) -> Tuple[float, int]:
    """
    Sample syndromes from DEM and decode using pymatching.
    Returns (logical_error_rate, num_shots)
    """
    sampler = dem.compile_sampler()
    det_data, obs_data, _ = sampler.sample(num_shots)
    matcher = pymatching.Matching.from_detector_error_model(dem)
    predicted = matcher.decode_batch(det_data)
    errors = np.any(obs_data != predicted, axis=1)
    return np.mean(errors), num_shots


# =============================================================================
# Main processing
# =============================================================================

def process_single_instance(data_dir: Path, num_samples: int = 50000,
                            save_dem_file: bool = True, verbose: bool = True) -> Dict:
    """
    Process a single code instance.

    Returns dict with:
    - distance, rounds, shots
    - ler_real: logical error rate decoding real data with p_ij DEM
    - ler_simulated: logical error rate decoding simulated data from p_ij DEM
    """
    with open(data_dir / "metadata.json") as f:
        metadata = json.load(f)

    distance = metadata["distance"]
    rounds = metadata["rounds"]
    original_shots = metadata["shots"]

    if verbose:
        print(f"  Processing d={distance}, r={rounds}...", flush=True)

    # Load noisy circuit to get DEM structure
    circuit = stim.Circuit.from_file(str(data_dir / "circuit_noisy_si1000.stim"))
    dem_template = circuit.detector_error_model()
    num_detectors = circuit.num_detectors

    # Extract DEM structure
    p_i_keys, p_ij_keys, p_ijk_keys, p_ijkl_keys, all_errors = extract_dem_structure(dem_template)

    # Load real detection events
    det_data = stim.read_shot_data_file(
        path=str(data_dir / "detection_events.b8"),
        format="b8",
        bit_packed=False,
        num_detectors=num_detectors
    )

    # Load actual observable flips
    obs_actual = stim.read_shot_data_file(
        path=str(data_dir / "obs_flips_actual.b8"),
        format="b8",
        bit_packed=False,
        num_observables=1
    )

    # Compute error probabilities from data using Appendix E formulas
    p_i, p_ij, p_ijk, p_ijkl = compute_p_ij_from_data(
        det_data,
        list(p_i_keys.keys()),
        list(p_ij_keys.keys()),
        list(p_ijk_keys.keys()),
        list(p_ijkl_keys.keys())
    )

    # Build DEM with computed probabilities
    dem = build_dem_from_computed_probs(all_errors, p_i, p_ij, p_ijk, p_ijkl)

    # Save DEM to folder
    if save_dem_file:
        dem_path = data_dir / "decoding_results" / "pij_model"
        dem_path.mkdir(parents=True, exist_ok=True)
        save_dem(dem, dem_path / "error_model.dem")

    # Decode REAL detection events with p_ij DEM
    ler_real, n_real = decode_real_data(dem, det_data, obs_actual)

    # Sample and decode from DEM (simulated)
    ler_sim, n_sim = sample_and_decode(dem, num_samples)

    if verbose:
        print(f"    Real data LER: {ler_real:.4f}, Simulated LER: {ler_sim:.4f}", flush=True)

    return {
        "distance": distance,
        "rounds": rounds,
        "shots_real": n_real,
        "shots_sim": n_sim,
        "ler_real": ler_real,
        "ler_simulated": ler_sim,
        "path": str(data_dir)
    }


def find_all_z_instances(base_dir: Path,
                         max_patches_per_distance: int = None,
                         skip_every_other_round: bool = False) -> List[Path]:
    """Find all Z-type instance directories."""
    patches_by_d = defaultdict(list)
    for patch_dir in sorted(base_dir.iterdir()):
        if patch_dir.is_dir() and patch_dir.name.startswith("d"):
            d = int(patch_dir.name.split("_")[0][1:])
            patches_by_d[d].append(patch_dir)

    instances = []
    for d in sorted(patches_by_d.keys()):
        patches = patches_by_d[d]
        if max_patches_per_distance:
            patches = patches[:max_patches_per_distance]

        for patch_dir in patches:
            z_dir = patch_dir / "Z"
            if z_dir.exists():
                rounds_dirs = sorted(z_dir.iterdir(),
                                    key=lambda x: int(x.name[1:]) if x.name[1:].isdigit() else 0)
                if skip_every_other_round:
                    rounds_dirs = rounds_dirs[::2]

                for rounds_dir in rounds_dirs:
                    if rounds_dir.is_dir() and rounds_dir.name.startswith("r"):
                        instances.append(rounds_dir)

    return instances


def get_google_decoder_results(base_dir: Path, instances: List[Path]) -> List[Dict]:
    """Read Google RL decoder results for comparison."""
    results = []
    for data_dir in instances:
        meta_file = data_dir / "metadata.json"
        if not meta_file.exists():
            continue

        with open(meta_file) as f:
            meta = json.load(f)

        actual_file = data_dir / "obs_flips_actual.b8"
        pred_file = data_dir / "decoding_results" / "correlated_matching_decoder_with_rl_optimized_prior" / "obs_flips_predicted.b8"

        if not actual_file.exists() or not pred_file.exists():
            continue

        actual = stim.read_shot_data_file(path=str(actual_file), format="b8",
                                          bit_packed=False, num_observables=1)
        predicted = stim.read_shot_data_file(path=str(pred_file), format="b8",
                                             bit_packed=False, num_observables=1)

        errors = np.any(actual != predicted, axis=1)
        ler = np.mean(errors)

        results.append({
            "distance": meta["distance"],
            "rounds": meta["rounds"],
            "shots": meta["shots"],
            "ler": ler
        })

    return results


def fit_error_per_cycle(rounds: np.ndarray, p_logical: np.ndarray) -> Tuple[float, float]:
    """Fit to extract eps (error per cycle)."""
    valid = (p_logical < 0.49) & (p_logical > 0.01)
    r_valid = rounds[valid]
    p_valid = p_logical[valid]

    if len(r_valid) < 2:
        return np.nan, np.nan

    y = np.log(1 - 2 * p_valid)
    slope, intercept = np.polyfit(r_valid, y, 1)
    eps = 0.5 * (1 - np.exp(slope))
    return eps, slope


def main():
    parser = argparse.ArgumentParser(description="Replicate Fig 1c from arXiv:2408.13687v1")
    parser.add_argument("--data-dir", type=str,
                        default="./2024_google_105Q_surface_code_d3_d5_d7",
                        help="Path to data directory")
    parser.add_argument("--num-samples", type=int, default=50000,
                        help="Number of syndrome samples for simulation")
    parser.add_argument("--output", type=str, default="figures/fig1c_replica.png",
                        help="Output plot filename")
    parser.add_argument("--max-patches", type=int, default=None,
                        help="Maximum patches per distance (default: all)")
    parser.add_argument("--skip-rounds", action="store_true",
                        help="Take every other round")
    args = parser.parse_args()

    base_dir = Path(args.data_dir)
    print(f"Base directory: {base_dir}", flush=True)
    if not base_dir.exists():
        print(f"Error: Data directory {base_dir} does not exist", flush=True)
        return

    # Create output directory
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all Z-type instances
    print("Finding Z-type instances...", flush=True)
    instances = find_all_z_instances(base_dir,
                                     max_patches_per_distance=args.max_patches,
                                     skip_every_other_round=args.skip_rounds)
    print(f"Found {len(instances)} Z-type instances", flush=True)

    # Process each instance (build DEM, decode real + simulated)
    print("\nProcessing instances (building p_ij DEM, decoding)...", flush=True)
    results = []
    for inst_path in instances:
        try:
            result = process_single_instance(inst_path, args.num_samples)
            results.append(result)
        except Exception as e:
            print(f"  Error processing {inst_path}: {e}", flush=True)

    # Get Google decoder results for comparison
    print("\nReading Google RL decoder results...", flush=True)
    google_results = get_google_decoder_results(base_dir, instances)

    # Save raw data
    csv_file = args.output.replace(".png", ".csv")
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["distance", "rounds", "shots_real", "shots_sim",
                                                "ler_real", "ler_simulated", "path"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Data saved to {csv_file}", flush=True)

    # Organize results by (d, r) and average
    def average_results(results_list, ler_key, shots_key):
        by_d_r = defaultdict(list)
        by_d_r_n = defaultdict(list)
        for r in results_list:
            by_d_r[(r["distance"], r["rounds"])].append(r[ler_key])
            by_d_r_n[(r["distance"], r["rounds"])].append(r[shots_key])

        averaged = {}
        for (d, r), values in by_d_r.items():
            p = np.mean(values)
            n_total = np.sum(by_d_r_n[(d, r)])
            std = np.sqrt(p * (1 - p) / n_total)  # binomial std
            averaged[(d, r)] = {"mean": p, "std": std, "n": n_total}
        return averaged

    pij_real_avg = average_results(results, "ler_real", "shots_real")
    pij_sim_avg = average_results(results, "ler_simulated", "shots_sim")

    # Google results
    google_by_d_r = defaultdict(list)
    google_by_d_r_n = defaultdict(list)
    for r in google_results:
        google_by_d_r[(r["distance"], r["rounds"])].append(r["ler"])
        google_by_d_r_n[(r["distance"], r["rounds"])].append(r["shots"])
    google_avg = {}
    for (d, r), values in google_by_d_r.items():
        p = np.mean(values)
        n_total = np.sum(google_by_d_r_n[(d, r)])
        std = np.sqrt(p * (1 - p) / n_total)
        google_avg[(d, r)] = {"mean": p, "std": std, "n": n_total}

    # Organize by distance
    def organize_by_d(averaged):
        by_d = defaultdict(dict)
        for (d, r), stats in averaged.items():
            by_d[d][r] = stats
        return by_d

    pij_real_by_d = organize_by_d(pij_real_avg)
    pij_sim_by_d = organize_by_d(pij_sim_avg)
    google_by_d = organize_by_d(google_avg)

    # =========================================================================
    # Plot: P_L vs rounds - all three methods
    # =========================================================================
    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    dist_colors = {3: "#66c2a5", 5: "#fc8d62", 7: "#8da0cb"}
    all_distances = sorted(set(pij_real_by_d.keys()) | set(google_by_d.keys()))

    for d in all_distances:
        color = dist_colors.get(d, "gray")

        # Google RL decoder (solid line, circles)
        if d in google_by_d:
            rd = google_by_d[d]
            rounds = np.array(sorted(rd.keys()))
            p_log = np.array([rd[r]["mean"] for r in rounds])
            p_std = np.array([rd[r]["std"] for r in rounds])
            ax.errorbar(rounds, p_log, yerr=p_std, marker='o', color=color,
                       linestyle='-', label=f"d={d} Google RL", markersize=4,
                       linewidth=1, capsize=2, alpha=1)

        # p_ij model - real data (dashed line, squares)
        if d in pij_real_by_d:
            rd = pij_real_by_d[d]
            rounds = np.array(sorted(rd.keys()))
            p_log = np.array([rd[r]["mean"] for r in rounds])
            p_std = np.array([rd[r]["std"] for r in rounds])
            ax.errorbar(rounds, p_log, yerr=p_std, marker='s', color=color,
                       linestyle='--', label=f"d={d} $p_{{ij}}$ (real)", markersize=4,
                       linewidth=1, capsize=2, alpha=1)

        # p_ij model - simulated (dotted line, triangles)
        if d in pij_sim_by_d:
            rd = pij_sim_by_d[d]
            rounds = np.array(sorted(rd.keys()))
            p_log = np.array([rd[r]["mean"] for r in rounds])
            p_std = np.array([rd[r]["std"] for r in rounds])
            ax.errorbar(rounds, p_log, yerr=p_std, marker='^', color=color,
                       linestyle=':', label=f"d={d} $p_{{ij}}$ (sim)", markersize=4,
                       linewidth=1, capsize=2, alpha=1)

    ax.set_xlabel("Number of rounds")
    ax.set_ylabel(r"Logical error rate $P_L$")
    ax.legend(fontsize=5, ncol=3, loc='lower right')
    ax.set_ylim(0, 0.55)

    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print(f"\nPlot saved to {args.output}", flush=True)
    plt.close()

    # =========================================================================
    # Fit and compare error per cycle
    # =========================================================================
    print("\n=== Error per cycle comparison ===", flush=True)
    print(f"{'d':<5} {'Google RL':<12} {'p_ij real':<12} {'p_ij sim':<12}", flush=True)
    print("-" * 45, flush=True)

    eps_results = []
    for d in all_distances:
        eps_google = np.nan
        eps_real = np.nan
        eps_sim = np.nan

        if d in google_by_d:
            rd = google_by_d[d]
            rounds = np.array(sorted(rd.keys()))
            p_log = np.array([rd[r]["mean"] for r in rounds])
            eps_google, _ = fit_error_per_cycle(rounds, p_log)

        if d in pij_real_by_d:
            rd = pij_real_by_d[d]
            rounds = np.array(sorted(rd.keys()))
            p_log = np.array([rd[r]["mean"] for r in rounds])
            eps_real, _ = fit_error_per_cycle(rounds, p_log)

        if d in pij_sim_by_d:
            rd = pij_sim_by_d[d]
            rounds = np.array(sorted(rd.keys()))
            p_log = np.array([rd[r]["mean"] for r in rounds])
            eps_sim, _ = fit_error_per_cycle(rounds, p_log)

        print(f"{d:<5} {eps_google:<12.6f} {eps_real:<12.6f} {eps_sim:<12.6f}", flush=True)
        eps_results.append({"d": d, "eps_google": eps_google, "eps_real": eps_real, "eps_sim": eps_sim})

    # Save fit results
    fit_file = args.output.replace(".png", "_fit_results.csv")
    with open(fit_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["d", "eps_google", "eps_real", "eps_sim"])
        writer.writeheader()
        writer.writerows(eps_results)
    print(f"\nFit results saved to {fit_file}", flush=True)

    # =========================================================================
    # Plot 2: eps vs d
    # =========================================================================
    fig2, ax2 = plt.subplots(figsize=(3, 2.5))

    distances = np.array([r["d"] for r in eps_results])
    eps_g = np.array([r["eps_google"] for r in eps_results])
    eps_r = np.array([r["eps_real"] for r in eps_results])
    eps_s = np.array([r["eps_sim"] for r in eps_results])

    ax2.semilogy(distances, eps_g, 'o-', color='#66c2a5', markersize=6,
                 linewidth=1.5, label='Google RL', alpha=1)
    ax2.semilogy(distances, eps_r, 's--', color='#fc8d62', markersize=6,
                 linewidth=1.5, label='$p_{ij}$ (real)', alpha=1)
    ax2.semilogy(distances, eps_s, '^:', color='#8da0cb', markersize=6,
                 linewidth=1.5, label='$p_{ij}$ (sim)', alpha=1)

    ax2.set_xlabel("Code distance $d$")
    ax2.set_ylabel(r"Error per cycle $\epsilon_L$")
    ax2.set_xticks(distances)
    ax2.legend(fontsize=7)

    plt.tight_layout()
    eps_plot_file = args.output.replace(".png", "_eps_vs_d.png")
    plt.savefig(eps_plot_file, dpi=300)
    print(f"Eps plot saved to {eps_plot_file}", flush=True)
    plt.close()

    print("\nDone!", flush=True)


if __name__ == "__main__":
    main()
