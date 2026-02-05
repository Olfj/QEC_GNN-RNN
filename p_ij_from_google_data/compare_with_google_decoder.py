"""
Compare p_ij model results with Google's correlated_matching_decoder_with_rl_optimized_prior.
"""
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')
plt.rcParams['figure.dpi'] = 300
import matplotlib.colors as colors
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[
    "#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3",
    "#a6d854", "#ffd92f", "#e5c494", "#b3b3b3"])
from collections import defaultdict
from pathlib import Path
import stim
import json
import csv


def get_google_decoder_results(base_dir: Path, max_patches: int = 3, skip_rounds: bool = True):
    """
    Read logical error rates from Google's decoder results.
    """
    results = []

    # Group patches by distance
    patches_by_d = defaultdict(list)
    for patch_dir in sorted(base_dir.iterdir()):
        if patch_dir.is_dir() and patch_dir.name.startswith("d"):
            d = int(patch_dir.name.split("_")[0][1:])
            patches_by_d[d].append(patch_dir)

    for d in sorted(patches_by_d.keys()):
        patches = patches_by_d[d][:max_patches]

        for patch_dir in patches:
            z_dir = patch_dir / "Z"
            if not z_dir.exists():
                continue

            rounds_dirs = sorted(z_dir.iterdir(),
                                key=lambda x: int(x.name[1:]) if x.name[1:].isdigit() else 0)

            if skip_rounds:
                rounds_dirs = rounds_dirs[::2]

            for rounds_dir in rounds_dirs:
                if not rounds_dir.is_dir() or not rounds_dir.name.startswith("r"):
                    continue

                # Read metadata
                meta_file = rounds_dir / "metadata.json"
                if not meta_file.exists():
                    continue

                with open(meta_file) as f:
                    meta = json.load(f)

                distance = meta["distance"]
                rounds = meta["rounds"]
                shots = meta["shots"]

                # Read actual observable flips
                actual_file = rounds_dir / "obs_flips_actual.b8"
                if not actual_file.exists():
                    continue

                actual = stim.read_shot_data_file(
                    path=str(actual_file),
                    format="b8",
                    bit_packed=False,
                    num_observables=1
                )

                # Read predicted observable flips from Google decoder
                pred_file = rounds_dir / "decoding_results" / "correlated_matching_decoder_with_rl_optimized_prior" / "obs_flips_predicted.b8"
                if not pred_file.exists():
                    continue

                predicted = stim.read_shot_data_file(
                    path=str(pred_file),
                    format="b8",
                    bit_packed=False,
                    num_observables=1
                )

                # Compute logical error rate
                errors = np.any(actual != predicted, axis=1)
                ler = np.mean(errors)

                results.append({
                    "distance": distance,
                    "rounds": rounds,
                    "logical_error_rate": ler,
                    "shots": shots
                })

                print(f"  Google decoder d={distance}, r={rounds}: LER={ler:.4f}", flush=True)

    return results


def parse_pij_log(log_path: str):
    """Parse the p_ij model log file."""
    import re
    results = []

    with open(log_path, 'r') as f:
        lines = f.readlines()

    current_d = None
    current_r = None

    for line in lines:
        match_proc = re.search(r'Processing d=(\d+), r=(\d+)', line)
        if match_proc:
            current_d = int(match_proc.group(1))
            current_r = int(match_proc.group(2))
            continue

        match_result = re.search(r'Logical error rate: ([\d.]+)', line)
        if match_result and current_d is not None:
            ler = float(match_result.group(1))
            results.append({
                'distance': current_d,
                'rounds': current_r,
                'logical_error_rate': ler
            })

    return results


def fit_error_per_cycle(rounds: np.ndarray, p_logical: np.ndarray):
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
    base_dir = Path("./2024_google_105Q_surface_code_d3_d5_d7")
    output_base = "figures/comparison"

    # Get Google decoder results
    print("Reading Google decoder results...", flush=True)
    google_results = get_google_decoder_results(base_dir, max_patches=3, skip_rounds=True)

    # Get p_ij model results from log
    print("\nReading p_ij model results...", flush=True)
    pij_results = parse_pij_log("replicate_fig1c.log")

    # Average results by (d, r) with binomial error bars
    def average_results(results, default_n=50000):
        by_d_r = defaultdict(list)
        by_d_r_n = defaultdict(list)
        for r in results:
            by_d_r[(r["distance"], r["rounds"])].append(r["logical_error_rate"])
            by_d_r_n[(r["distance"], r["rounds"])].append(r.get("shots", default_n))

        averaged = {}
        for (d, r), values in by_d_r.items():
            p = np.mean(values)
            n_total = np.sum(by_d_r_n[(d, r)])  # total shots across all patches
            # Binomial standard deviation: sqrt(p*(1-p)/n)
            std = np.sqrt(p * (1 - p) / n_total)
            averaged[(d, r)] = {
                "mean": p,
                "std": std,
                "n": n_total
            }
        return averaged

    google_avg = average_results(google_results)
    pij_avg = average_results(pij_results, default_n=50000)

    # Organize by distance
    def organize_by_d(averaged):
        by_d = defaultdict(dict)
        for (d, r), stats in averaged.items():
            by_d[d][r] = stats
        return by_d

    google_by_d = organize_by_d(google_avg)
    pij_by_d = organize_by_d(pij_avg)

    # =========================================================================
    # Plot 1: P_L vs rounds comparison
    # =========================================================================
    fig1, ax1 = plt.subplots(figsize=(4, 3))

    dist_colors = {3: "#66c2a5", 5: "#fc8d62", 7: "#8da0cb"}

    for d in sorted(set(google_by_d.keys()) | set(pij_by_d.keys())):
        color = dist_colors.get(d, "gray")

        # Google decoder (solid line)
        if d in google_by_d:
            rounds_dict = google_by_d[d]
            rounds = np.array(sorted(rounds_dict.keys()))
            p_logical = np.array([rounds_dict[r]["mean"] for r in rounds])
            p_std = np.array([rounds_dict[r]["std"] for r in rounds])
            ax1.errorbar(rounds, p_logical, yerr=p_std,
                        marker='o', color=color, linestyle='-',
                        label=f"d={d} Google RL", markersize=4, linewidth=1, capsize=2, alpha=1)

        # p_ij model (dashed line)
        if d in pij_by_d:
            rounds_dict = pij_by_d[d]
            rounds = np.array(sorted(rounds_dict.keys()))
            p_logical = np.array([rounds_dict[r]["mean"] for r in rounds])
            p_std = np.array([rounds_dict[r]["std"] for r in rounds])
            ax1.errorbar(rounds, p_logical, yerr=p_std,
                        marker='s', color=color, linestyle='--',
                        label=f"d={d} $p_{{ij}}$ model", markersize=4, linewidth=1, capsize=2, alpha=1)

    ax1.set_xlabel("Number of rounds")
    ax1.set_ylabel(r"Logical error rate $P_L$")
    ax1.legend(fontsize=6, ncol=2, loc='lower right')
    ax1.set_ylim(0, 0.55)

    plt.tight_layout()
    plt.savefig(f"{output_base}_p_logical_vs_rounds.png", dpi=300)
    print(f"\nPlot 1 saved to {output_base}_p_logical_vs_rounds.png", flush=True)
    plt.close()

    # =========================================================================
    # Compute and compare eps_L
    # =========================================================================
    print("\n=== Error per cycle comparison ===", flush=True)

    eps_google = {}
    eps_pij = {}

    for d in sorted(set(google_by_d.keys()) | set(pij_by_d.keys())):
        if d in google_by_d:
            rounds_dict = google_by_d[d]
            rounds = np.array(sorted(rounds_dict.keys()))
            p_logical = np.array([rounds_dict[r]["mean"] for r in rounds])
            eps, _ = fit_error_per_cycle(rounds, p_logical)
            eps_google[d] = eps

        if d in pij_by_d:
            rounds_dict = pij_by_d[d]
            rounds = np.array(sorted(rounds_dict.keys()))
            p_logical = np.array([rounds_dict[r]["mean"] for r in rounds])
            eps, _ = fit_error_per_cycle(rounds, p_logical)
            eps_pij[d] = eps

    print(f"\n{'d':<5} {'Google RL':<12} {'p_ij model':<12} {'Ratio':<10}")
    print("-" * 40)
    for d in sorted(set(eps_google.keys()) | set(eps_pij.keys())):
        g = eps_google.get(d, np.nan)
        p = eps_pij.get(d, np.nan)
        ratio = p / g if g > 0 else np.nan
        print(f"{d:<5} {g:<12.6f} {p:<12.6f} {ratio:<10.2f}")

    # =========================================================================
    # Plot 2: eps_L vs distance comparison
    # =========================================================================
    fig2, ax2 = plt.subplots(figsize=(3, 2.5))

    distances = np.array(sorted(set(eps_google.keys()) | set(eps_pij.keys())))

    # Google decoder
    eps_g_vals = np.array([eps_google.get(d, np.nan) for d in distances])
    valid_g = ~np.isnan(eps_g_vals)
    ax2.semilogy(distances[valid_g], eps_g_vals[valid_g], 'o-',
                 color='#66c2a5', markersize=6, linewidth=1.5, label='Google RL decoder', alpha=1)

    # p_ij model
    eps_p_vals = np.array([eps_pij.get(d, np.nan) for d in distances])
    valid_p = ~np.isnan(eps_p_vals)
    ax2.semilogy(distances[valid_p], eps_p_vals[valid_p], 's--',
                 color='#fc8d62', markersize=6, linewidth=1.5, label='$p_{ij}$ model', alpha=1)

    ax2.set_xlabel("Code distance $d$")
    ax2.set_ylabel(r"Error per cycle $\epsilon_L$")
    ax2.set_xticks(distances)
    ax2.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(f"{output_base}_eps_vs_d.png", dpi=300)
    print(f"\nPlot 2 saved to {output_base}_eps_vs_d.png", flush=True)
    plt.close()

    # Save comparison data
    with open(f"{output_base}_data.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["distance", "eps_google_rl", "eps_pij_model", "ratio"])
        for d in distances:
            g = eps_google.get(d, np.nan)
            p = eps_pij.get(d, np.nan)
            ratio = p / g if g > 0 else np.nan
            writer.writerow([d, g, p, ratio])

    print(f"\nData saved to {output_base}_data.csv", flush=True)
    print("\nDone!", flush=True)


if __name__ == "__main__":
    main()
