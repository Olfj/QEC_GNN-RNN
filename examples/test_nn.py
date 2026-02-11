"""
Evaluate GNN-RNN decoder(s) and MWPM across different round counts.

Plots logical error rate P_L vs rounds on a log-log scale, with
1-sigma error bars from the binomial distribution: std = sqrt(P_L*(1-P_L)/N).

Usage:
    python examples/test_nn.py --d 3 --p 0.001 \
        --model_last models/d3_last.pt \
        --model_error_chain models/d3_ec.pt \
        --model_mpp models/d3_mpp.pt

    # Quick sanity check:
    python examples/test_nn.py --d 3 --p 0.001 \
        --shots 1000 --batch_size 100 --rounds 5 10 20
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')
import matplotlib.colors as colors
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[
    "#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3",
    "#a6d854", "#ffd92f", "#e5c494", "#b3b3b3"])
from collections import OrderedDict

from gru_decoder import GRUDecoder
from data import Dataset, Args
from mwmp import test_mwpm


def load_model(model_path, args):
    """Load a GRUDecoder from a checkpoint."""
    decoder = GRUDecoder(args)
    decoder.load_state_dict(
        torch.load(model_path, weights_only=True, map_location=args.device)
    )
    decoder.to(args.device)
    decoder.eval()
    return decoder


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GNN-RNN decoders vs MWPM across round counts")
    parser.add_argument("--d", type=int, required=True, help="Code distance")
    parser.add_argument("--p", type=float, default=0.001, help="Error rate")
    parser.add_argument("--dt", type=int, default=2, help="Sliding window size")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--shots", type=int, default=1_000_000,
                        help="Total shots per round count (n_iter = shots // batch_size)")
    parser.add_argument("--rounds", type=int, nargs="+",
                        default=[5, 10, 20, 50, 100, 200, 500, 1000])
    parser.add_argument("--model_last", type=str, default=None,
                        help="Path to model trained with label_mode=last")
    parser.add_argument("--model_error_chain", type=str, default=None,
                        help="Path to model trained with label_mode=error_chain")
    parser.add_argument("--model_mpp", type=str, default=None,
                        help="Path to model trained with label_mode=mpp")
    parser.add_argument("--out", type=str, default=None,
                        help="Output prefix for CSV and figure (default: auto)")
    cli = parser.parse_args()

    # Determine output prefix
    out_prefix = cli.out or f"eval_d{cli.d}_p{cli.p}"
    os.makedirs("results", exist_ok=True)
    out_prefix = os.path.join("results", out_prefix)

    # Shared model args (must match training config)
    model_args = Args(
        distance=cli.d,
        error_rate=cli.p,
        t=cli.rounds[0],  # placeholder, overridden per round
        dt=cli.dt,
        label_mode="last",
        batch_size=cli.batch_size,
        embedding_features=[3, 32, 64, 128, 256, 512],
        hidden_size=512,
        n_gru_layers=4,
    )

    # Load models
    models = OrderedDict()
    for name, path in [("last", cli.model_last),
                       ("error_chain", cli.model_error_chain),
                       ("mpp", cli.model_mpp)]:
        if path is not None:
            print(f"Loading {name} model from {path}")
            models[name] = load_model(path, model_args)

    if not models:
        print("No NN models provided — evaluating MWPM only.")

    rounds_list = sorted(cli.rounds)
    n_iter = max(1, cli.shots // cli.batch_size)
    total_shots = n_iter * cli.batch_size
    print(f"d={cli.d}, p={cli.p}, dt={cli.dt}, "
          f"batch_size={cli.batch_size}, n_iter={n_iter} ({total_shots} shots/round)")
    print(f"Rounds: {rounds_list}\n")

    # Results: method -> {t: (P_L, std)}
    results = {name: {} for name in list(models.keys()) + ["MWPM"]}

    for t in rounds_list:
        print(f"--- t = {t} ---")
        dataset_args = Args(
            distance=cli.d,
            error_rate=cli.p,
            t=t,
            dt=cli.dt,
            label_mode="last",
            batch_size=cli.batch_size,
            embedding_features=[3, 32, 64, 128, 256, 512],
            hidden_size=512,
            n_gru_layers=4,
        )
        dataset = Dataset(dataset_args)

        # Evaluate NN models
        for name, decoder in models.items():
            with torch.no_grad():
                acc, std = decoder.test_model(dataset, n_iter=n_iter, verbose=False)
            acc = float(acc)
            std = float(std)
            p_l = 1 - acc
            print(f"  {name:15s}  acc={acc:.6f}  P_L={p_l:.6f} +/- {std:.6f}")
            results[name][t] = (p_l, std)

        # Evaluate MWPM
        acc_mwpm, std_mwpm = test_mwpm(dataset, n_iter=n_iter, verbose=False)
        acc_mwpm = float(acc_mwpm)
        std_mwpm = float(std_mwpm)
        p_l_mwpm = 1 - acc_mwpm
        print(f"  {'MWPM':15s}  acc={acc_mwpm:.6f}  P_L={p_l_mwpm:.6f} +/- {std_mwpm:.6f}")
        results["MWPM"][t] = (p_l_mwpm, std_mwpm)
        print()

    # --- Save CSV ---
    header = "rounds," + ",".join(
        f"{m}_PL,{m}_std" for m in results.keys()
    )
    rows = []
    for t in rounds_list:
        row = [str(t)]
        for m in results:
            if t in results[m]:
                p_l, std = results[m][t]
                row.extend([f"{p_l:.8f}", f"{std:.8f}"])
            else:
                row.extend(["", ""])
        rows.append(",".join(row))

    csv_path = out_prefix + ".csv"
    with open(csv_path, "w") as f:
        f.write(header + "\n")
        for row in rows:
            f.write(row + "\n")
    print(f"Saved CSV to {csv_path}")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 6))
    markers = {"last": "o", "error_chain": "s", "mpp": "^", "MWPM": "D"}

    for method, data in results.items():
        if not data:
            continue
        ts = sorted(data.keys())
        p_ls = np.array([data[t][0] for t in ts])
        stds = np.array([data[t][1] for t in ts])
        ax.errorbar(
            ts, p_ls, yerr=stds,
            label=method,
            marker=markers.get(method, "x"),
            capsize=3, linewidth=1.5, markersize=5,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Rounds")
    ax.set_ylabel("Logical Error Rate $P_L$")
    ax.set_title(f"d={cli.d}, p={cli.p}")
    ax.legend()
    fig.tight_layout()

    fig.savefig(f"{out_prefix}.pdf")
    print(f"Saved figure to {out_prefix}.pdf")

    plt.show()


if __name__ == "__main__":
    main()
