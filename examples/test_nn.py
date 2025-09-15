import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
from gru_decoder import GRUDecoder
from data import Dataset, Args
from mwmp import test_mwpm
import torch
import argparse
import numpy as np
# python examples/test_nn.py --d 5 --t 49 --dt 2 --p 0.001 --batch_size 100 --load_path distance3

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=int, default=5)
    parser.add_argument('--t', type=int, default=49)
    parser.add_argument('--dt', type=int, default=2)
    parser.add_argument('--p', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--load_path', type=str, default=None)

    args_cli = parser.parse_args()
    load_path = args_cli.load_path

    args = Args(
        distance=args_cli.d,
        error_rate=args_cli.p,
        t=args_cli.t,
        dt=args_cli.dt,
        sliding=True,
        batch_size=args_cli.batch_size,
        embedding_features=[3, 32, 64, 128, 256, 512],
        hidden_size=512,
        n_gru_layers=4,
        seed=42 
    )

    decoder = GRUDecoder(args)
    if load_path is not None:
        decoder.load_state_dict(torch.load("./models/" + load_path + ".pt", weights_only=True,  map_location=args.device))
    n_iter = 1000
    decoder.to(args.device)  # Move model to MPS or appropriate device
    accuracies = []
    for t in [250]:
        print('Starting with t=',t)
        args = Args(
            distance=args_cli.d,
            error_rate=args_cli.p,
            t=t,
            dt=args_cli.dt,
            sliding=True,
            batch_size=args_cli.batch_size,
            embedding_features=[3, 32, 64, 128, 256, 512],
            hidden_size=512,
            n_gru_layers=4)
        acc, std = decoder.extract_epsilon(Dataset(args), n_iter=n_iter)
        accuracies.append(acc)
    # np.savetxt(f"results_{load_path}_p_{args_cli.p}.csv", accuracies)