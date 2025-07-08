import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
from gru_decoder import GRUDecoder
from data import Args
from utils import TrainingLogger
import torch
from datetime import datetime
import argparse
# python examples/train_nn.py --d 5 --t 49 --dt 2 --batch_size 32 --n_batches 10 --n_epochs 2 --load_path distance3

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=int, default=5)
    parser.add_argument('--t', type=int, default=49)
    parser.add_argument('--dt', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--n_batches', type=int, default=256)
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--load_path', type=str, default=None)

    args_cli = parser.parse_args()

    d = args_cli.d
    t = args_cli.t
    dt = args_cli.dt
    load_path = args_cli.load_path

    args = Args(
        distance=d,
        error_rates=[0.001, 0.002, 0.003, 0.004, 0.005],
        t=[t],
        dt=dt,
        sliding=True,
        train_all_times = False,
        batch_size=args_cli.batch_size,
        n_batches=args_cli.n_batches,
        n_epochs=args_cli.n_epochs,
        embedding_features=[5, 32, 64, 128, 256],
        hidden_size=128,
        n_gru_layers=4,
        log_wandb=True
    )
    current_datetime = datetime.now().strftime("%y%m%d_%H%M%S")
    model_name = 'd' + str(d) + '_t' + str(t) + '_dt' + str(dt) + '_' + current_datetime

    decoder = GRUDecoder(args)
    if load_path is not None:
        decoder.load_state_dict(torch.load("./models/" + load_path + ".pt", weights_only=True,  map_location=args.device))
        run_id = load_path[-6:]
        model_name = model_name + '_load_' + run_id
    logger = TrainingLogger(logfile=model_name, statsfile=model_name)
    decoder.to(args.device)  # Move model to MPS or appropriate device
    decoder = torch.compile(decoder)  # Then compile
    decoder.train_model(logger, save=model_name)
