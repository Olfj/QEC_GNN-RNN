import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
from gru_decoder import GRUDecoder
from data import Args
from utils import TrainingLogger
import torch
from datetime import datetime
import argparse
# python examples/train_nn.py --d 5 --p 0.001 --t 50 --dt 2 --batch_size 32 --n_batches 10 --n_epochs 2
# python examples/train_nn.py --d 3 --p 0.005 --t 10 --dt 2 --label_mode mpp --note test_run
# python examples/train_nn.py --d 5 --p 0.001 --t 50 --dt 2 --label_mode error_chain --load_path my_model
# python examples/train_nn.py --d 5 --p 0.001 --t 50 --dt 2 --wandb --wandb_project GNN-RNN-mpp
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=int, default=5)
    parser.add_argument('--p', type=float, default=0.001)
    parser.add_argument('--t', type=int, default=50)
    parser.add_argument('--dt', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--n_batches', type=int, default=256)
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--label_mode', type=str, default='last', choices=['last', 'error_chain', 'mpp'])
    parser.add_argument('--note', type=str, default='')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='GNN-RNN-google')

    args_cli = parser.parse_args()

    d = args_cli.d
    p = args_cli.p
    t = args_cli.t
    dt = args_cli.dt
    load_path = args_cli.load_path or None  # treat empty string as None
    label_mode = args_cli.label_mode

    args = Args(
        distance=d,
        error_rate=p,
        t=t,
        dt=dt,

        label_mode=label_mode,
        batch_size=args_cli.batch_size,
        n_batches=args_cli.n_batches,
        n_epochs=args_cli.n_epochs,
        embedding_features=[3, 32, 64, 128, 256, 512],
        hidden_size=512,
        n_gru_layers=4,
        log_wandb=args_cli.wandb,
        wandb_project=args_cli.wandb_project
    )
    current_datetime = datetime.now().strftime("%y%m%d_%H%M%S")
    job_id = os.environ.get("SLURM_JOB_ID", "")
    model_name = f"d{d}_p{p}_t{t}_dt{dt}_{label_mode}_{current_datetime}"
    if job_id:
        model_name += f"_{job_id}"
    if args_cli.note:
        model_name += f"_{args_cli.note}"

    decoder = GRUDecoder(args)
    if load_path is not None:
        decoder.load_state_dict(torch.load("./models/" + load_path + ".pt", weights_only=True,  map_location=args.device))
        run_id = load_path[-6:]
        model_name = model_name + '_load_' + run_id
    logger = TrainingLogger(logfile=model_name, statsfile=model_name)
    decoder.to(args.device)  # Move model to MPS or appropriate device
    decoder = torch.compile(decoder)  # Then compile
    decoder.train_model(logger, save=model_name)
