import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
from gru_decoder import GRUDecoder
from data import Args
import torch

if __name__ == "__main__":
    """
    Distance 3 model. 
    Trained on t = 99, dt = 5, with error rates [0.001, 0.002, 0.003, 0.004, 0.005]
    """
    args = Args(
        distance=3,
        error_rates=[0.001],
        t=[99],
        dt=5,

        batch_size=2048,
        embedding_features=[5, 32, 64, 128, 256],
        hidden_size=128,
        n_layers=4,
        seed=42
    )
    decoder_d3 = GRUDecoder(args)
    decoder_d3.load_state_dict(torch.load("./models/distance3.pt", weights_only=True))

    """
    Distance 5 model. 
    Trained on t = 49, dt = 2, with error rates [0.001, 0.002, 0.003, 0.004, 0.005]
    """
    args = Args(
        distance=5,
        error_rates=[0.001],
        t=[99],
        dt=2,

        batch_size=2048,
        embedding_features=[5, 32, 64, 128, 256],
        hidden_size=128,
        n_layers=4,
        seed=42
    )
    decoder_d5 = GRUDecoder(args)
    decoder_d5.load_state_dict(torch.load("./models/distance5.pt", weights_only=True))

    """
    Distance 7 model. 
    Trained on t = 49, dt = 2, with error rates [0.001, 0.002, 0.003, 0.004, 0.005]
    """
    args = Args(
        distance=7,
        error_rates=[0.001],
        t=[99],
        dt=2,

        batch_size=2048,
        embedding_features=[5, 32, 64, 128, 256, 512],
        hidden_size=256,
        n_layers=4,
        seed=42
    )
    decoder_d7 = GRUDecoder(args)
    decoder_d7.load_state_dict(torch.load("./models/distance7.pt", weights_only=True))
