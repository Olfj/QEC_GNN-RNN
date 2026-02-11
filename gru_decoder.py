import torch, time, os
import torch.nn as nn
from data import Dataset
from args import Args
from utils import GraphConvLayer, TrainingLogger, group, standard_deviation
from torch_geometric.nn import global_mean_pool
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
from copy import deepcopy
import wandb
import numpy as np
os.environ["WANDB_SILENT"] = "True"

class GRUDecoder(nn.Module):
    """
    A QEC decoder combining a GNN and an RNN.
    """
    def __init__(self, args: Args):
        super().__init__()
        self.args = args
        self.g_max = args.t - args.dt + 2

        features = list(zip(args.embedding_features[:-1], args.embedding_features[1:]))
        self.embedding = nn.ModuleList([GraphConvLayer(a, b) for a, b in features])

        self.empty_embedding = nn.Parameter(torch.zeros(args.embedding_features[-1]))

        self.rnn = nn.GRU(
            args.embedding_features[-1],
            args.hidden_size, num_layers=args.n_gru_layers,
            batch_first=True
        )

        self.decoder = nn.Sequential(
            nn.Linear(args.hidden_size, 1),
            nn.Sigmoid()
        )

    def embed(self, x, edge_index, edge_attr, batch_labels):
        for layer in self.embedding:
            x = layer(x, edge_index, edge_attr)
        return global_mean_pool(x, batch_labels)

    def forward(self, x, edge_index, edge_attr, batch_labels, label_map):
        x = self.embed(x, edge_index, edge_attr, batch_labels)
        B = int(label_map[:, 0].max().item()) + 1
        x = group(x, label_map, B, self.g_max, self.empty_embedding)
        # x shape: [B, g_max, embed_dim] — fixed size, no packing

        out, h = self.rnn(x)
        # out shape: [B, g_max, hidden_size]

        predictions = self.decoder(out).squeeze(-1)
        # predictions shape: [B, g_max]

        final_prediction = self.decoder(h[-1])
        # final_prediction shape: [B, 1]

        return predictions, final_prediction



    def train_model(
            self, 
            logger: TrainingLogger | None = None, 
            save: str | None = None
        ) -> None:
        local_log = isinstance(logger, TrainingLogger)
        best_model = self.state_dict()

        if self.args.log_wandb:
            wandb.init(project=self.args.wandb_project, name=save, config=self.args)

        if local_log:
            logger.on_training_begin(self.args)
        
        self.train()
        dataset = Dataset(self.args)

        # Compute MWPM baseline accuracy once for wandb reference line.
        # Adaptively samples until std/P_L < 1%, capped at max_shots.
        mwpm_accuracy = None
        if self.args.log_wandb:
            import pymatching
            mwpm_args = deepcopy(self.args)
            mwpm_args.label_mode = "last"
            mwpm_dataset = Dataset(mwpm_args)
            dem = mwpm_dataset.circuits[0].detector_error_model(decompose_errors=True)
            matcher = pymatching.Matching.from_detector_error_model(dem)

            total_correct = 0
            total_shots = 0
            max_shots = 10_000_000
            target_rel_std = 0.01
            while total_shots < max_shots:
                det_events, flips = mwpm_dataset.sample_syndromes(0)
                preds = matcher.decode_batch(det_events)
                total_correct += int(np.sum(preds == flips))
                total_shots += mwpm_args.batch_size
                p_l = 1 - total_correct / total_shots
                if p_l > 0:
                    rel_std = np.sqrt((1 - p_l) / (p_l * total_shots))
                    if rel_std < target_rel_std:
                        break

            mwpm_accuracy = total_correct / total_shots
            p_l = 1 - mwpm_accuracy
            std = np.sqrt(p_l * (1 - p_l) / total_shots)
            print(f"MWPM baseline: acc={mwpm_accuracy:.6f}, "
                  f"P_L={p_l:.6f} +/- {std:.6f} ({total_shots} shots)")
            del mwpm_dataset

        optim = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        schedule = lambda epoch: max(0.95 ** epoch, self.args.min_lr / self.args.lr)
        scheduler = LambdaLR(optim, lr_lambda=schedule)
        best_accuracy = 0
        
        for i in range(1, self.args.n_epochs + 1):
            if local_log:
                logger.on_epoch_begin(i)
        
            epoch_loss = 0
            epoch_acc = 0
            data_time = 0
            model_time = 0
        
            for _ in range(self.args.n_batches):
                optim.zero_grad()

                t0 = time.perf_counter()
                x, edge_index, batch_labels, label_map, edge_attr, last_label, flips_full = dataset.generate_batch()

                t1 = time.perf_counter()
                # out shape: [B, g_max] — fixed size, predictions for all chunks
                out, final_prediction = self.forward(x, edge_index, edge_attr, batch_labels, label_map)

                if self.args.label_mode != "last":
                    loss_raw = nn.functional.binary_cross_entropy(out, flips_full, reduction='none')
                    if self.args.weight_last:
                        weights = torch.ones_like(loss_raw)
                        weights[:, -1] = self.args.t
                        loss_raw = loss_raw * weights
                    loss = loss_raw.mean()
                else:
                    loss = nn.functional.binary_cross_entropy(final_prediction, last_label)

                loss.backward()
                optim.step()

                t2 = time.perf_counter()

                data_time += t1 - t0
                model_time += t2 - t1
                epoch_loss += loss.item()
                epoch_acc += (torch.sum(torch.round(final_prediction) == last_label) / torch.numel(last_label)).item()
            epoch_loss /= self.args.n_batches
            epoch_acc /= self.args.n_batches

            metrics = {
                "loss":  epoch_loss,
                "accuracy": epoch_acc,
                "lr": scheduler.get_last_lr()[0],
                "data_time": data_time,
                "model_time": model_time
            }

            if self.args.log_wandb:
                metrics["mwpm_accuracy"] = mwpm_accuracy
                wandb.log(metrics)
            if local_log:
                logger.on_epoch_end(logs=metrics)

            if epoch_acc > best_accuracy:
                best_accuracy = epoch_acc
                if save:
                    os.makedirs("./models", exist_ok=True)
                    torch.save(self.state_dict(), f"./models/{save}.pt")
        
            scheduler.step()
            
        if local_log:
            logger.on_training_end()


    def test_model(self, dataset: Dataset, n_iter=1000, verbose=True):
        """
        Evaluates the model by feeding n_iter batches to the decoder and
        calculating the mean and standard deviation of the accuracy.
        """
        self.eval()
        accuracy_list = torch.zeros(n_iter)
        data_time, model_time = 0, 0
        for i in tqdm(range(n_iter), disable=verbose):
            t0 = time.perf_counter()
            x, edge_index, batch_labels, label_map, edge_attr, last_label, flips_full = dataset.generate_batch()
            t1 = time.perf_counter()
            out, final_prediction = self.forward(x, edge_index, edge_attr, batch_labels, label_map)
            t2 = time.perf_counter()
            accuracy_list[i] = (torch.sum(torch.round(final_prediction) == last_label) / torch.numel(last_label)).item()
            data_time += t1 - t0
            model_time += t2 - t1
        accuracy = accuracy_list.mean()
        std = standard_deviation(accuracy, n_iter * dataset.batch_size)
        if verbose:
            print(f"Accuracy: {accuracy:.4f}, data time = {data_time:.3f}, model time = {model_time:.3f}")
        return accuracy, std
