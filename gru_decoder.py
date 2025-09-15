import torch, time, os
import torch.nn as nn 
from data import Dataset
from args import Args
from utils import GraphConvLayer, TrainingLogger, group, standard_deviation
from torch_geometric.nn import global_mean_pool
from torch.nn.utils.rnn import pad_packed_sequence
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
from copy import deepcopy
import wandb
import numpy as np
from scipy.stats import linregress
os.environ["WANDB_SILENT"] = "True"

class GRUDecoder(nn.Module):
    """
    A QEC decoder combining a GNN and an RNN.
    """
    def __init__(self, args: Args):
        super().__init__()
        self.args = args
        
        features = list(zip(args.embedding_features[:-1], args.embedding_features[1:]))
        self.embedding =  nn.ModuleList([GraphConvLayer(a, b) for a, b in features])

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
        # Run embedding + group
        x = self.embed(x, edge_index, edge_attr, batch_labels)
        x = group(x, label_map)

        # GRU output: out_packed is packed sequence, h is final hidden state
        out_packed, h = self.rnn(x)

        # Unpack the output to get predictions over all chunks
        # out shape: [batch_size, g_actual, hidden_size]
        out, _ = pad_packed_sequence(out_packed, batch_first=True)

        # Apply decoder to get chunkwise predictions (e.g. for time-resolved loss)
        # predictions shape: [batch_size, g_actual]
        predictions = self.decoder(out).squeeze(-1)

        # Get final prediction from the last hidden layer for each sample
        # h shape: [n_layers, batch_size, hidden_size]
        # h[-1] is the final layer's output → shape: [batch_size, hidden_size]
        # final_prediction shape: [batch_size, 1]
        final_prediction = self.decoder(h[-1])

        # Return both time-resolved and final prediction
        return predictions, final_prediction



    def train_model(
            self, 
            logger: TrainingLogger | None = None, 
            save: str | None = None
        ) -> None:
        local_log = isinstance(logger, TrainingLogger)
        best_model = self.state_dict()

        if self.args.log_wandb:
            wandb.init(project="GNN-RNN-google", name = save, config = self.args)

        if local_log:
            logger.on_training_begin(self.args)
        
        self.train()
        dataset = Dataset(self.args)
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
                x, edge_index, batch_labels, label_map, edge_attr, aligned_flips, lengths, last_label, flips_full = dataset.generate_batch()

                t1 = time.perf_counter()
                # Forward pass through the model
                # out has shape [B, g_actual], where:
                #   B = batch size
                #   g_actual = maximum number of non-empty chunks in batch
                # (can vary between batches, <= t - dt + 2)
                out, final_prediction = self.forward(x, edge_index, edge_attr, batch_labels, label_map)
                if self.args.train_all_times:
                    # Create a boolean mask of shape [B, g_actual] indicating valid chunk positions
                    # For each batch element b, mask[b, i] = True if i < lengths[b]
                    # lengths[b] is the number of non-empty chunks for batch element b
                    mask = torch.arange(out.size(1), device=out.device)[None, :] < lengths[:, None]

                    # Compute binary cross-entropy loss for each element without reduction
                    # loss_raw has shape [B, g_actual], matching the shape of out and aligned_flips
                    loss_raw = nn.functional.binary_cross_entropy(out, aligned_flips, reduction='none')
                    # make a weight tensor: all 1's except last element is 50
                    weights = torch.ones_like(loss_raw)
                    weights[:, -1] = self.args.t
                    loss_raw = loss_raw * weights
                    # Apply the mask to zero out the loss from padded (non-existent) chunks
                    # Then compute the mean loss over all valid elements
                    loss = (loss_raw * mask).sum() / mask.sum()
                # If not training all times, we only consider the final label
                else:
                    loss = nn.functional.binary_cross_entropy(final_prediction, last_label)

                # Backpropagation and optimization step
                loss.backward()
                optim.step()
                
                t2 = time.perf_counter()
                
                # Statistics
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
            x, edge_index, batch_labels, label_map, edge_attr, aligned_flips, lengths, last_label, flips_full = dataset.generate_batch()
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
    def extract_epsilon(self, dataset: Dataset, n_iter=1000, verbose=True):
        """
        Evaluates the model by feeding n_iter batches to the decoder and 
        calculating the mean and standard deviation of the accuracy. 
        """
        self.eval()
        epsilon_list = torch.zeros(n_iter)
        data_time, model_time = 0, 0
        n_valid_batches = 0
        failure_rate_per_chunk_mean = np.zeros(self.args.t)
        for i in tqdm(range(n_iter), disable=verbose):
            t0 = time.perf_counter()
            x, edge_index, batch_labels, label_map, edge_attr, aligned_flips, lengths, last_label, flips_full = dataset.generate_batch()
            t1 = time.perf_counter() 

            out, final_prediction = self.forward(x, edge_index, edge_attr, batch_labels, label_map)
            t2 = time.perf_counter()
            # print(torch.sum(out, dim=0))
            label_map = label_map.int()
            g_max = self.args.t - self.args.dt + 2
            # Build index matrix from label_map
            idx = torch.full((self.args.batch_size, g_max), -1, dtype=torch.long)
            idx[label_map[:,0], label_map[:,1]] = torch.arange(label_map.size(0))

            # Forward-fill indices
            mask = idx != -1
            ffill = torch.where(mask, idx, torch.zeros_like(idx))
            ffill = ffill.cummax(dim=1).values    # forward fill each row

            # Gather values
            vals = torch.cat([out[b, :sum(mask[b]).item()] for b in range(self.args.batch_size)])
            out_per_chunk = vals[ffill]

            correct_per_chunk_and_shot = (torch.round(out_per_chunk) == flips_full)
            correct_per_chunk = torch.sum(correct_per_chunk_and_shot, dim = 0) / self.args.batch_size

            # extract epsilon_L:
            failure_rate_per_chunk = 1 - correct_per_chunk.cpu().numpy()
            logPL = np.log(1 - 2 * failure_rate_per_chunk)
            failure_rate_per_chunk_mean += failure_rate_per_chunk
            t = np.arange(self.args.dt - 1, self.args.t + 1)
            slope, _, _, _, _ = linregress(t, logPL)
            e_L = 0.5 * (1-np.exp(slope))
            if not np.isnan(e_L):
                epsilon_list[i] = e_L
                n_valid_batches += 1
            data_time += t1 - t0
            model_time += t2 - t1
        print(np.array2string(failure_rate_per_chunk_mean / n_iter, separator=", "))
        epsilon = epsilon_list.mean()
        std = standard_deviation(epsilon, n_iter * dataset.batch_size)
        if verbose:
            print(f"Epsilon_L: {epsilon:.6f}, no.valid batches: {n_valid_batches}, data time = {data_time:.3f}, model time = {model_time:.3f}")
        return epsilon, std
